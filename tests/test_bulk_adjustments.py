"""
Verifica la regla genérica de detección de ajustes masivos de inventario
(remove_bulk_adjustments, src/ventas_forecast/data/cleaning.py) y su
integración en SalesForecastModel.load_and_clean().

Caso confirmado por el negocio (2026-09-23): el 19/02/2026, 1.761 productos
DISTINTOS registran exactamente 12 unidades a 111 COP (1.332 COP) cada uno,
en una sola fila por producto — el propietario confirmó que es una carga o
ajuste de inventario, no una venta real (es estadísticamente imposible que
tantos productos sin relación se vendan el mismo día con la misma cantidad
y el mismo precio). La regla NO usa esa fecha como valor fijo: cualquier
combinación (fecha, cantidad, precio) compartida por al menos
min_products productos distintos se trata como ajuste masivo y se elimina.

Casos cubiertos:
  (a) elimina un grupo sintético que alcanza el umbral (>= min_products
      productos distintos con la misma fecha/cantidad/precio).
  (b) conserva un grupo por debajo del umbral.
  (c) conserva la misma cantidad/precio si ocurre en fechas DISTINTAS (cada
      fecha se evalúa por separado).
  (d) cuenta productos DISTINTOS, no filas: un producto con varias filas del
      mismo día/cantidad/precio no infla el conteo de productos.
  (e) min_products=None (en el modelo) desactiva la regla por completo.
  (f) SalesForecastModel.load_and_clean() aplica la regla justo después de
      parsear los numéricos y antes de agregar a producto-mes, y guarda el
      resumen en self.cleaning_summary.

Ejecutar:  python -m unittest tests.test_bulk_adjustments
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.data.cleaning import remove_bulk_adjustments  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402


def _synthetic_rows(n_products, fecha, ventas, precio, start_id=0):
    """n_products filas, una por producto, con la misma fecha/cantidad/precio."""
    return [
        {
            "CODIGO": start_id + i,
            "FECHA": fecha,
            "VENTAS": ventas,
            "PRECIO_PROMEDIO": precio,
        }
        for i in range(n_products)
    ]


class RemovesBulkGroupAtOrAboveThresholdTest(unittest.TestCase):
    """(a) un grupo (fecha, cantidad, precio) con >= min_products productos
    distintos se elimina por completo."""

    def setUp(self):
        # 5 productos distintos comparten fecha/cantidad/precio (el "ajuste").
        ajuste = _synthetic_rows(5, "2026-02-19", 12.0, 111.0, start_id=0)
        # Filas normales, sin relación con el ajuste (otro producto, otra
        # fecha/cantidad/precio), que deben conservarse intactas.
        normales = [
            {"CODIGO": 100, "FECHA": "2024-01-09", "VENTAS": 13.0, "PRECIO_PROMEDIO": 32000.0},
            {"CODIGO": 101, "FECHA": "2024-02-10", "VENTAS": 3.0, "PRECIO_PROMEDIO": 8000.0},
        ]
        self.df = pd.DataFrame(ajuste + normales)

    def test_all_rows_of_the_bulk_group_are_removed(self):
        df_limpio, _ = remove_bulk_adjustments(
            self.df, "FECHA", "CODIGO", "VENTAS", "PRECIO_PROMEDIO", min_products=5
        )
        self.assertEqual(len(df_limpio), 2)
        self.assertTrue((df_limpio["CODIGO"] >= 100).all())

    def test_summary_counts_rows_products_dates_and_cop_amount(self):
        _, resumen = remove_bulk_adjustments(
            self.df, "FECHA", "CODIGO", "VENTAS", "PRECIO_PROMEDIO", min_products=5
        )
        self.assertEqual(resumen["rows_removed"], 5)
        self.assertEqual(resumen["distinct_products"], 5)
        self.assertEqual(resumen["affected_dates"], ["2026-02-19"])
        self.assertEqual(resumen["cop_amount"], 5 * 12.0 * 111.0)

    def test_returns_a_copy_and_does_not_mutate_the_original(self):
        original_len = len(self.df)
        remove_bulk_adjustments(
            self.df, "FECHA", "CODIGO", "VENTAS", "PRECIO_PROMEDIO", min_products=5
        )
        self.assertEqual(len(self.df), original_len)


class KeepsGroupsBelowThresholdTest(unittest.TestCase):
    """(b) un grupo con menos de min_products productos distintos se conserva."""

    def test_group_below_threshold_is_not_removed(self):
        # Solo 4 productos comparten fecha/cantidad/precio; el umbral es 5.
        df = pd.DataFrame(_synthetic_rows(4, "2026-02-19", 12.0, 111.0))

        df_limpio, resumen = remove_bulk_adjustments(
            df, "FECHA", "CODIGO", "VENTAS", "PRECIO_PROMEDIO", min_products=5
        )

        self.assertEqual(len(df_limpio), len(df))
        self.assertEqual(resumen["rows_removed"], 0)
        self.assertEqual(resumen["distinct_products"], 0)
        self.assertEqual(resumen["affected_dates"], [])
        self.assertEqual(resumen["cop_amount"], 0.0)


class SameQtyPriceOnDifferentDatesIsKeptTest(unittest.TestCase):
    """(c) la misma cantidad/precio en fechas DISTINTAS no se combina en un
    solo grupo: cada fecha se evalúa por separado."""

    def test_split_across_dates_below_threshold_each_is_kept(self):
        # 3 productos el 2026-02-19 y otros 3 productos distintos el
        # 2026-02-20, misma cantidad/precio; ninguna fecha alcanza el
        # umbral de 5 por separado, aunque juntas sumen 6.
        dia_1 = _synthetic_rows(3, "2026-02-19", 12.0, 111.0, start_id=0)
        dia_2 = _synthetic_rows(3, "2026-02-20", 12.0, 111.0, start_id=3)
        df = pd.DataFrame(dia_1 + dia_2)

        df_limpio, resumen = remove_bulk_adjustments(
            df, "FECHA", "CODIGO", "VENTAS", "PRECIO_PROMEDIO", min_products=5
        )

        self.assertEqual(len(df_limpio), len(df))
        self.assertEqual(resumen["rows_removed"], 0)


class CountsDistinctProductsNotRowsTest(unittest.TestCase):
    """(d) varias filas del MISMO producto con igual fecha/cantidad/precio
    cuentan como UN producto, no como varias."""

    def test_repeated_rows_of_one_product_do_not_inflate_the_count(self):
        # Un solo producto (CODIGO=1) con 6 filas idénticas de fecha,
        # cantidad y precio: 6 filas pero 1 solo producto distinto.
        df = pd.DataFrame([
            {"CODIGO": 1, "FECHA": "2026-02-19", "VENTAS": 12.0, "PRECIO_PROMEDIO": 111.0}
            for _ in range(6)
        ])

        df_limpio, resumen = remove_bulk_adjustments(
            df, "FECHA", "CODIGO", "VENTAS", "PRECIO_PROMEDIO", min_products=2
        )

        # Aunque hay 6 filas (>= 2), solo hay 1 producto distinto (< 2):
        # no debe eliminarse nada.
        self.assertEqual(len(df_limpio), 6)
        self.assertEqual(resumen["rows_removed"], 0)
        self.assertEqual(resumen["distinct_products"], 0)


class LoadAndCleanAppliesBulkAdjustmentRuleTest(unittest.TestCase):
    """(e)/(f) SalesForecastModel(bulk_adjustment_min_products=...) aplica
    (o desactiva) la regla dentro de load_and_clean(), justo después de
    parsear los numéricos y antes de agregar a producto-mes, y guarda el
    resumen en self.cleaning_summary."""

    CSV_HEADER = (
        "CODIGO;DESCRIPCION;FECHA;ANIO;MES;DIA;DIA_SEMANA;VENTAS;"
        "TOTAL_VENDIDO;PRECIO_PROMEDIO;FRECUENCIA;HUBO_PROMOCION;TEMPORADA\n"
    )

    def _write_csv(self, tmp_path):
        filas = []
        # Historial normal: 3 productos con ventas en varios meses.
        for prod in range(1, 4):
            for mes, dia in ((1, "09"), (2, "10"), (3, "11")):
                filas.append(
                    f"{prod};PRODUCTO {prod};{dia}/0{mes}/2024;2024;{mes};{dia};2;"
                    f"10,00;100.000,00;10.000,00;5;0;Baja"
                )
        # Ajuste masivo sintético: 6 productos distintos, misma fecha,
        # cantidad y precio (umbral de la prueba: min_products=5).
        for prod in range(50, 56):
            filas.append(
                f"{prod};AJUSTE {prod};19/02/2026;2026;2;19;4;"
                f"12,00;1.332,00;111,00;1;0;Baja"
            )

        contenido = self.CSV_HEADER + "\n".join(filas) + "\n"
        csv_path = tmp_path / "raw.csv"
        csv_path.write_text(contenido, encoding="utf-8")
        return csv_path

    def test_enabled_removes_the_bulk_rows_and_stores_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(Path(tmp))
            model = SalesForecastModel(
                filepath=csv_path, bulk_adjustment_min_products=5
            )
            model.load_and_clean()

            self.assertIsNotNone(model.cleaning_summary)
            self.assertEqual(model.cleaning_summary["rows_removed"], 6)
            self.assertEqual(model.cleaning_summary["distinct_products"], 6)

            # Los 6 productos del ajuste no deben quedar en el histórico
            # agregado (ni siquiera como fila de venta 0 tras el calendario,
            # porque nunca tuvieron una venta real que abriera su historial).
            productos_finales = set(model.df["CODIGO"].unique())
            self.assertTrue(productos_finales.isdisjoint(range(50, 56)))

    def test_disabled_keeps_all_rows_and_summary_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(Path(tmp))
            model = SalesForecastModel(
                filepath=csv_path, bulk_adjustment_min_products=None
            )
            model.load_and_clean()

            self.assertIsNone(model.cleaning_summary)
            productos_finales = set(model.df["CODIGO"].unique())
            self.assertTrue(set(range(50, 56)).issubset(productos_finales))


if __name__ == "__main__":
    unittest.main()
