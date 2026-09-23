"""
Verifica el completado de calendario mensual (SalesForecastModel._complete_calendar)
del modelo de producción, que corrige el HALLAZGO CRÍTICO de 2026-09-23: sin
completar el calendario, groupby(id).shift()/rolling() opera "por fila" (mes
con ventas anterior) en vez de "por mes calendario", porque los meses sin
ninguna venta simplemente no tienen fila en el CSV crudo.

Casos cubiertos:
  (a) un producto con un mes faltante recibe una fila con total_sold=0 y
      precio con forward-fill.
  (b) no quedan huecos de calendario: filas consecutivas de cada producto
      difieren en exactamente 1 mes.
  (c) el calendario arranca en el primer mes con ventas de cada producto y
      termina en el último mes global del dataset.
  (d) el target del mes anterior a un mes en cero es 0 y su target_class es 0
      (Reducir) en self._df_train.
  (e) lag_1 es igual al total_sold del mes CALENDARIO anterior (0 si ese mes
      no tuvo ventas).
  (f) las filas cuyo mes actual tiene total_sold=0 no entran al frame de
      entrenamiento (self._df_train).
  (g) min_months cuenta meses CON ventas, no filas de calendario.

Ejecutar:  python -m unittest tests.test_monthly_calendar
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402

# Producto 0 no vende en la posición de mes 3 (abril 2024, dentro de un
# calendario de 6 meses: enero-junio 2024). Productos 1 y 2 venden todos los
# meses, para poder comparar contra un caso sin huecos.
DROP_MONTHS = {0: [3]}


def _build_calendar_only(n_products=3, n_months=6, drop_months=DROP_MONTHS):
    """Modelo con self.df ya agregado (una fila por producto-mes, solo meses
    con ventas) pero SIN completar el calendario todavía — el estado en el
    que quedaría self.df justo antes de llamar a _complete_calendar() dentro
    de load_and_clean()."""
    model = SalesForecastModel(filepath="unused.csv")
    model.df = synthetic_history(
        n_products=n_products, n_months=n_months, drop_months=drop_months
    )
    model._ultimo_mes = model.df["fecha"].max()
    model._mes_pred = model._ultimo_mes + pd.DateOffset(months=1)
    return model


def _build_full_model(min_months=6, n_products=3, n_months=6, drop_months=DROP_MONTHS):
    """Modelo con el calendario ya completado y el pipeline corrido hasta
    prepare_split(), igual que lo haría run() tras load_and_clean()."""
    model = _build_calendar_only(
        n_products=n_products, n_months=n_months, drop_months=drop_months
    )
    model.min_months = min_months
    model.df = model._complete_calendar(model.df)
    model.build_features()
    model.cluster()
    model.prepare_split()
    return model


class CompleteCalendarFillsZeroRowsTest(unittest.TestCase):
    """(a) mes faltante -> fila con total_sold=0 y precio forward-filled."""

    def setUp(self):
        self.model = _build_calendar_only()
        self.df_completo = self.model._complete_calendar(self.model.df)

    def test_missing_month_gets_zero_sales_row(self):
        c = self.model._c
        fila_marzo = self.df_completo[
            (self.df_completo[c("id")] == 0) &
            (self.df_completo["fecha"] == pd.Timestamp("2024-03-01"))
        ]
        fila_abril = self.df_completo[
            (self.df_completo[c("id")] == 0) &
            (self.df_completo["fecha"] == pd.Timestamp("2024-04-01"))
        ]
        self.assertEqual(len(fila_abril), 1, "abril debía crearse por completado de calendario")
        self.assertEqual(fila_abril[c("total_sold")].iloc[0], 0.0)
        self.assertEqual(fila_abril[c("sales")].iloc[0], 0.0)
        self.assertEqual(fila_abril[c("frequency")].iloc[0], 0.0)

        # Forward-fill: el precio de abril (sin venta) es el último precio
        # conocido (marzo), no NaN ni cero.
        self.assertEqual(
            fila_abril[c("avg_price")].iloc[0], fila_marzo[c("avg_price")].iloc[0]
        )

    def test_row_count_increases_by_number_of_missing_months(self):
        # Solo el producto 0 tiene un mes faltante (abril): +1 fila.
        self.assertEqual(len(self.df_completo), len(self.model.df) + 1)


class CompleteCalendarNoGapsTest(unittest.TestCase):
    """(b) no quedan huecos: filas consecutivas por producto difieren en 1 mes."""

    def setUp(self):
        self.model = _build_calendar_only()
        self.df_completo = self.model._complete_calendar(self.model.df)

    def test_consecutive_rows_differ_by_exactly_one_month(self):
        c = self.model._c
        df = self.df_completo.copy()
        df["ym"] = df["fecha"].dt.year * 12 + df["fecha"].dt.month
        for _, g in df.sort_values("fecha").groupby(c("id")):
            diffs = g["ym"].diff().dropna().unique()
            self.assertTrue(
                (diffs == 1).all(),
                f"huecos de calendario para producto {g[c('id')].iloc[0]}: {diffs}",
            )


class CompleteCalendarRangeTest(unittest.TestCase):
    """(c) el calendario arranca en el primer mes con ventas y termina en el
    último mes global del dataset."""

    def setUp(self):
        self.model = _build_calendar_only()
        self.df_completo = self.model._complete_calendar(self.model.df)

    def test_calendar_starts_at_first_sale_and_ends_at_global_last_month(self):
        c = self.model._c
        primer_mes_global = pd.Timestamp("2024-01-01")
        for prod_id, g in self.df_completo.groupby(c("id")):
            self.assertEqual(g["fecha"].min(), primer_mes_global)
            self.assertEqual(g["fecha"].max(), self.model._ultimo_mes)


class TargetAndLagUseCalendarMonthsTest(unittest.TestCase):
    """(d) target/target_class del mes anterior a un mes en cero, y
    (e) lag_1 usa el mes calendario anterior (0 si no tuvo ventas)."""

    def setUp(self):
        self.model = _build_full_model()

    def test_target_of_month_before_zero_month_is_zero_and_class_is_reducir(self):
        c = self.model._c
        fila_marzo = self.model.df[
            (self.model.df[c("id")] == 0) &
            (self.model.df["fecha"] == pd.Timestamp("2024-03-01"))
        ]
        self.assertEqual(len(fila_marzo), 1)
        self.assertEqual(fila_marzo["target"].iloc[0], 0.0)

        df_train = self.model._df_train
        fila_marzo_train = df_train[
            (df_train[c("id")] == 0) &
            (df_train["fecha"] == pd.Timestamp("2024-03-01"))
        ]
        self.assertEqual(len(fila_marzo_train), 1)
        self.assertEqual(fila_marzo_train["target_class"].iloc[0], 0)

    def test_lag_1_equals_previous_calendar_month_total_sold(self):
        c = self.model._c
        fila_abril = self.model.df[
            (self.model.df[c("id")] == 0) &
            (self.model.df["fecha"] == pd.Timestamp("2024-04-01"))
        ]
        fila_mayo = self.model.df[
            (self.model.df[c("id")] == 0) &
            (self.model.df["fecha"] == pd.Timestamp("2024-05-01"))
        ]
        self.assertEqual(fila_abril[c("total_sold")].iloc[0], 0.0)
        # lag_1 de mayo debe ser el total_sold del mes calendario anterior
        # (abril, con venta 0) y NO el de marzo (el mes con ventas previo,
        # que era el comportamiento erróneo antes de completar el calendario).
        self.assertEqual(fila_mayo["lag_1"].iloc[0], 0.0)
        self.assertEqual(
            fila_mayo["lag_1"].iloc[0], fila_abril[c("total_sold")].iloc[0]
        )


class ZeroSalesRowsExcludedFromTrainingTest(unittest.TestCase):
    """(f) filas con total_sold actual = 0 no entran al frame de entrenamiento."""

    def test_zero_sales_current_month_excluded_from_train_frame(self):
        model = _build_full_model()
        c = model._c
        df_train = model._df_train

        fila_abril_en_train = df_train[
            (df_train[c("id")] == 0) &
            (df_train["fecha"] == pd.Timestamp("2024-04-01"))
        ]
        self.assertEqual(
            len(fila_abril_en_train), 0,
            "la fila con ventas actuales en 0 no debe entrar al entrenamiento",
        )


class MinMonthsCountsSalesMonthsTest(unittest.TestCase):
    """(g) min_months cuenta meses CON ventas, no filas de calendario."""

    def test_product_with_calendar_gap_excluded_when_sales_months_below_min(self):
        # Producto 0 tiene 6 filas de calendario pero solo 5 meses CON
        # ventas (abril quedó en 0). Con min_months=6, debe quedar excluido
        # de la predicción aunque tenga 6 filas de calendario.
        model = _build_full_model(min_months=6)
        c = model._c

        productos_predichos = set(model.df_predict[c("id")].unique())
        self.assertNotIn(
            0, productos_predichos,
            "producto con solo 5 meses de VENTAS (aunque 6 filas de calendario) "
            "no debía cumplir min_months=6",
        )
        # Los productos 1 y 2 vendieron los 6 meses: sí deben cumplir.
        self.assertIn(1, productos_predichos)
        self.assertIn(2, productos_predichos)

    def test_product_qualifies_once_min_months_matches_its_sales_months(self):
        model = _build_full_model(min_months=5)
        c = model._c
        productos_predichos = set(model.df_predict[c("id")].unique())
        self.assertIn(0, productos_predichos)


if __name__ == "__main__":
    unittest.main()
