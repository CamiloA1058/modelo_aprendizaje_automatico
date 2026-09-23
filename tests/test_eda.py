"""
Verifica las funciones puras de src/ventas_forecast/eda.py, usadas por
scripts/eda.py para el análisis exploratorio de datos (fase 2 CRISP-DM):

  - data_quality_summary: nulos por columna, duplicados exactos, filas con
    id de producto nulo, número de productos y rango de fechas (crudo, antes
    de cualquier agregación).
  - monthly_totals / seasonal_profile: total por mes calendario y perfil
    estacional (promedio por número de mes 1-12 y años observados).
  - abc_classification: clasificación ABC por producto, incluida la regla
    de frontera cuando un producto hace que el acumulado CRUCE el 80 %.
  - intermittency_summary: fracción de meses en cero por producto.

Ejecutar:  python -m unittest tests.test_eda
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.eda import (  # noqa: E402
    abc_classification,
    data_quality_summary,
    intermittency_summary,
    monthly_totals,
    seasonal_profile,
)


class DataQualitySummaryTest(unittest.TestCase):
    """data_quality_summary sobre un dataframe crudo pequeño y conocido."""

    def setUp(self):
        # Fila 2 es un duplicado exacto de la fila 1; fila 3 tiene un nulo en
        # 'val'; fila 4 tiene 'id' nulo (como los 8 registros "VARIOS" reales
        # de Query_Result_V5.csv).
        self.df = pd.DataFrame({
            "id":   [1, 1, 2, np.nan],
            "anio": [2024, 2024, 2024, 2024],
            "mes":  [1, 1, 2, 3],
            "val":  [10.0, 10.0, np.nan, 5.0],
        })
        self.col_map = {"id": "id", "year": "anio", "month": "mes"}

    def test_counts_rows_and_columns(self):
        resumen = data_quality_summary(self.df, self.col_map)
        self.assertEqual(resumen["n_rows"], 4)
        self.assertEqual(resumen["n_cols"], 4)

    def test_counts_nulls_per_column(self):
        resumen = data_quality_summary(self.df, self.col_map)
        self.assertEqual(resumen["nulls"]["val"], 1)
        self.assertEqual(resumen["nulls"]["id"], 1)
        self.assertEqual(resumen["nulls"]["anio"], 0)

    def test_counts_exact_duplicate_rows(self):
        resumen = data_quality_summary(self.df, self.col_map)
        # pandas.duplicated() marca solo la SEGUNDA ocurrencia como True.
        self.assertEqual(resumen["n_duplicates"], 1)

    def test_counts_null_product_id_rows_and_product_count(self):
        resumen = data_quality_summary(self.df, self.col_map)
        self.assertEqual(resumen["n_null_id"], 1)
        # Productos distintos SIN contar el id nulo: {1, 2} -> 2.
        self.assertEqual(resumen["n_products"], 2)

    def test_date_range_and_distinct_months_from_year_month(self):
        resumen = data_quality_summary(self.df, self.col_map)
        self.assertEqual(resumen["date_min"], pd.Timestamp("2024-01-01"))
        self.assertEqual(resumen["date_max"], pd.Timestamp("2024-03-01"))
        # Combinaciones (anio, mes) distintas: (2024,1), (2024,2), (2024,3).
        self.assertEqual(resumen["n_distinct_months"], 3)


class MonthlyTotalsAndSeasonalProfileTest(unittest.TestCase):
    """monthly_totals agrega por mes calendario; seasonal_profile promedia
    por número de mes (1-12) y cuenta los años que aportaron observación."""

    def setUp(self):
        self.df = pd.DataFrame({
            "fecha": pd.to_datetime([
                "2024-01-01", "2024-02-01", "2025-01-01",
            ]),
            "valor": [10.0, 5.0, 30.0],
        })

    def test_monthly_totals_sums_per_calendar_month(self):
        mensual = monthly_totals(self.df, "fecha", "valor")
        self.assertEqual(mensual.loc[pd.Timestamp("2024-01-01")], 10.0)
        self.assertEqual(mensual.loc[pd.Timestamp("2024-02-01")], 5.0)
        self.assertEqual(mensual.loc[pd.Timestamp("2025-01-01")], 30.0)

    def test_seasonal_profile_averages_and_counts_years_observed(self):
        mensual = monthly_totals(self.df, "fecha", "valor")
        perfil = seasonal_profile(mensual)

        # Enero (mes=1) tiene datos de 2024 y 2025: promedio (10+30)/2=20,
        # 2 años observados.
        self.assertEqual(perfil.loc[1, "promedio"], 20.0)
        self.assertEqual(perfil.loc[1, "n_anios"], 2)

        # Febrero (mes=2) solo tiene datos de 2024: promedio 5, 1 año.
        self.assertEqual(perfil.loc[2, "promedio"], 5.0)
        self.assertEqual(perfil.loc[2, "n_anios"], 1)

        # Marzo (mes=3) no tiene ninguna observación: 0 años, promedio NaN.
        self.assertEqual(perfil.loc[3, "n_anios"], 0)
        self.assertTrue(pd.isna(perfil.loc[3, "promedio"]))

        # El índice cubre siempre los 12 meses del año, aunque falten datos.
        self.assertListEqual(list(perfil.index), list(range(1, 13)))


class AbcClassificationTest(unittest.TestCase):
    """Clasificación ABC con cortes a=0.80 (A) y b=0.95 (B), resto C.

    Regla de frontera DECIDIDA para este módulo: la clase de un producto se
    define por su participación ACUMULADA incluyéndolo a él mismo (A si
    acumulado <= a, B si <= b, si no C). Un producto que por sí solo hace
    que el acumulado SUPERE el 80 % (por ejemplo, un producto que concentra
    el 82 % de las ventas) queda clasificado como B, no como A: la regla
    describe el CORTE acumulado, no la importancia individual del producto.
    """

    def test_products_landing_exactly_on_cut_points(self):
        # Ventas: 500, 300, 100, 70, 30 (total 1000).
        # Acumulado: 0.50, 0.80, 0.90, 0.97, 1.00.
        df = pd.DataFrame({
            "id":  ["A", "B", "C", "D", "E"],
            "val": [500, 300, 100, 70, 30],
        })
        tabla, resumen = abc_classification(df, "id", "val")

        clases = tabla.set_index("id")["clase"]
        self.assertEqual(clases["A"], "A")
        # B cae EXACTO en el corte 0.80 -> A (regla <=, inclusive).
        self.assertEqual(clases["B"], "A")
        self.assertEqual(clases["C"], "B")
        # D cae EXACTO en 0.97 (> 0.95) -> C.
        self.assertEqual(clases["D"], "C")
        self.assertEqual(clases["E"], "C")

        self.assertEqual(resumen["A"]["n_products"], 2)
        self.assertAlmostEqual(resumen["A"]["pct_sales"], 80.0)
        self.assertEqual(resumen["B"]["n_products"], 1)
        self.assertAlmostEqual(resumen["B"]["pct_sales"], 10.0)
        self.assertEqual(resumen["C"]["n_products"], 2)
        self.assertAlmostEqual(resumen["C"]["pct_sales"], 10.0)

    def test_product_that_crosses_80_percent_alone_is_class_b(self):
        # Un solo producto concentra 82 % de las ventas: su acumulado
        # (0.82) YA supera el corte de 80 %, así que por la regla
        # documentada arriba queda en B, no en A.
        df = pd.DataFrame({
            "id":  ["X", "Y", "Z"],
            "val": [820, 100, 80],
        })
        tabla, _ = abc_classification(df, "id", "val")
        clases = tabla.set_index("id")["clase"]
        self.assertEqual(clases["X"], "B")

    def test_table_is_sorted_descending_by_value(self):
        df = pd.DataFrame({
            "id":  ["A", "B", "C"],
            "val": [10, 100, 50],
        })
        tabla, _ = abc_classification(df, "id", "val")
        self.assertListEqual(list(tabla["id"]), ["B", "C", "A"])
        self.assertTrue((tabla["cum_share"].diff().dropna() >= 0).all())


class IntermittencySummaryTest(unittest.TestCase):
    """Fracción de meses en cero, global y por producto."""

    def setUp(self):
        self.df = pd.DataFrame({
            "id":  ["A", "A", "A", "A", "B", "B", "B", "B"],
            "val": [0, 0, 10, 20, 5, 5, 5, 5],
        })

    def test_overall_zero_share(self):
        resumen = intermittency_summary(self.df, "id", "val")
        # 2 filas en cero de 8 filas totales.
        self.assertAlmostEqual(resumen["share_zero_overall"], 2 / 8)

    def test_per_product_zero_share_and_distribution(self):
        resumen = intermittency_summary(self.df, "id", "val")
        por_producto = resumen["share_zero_per_product"]
        self.assertAlmostEqual(por_producto["A"], 0.5)
        self.assertAlmostEqual(por_producto["B"], 0.0)

        esperado = pd.Series([0.5, 0.0])
        self.assertAlmostEqual(resumen["median"], esperado.median())
        self.assertAlmostEqual(resumen["q1"], esperado.quantile(0.25))
        self.assertAlmostEqual(resumen["q3"], esperado.quantile(0.75))


if __name__ == "__main__":
    unittest.main()
