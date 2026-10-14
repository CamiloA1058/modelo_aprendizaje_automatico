"""Ayudantes puros del estudio de caso fuera de muestra (origen rodante)."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ventas_forecast import case_study as cs  # noqa: E402


class TestRealClass(unittest.TestCase):
    def test_reglas_de_etiqueta(self):
        actual = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        siguiente = np.array([116.0, 115.0, 90.0, 89.0, 0.0])
        out = cs.real_class(siguiente, actual, growth=1.15, reduce=0.90)
        # 1,16 > 1,15 Reforzar; 1,15 no supera el umbral -> Mantener;
        # 0,90 no es < 0,90 -> Mantener; 0,89 y cero -> Reducir
        self.assertEqual(out.tolist(), [2, 1, 1, 0, 0])


class TestClassificationSummary(unittest.TestCase):
    def test_metricas_y_matriz(self):
        y_true = [0, 0, 1, 2, 2, 2]
        y_pred = [0, 1, 1, 2, 2, 0]
        m = cs.classification_summary(y_true, y_pred)
        self.assertEqual(m["confusion"], [[1, 1, 0], [0, 1, 0], [1, 0, 2]])
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertAlmostEqual(m["balanced_accuracy"], (0.5 + 1.0 + 2 / 3) / 3)
        self.assertIn("f1_macro", m)
        self.assertEqual(m["n"], 6)

    def test_clase_ausente_no_rompe(self):
        m = cs.classification_summary([0, 0, 0], [0, 0, 0])
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertEqual(m["confusion"][0], [3, 0, 0])

    def test_linea_base_mayoritaria(self):
        m = cs.majority_baseline_summary([0, 0, 0, 2], majority_class=0)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.5)


class TestRegressionSummary(unittest.TestCase):
    def test_mae_rmse_mase_y_razon(self):
        real = np.array([100.0, 200.0])
        pred = np.array([110.0, 150.0])
        m = cs.regression_summary(real, pred, scale=20.0)
        self.assertAlmostEqual(m["mae"], 30.0)
        self.assertAlmostEqual(m["rmse"], np.sqrt((100 + 2500) / 2))
        self.assertAlmostEqual(m["mase"], 1.5)
        self.assertAlmostEqual(m["sum_ratio"], 260 / 300)
        self.assertEqual(m["n"], 2)

    def test_escala_cero_da_nan(self):
        m = cs.regression_summary(np.array([1.0]), np.array([2.0]), scale=0)
        self.assertTrue(np.isnan(m["mase"]))


class TestNaiveScale(unittest.TestCase):
    def test_media_del_error_ingenuo_de_entrenamiento(self):
        df = pd.DataFrame({"id": ["a", "a", "b"], "total": [10.0, 20.0, 5.0],
                           "target": [15.0, 10.0, 5.0]})
        self.assertAlmostEqual(cs.naive_scale(df, "id", "total", top_ids=["a"]), 7.5)


class TestMonthlySales(unittest.TestCase):
    CSV = (
        "CODIGO;DESCRIPCION;FECHA;ANIO;MES;DIA;DIA_SEMANA;VENTAS;TOTAL_VENDIDO;"
        "PRECIO_PROMEDIO;FRECUENCIA;HUBO_PROMOCION;TEMPORADA\r\n"
        "1;A;02/04/2026;2026;4;2;3;1,00;1.000,00;1.000,00;1;0;Baja \r\n"
        "1;A;20/04/2026;2026;4;20;1;2,00;2.500,50;1.250,25;1;0;Baja \r\n"
        "T9;B;03/05/2026;2026;5;3;6;1,00;700,00;700,00;1;0;Baja \r\n"
    )

    def test_suma_por_producto_en_el_mes(self):
        s = cs.monthly_sales(io.StringIO(self.CSV), "2026-04")
        self.assertEqual(s.to_dict(), {"1": 3500.5})

    def test_codigos_siempre_texto(self):
        s = cs.monthly_sales(io.StringIO(self.CSV), "2026-05")
        self.assertEqual(list(s.index), ["T9"])

    def test_meses_disponibles(self):
        self.assertEqual(cs.months_in(io.StringIO(self.CSV)), ["2026-04", "2026-05"])


class TestCombineTraining(unittest.TestCase):
    def test_incluye_solo_holdout_anterior_al_mes(self):
        base = self.__class__.BASE
        hold = TestMonthlySales.CSV
        out = io.StringIO()
        n = cs.combine_training(io.StringIO(base), io.StringIO(hold), "2026-05", out)
        text = out.getvalue()
        self.assertEqual(n, 3)  # 1 base + 2 filas de abril
        self.assertNotIn("03/05/2026", text)
        self.assertIn("20/04/2026", text)
        self.assertTrue(text.startswith("CODIGO;DESCRIPCION;FECHA"))

    BASE = (
        "CODIGO;DESCRIPCION;FECHA;ANIO;MES;DIA;DIA_SEMANA;VENTAS;TOTAL_VENDIDO;"
        "PRECIO_PROMEDIO;FRECUENCIA;HUBO_PROMOCION;TEMPORADA\r\n"
        "9;X;27/03/2026;2026;3;27;5;1,00;10,00;10,00;1;0;Baja \r\n"
    )


class TestPooled(unittest.TestCase):
    def test_agrupa_filas_de_varios_meses(self):
        a = pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]})
        b = pd.DataFrame({"y_true": [2], "y_pred": [0]})
        m = cs.classification_summary(*cs.pool([a, b], "y_true", "y_pred"))
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["accuracy"], 2 / 3)

    def test_escala_agrupada_ponderada_por_filas(self):
        self.assertAlmostEqual(cs.pooled_scale([(10.0, 1), (20.0, 3)]), 17.5)


if __name__ == "__main__":
    unittest.main()
