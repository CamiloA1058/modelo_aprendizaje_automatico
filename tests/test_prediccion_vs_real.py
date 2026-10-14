"""Tabla de prediccion vs realidad por producto (ventas_forecast.prediccion_vs_real):
columnas, error, clase real, acierto y resumen mensual."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import prediccion_vs_real as pr  # noqa: E402
from ventas_forecast.model import ETIQUETA_SIN_RECOMENDACION  # noqa: E402

SIN_REC = ETIQUETA_SIN_RECOMENDACION


def _resultados():
    return pd.DataFrame({
        "CODIGO": ["A", "B", "C", "D", "P"],
        "DESCRIPCION": ["a", "b", "c", "d", "p"],
        "TOTAL_VENDIDO": [100.0, 100.0, 100.0, 100.0, 0.0],
        "ventas_predichas": [130.0, 90.0, 120.0, np.nan, 50.0],
        "ventas_predichas_rf": [150.0, 80.0, 100.0, np.nan, np.nan],
        "prob_crecimiento": [0.8, 0.1, 0.4, 0.3, np.nan],
        "decision": ["Reforzar stock", "Reducir stock", "Mantener stock",
                     "Reducir stock", SIN_REC],
        "origen_estimacion": ["modelo"] * 4 + ["promedio"],
    })


REAL = pd.Series({"A": 200.0, "B": 0.0, "C": 105.0, "P": 40.0})  # D sin ventas


class TablaTest(unittest.TestCase):
    def setUp(self):
        self.t = pr.tabla_producto(_resultados(), REAL, "CODIGO", "DESCRIPCION",
                                   "TOTAL_VENDIDO", 1.15, 0.90)
        self.f = self.t.set_index("Código")

    def test_columnas_en_orden(self):
        self.assertEqual(list(self.t.columns), pr.COLUMNAS)

    def test_diferencia_y_error(self):
        self.assertEqual(self.f.loc["A", "Diferencia (real − predicción)"], 70.0)
        self.assertAlmostEqual(self.f.loc["A", "Error %"], 70.0 / 200.0)
        self.assertEqual(self.f.loc["C", "Real del mes (COP)"], 105.0)

    def test_error_vacio_si_real_cero(self):
        self.assertTrue(np.isnan(self.f.loc["B", "Error %"]))
        self.assertEqual(self.f.loc["B", "Diferencia (real − predicción)"], -90.0)

    def test_producto_sin_monto_deja_prediccion_vacia(self):
        self.assertTrue(np.isnan(self.f.loc["D", "Predicción combinada (COP)"]))
        self.assertTrue(np.isnan(self.f.loc["D", "Diferencia (real − predicción)"]))
        self.assertEqual(self.f.loc["D", "Real del mes (COP)"], 0.0)

    def test_clase_real_y_acierto(self):
        self.assertEqual(self.f.loc["A", "Clase real"], "Reforzar")
        self.assertEqual(self.f.loc["B", "Clase real"], "Reducir")
        self.assertEqual(self.f.loc["C", "Clase real"], "Mantener")
        self.assertEqual(self.f.loc["D", "Clase real"], "Reducir")
        self.assertEqual(self.f.loc["A", "¿Acertó la decisión?"], "Sí")
        self.assertEqual(self.f.loc["B", "¿Acertó la decisión?"], "Sí")
        self.assertEqual(self.f.loc["C", "¿Acertó la decisión?"], "Sí")
        self.assertEqual(self.f.loc["D", "¿Acertó la decisión?"], "Sí")

    def test_acierto_no(self):
        r = _resultados()
        r.loc[0, "decision"] = "Reducir stock"
        t = pr.tabla_producto(r, REAL, "CODIGO", "DESCRIPCION", "TOTAL_VENDIDO",
                              1.15, 0.90).set_index("Código")
        self.assertEqual(t.loc["A", "¿Acertó la decisión?"], "No")

    def test_promedio_sin_probabilidad_clase_ni_acierto(self):
        p = self.f.loc["P"]
        self.assertEqual(p["Origen"], "promedio")
        self.assertEqual(p["Decisión recomendada"], SIN_REC)
        self.assertTrue(np.isnan(p["Prob. crecimiento"]))
        self.assertTrue(np.isnan(p["Solo modelo (COP)"]))
        self.assertEqual(p["Clase real"], "")
        self.assertEqual(p["¿Acertó la decisión?"], "")
        self.assertEqual(p["Diferencia (real − predicción)"], -10.0)

    def test_sin_columna_origen_todo_es_modelo(self):
        r = _resultados().drop(columns="origen_estimacion").iloc[:4]
        t = pr.tabla_producto(r, REAL, "CODIGO", "DESCRIPCION", "TOTAL_VENDIDO",
                              1.15, 0.90)
        self.assertEqual(set(t["Origen"]), {"modelo"})


class ResumenTest(unittest.TestCase):
    def test_totales_aciertos_y_cobertura(self):
        t = pr.tabla_producto(_resultados(), REAL, "CODIGO", "DESCRIPCION",
                              "TOTAL_VENDIDO", 1.15, 0.90)
        r = pr.resumen_mes("2026-04", t, real_top_total=1000.0)
        self.assertEqual(r["Productos modelo"], 4)
        self.assertEqual(r["Productos promedio"], 1)
        self.assertEqual(r["Predicción modelo (COP)"], 340.0)
        self.assertEqual(r["Predicción promedio (COP)"], 50.0)
        self.assertEqual(r["Predicción total (COP)"], 390.0)
        self.assertEqual(r["Real modelo (COP)"], 305.0)      # A, B, C (con monto)
        self.assertEqual(r["Real promedio (COP)"], 40.0)
        self.assertAlmostEqual(r["Razón predicción / real"], 390.0 / 345.0)
        self.assertAlmostEqual(r["Cobertura del real top 1900"], 0.39)
        self.assertEqual(r["Decisiones acertadas"], 4)
        self.assertAlmostEqual(r["% decisiones acertadas"], 1.0)
        self.assertAlmostEqual(r["MAE modelo (COP)"], (70 + 90 + 15) / 3)
        self.assertAlmostEqual(r["MAE promedio (COP)"], 10.0)


if __name__ == "__main__":
    unittest.main()
