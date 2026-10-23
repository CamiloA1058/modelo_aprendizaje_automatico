"""Reporte de predicción vs realidad con la acción única: columnas de acción y
monto por acción, acierto de la acción y resumen por acción.

Ejecutar:  python -m unittest tests.test_prediccion_vs_real_accion
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import prediccion_vs_real as pr  # noqa: E402
from ventas_forecast.model import ETIQUETA_SIN_RECOMENDACION  # noqa: E402


def _resultados():
    return pd.DataFrame({
        "CODIGO": ["A", "B", "C", "D", "R", "P"],
        "DESCRIPCION": ["a", "b", "c", "d", "r", "p"],
        "TOTAL_VENDIDO": [100.0, 100.0, 100.0, 100.0, 100.0, 0.0],
        "ventas_predichas": [130.0, 90.0, 120.0, np.nan, 80.0, 50.0],
        "ventas_predichas_rf": [150.0, 80.0, 100.0, np.nan, 70.0, np.nan],
        "prob_crecimiento": [0.8, 0.1, 0.4, 0.3, 0.2, np.nan],
        "decision": ["Reforzar stock", "Reducir stock", "Mantener stock", "Reducir stock",
                     "Mantener stock", ETIQUETA_SIN_RECOMENDACION],
        "origen_estimacion": ["modelo"] * 5 + ["promedio"],
        "prob_venta": [0.9, 0.2, 0.7, 0.3, 0.62, 0.3],
        "se_vendera": ["Sí", "No", "Sí", "No", "No", "No"],
        "stock_sugerido": [200.0, 0.0, 150.0, np.nan, 80.0, 60.0],
        "stock_sugerido_unidades": [2.0, 0.0, 2.0, np.nan, 1.0, 1.0],
        "accion": ["comprar_mas", "no_reponer", "mantener", "no_reponer", "revisar",
                   "por_promedio"],
        "accion_nombre": ["Comprar más", "No reponer este mes", "Mantener",
                          "No reponer este mes", "Revisar", "Por promedio"],
        "monto_si_vende": [140.0, 95.0, 110.0, np.nan, 90.0, 60.0],
        "venta_esperada": [126.0, 19.0, 77.0, np.nan, 55.8, 18.0],
        "monto_mostrado": [126.0, 0.0, 77.0, np.nan, 55.8, 0.0],
        "monto_rango": [False, False, False, False, True, False],
    })


REAL = pd.Series({"A": 200.0, "B": 0.0, "C": 105.0, "R": 50.0, "P": 40.0})  # D sin ventas


class TablaConAccionTest(unittest.TestCase):
    def setUp(self):
        self.t = pr.tabla_producto(_resultados(), REAL, "CODIGO", "DESCRIPCION",
                                   "TOTAL_VENDIDO", 1.15, 0.90)
        self.f = self.t.set_index("Código")

    def test_columnas(self):
        self.assertEqual(list(self.t.columns), pr.COLUMNAS)
        for col in ("Acción", "Monto esperado (COP)", "Rango del monto",
                    "Venta esperada (COP)", "¿Acertó la acción?", "Stock sugerido"):
            self.assertIn(col, pr.COLUMNAS)
        self.assertNotIn("Stock sugerido (70 %)", pr.COLUMNAS)

    def test_valores_por_accion(self):
        f = self.f
        self.assertEqual(f.loc["A", "Acción"], "Comprar más")
        self.assertEqual(f.loc["A", "Monto esperado (COP)"], 126.0)
        self.assertEqual(f.loc["B", "Monto esperado (COP)"], 0.0)
        self.assertEqual(f.loc["R", "Rango del monto"], "Entre 0 y 90 si se vende")
        self.assertEqual(f.loc["A", "Rango del monto"], "")
        self.assertEqual(f.loc["C", "Venta esperada (COP)"], 77.0)

    def test_acierto_de_la_accion(self):
        f = self.f
        self.assertEqual(f.loc["A", "¿Acertó la acción?"], "Sí")   # creció > 15 %
        self.assertEqual(f.loc["B", "¿Acertó la acción?"], "Sí")   # no vendió
        self.assertEqual(f.loc["D", "¿Acertó la acción?"], "Sí")   # no vendió
        self.assertEqual(f.loc["C", "¿Acertó la acción?"], "")     # Mantener: sin regla
        self.assertEqual(f.loc["P", "¿Acertó la acción?"], "")

    def test_sin_accion_las_columnas_quedan_vacias(self):
        r = _resultados().drop(columns=["accion", "accion_nombre", "monto_si_vende",
                                        "venta_esperada", "monto_mostrado", "monto_rango"])
        t = pr.tabla_producto(r, REAL, "CODIGO", "DESCRIPCION", "TOTAL_VENDIDO", 1.15, 0.90)
        self.assertTrue((t["Acción"] == "").all())
        self.assertTrue(t["Venta esperada (COP)"].isna().all())


class ResumenPorAccionTest(unittest.TestCase):
    def setUp(self):
        t = pr.tabla_producto(_resultados(), REAL, "CODIGO", "DESCRIPCION",
                              "TOTAL_VENDIDO", 1.15, 0.90)
        self.r = pr.resumen_mes("2026-04", t, real_top_total=1000.0)

    def test_conteo_y_precision_por_accion(self):
        r = self.r
        self.assertEqual(r["Comprar más (productos)"], 1)
        self.assertAlmostEqual(r["% acierto Comprar más"], 1.0)
        self.assertEqual(r["No reponer (productos)"], 2)
        self.assertAlmostEqual(r["% acierto No reponer"], 1.0)
        self.assertEqual(r["Mantener (productos)"], 1)
        self.assertEqual(r["Revisar (productos)"], 1)
        self.assertEqual(r["Por promedio (productos)"], 1)
        self.assertEqual(r["Comprar más con «se venderá: No»"], 0)

    def test_venta_esperada_e_incoherencias_de_monto(self):
        r = self.r
        self.assertAlmostEqual(r["Venta esperada (COP)"], 126 + 19 + 77 + 55.8 + 18)
        self.assertAlmostEqual(r["Razón venta esperada / real"],
                               (126 + 19 + 77 + 55.8 + 18) / (200 + 0 + 105 + 50 + 40))
        # R es "se venderá: No" pero con rango (no cuenta); B y P muestran 0.
        self.assertEqual(r["Incoherencias de monto"], 0)

    def test_comprar_mas_bajo_el_mes_anterior_cuenta_como_incoherencia(self):
        res = _resultados()
        res.loc[0, "monto_mostrado"] = 90.0
        t = pr.tabla_producto(res, REAL, "CODIGO", "DESCRIPCION", "TOTAL_VENDIDO", 1.15, 0.90)
        self.assertEqual(pr.resumen_mes("2026-04", t)["Incoherencias de monto"], 1)

    def test_grupos_suman_la_venta_esperada(self):
        t = pr.tabla_producto(_resultados(), REAL, "CODIGO", "DESCRIPCION",
                              "TOTAL_VENDIDO", 1.15, 0.90)
        g = pr.tabla_grupos_mes(t, pd.Series({"A": "X", "B": "X", "C": "Y"}))
        self.assertAlmostEqual(g.set_index("Grupo").loc["X", "Predicción (COP)"], 145.0)


if __name__ == "__main__":
    unittest.main()
