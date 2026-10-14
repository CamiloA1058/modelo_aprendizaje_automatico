"""Excel prediccion_vs_real con el stock sugerido: columnas por producto
(stock en COP y unidades, unidades reales, cubrio la demanda) y cobertura en
el resumen mensual y agrupada."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import prediccion_vs_real as pr  # noqa: E402
from ventas_forecast.model import ETIQUETA_SIN_RECOMENDACION as SIN_REC  # noqa: E402

REAL = pd.Series({"A": 200.0, "B": 0.0, "C": 105.0, "P": 40.0})
REAL_U = pd.Series({"A": 3.0, "B": 0.0, "C": 4.0, "P": 6.0})


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
        "stock_sugerido": [210.0, 90.0, 100.0, np.nan, 45.0],
        "stock_sugerido_unidades": [4.0, 2.0, 3.0, np.nan, 5.0],
    })


def _tabla(r=None):
    return pr.tabla_producto(_resultados() if r is None else r, REAL, "CODIGO",
                             "DESCRIPCION", "TOTAL_VENDIDO", 1.15, 0.90,
                             real_unid=REAL_U)


class TablaStockTest(unittest.TestCase):
    def setUp(self):
        self.t = _tabla()
        self.f = self.t.set_index("Código")

    def test_columnas_nuevas_al_final_y_en_orden(self):
        nuevas = ["Stock sugerido (70 %)", "Stock sugerido (unidades)",
                  "Unidades reales", "¿Cubrió la demanda?"]
        self.assertEqual(pr.COLUMNAS[-4:], nuevas)
        self.assertEqual(list(self.t.columns), pr.COLUMNAS)

    def test_valores(self):
        f = self.f
        self.assertEqual(f.loc["A", "Stock sugerido (70 %)"], 210.0)
        self.assertEqual(f.loc["A", "Stock sugerido (unidades)"], 4.0)
        self.assertEqual(f.loc["A", "Unidades reales"], 3.0)
        self.assertEqual(f.loc["D", "Unidades reales"], 0.0)

    def test_cubrio_la_demanda(self):
        f = self.f
        self.assertEqual(list(f["¿Cubrió la demanda?"]), ["Sí", "Sí", "No", "", "Sí"])

    def test_sin_stock_en_los_resultados_queda_vacio(self):
        r = _resultados().drop(columns=["stock_sugerido", "stock_sugerido_unidades"])
        t = _tabla(r)
        self.assertTrue(t["Stock sugerido (70 %)"].isna().all())
        self.assertEqual(set(t["¿Cubrió la demanda?"]), {""})

    def test_sin_unidades_reales_queda_nan(self):
        t = pr.tabla_producto(_resultados(), REAL, "CODIGO", "DESCRIPCION",
                              "TOTAL_VENDIDO", 1.15, 0.90)
        self.assertTrue(t["Unidades reales"].isna().all())


class ResumenStockTest(unittest.TestCase):
    def setUp(self):
        self.r = pr.resumen_mes("2026-04", _tabla(), real_top_total=1000.0)

    def test_cobertura_del_mes(self):
        self.assertEqual(self.r["Productos evaluados (stock)"], 4)
        self.assertAlmostEqual(self.r["Cobertura del stock sugerido"], 3 / 4)

    def test_cobertura_entre_los_que_vendieron_y_unidades(self):
        self.assertAlmostEqual(self.r["Cobertura (los que vendieron)"], 2 / 3)
        self.assertAlmostEqual(self.r["Cobertura (unidades)"], 2 / 4)

    def test_stock_sobre_real(self):
        self.assertAlmostEqual(self.r["Stock total / real"], 445.0 / 345.0)

    def test_sin_stock_es_nan(self):
        r = _resultados().drop(columns=["stock_sugerido", "stock_sugerido_unidades"])
        res = pr.resumen_mes("2026-04", _tabla(r), real_top_total=1000.0)
        self.assertTrue(np.isnan(res["Cobertura del stock sugerido"]))
        self.assertEqual(res["Productos evaluados (stock)"], 0)

    def test_cobertura_agrupada_apila_los_meses(self):
        t = _tabla()
        m = pr.cobertura_agrupada({"2026-04": t, "2026-05": t})
        self.assertEqual(m["n"], 8)
        self.assertAlmostEqual(m["cobertura"], 3 / 4)


if __name__ == "__main__":
    unittest.main()
