"""
Dashboard con la estimacion por promedio: los productos sin ventas el ultimo
mes (origen_estimacion = "promedio") suman al total predicho, pero no cuentan
en las decisiones ni en la dona y no muestran probabilidad.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))
import app as dashboard_app  # noqa: E402
from ventas_forecast.model import (  # noqa: E402
    ETIQUETA_SIN_RECOMENDACION, SalesForecastModel,
)


class PromedioApiTest(unittest.TestCase):

    def setUp(self):
        model = SalesForecastModel(filepath="unused.csv")
        model.blend_weight = 0.3
        c = model._c
        m = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
        # A y B: productos del modelo (B sin monto). P: sin ventas en marzo.
        model.df = pd.DataFrame({
            c("id"):          ["A"] * 3 + ["B"] * 3 + ["P"] * 3,
            c("description"): ["Producto A"] * 3 + ["Producto B"] * 3 + ["Producto P"] * 3,
            "fecha":          list(m) * 3,
            c("total_sold"):  [100.0, 100.0, 200.0, 5000.0, 6000.0, 7000.0,
                               300.0, 600.0, 0.0],
            c("frequency"):   [1] * 9,
            c("avg_price"):   [10.0] * 9,
        })
        model.metrics = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
        results = pd.DataFrame({
            c("id"):              ["A", "B", "P"],
            c("description"):     ["Producto A", "Producto B", "Producto P"],
            c("total_sold"):      [200.0, 7000.0, 0.0],
            "ventas_predichas":   [184.0, None, 300.0],
            "ventas_predichas_rf": [100.0, None, None],
            "decision":           ["Reducir stock", "Reforzar stock",
                                   ETIQUETA_SIN_RECOMENDACION],
            "prob_crecimiento":   [0.2, 0.7, None],
            "prob_reduccion":     [0.6, 0.1, None],
            "cluster":            [0, 1, 0],
            "origen_estimacion":  ["modelo", "modelo", "promedio"],
        })
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._saved = dict(dashboard_app._state)
        self._saved_out = dashboard_app.DEFAULT_OUTPUT
        dashboard_app.DEFAULT_OUTPUT = Path(self._tmp.name) / "pred.csv"
        results.to_csv(dashboard_app.DEFAULT_OUTPUT, index=False)
        dashboard_app._state.update(model=model, results=results, mes_predicho="2026-04")
        self.client = dashboard_app.app.test_client()

    def tearDown(self):
        dashboard_app.DEFAULT_OUTPUT = self._saved_out
        dashboard_app._state.clear()
        dashboard_app._state.update(self._saved)

    def test_summary_totals_split_model_and_promedio(self):
        d = self.client.get("/api/summary").get_json()
        self.assertEqual(d["pred_total"], 484.0)
        self.assertEqual(d["pred_total_modelo"], 184.0)
        self.assertEqual(d["pred_total_promedio"], 300.0)
        self.assertEqual(d["n_promedio"], 1)
        self.assertEqual(d["pred_total_rf"], 100.0)
        self.assertEqual(d["blend_weight"], 0.3)

    def test_summary_decisions_count_only_model_products(self):
        d = self.client.get("/api/summary").get_json()
        self.assertEqual(d["total"], 2)
        self.assertEqual((d["reforzar"], d["mantener"], d["reducir"]), (1, 0, 1))
        self.assertAlmostEqual(d["avg_prob"], 45.0)

    def test_summary_history_covers_model_and_promedio_products(self):
        d = self.client.get("/api/summary").get_json()
        self.assertEqual([h["ventas_reales"] for h in d["historial"]],
                         [400.0, 700.0, 200.0])
        self.assertEqual(d["con_pred"], 2)

    def test_summary_variation_same_product_set(self):
        d = self.client.get("/api/summary").get_json()
        # base de A (200) + P (0) = 200 ; pred 484
        self.assertEqual(d["variacion"], 142.0)

    def test_summary_top_tables_exclude_promedio(self):
        d = self.client.get("/api/summary").get_json()
        for tabla in ("top_growth", "top_risk"):
            self.assertNotIn("P", [r["CODIGO"] for r in d[tabla]])

    def test_products_filter_sin_recomendacion(self):
        d = self.client.get("/api/products", query_string={
            "decision": ETIQUETA_SIN_RECOMENDACION}).get_json()
        self.assertEqual([r["CODIGO"] for r in d["data"]], ["P"])
        self.assertEqual(d["data"][0]["origen_estimacion"], "promedio")
        self.assertEqual(d["data"][0]["ventas_predichas"], 300.0)

    def test_product_detail_promedio_has_no_probability(self):
        d = self.client.get("/api/product/P").get_json()
        p = d["prediccion"]
        self.assertEqual(p["origen_estimacion"], "promedio")
        self.assertIsNone(p["prob_crecimiento"])
        self.assertIsNone(p["prob_reduccion"])
        self.assertIsNone(p["ventas_predichas_rf"])
        self.assertEqual(p["ventas_predichas"], 300.0)

    def test_product_detail_model_product_keeps_probability(self):
        p = self.client.get("/api/product/A").get_json()["prediccion"]
        self.assertEqual(p["origen_estimacion"], "modelo")
        self.assertEqual(p["prob_crecimiento"], 0.2)

    def test_results_without_origin_column_behave_as_before(self):
        res = dashboard_app._state["results"]
        dashboard_app._state["results"] = res[res["origen_estimacion"] == "modelo"] \
            .drop(columns="origen_estimacion")
        d = self.client.get("/api/summary").get_json()
        self.assertEqual(d["n_promedio"], 0)
        self.assertEqual(d["pred_total"], 184.0)
        self.assertEqual(d["pred_total_promedio"], 0.0)
        self.assertEqual(d["total"], 2)


if __name__ == "__main__":
    unittest.main()
