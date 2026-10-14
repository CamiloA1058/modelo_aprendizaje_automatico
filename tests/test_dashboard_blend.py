"""
API del dashboard con la prediccion combinada: ventas_predichas es la mezcla
(modelo + media de 3 meses) y ventas_predichas_rf conserva el RF puro.
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
from ventas_forecast.model import SalesForecastModel  # noqa: E402


class BlendApiTest(unittest.TestCase):

    def setUp(self):
        model = SalesForecastModel(filepath="unused.csv")
        model.blend_weight = 0.3
        c = model._c
        months = pd.to_datetime(["2026-02-01", "2026-03-01"])
        model.df = pd.DataFrame({
            c("id"):          ["A", "A", "B", "B"],
            c("description"): ["Producto A", "Producto A", "Producto B", "Producto B"],
            "fecha":          [months[0], months[1], months[0], months[1]],
            c("total_sold"):  [100.0, 200.0, 5000.0, 7000.0],
            c("frequency"):   [1, 2, 3, 4],
            c("avg_price"):   [10.0, 10.0, 20.0, 20.0],
        })
        model.metrics = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
        # A: blend 0.3 * 100 + 0.7 * 220 = 184 ; B sin prediccion de monto.
        results = pd.DataFrame({
            c("id"):              ["A", "B"],
            c("description"):     ["Producto A", "Producto B"],
            c("total_sold"):      [200.0, 7000.0],
            "ventas_predichas":   [184.0, None],
            "ventas_predichas_rf": [100.0, None],
            "decision":           ["Reducir stock", "Reforzar stock"],
            "prob_crecimiento":   [0.2, 0.7],
            "prob_reduccion":     [0.6, 0.1],
            "cluster":            [0, 1],
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

    def test_summary_returns_blend_total_rf_total_and_weight(self):
        data = self.client.get("/api/summary").get_json()
        self.assertEqual(data["pred_total"], 184.0)
        self.assertEqual(data["pred_total_rf"], 100.0)
        self.assertEqual(data["blend_weight"], 0.3)

    def test_summary_variation_uses_blend(self):
        data = self.client.get("/api/summary").get_json()
        self.assertEqual(data["variacion"], -8.0)

    def test_summary_top_tables_carry_both_columns(self):
        data = self.client.get("/api/summary").get_json()
        for tabla in ("top_growth", "top_risk"):
            fila = next(r for r in data[tabla] if r["CODIGO"] == "A")
            self.assertEqual(fila["ventas_predichas"], 184.0)
            self.assertEqual(fila["ventas_predichas_rf"], 100.0)

    def test_summary_without_weight_reports_null(self):
        dashboard_app._state["model"].blend_weight = None
        data = self.client.get("/api/summary").get_json()
        self.assertIsNone(data["blend_weight"])

    def test_summary_without_rf_column_falls_back_to_blend_total(self):
        dashboard_app._state["results"] = dashboard_app._state["results"].drop(
            columns=["ventas_predichas_rf"])
        data = self.client.get("/api/summary").get_json()
        self.assertEqual(data["pred_total_rf"], 184.0)

    def test_products_include_both_columns(self):
        data = self.client.get("/api/products").get_json()
        fila = next(r for r in data["data"] if r["CODIGO"] == "A")
        self.assertEqual(fila["ventas_predichas"], 184.0)
        self.assertEqual(fila["ventas_predichas_rf"], 100.0)

    def test_products_sorted_by_blend(self):
        data = self.client.get("/api/products?sort=ventas").get_json()
        self.assertEqual(data["data"][0]["CODIGO"], "A")

    def test_product_detail_has_blend_and_model_only(self):
        data = self.client.get("/api/product/A").get_json()
        self.assertEqual(data["prediccion"]["ventas_predichas"], 184.0)
        self.assertEqual(data["prediccion"]["ventas_predichas_rf"], 100.0)

    def test_download_csv_has_both_columns(self):
        resp = self.client.get("/api/download")
        cabecera = resp.data.decode("utf-8").splitlines()[0].split(",")
        resp.close()
        self.assertIn("ventas_predichas", cabecera)
        self.assertIn("ventas_predichas_rf", cabecera)


if __name__ == "__main__":
    unittest.main()
