"""
/api/summary: the history line must cover the same products as the predicted
total, and the total is the plain sum of per-product predictions.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))
import app as dashboard_app  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402


class SummaryHistoryComparableTest(unittest.TestCase):

    def setUp(self):
        model = SalesForecastModel(filepath="unused.csv")
        c = model._c
        months = pd.to_datetime(["2026-01-01", "2026-02-01"])
        # "A" has an amount prediction; "B" does not (outside the top products).
        model.df = pd.DataFrame({
            c("id"):         ["A", "A", "B", "B"],
            "fecha":         [months[0], months[1], months[0], months[1]],
            c("total_sold"): [100.0, 200.0, 5000.0, 7000.0],
        })
        model.metrics = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
        results = pd.DataFrame({
            c("id"):            ["A", "B"],
            c("description"):   ["Producto A", "Producto B"],
            c("total_sold"):    [200.0, 7000.0],
            "ventas_predichas": [150.0, None],
            "decision":         ["Reducir stock", "Reforzar stock"],
            "prob_crecimiento": [0.2, 0.7],
            "prob_reduccion":   [0.6, 0.1],
            "cluster":          [0, 1],
        })
        self._saved = dict(dashboard_app._state)
        dashboard_app._state.update(model=model, results=results, mes_predicho="2026-03")
        self.client = dashboard_app.app.test_client()

    def tearDown(self):
        dashboard_app._state.clear()
        dashboard_app._state.update(self._saved)

    def test_history_only_includes_products_with_amount_prediction(self):
        data = self.client.get("/api/summary").get_json()
        self.assertEqual([h["ventas_reales"] for h in data["historial"]], [100.0, 200.0])

    def test_total_is_uncalibrated_sum_and_variation_uses_same_products(self):
        data = self.client.get("/api/summary").get_json()
        self.assertEqual(data["pred_total"], 150.0)
        self.assertEqual(data["con_pred"], 1)
        self.assertEqual(data["variacion"], -25.0)
        self.assertNotIn("factor_calibracion", data)


if __name__ == "__main__":
    unittest.main()
