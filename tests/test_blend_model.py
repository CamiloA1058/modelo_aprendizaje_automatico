"""
Mezcla del sistema: ventas_predichas = w * RF + (1 - w) * media_3, con w
elegido SOLO con la particion de prueba interna del regresor (criterio de
sesgo de la suma). El RF puro se conserva en ventas_predichas_rf.

Ejecutar:  python -m unittest tests.test_blend_model
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ventas_forecast.model as tk  # noqa: E402
from ventas_forecast import improvements as im  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402


def _trained_model(tmp, top_n_products=1900, n_products=6, n_months=24):
    model = SalesForecastModel(
        filepath="unused.csv", output_path=Path(tmp) / "pred.csv",
        top_n_products=top_n_products,
    )
    model.df = synthetic_history(n_products=n_products, n_months=n_months)
    model._ultimo_mes = model.df["fecha"].max()
    model._mes_pred = model._ultimo_mes + pd.DateOffset(months=1)
    model.build_features()
    model.cluster()
    model.prepare_split()
    return model


class BlendWeightUsesOnlyInternalValidationTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model = _trained_model(self._tmp.name)

    def test_weight_chosen_with_internal_test_partition_and_bias_criterion(self):
        captured = {}
        real_choose = im.choose_blend_weight

        def spy(real, rf, base, *args, **kwargs):
            captured.update(real=np.asarray(real), rf=np.asarray(rf),
                            base=np.asarray(base), kwargs=kwargs)
            return real_choose(real, rf, base, *args, **kwargs)

        with mock.patch.object(tk.im, "choose_blend_weight", spy):
            self.model.train_regressor()

        res = self.model.reg_test_results
        np.testing.assert_allclose(captured["real"], res["real"].to_numpy())
        np.testing.assert_allclose(captured["rf"], res["predicho"].to_numpy())
        self.assertEqual(captured["kwargs"].get("criterion"), "sesgo")
        self.assertEqual(len(captured["real"]), self.model.reg_metrics["n_test"])

    def test_weight_is_on_the_grid_and_stored(self):
        self.model.train_regressor()
        self.assertIn(self.model.blend_weight, im.BLEND_WEIGHTS)
        self.assertEqual(self.model.blend_metrics["w"], self.model.blend_weight)

    def test_blend_metrics_next_to_rf_metrics_without_removing_anything(self):
        self.model.train_regressor()
        for k in ("mae", "mse", "rmse", "r2", "mase", "n_test"):
            self.assertIn(k, self.model.reg_metrics)
        for k in ("mae", "rmse", "r2", "mase", "sum_ratio", "n_test", "w"):
            self.assertIn(k, self.model.blend_metrics)
        self.assertIn("sum_ratio", self.model.reg_metrics)

    def test_blend_metrics_match_independent_computation(self):
        self.model.train_regressor()
        res = self.model.reg_test_results
        y, rf = res["real"].to_numpy(), res["predicho"].to_numpy()
        m3 = self.model.reg_test_media_3
        w = self.model.blend_weight
        pred = w * rf + (1 - w) * m3
        self.assertAlmostEqual(self.model.blend_metrics["mae"],
                               round(float(np.mean(np.abs(y - pred))), 4))
        self.assertAlmostEqual(self.model.blend_metrics["sum_ratio"],
                               round(float(pred.sum() / y.sum()), 4))
        self.assertAlmostEqual(self.model.reg_metrics["sum_ratio"],
                               round(float(rf.sum() / y.sum()), 4))

    def test_classifier_and_regressor_hyperparameters_unchanged(self):
        self.model.train_regressor()
        self.assertEqual(self.model.reg.n_estimators, 200)
        self.assertEqual(self.model.reg.max_depth, 12)


class _StubClassifier:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.2, 0.5, 0.3], (len(X), 1))


class PredictNextMonthBlendTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model = _trained_model(self._tmp.name, top_n_products=3)
        self.model.train_regressor()
        # El historial sintetico no tiene las 3 clases: clasificador simulado.
        self.model.clf = _StubClassifier()

    def _predict(self):
        self.model.predict_next_month()
        return self.model.results

    def test_ventas_predichas_is_blend_and_rf_is_preserved(self):
        self.model.blend_weight = 0.4
        res = self._predict()
        c = self.model._c
        dfp = self.model.df_predict
        self.assertIn("ventas_predichas_rf", res.columns)
        top = res[res["ventas_predichas"].notna()]
        self.assertGreater(len(top), 0)
        m3 = dfp.loc[top.index, "media_3_actual"].to_numpy()
        m3 = np.where(np.isnan(m3), dfp.loc[top.index, c("total_sold")].to_numpy(), m3)
        rf = top["ventas_predichas_rf"].to_numpy()
        np.testing.assert_allclose(top["ventas_predichas"].to_numpy(),
                                   0.4 * rf + 0.6 * m3)

    def test_rf_column_equals_raw_regressor_output(self):
        res = self._predict()
        top = res[res["ventas_predichas_rf"].notna()]
        dfp = self.model.df_predict
        X = self.model.scaler.transform(dfp.loc[top.index, self.model._features].fillna(0))
        np.testing.assert_allclose(top["ventas_predichas_rf"].to_numpy(),
                                   np.expm1(self.model.reg.predict(X)))

    def test_weight_one_gives_pure_rf(self):
        self.model.blend_weight = 1.0
        res = self._predict()
        top = res[res["ventas_predichas"].notna()]
        np.testing.assert_allclose(top["ventas_predichas"], top["ventas_predichas_rf"])

    def test_products_without_amount_prediction_stay_nan_in_both_columns(self):
        res = self._predict()
        self.assertTrue((res["ventas_predichas"].isna()
                         == res["ventas_predichas_rf"].isna()).all())
        c = self.model._c
        fuera = ~res[c("id")].isin(self.model._top_prods)
        self.assertTrue(res.loc[fuera, "ventas_predichas"].isna().all())

    def test_missing_media_3_falls_back_to_current_month_sales(self):
        self.model.blend_weight = 0.0
        self.model.df_predict["media_3_actual"] = np.nan
        res = self._predict()
        c = self.model._c
        top = res[res["ventas_predichas"].notna()]
        np.testing.assert_allclose(top["ventas_predichas"], top[c("total_sold")])
        self.assertFalse(top["ventas_predichas"].isna().any())

    def test_missing_weight_falls_back_to_pure_rf(self):
        self.model.blend_weight = None
        res = self._predict()
        top = res[res["ventas_predichas"].notna()]
        np.testing.assert_allclose(top["ventas_predichas"], top["ventas_predichas_rf"])


if __name__ == "__main__":
    unittest.main()
