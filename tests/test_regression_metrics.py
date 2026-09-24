"""
Verifica que train_regressor() del modelo de producción:
  - Seleccione self._top_prods y entrene el RandomForestRegressor SOLO con
    la partición de entrenamiento (sin fuga de datos del conjunto de prueba).
  - Evalúe en la partición de prueba y calcule self.reg_metrics (MAE, MSE,
    RMSE, R2, MASE, n_test).
  - Calcule self.baseline_metrics para los pronósticos ingenuo y media_3
    sobre las mismas filas de prueba.

Ejecutar:  python -m unittest tests.test_regression_metrics
"""

import sys
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_kmeans_rf_prod as tk  # noqa: E402
from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402


def _build_model(top_n_products=1900, n_products=6, n_months=24):
    model = SalesForecastModel(filepath="unused.csv", top_n_products=top_n_products)
    model.df = synthetic_history(n_products=n_products, n_months=n_months)
    model._ultimo_mes = model.df["fecha"].max()
    model._mes_pred = model._ultimo_mes + pd.DateOffset(months=1)
    model.build_features()
    model.cluster()
    model.prepare_split()
    return model


class RegressorTrainedOnlyOnTrainRowsTest(unittest.TestCase):
    """(a) el regresor se ajusta SOLO con filas de entrenamiento."""

    def setUp(self):
        self.model = _build_model()

    def test_fit_uses_only_training_rows_of_top_products(self):
        captured = {}
        real_fit = tk.RandomForestRegressor.fit

        def spy_fit(self_rf, X, y, **kwargs):
            # y conserva el índice de _df_train hasta este punto (no se
            # convierte a numpy antes de llegar aquí).
            captured["y_index"] = list(y.index)
            captured["n_rows"] = X.shape[0]
            return real_fit(self_rf, X, y, **kwargs)

        with mock.patch.object(tk.RandomForestRegressor, "fit", spy_fit):
            self.model.train_regressor()

        train_idx = set(self.model._train_idx)
        test_idx = set(self.model._test_idx)
        seen = set(captured["y_index"])

        # Ninguna fila de prueba participó en el ajuste
        self.assertTrue(seen.isdisjoint(test_idx))
        self.assertTrue(seen.issubset(train_idx))

        # El número (y el conjunto) de filas ajustadas coincide con una
        # cuenta independiente: filas de ENTRENAMIENTO de top productos con
        # target > 0.
        c = self.model._c
        df_train = self.model._df_train
        esperado = df_train[
            df_train.index.isin(train_idx) &
            df_train[c("id")].isin(self.model._top_prods) &
            (df_train["target"] > 0)
        ]
        self.assertEqual(captured["n_rows"], len(esperado))
        self.assertEqual(seen, set(esperado.index))


class RegMetricsTest(unittest.TestCase):
    """(b) self.reg_metrics tiene las claves correctas con valores finitos."""

    def setUp(self):
        self.model = _build_model()
        self.model.train_regressor()

    def test_reg_metrics_has_expected_keys_and_finite_values(self):
        m = self.model.reg_metrics
        for k in ("mae", "mse", "rmse", "r2", "mase", "n_test"):
            self.assertIn(k, m)

        for k in ("mae", "mse", "rmse", "r2", "mase"):
            self.assertTrue(math.isfinite(m[k]), f"{k} no es finito: {m[k]}")

        self.assertGreater(m["n_test"], 0)

    def test_rmse_equals_sqrt_mse(self):
        m = self.model.reg_metrics
        self.assertAlmostEqual(m["rmse"], math.sqrt(m["mse"]), places=3)


class MaseScaleTest(unittest.TestCase):
    """(c) MASE = mae_modelo / escala-ingenua, escala calculada en ENTRENAMIENTO."""

    def setUp(self):
        self.model = _build_model()
        self.model.train_regressor()

    def test_mase_equals_mae_over_independent_naive_scale(self):
        c = self.model._c
        df_train = self.model._df_train
        train_idx = set(self.model._train_idx)

        filas_train_top = df_train[
            df_train.index.isin(train_idx) &
            df_train[c("id")].isin(self.model._top_prods) &
            (df_train["target"] > 0)
        ]
        escala_esperada = (
            filas_train_top["target"] - filas_train_top[c("total_sold")]
        ).abs().mean()

        mase_esperado = round(self.model.reg_metrics["mae"] / escala_esperada, 4)
        self.assertAlmostEqual(
            self.model.reg_metrics["mase"], mase_esperado, places=3
        )


class BaselineMetricsTest(unittest.TestCase):
    """(d) self.baseline_metrics["ingenuo"] calculado sobre filas de prueba."""

    def setUp(self):
        self.model = _build_model()
        self.model.train_regressor()

    def test_baseline_metrics_has_ingenuo_and_media_3(self):
        self.assertIn("ingenuo", self.model.baseline_metrics)
        self.assertIn("media_3", self.model.baseline_metrics)

    def test_ingenuo_mase_is_finite(self):
        mase = self.model.baseline_metrics["ingenuo"]["mase"]
        self.assertTrue(math.isfinite(mase))

    def test_ingenuo_mae_equals_mean_abs_error_on_test_rows(self):
        c = self.model._c
        df_train = self.model._df_train
        test_idx = set(self.model._test_idx)

        filas_test_top = df_train[
            df_train.index.isin(test_idx) &
            df_train[c("id")].isin(self.model._top_prods) &
            (df_train["target"] > 0)
        ]
        mae_esperado = round(
            (filas_test_top["target"] - filas_test_top[c("total_sold")])
            .abs().mean(),
            4,
        )
        self.assertAlmostEqual(
            self.model.baseline_metrics["ingenuo"]["mae"], mae_esperado, places=4
        )


class TopProdsIgnoresTestRowsTest(unittest.TestCase):
    """(e) _top_prods se calcula SOLO con filas de entrenamiento."""

    def test_top_prods_selection_ignores_test_rows(self):
        # top_n_products=1 para que un solo producto "gane" la selección:
        # así un valor inflado en las filas de PRUEBA de un producto que no
        # sería top en entrenamiento revela si la fuga ocurre.
        model = _build_model(top_n_products=1, n_products=3, n_months=12)

        c = model._c
        test_idx = set(model._test_idx)
        mask_test_prod0 = (
            model._df_train.index.isin(test_idx) &
            (model._df_train[c("id")] == 0)
        )
        self.assertTrue(mask_test_prod0.any(), "el producto 0 no tiene filas de prueba")

        # Inflar artificialmente las ventas del PRODUCTO 0 solo en sus filas
        # de PRUEBA: si _top_prods se calculara con todas las filas de
        # _df_train (como antes de la corrección), el producto 0 pasaría a
        # ser el top-1 por esta inflación; si se calcula solo con
        # entrenamiento, el resultado no debe cambiar.
        model._df_train.loc[mask_test_prod0, c("total_sold")] = 10_000_000.0

        esperado = (
            model._df_train.loc[model._train_idx]
            .groupby(c("id"))[c("total_sold")]
            .sum().nlargest(model.top_n_products).index
        )

        model.train_regressor()

        self.assertEqual(list(model._top_prods), list(esperado))
        self.assertNotIn(0, list(model._top_prods))


if __name__ == "__main__":
    unittest.main()
