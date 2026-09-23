"""
Verifica el ajuste de hiperparámetros del clasificador (GridSearchCV +
validación cruzada temporal por mes) del modelo de producción:
  - El splitter mensual no mezcla meses entre entrenamiento y validación.
  - tune_classifier() usa SOLO la partición de entrenamiento (sin fuga de
    datos del conjunto de prueba).
  - clf_params reproduce el RF por defecto y admite parámetros personalizados.

Ejecutar:  python -m unittest tests.test_rf_tuning
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_kmeans_rf_prod as tk  # noqa: E402
from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402


def _build_model(clf_params=None, n_products=6, n_months=24):
    model = SalesForecastModel(filepath="unused.csv", clf_params=clf_params)
    model.df = synthetic_history(n_products=n_products, n_months=n_months)
    model._ultimo_mes = model.df["fecha"].max()
    model._mes_pred = model._ultimo_mes + pd.DateOffset(months=1)
    model.build_features()
    model.cluster()
    model.prepare_split()
    return model


class MonthlyFoldsTest(unittest.TestCase):

    def setUp(self):
        self.model = _build_model()
        self.dates_train = (
            self.model._df_train.loc[self.model._train_idx, "fecha"]
            .reset_index(drop=True)
        )

    def test_train_months_are_strictly_before_val_months(self):
        folds = self.model._monthly_time_series_folds(self.dates_train, n_splits=3)
        self.assertEqual(len(folds), 3)
        for train_pos, val_pos in folds:
            train_dates = self.dates_train.iloc[train_pos]
            val_dates = self.dates_train.iloc[val_pos]
            self.assertLess(train_dates.max(), val_dates.min())

    def test_train_and_val_positions_are_disjoint_and_valid(self):
        folds = self.model._monthly_time_series_folds(self.dates_train, n_splits=3)
        n = len(self.dates_train)
        for train_pos, val_pos in folds:
            self.assertTrue(set(train_pos).isdisjoint(set(val_pos)))
            for pos in list(train_pos) + list(val_pos):
                self.assertGreaterEqual(pos, 0)
                self.assertLess(pos, n)

    def test_no_month_is_split_across_train_and_val(self):
        folds = self.model._monthly_time_series_folds(self.dates_train, n_splits=3)
        for train_pos, val_pos in folds:
            meses_train = set(self.dates_train.iloc[train_pos])
            meses_val = set(self.dates_train.iloc[val_pos])
            self.assertTrue(meses_train.isdisjoint(meses_val))


class TuneClassifierTest(unittest.TestCase):

    def setUp(self):
        self.model = _build_model()
        self.grid = {
            "rf__n_estimators": [10],
            "rf__max_depth": [2, 3],
            "rf__min_samples_leaf": [1],
        }

    def test_best_params_belong_to_grid_and_results_have_one_row_per_combo(self):
        best = self.model.tune_classifier(param_grid=self.grid, n_splits=3)

        self.assertIn(best["n_estimators"], self.grid["rf__n_estimators"])
        self.assertIn(best["max_depth"], self.grid["rf__max_depth"])
        self.assertIn(best["min_samples_leaf"], self.grid["rf__min_samples_leaf"])

        # 1 * 2 * 1 = 2 combinaciones de hiperparámetros
        self.assertEqual(len(self.model.tuning_results), 2)
        self.assertEqual(self.model.best_clf_params, best)

    def test_does_not_modify_clf_params_automatically(self):
        original = dict(self.model.clf_params)
        self.model.tune_classifier(param_grid=self.grid, n_splits=3)
        self.assertEqual(self.model.clf_params, original)

    def test_uses_only_training_rows(self):
        captured = {}
        real_fit = tk.GridSearchCV.fit

        def spy_fit(self_grid, X, y=None, **kwargs):
            captured["index"] = list(X.index)
            return real_fit(self_grid, X, y, **kwargs)

        with mock.patch.object(tk.GridSearchCV, "fit", spy_fit):
            self.model.tune_classifier(param_grid=self.grid, n_splits=3)

        seen = set(captured["index"])
        train_idx = set(self.model._train_idx)
        test_idx = set(self.model._test_idx)

        self.assertEqual(seen, train_idx)
        self.assertTrue(seen.isdisjoint(test_idx))

    def test_raises_if_prepare_split_not_called(self):
        model = SalesForecastModel(filepath="unused.csv")
        with self.assertRaises(RuntimeError):
            model.tune_classifier()


class ClfParamsTest(unittest.TestCase):

    def test_default_clf_params_are_the_tuned_values(self):
        # Valores seleccionados con tune_classifier() sobre Query_Result_V5.csv
        model = _build_model()
        self.assertEqual(model.clf_params, tk.DEFAULT_CLF_PARAMS)

        model.train_classifier()
        self.assertEqual(model.clf.n_estimators, 300)
        self.assertEqual(model.clf.max_depth, 14)
        self.assertEqual(model.clf.min_samples_leaf, 1)

    def test_custom_clf_params_are_applied_to_classifier(self):
        model = _build_model(clf_params={"n_estimators": 7, "max_depth": 4})
        model.train_classifier()

        self.assertEqual(model.clf.n_estimators, 7)
        self.assertEqual(model.clf.max_depth, 4)
        # No sobreescrito: conserva el valor por defecto
        self.assertEqual(model.clf.min_samples_leaf, 1)


if __name__ == "__main__":
    unittest.main()
