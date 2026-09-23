"""
Verifica el ajuste de hiperparámetros del clasificador (GridSearchCV +
validación cruzada temporal por mes) del modelo de producción:
  - El splitter mensual no mezcla meses entre entrenamiento y validación.
  - tune_classifier() usa SOLO la partición de entrenamiento (sin fuga de
    datos del conjunto de prueba) y selecciona por F1 macro por defecto
    (recompensa las clases minoritarias, a diferencia del F1 ponderado).
  - clf_params reproduce el RF por defecto y admite parámetros personalizados.
  - train_classifier() reporta F1 macro y accuracy balanceada además de las
    métricas ponderadas (necesarias para leer el desempeño bajo desbalance
    de clases, ver CHANGELOG 2026-09-23 "métrica macro").

Ejecutar:  python -m unittest tests.test_rf_tuning
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

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

    def test_default_scoring_is_f1_macro(self):
        # F1 macro promedia el F1 de cada clase con el mismo peso, así que
        # premia acertar en las clases minoritarias (Mantener/Reforzar);
        # f1_weighted (el valor previo) premia sobre todo la clase mayoritaria
        # ("Reducir", 79,30 % de la prueba tras el completado de calendario).
        captured = {}
        real_grid_search_cv = tk.GridSearchCV

        def fake_grid_search_cv(*args, **kwargs):
            captured["scoring"] = kwargs.get("scoring")
            return real_grid_search_cv(*args, **kwargs)

        with mock.patch.object(tk, "GridSearchCV", side_effect=fake_grid_search_cv):
            self.model.tune_classifier(param_grid=self.grid, n_splits=3)

        self.assertEqual(captured["scoring"], "f1_macro")

    def test_explicit_scoring_overrides_default(self):
        captured = {}
        real_grid_search_cv = tk.GridSearchCV

        def fake_grid_search_cv(*args, **kwargs):
            captured["scoring"] = kwargs.get("scoring")
            return real_grid_search_cv(*args, **kwargs)

        with mock.patch.object(tk, "GridSearchCV", side_effect=fake_grid_search_cv):
            self.model.tune_classifier(
                param_grid=self.grid, n_splits=3, scoring="f1_weighted"
            )

        self.assertEqual(captured["scoring"], "f1_weighted")


class ClfParamsTest(unittest.TestCase):

    def test_default_clf_params_are_the_tuned_values(self):
        # Valores seleccionados con tune_classifier(scoring="f1_macro") sobre
        # Query_Result_V5.csv (ver CHANGELOG 2026-09-23 "métrica macro":
        # max_depth pasa de None a 14, min_samples_leaf de 1 a 5).
        model = _build_model()
        self.assertEqual(model.clf_params, tk.DEFAULT_CLF_PARAMS)

        model.train_classifier()
        self.assertEqual(model.clf.n_estimators, 300)
        self.assertEqual(model.clf.max_depth, 14)
        self.assertEqual(model.clf.min_samples_leaf, 5)

    def test_custom_clf_params_are_applied_to_classifier(self):
        model = _build_model(clf_params={"n_estimators": 7, "max_depth": 4})
        model.train_classifier()

        self.assertEqual(model.clf.n_estimators, 7)
        self.assertEqual(model.clf.max_depth, 4)
        # No sobreescrito: conserva el valor por defecto (DEFAULT_CLF_PARAMS)
        self.assertEqual(model.clf.min_samples_leaf, tk.DEFAULT_CLF_PARAMS["min_samples_leaf"])


class ClassifierMetricsTest(unittest.TestCase):
    """(b) self.metrics agrega f1_macro y balanced_accuracy, con valores
    que coinciden con los calculados independientemente por sklearn sobre
    las mismas predicciones de prueba (además de las claves ponderadas
    existentes, que scripts/app.py sigue leyendo sin cambios)."""

    def setUp(self):
        self.model = _build_model()
        self.model.train_classifier()

    def test_existing_weighted_keys_are_preserved(self):
        for clave in ("accuracy", "precision", "recall", "f1"):
            self.assertIn(clave, self.model.metrics)

    def test_metrics_include_macro_f1_and_balanced_accuracy(self):
        X_te = self.model._X_scaled[self.model._test_idx]
        y_te = self.model._y.iloc[self.model._test_idx]
        pred = self.model.clf.predict(X_te)

        f1_macro_esperado = round(
            f1_score(y_te, pred, average="macro", zero_division=0), 4
        )
        balanced_acc_esperado = round(balanced_accuracy_score(y_te, pred), 4)

        self.assertIn("f1_macro", self.model.metrics)
        self.assertIn("balanced_accuracy", self.model.metrics)
        self.assertEqual(self.model.metrics["f1_macro"], f1_macro_esperado)
        self.assertEqual(
            self.model.metrics["balanced_accuracy"], balanced_acc_esperado
        )


if __name__ == "__main__":
    unittest.main()
