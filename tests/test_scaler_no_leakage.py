"""
Verifica que el escalador del modelo de producción se ajuste solo con el
conjunto de entrenamiento, para que las métricas de validación no usen
información del conjunto de prueba.

Ejecutar:  python -m unittest tests.test_scaler_no_leakage
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history as _synthetic_history  # noqa: E402


class ScalerNoLeakageTest(unittest.TestCase):

    def setUp(self):
        self.model = SalesForecastModel(filepath="unused.csv")
        self.model.df = _synthetic_history()
        self.model._ultimo_mes = self.model.df["fecha"].max()
        self.model._mes_pred = self.model._ultimo_mes + pd.DateOffset(months=1)
        self.model.build_features()
        self.model.cluster()
        self.model.prepare_split()

    def test_scaler_is_fitted_on_training_rows_only(self):
        X_train = self.model._X.loc[self.model._train_idx]
        np.testing.assert_allclose(
            self.model.scaler.center_, X_train.median().to_numpy()
        )

    def test_scaled_matrix_is_transform_of_all_rows(self):
        expected = self.model.scaler.transform(self.model._X)
        np.testing.assert_allclose(self.model._X_scaled, expected)


if __name__ == "__main__":
    unittest.main()
