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


def _synthetic_history(n_products=6, n_months=24):
    """Historial mensual con tendencia creciente: los meses de prueba
    tienen valores mayores que los de entrenamiento, así la mediana del
    conjunto completo difiere de la mediana del entrenamiento."""
    rows = []
    fechas = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    rng = np.random.default_rng(0)
    for p in range(n_products):
        for i, fecha in enumerate(fechas):
            total = 1000.0 * (i + 1) * (p + 1) + rng.integers(0, 500)
            rows.append({
                "CODIGO": p,
                "DESCRIPCION": f"PRODUCTO {p}",
                "fecha": fecha,
                "VENTAS": float(i + 1 + p),
                "TOTAL_VENDIDO": total,
                "PRECIO_PROMEDIO": total / (i + 1 + p),
                "FRECUENCIA": i + 1,
                "ANIO": fecha.year,
                "MES": fecha.month,
            })
    return pd.DataFrame(rows)


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
