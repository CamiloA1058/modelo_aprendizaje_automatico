"""
Verifica que el clustering (KMeans) del modelo de producción no incurra en
fuga de datos (el escalador/KMeans de clustering se ajustan solo con las
filas de entrenamiento) y que la selección de k (evaluate_k) funcione sin
alterar el estado del pipeline.

Ejecutar:  python -m unittest tests.test_kmeans_selection
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402


def _build_model():
    model = SalesForecastModel(filepath="unused.csv")
    model.df = synthetic_history()
    model._ultimo_mes = model.df["fecha"].max()
    model._mes_pred = model._ultimo_mes + pd.DateOffset(months=1)
    model.build_features()
    return model


class ClusterNoLeakageTest(unittest.TestCase):

    def setUp(self):
        self.model = _build_model()
        self.model.cluster()
        self.model.prepare_split()

    def test_cluster_scaler_center_is_median_of_training_partition(self):
        c = self.model._c
        cluster_cols = [c("total_sold"), c("frequency"), c("avg_price")]

        # Partición de entrenamiento (mismo filtro y split temporal que
        # prepare_split): filas con target conocido y positivo, primer
        # train_ratio de cada producto ordenado por fecha.
        df_train = self.model.df[
            self.model.df["target"].notna() &
            (self.model.df["target"] > 0) &
            (self.model.df[c("total_sold")] > 0)
        ].copy().reset_index(drop=True)

        train_idx = []
        for _, g in df_train.groupby(c("id")):
            g = g.sort_values("fecha")
            cut = int(len(g) * self.model.train_ratio)
            train_idx += g.index[:cut].tolist()

        X_train = df_train.loc[train_idx, cluster_cols].fillna(0)
        X_full = self.model.df[cluster_cols].fillna(0)

        # La mediana de entrenamiento debe diferir de la mediana global
        # (por la tendencia sintética), para que el test sea significativo.
        self.assertFalse(
            np.allclose(X_train.median().to_numpy(), X_full.median().to_numpy())
        )

        np.testing.assert_allclose(
            self.model.cluster_scaler.center_, X_train.median().to_numpy()
        )

    def test_every_row_has_cluster_label(self):
        self.assertNotIn(np.nan, self.model.df["cluster"].values)
        self.assertEqual(self.model.df["cluster"].notna().sum(), len(self.model.df))


class EvaluateKTest(unittest.TestCase):

    def setUp(self):
        self.model = _build_model()

    def test_returns_one_row_per_k_with_finite_metrics(self):
        k_range = range(2, 6)
        tabla = self.model.evaluate_k(k_range=k_range)

        self.assertEqual(list(tabla["k"]), list(k_range))
        self.assertTrue(np.isfinite(tabla["inercia"]).all())
        self.assertTrue(((tabla["silueta"] >= -1) & (tabla["silueta"] <= 1)).all())

    def test_evaluate_k_does_not_modify_n_clusters(self):
        original = self.model.n_clusters
        self.model.evaluate_k(k_range=range(2, 5))
        self.assertEqual(self.model.n_clusters, original)


if __name__ == "__main__":
    unittest.main()
