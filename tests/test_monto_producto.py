"""Experimento "acierto del monto por producto": objetivo a varios meses,
combinacion en dos pasos (hurdle), Tweedie, correccion por tamano y metricas
de acierto por producto."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import monto_producto as mp  # noqa: E402

ID, TOT = "CODIGO", "TOTAL_VENDIDO"


def producto(pid, ventas, inicio="2024-01-01"):
    return pd.DataFrame({
        ID: pid,
        "fecha": pd.date_range(inicio, periods=len(ventas), freq="MS"),
        TOT: [float(v) for v in ventas],
    })


class ObjetivoVariosMesesTest(unittest.TestCase):
    def test_horizonte_uno_es_el_mes_siguiente(self):
        df = producto(1, [10, 20, 30, 40])
        t = mp.multi_month_target(df, ID, TOT, 1)
        np.testing.assert_allclose(t.to_numpy(), [20, 30, 40, np.nan])

    def test_horizonte_dos_suma_los_dos_meses_siguientes(self):
        df = producto(1, [10, 20, 30, 40])
        t = mp.multi_month_target(df, ID, TOT, 2)
        np.testing.assert_allclose(t.to_numpy(), [50, 70, np.nan, np.nan])

    def test_horizonte_tres_exige_ventana_completa(self):
        df = producto(1, [1, 2, 3, 4, 5])
        t = mp.multi_month_target(df, ID, TOT, 3)
        np.testing.assert_allclose(t.to_numpy(), [9, 12, np.nan, np.nan, np.nan])

    def test_productos_no_se_mezclan_y_conserva_el_indice(self):
        a = producto(1, [1, 2, 3])
        b = producto(2, [100, 200, 300])
        df = pd.concat([b, a], ignore_index=True).sample(frac=1, random_state=3)
        t = mp.multi_month_target(df, ID, TOT, 2)
        self.assertTrue(t.index.equals(df.index))
        por_id = {(r[ID], r["fecha"].month): t[i] for i, r in df.iterrows()}
        self.assertEqual(por_id[(1, 1)], 5.0)
        self.assertEqual(por_id[(2, 1)], 500.0)
        self.assertTrue(np.isnan(por_id[(2, 2)]))

    def test_horizonte_invalido(self):
        with self.assertRaises(ValueError):
            mp.multi_month_target(producto(1, [1, 2]), ID, TOT, 0)


class HurdleTest(unittest.TestCase):
    def test_sin_umbral_es_probabilidad_por_monto(self):
        out = mp.hurdle_combine([0.5, 1.0, 0.0], [100.0, 40.0, 10.0])
        np.testing.assert_allclose(out, [50.0, 40.0, 0.0])

    def test_con_umbral_da_cero_debajo_y_el_monto_condicional_arriba(self):
        out = mp.hurdle_combine([0.2, 0.5, 0.9], [100.0, 40.0, 10.0], threshold=0.5)
        np.testing.assert_allclose(out, [0.0, 40.0, 10.0])


class MetricasTest(unittest.TestCase):
    def test_acierto_por_producto_recorta_y_omite_reales_en_cero(self):
        acc = mp.product_accuracy([100, 100, 100, 0], [100, 50, 300, 5])
        np.testing.assert_allclose(acc[:3], [1.0, 0.5, 0.0])
        self.assertTrue(np.isnan(acc[3]))

    def test_metricas_agregadas(self):
        real = np.array([100.0, 100.0, 100.0, 0.0])
        pred = np.array([100.0, 60.0, 300.0, 40.0])
        m = mp.accuracy_metrics(real, pred)
        self.assertEqual(m["n"], 4)
        self.assertEqual(m["n_pos"], 3)
        self.assertAlmostEqual(m["mean_acc"], (1.0 + 0.6 + 0.0) / 3)
        self.assertAlmostEqual(m["within30"], 1 / 3)
        self.assertAlmostEqual(m["within50"], 2 / 3)
        self.assertAlmostEqual(m["acc_wape"], 1 - (0 + 40 + 200 + 40) / 300)
        self.assertAlmostEqual(m["sum_ratio"], 500 / 300)
        self.assertAlmostEqual(m["mae"], 280 / 4)

    def test_dentro_de_30_y_50_son_inclusivos(self):
        m = mp.accuracy_metrics([100, 100, 100], [130, 150, 151])
        self.assertAlmostEqual(m["within30"], 1 / 3)
        self.assertAlmostEqual(m["within50"], 2 / 3)

    def test_sin_reales_positivos_da_nan(self):
        m = mp.accuracy_metrics([0, 0], [1, 2])
        self.assertTrue(np.isnan(m["mean_acc"]))
        self.assertTrue(np.isnan(m["acc_wape"]))
        self.assertEqual(m["n_pos"], 0)

    def test_acierto_de_vende_o_no(self):
        self.assertAlmostEqual(
            mp.sells_accuracy([10, 0, 5, 0], [3, 0, 0, 9]), 0.5)


class SegmentosTest(unittest.TestCase):
    def test_top_todos_los_meses_y_resto(self):
        ventana = np.array([
            [100.0, 100.0],   # top y vende siempre
            [90.0, 0.0],      # top, no vende siempre
            [5.0, 5.0],       # vende siempre, fuera del top
            [1.0, 0.0],       # resto
            [0.0, 0.0],       # resto
        ])
        s = mp.segment_masks(ventana, top_n=2)
        self.assertEqual(s["top"].tolist(), [True, True, False, False, False])
        self.assertEqual(s["todos_los_meses"].tolist(), [True, False, True, False, False])
        self.assertEqual(s["resto"].tolist(), [False, False, False, True, True])


class CorreccionPorTamanoTest(unittest.TestCase):
    def test_modo_suma_corrige_cada_cubeta(self):
        pred = np.array([1.0] * 4 + [10.0] * 4)
        real = np.array([2.0] * 4 + [5.0] * 4)
        corr = mp.fit_bucket_correction(pred, real, n_buckets=2, mode="sum", min_n=1)
        out = mp.apply_bucket_correction(np.array([1.0, 10.0]), corr)
        np.testing.assert_allclose(out, [2.0, 5.0])

    def test_modo_acierto_usa_la_mediana_ponderada_de_real_sobre_pred(self):
        # razones real/pred 0,5 / 1 / 2 con pesos pred/real 2 / 1 / 0,5
        pred = np.ones(3)
        real = np.array([0.5, 1.0, 2.0])
        corr = mp.fit_bucket_correction(pred, real, n_buckets=1, mode="acc", min_n=1,
                                        clip=(0.1, 10.0))
        self.assertAlmostEqual(corr["factors"][0], 0.5)

    def test_el_factor_se_recorta(self):
        corr = mp.fit_bucket_correction([1.0, 1.0], [100.0, 100.0], n_buckets=1,
                                        mode="sum", min_n=1, clip=(0.25, 4.0))
        self.assertEqual(corr["factors"][0], 4.0)

    def test_cubetas_pequenas_no_se_corrigen(self):
        corr = mp.fit_bucket_correction([1.0, 1.0], [3.0, 3.0], n_buckets=1,
                                        mode="sum", min_n=5)
        self.assertEqual(corr["factors"][0], 1.0)

    def test_modo_invalido(self):
        with self.assertRaises(ValueError):
            mp.fit_bucket_correction([1.0], [1.0], mode="otro")

    def test_aplicar_conserva_ceros(self):
        corr = mp.fit_bucket_correction(np.arange(1.0, 21.0), np.arange(1.0, 21.0) * 2,
                                        n_buckets=4, mode="sum", min_n=1)
        out = mp.apply_bucket_correction(np.array([0.0, 5.0]), corr)
        self.assertEqual(out[0], 0.0)
        self.assertGreater(out[1], 5.0)


class TweedieTest(unittest.TestCase):
    def datos(self, n=300):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(n, 3))
        mu = np.exp(1 + 3 * X[:, 0])
        y = np.where(rng.uniform(size=n) < 0.6, rng.gamma(2.0, mu / 2.0), 0.0)
        return X, y

    def test_prediccion_no_negativa_y_finita(self):
        X, y = self.datos()
        m = mp.fit_tweedie(X, y, power=1.5, n_estimators=50)
        p = mp.predict_nonneg(m, X)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all(p >= 0))
        self.assertGreater(np.corrcoef(p, y)[0, 1], 0.3)

    def test_elige_la_potencia_con_menor_mae_de_validacion(self):
        X, y = self.datos(400)
        potencia, modelo, mae = mp.choose_tweedie(
            X[:300], y[:300], X[300:], y[300:], powers=(1.2, 1.5, 1.8), n_estimators=50)
        self.assertIn(potencia, (1.2, 1.5, 1.8))
        self.assertEqual(set(mae), {1.2, 1.5, 1.8})
        self.assertEqual(potencia, min(mae, key=mae.get))
        self.assertEqual(len(mp.predict_nonneg(modelo, X[300:])), 100)


if __name__ == "__main__":
    unittest.main()
