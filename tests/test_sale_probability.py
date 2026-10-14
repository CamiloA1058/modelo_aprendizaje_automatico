"""Probabilidad de venta (experimento): variables de recencia/frecuencia sin
mirar al futuro, marco de entrenamiento del objetivo binario 'vende', umbral
elegido con validacion interna y metricas."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import sale_probability as sp  # noqa: E402

ID, TOT, FREQ = "CODIGO", "TOTAL_VENDIDO", "FRECUENCIA"


def calendario(ventas, freq=None, pid=1, inicio="2024-01-01"):
    """Producto con calendario completo: una fila por mes."""
    n = len(ventas)
    f = freq if freq is not None else [1.0 if v > 0 else 0.0 for v in ventas]
    return pd.DataFrame({
        ID: pid,
        "fecha": pd.date_range(inicio, periods=n, freq="MS"),
        TOT: [float(v) for v in ventas],
        FREQ: [float(x) for x in f],
    })


def con_target(df):
    df = df.copy()
    df["target"] = df.groupby(ID)[TOT].shift(-1)
    return df


class FeaturesTest(unittest.TestCase):
    def setUp(self):
        # meses: 0:5 1:0 2:0 3:7 4:0 5:0 6:0 7:9
        self.df = calendario([5, 0, 0, 7, 0, 0, 0, 9],
                             freq=[2, 0, 0, 3, 0, 0, 0, 4])
        self.f = sp.add_probability_features(self.df, ID, TOT, FREQ)

    def test_meses_desde_la_ultima_venta(self):
        self.assertEqual(self.f["meses_desde_venta"].tolist(),
                         [0, 1, 2, 0, 1, 2, 3, 0])

    def test_meses_con_venta_en_ventanas_3_6_y_12(self):
        self.assertEqual(self.f["ventas_3m"].tolist(), [1, 1, 1, 1, 1, 1, 0, 1])
        self.assertEqual(self.f["ventas_6m"].tolist(), [1, 1, 1, 2, 2, 2, 1, 2])
        self.assertEqual(self.f["ventas_12m"].tolist(), [1, 1, 1, 2, 2, 2, 2, 3])

    def test_rezago_12_y_frecuencias_rezagadas(self):
        self.assertTrue(self.f["lag_12"].isna().all())
        self.assertTrue(np.isnan(self.f["frec_lag_1"].iloc[0]))
        self.assertEqual(self.f["frec_lag_1"].iloc[3], 0.0)
        self.assertEqual(self.f["frec_lag_3"].iloc[3], 2.0)
        self.assertEqual(self.f["frec_lag_1"].iloc[4], 3.0)
        self.assertAlmostEqual(self.f["frec_mean_3"].iloc[4], (3 + 0 + 0) / 3)

    def test_edad_y_fraccion_historica(self):
        self.assertEqual(self.f["edad_meses"].tolist(), list(range(8)))
        self.assertAlmostEqual(self.f["frac_hist"].iloc[3], 2 / 4)
        self.assertAlmostEqual(self.f["frac_hist"].iloc[7], 3 / 8)

    def test_mes_siguiente_ciclico(self):
        m = (self.df["fecha"].dt.month % 12) + 1
        np.testing.assert_allclose(self.f["mes_sig_sin"], np.sin(2 * np.pi * m / 12))
        np.testing.assert_allclose(self.f["mes_sig_cos"], np.cos(2 * np.pi * m / 12))

    def test_no_modifica_la_entrada(self):
        antes = self.df.copy()
        sp.add_probability_features(self.df, ID, TOT, FREQ)
        pd.testing.assert_frame_equal(self.df, antes)

    def test_sin_mirar_al_futuro(self):
        """Las variables de una fila no cambian al borrar los meses
        posteriores ni al alterar sus ventas."""
        largo = pd.concat([
            calendario([3, 0, 4, 0, 0, 2, 6, 0, 1, 0, 0, 5, 0, 7], pid=1),
            calendario([0.5, 2, 0, 0, 1, 1, 0, 3, 0, 0, 0, 2, 2, 0], pid=2),
        ], ignore_index=True)
        completo = sp.add_probability_features(largo, ID, TOT, FREQ)
        corte = pd.Timestamp("2024-08-01")
        recortado = sp.add_probability_features(
            largo[largo["fecha"] <= corte], ID, TOT, FREQ)
        alterado = largo.copy()
        alterado.loc[alterado["fecha"] > corte, [TOT, FREQ]] = 999.0
        alterado = sp.add_probability_features(alterado, ID, TOT, FREQ)
        previo = completo[completo["fecha"] <= corte].reset_index(drop=True)
        for col in sp.NEW_FEATURES + ["mes_sig_sin", "mes_sig_cos"]:
            pd.testing.assert_series_equal(
                previo[col], recortado.reset_index(drop=True)[col], check_names=False)
            pd.testing.assert_series_equal(
                previo[col],
                alterado[alterado["fecha"] <= corte].reset_index(drop=True)[col],
                check_names=False)

    def test_productos_independientes(self):
        dos = pd.concat([calendario([5, 0, 3], pid=1), calendario([0, 4, 4], pid=2)],
                        ignore_index=True)
        f = sp.add_probability_features(dos, ID, TOT, FREQ)
        b = f[f[ID] == 2]
        self.assertEqual(b["edad_meses"].tolist(), [0, 1, 2])
        self.assertEqual(b["ventas_3m"].tolist(), [0, 1, 2])


class TrainFrameTest(unittest.TestCase):
    def armar(self, ventas, pid=1):
        d = sp.add_probability_features(calendario(ventas, pid=pid), ID, TOT, FREQ)
        return con_target(d)

    def test_incluye_filas_con_mes_actual_en_cero_si_vendio_en_los_6_meses(self):
        d = self.armar([5, 0, 0, 0, 0, 0, 0, 0, 0, 4])
        frame, _, _ = sp.sale_train_frame(d, ID, TOT, 0.8)
        # filas con target: meses 0..8; vendio en la ventana (t-5..t): t=0..5
        self.assertEqual(frame["fecha"].dt.month.tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(frame["vende"].astype(int).tolist(), [0, 0, 0, 0, 0, 0])

    def test_excluye_fila_sin_venta_en_ventana_y_la_del_ultimo_mes(self):
        d = self.armar([5, 0, 0, 0, 0, 0, 0, 3])
        frame, _, _ = sp.sale_train_frame(d, ID, TOT, 0.8)
        meses = frame["fecha"].dt.month.tolist()
        self.assertNotIn(7, meses)
        self.assertNotIn(8, meses)
        self.assertEqual(max(meses), 6)

    def test_objetivo_vende_es_target_positivo(self):
        d = self.armar([5, 3, 0, 2, 0, 1])
        frame, _, _ = sp.sale_train_frame(d, ID, TOT, 0.8)
        self.assertEqual(frame["vende"].astype(int).tolist(), [1, 0, 1, 0, 1])

    def test_particion_temporal_80_20_por_producto(self):
        d = pd.concat([self.armar([1] * 11, pid=1), self.armar([2] * 6, pid=2)],
                      ignore_index=True)
        frame, tr, te = sp.sale_train_frame(d, ID, TOT, 0.8)
        self.assertEqual(len(frame), 10 + 5)
        self.assertEqual(sorted(tr + te), list(range(len(frame))))
        for pid, n_tr in ((1, 8), (2, 4)):
            g = frame[frame[ID] == pid]
            tr_g = [i for i in tr if i in g.index]
            te_g = [i for i in te if i in g.index]
            self.assertEqual(len(tr_g), n_tr)
            self.assertLess(frame.loc[tr_g, "fecha"].max(), frame.loc[te_g, "fecha"].min())


class ThresholdTest(unittest.TestCase):
    def test_maximiza_la_exactitud(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.35, 0.4, 0.8, 0.9])
        t = sp.choose_threshold(y, p)
        self.assertEqual(float(((p >= t) == y).mean()), 1.0)

    def test_empate_elige_el_mas_cercano_a_0_5(self):
        y = np.array([0, 1])
        p = np.array([0.2, 0.8])
        t = sp.choose_threshold(y, p)
        self.assertGreater(t, 0.2)
        self.assertLessEqual(t, 0.8)
        self.assertLess(abs(t - 0.5), 0.1)

    def test_umbral_alto_si_los_positivos_son_pocos(self):
        y = np.array([1, 0, 0, 0, 0, 0])
        p = np.array([0.9, 0.3, 0.2, 0.4, 0.35, 0.1])
        t = sp.choose_threshold(y, p)
        self.assertGreater(t, 0.4)
        self.assertLessEqual(t, 0.9)


class MetricsTest(unittest.TestCase):
    def test_metricas_de_clasificacion(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        pred = np.array([1, 1, 0, 0, 0, 1, 0, 1])
        p = np.array([.9, .8, .4, .2, .1, .6, .3, .7])
        m = sp.classification_metrics(y, pred, p)
        self.assertEqual(m["n"], 8)
        self.assertAlmostEqual(m["accuracy"], 6 / 8)
        self.assertAlmostEqual(m["precision_vende"], 3 / 4)
        self.assertAlmostEqual(m["recall_vende"], 3 / 4)
        self.assertAlmostEqual(m["precision_no"], 3 / 4)
        self.assertAlmostEqual(m["recall_no"], 3 / 4)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(m["auc"], 15 / 16)  # un par mal ordenado de 16
        self.assertAlmostEqual(m["tasa_vende"], 0.5)

    def test_sin_probabilidad_el_auc_es_nan(self):
        m = sp.classification_metrics([1, 0], [1, 0])
        self.assertTrue(np.isnan(m["auc"]))

    def test_una_sola_clase_no_falla(self):
        m = sp.classification_metrics([1, 1], [1, 0], [0.9, 0.2])
        self.assertTrue(np.isnan(m["auc"]))
        self.assertAlmostEqual(m["accuracy"], 0.5)

    def test_calibracion_por_deciles(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=2000)
        y = (rng.uniform(size=2000) < p).astype(int)
        t = sp.calibration_table(y, p, 10)
        self.assertEqual(len(t), 10)
        self.assertEqual(int(t["n"].sum()), 2000)
        self.assertTrue((t["prob_media"].diff().dropna() > 0).all())
        self.assertLess(float((t["prob_media"] - t["tasa_real"]).abs().max()), 0.1)


class RulesTest(unittest.TestCase):
    def test_reglas_simples(self):
        f = pd.DataFrame({TOT: [5.0, 0.0, 0.0, 2.0],
                          "ventas_3m": [3, 2, 1, 1],
                          "frac_hist": [0.9, 0.4, 0.6, 0.5]})
        r = sp.rule_predictions(f, TOT)
        self.assertEqual(r["vendio_mes_anterior"].tolist(), [True, False, False, True])
        self.assertEqual(r["2_de_3_meses"].tolist(), [True, True, False, False])
        self.assertEqual(r["fraccion_historica"].tolist(), [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
