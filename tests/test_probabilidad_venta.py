"""Probabilidad de venta (produccion, activada por defecto): un clasificador
binario "habra ventas el mes siguiente" entrenado solo con la particion de
entrenamiento escribe prob_venta (0-1) y se_vendera (Si/No) para todos los
productos de los resultados (modelo y estimacion por promedio). No cambia el
clasificador de 3 clases, el regresor, la mezcla ni la estimacion por
promedio; con la opcion en False la salida es la de siempre."""
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import sale_probability as sp  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402

N_MONTHS = 12
DROP = {0: [8, 11], 4: [9, 10, 11], 5: [0, 1, 2, 3, 4, 5, 11]}


def _trained(tmp, prob=None, drop=DROP, run_prob=True):
    kw = {} if prob is None else {"probabilidad_venta": prob}
    m = SalesForecastModel(filepath="unused.csv", min_months=6,
                           output_path=Path(tmp) / "pred.csv", **kw)
    m.df = synthetic_history(n_products=6, n_months=N_MONTHS, drop_months=drop)
    m._ultimo_mes = m.df["fecha"].max()
    m._mes_pred = m._ultimo_mes + pd.DateOffset(months=1)
    m.df = m._complete_calendar(m.df)
    m.build_features()
    m.cluster()
    m.prepare_split()
    m.train_classifier()
    m.train_regressor()
    m.predict_next_month()
    m.estimar_por_promedio()
    if run_prob:
        m.agregar_probabilidad_venta()
    return m


class ProbabilidadVentaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.m = _trained(cls._tmp.name)
        cls.res = cls.m.results
        cls.c = cls.m._c

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_activada_por_defecto(self):
        self.assertTrue(SalesForecastModel(filepath="x.csv").probabilidad_venta)

    def test_columnas_para_todos_los_productos(self):
        self.assertIn("prob_venta", self.res.columns)
        self.assertIn("se_vendera", self.res.columns)
        self.assertFalse(self.res["prob_venta"].isna().any())
        self.assertTrue(self.res["prob_venta"].between(0, 1).all())
        self.assertEqual(set(self.res["se_vendera"]) - {"Sí", "No"}, set())
        self.assertEqual(set(self.res["origen_estimacion"]), {"modelo", "promedio"})

    def test_se_vendera_sigue_el_umbral(self):
        t = self.m.sale_threshold
        self.assertGreater(t, 0)
        self.assertLess(t, 1)
        esperado = np.where(self.res["prob_venta"] >= t, "Sí", "No")
        self.assertEqual(list(self.res["se_vendera"]), list(esperado))

    def test_guarda_exactitud_de_validacion_interna(self):
        a = self.m.sale_val_accuracy
        self.assertTrue(0.0 <= a <= 1.0)

    def test_entrenado_solo_con_la_particion_de_entrenamiento(self):
        c = self.c
        d = sp.add_probability_features(
            self.m.df, c("id"), c("total_sold"), c("frequency"))
        frame, tr, te = sp.sale_train_frame(
            d, c("id"), c("total_sold"), self.m.train_ratio)
        self.assertEqual(self.m._sale_n_train, len(tr))
        self.assertEqual(self.m._sale_n_val, len(te))
        # el mes a predecir no tiene target: nunca entra al entrenamiento
        self.assertTrue(frame["target"].notna().all())
        self.assertNotIn(self.m._ultimo_mes, set(frame.loc[tr, "fecha"]))

    def test_csv_incluye_las_columnas(self):
        csv = pd.read_csv(self.m.output_path)
        self.assertIn("prob_venta", csv.columns)
        self.assertIn("se_vendera", csv.columns)
        self.assertEqual(len(csv), len(self.res))


class SinCambiosEnLoDemasTest(unittest.TestCase):
    def test_apagado_la_salida_es_la_de_siempre(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            off = _trained(t1, prob=False, run_prob=False)
            off2 = _trained(t2, prob=False)
            pd.testing.assert_frame_equal(off.results, off2.results)
            self.assertNotIn("prob_venta", off2.results.columns)
            self.assertNotIn("se_vendera", off2.results.columns)
            csv = pd.read_csv(off2.output_path)
            self.assertNotIn("prob_venta", csv.columns)
            self.assertIsNone(off2.sale_threshold)

    def test_encendido_no_cambia_clasificador_regresor_mezcla_ni_promedio(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            off = _trained(t1, prob=False)
            on = _trained(t2, prob=True)
            pd.testing.assert_frame_equal(
                off.results, on.results.drop(columns=["prob_venta", "se_vendera"]))
            self.assertEqual(off.blend_weight, on.blend_weight)
            pd.testing.assert_frame_equal(off.df_predict, on.df_predict)
            Xo = off.scaler.transform(off.df_predict[off._features].fillna(0))
            Xn = on.scaler.transform(on.df_predict[on._features].fillna(0))
            # tolerancia: el orden de suma de los arboles en paralelo puede
            # mover el ultimo bit (mismo comportamiento con la opcion apagada)
            np.testing.assert_allclose(off.clf.predict_proba(Xo),
                                       on.clf.predict_proba(Xn), atol=1e-12)
            np.testing.assert_allclose(off.reg.predict(Xo), on.reg.predict(Xn),
                                       atol=1e-9)


class UnaSolaClaseTest(unittest.TestCase):
    def test_sin_ceros_en_el_entrenamiento_no_falla(self):
        orig = sp.sale_train_frame

        def solo_ventas(*a, **k):
            frame, tr, te = orig(*a, **k)
            frame["vende"] = 1
            return frame, tr, te

        with tempfile.TemporaryDirectory() as tmp:
            m = _trained(tmp, run_prob=False)
            with mock.patch.object(sp, "sale_train_frame", solo_ventas):
                m.agregar_probabilidad_venta()
            self.assertTrue(m.results["prob_venta"].between(0, 1).all())
            self.assertEqual(set(m.results["se_vendera"]), {"Sí"})


if __name__ == "__main__":
    unittest.main()
