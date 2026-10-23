"""
Ninguna seleccion usa los datos sobre los que se reporta.

- El peso w de la mezcla w*RF + (1-w)*media_3, el umbral de la probabilidad de
  venta y los cuantiles del stock sugerido se eligen con una validacion interna
  temporal DENTRO de la particion de entrenamiento (ultimo ~20 % de los meses
  de entrenamiento de cada producto).
- El umbral t de Reforzar, s_baja (No reponer), el monto si se vende y la
  regla de stock por accion tambien se eligen con esa validacion interna.
- La particion de prueba interna (ultimo 20 % de los meses de cada producto)
  solo se usa para reportar: si se cambian sus objetivos, ninguna seleccion
  cambia.

Ejecutar:  python -m unittest tests.test_seleccion_sin_prueba
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ventas_forecast.model as tk  # noqa: E402
from ventas_forecast import improvements as im  # noqa: E402
from ventas_forecast import validacion_interna as vi  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402

DROP = {0: [8, 11], 4: [9, 10, 11], 5: [0, 1, 2, 3, 4, 5, 11]}


def _preparado(tmp, n_products=6, n_months=24, drop=None, **kw):
    m = SalesForecastModel(filepath="unused.csv", min_months=6,
                           output_path=Path(tmp) / "pred.csv", **kw)
    m.df = synthetic_history(n_products=n_products, n_months=n_months,
                             drop_months=drop)
    m._ultimo_mes = m.df["fecha"].max()
    m._mes_pred = m._ultimo_mes + pd.DateOffset(months=1)
    if drop:
        m.df = m._complete_calendar(m.df)
    m.build_features()
    m.cluster()
    m.prepare_split()
    return m


def _llaves(df, id_col):
    return set(zip(df[id_col], pd.to_datetime(df["fecha"])))


class ParticionTemporalTest(unittest.TestCase):

    def test_primeras_filas_de_cada_producto_por_fecha(self):
        ids = ["a"] * 5 + ["b"] * 4
        fechas = pd.to_datetime([
            "2024-03-01", "2024-01-01", "2024-05-01", "2024-02-01", "2024-04-01",
            "2024-02-01", "2024-01-01", "2024-04-01", "2024-03-01"])
        mask = vi.particion_temporal(ids, fechas, 0.8)
        # a: 5 filas -> 4 primeras (ene..abr); b: 4 filas -> 3 primeras (ene..mar)
        esperado = [True, True, False, True, True, True, True, False, True]
        self.assertEqual(mask.tolist(), esperado)

    def test_un_producto_con_una_fila_queda_en_validacion(self):
        # int(1 * 0.8) = 0, igual que _build_train_frame
        self.assertEqual(vi.particion_temporal(["a"], pd.to_datetime(["2024-01-01"]), 0.8)
                         .tolist(), [False])


class PesoMezclaSinPruebaTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _modelo(self):
        return _preparado(self._tmp.name)

    def _w_con_objetivos_de_prueba(self, transformar):
        m = self._modelo()
        idx = m._test_idx
        m._df_train.loc[idx, "target"] = transformar(m._df_train.loc[idx, "target"])
        m.train_regressor()
        return m

    def test_w_no_cambia_si_cambian_los_objetivos_de_prueba(self):
        base = self._modelo()
        base.train_regressor()
        por_cien = self._w_con_objetivos_de_prueba(lambda t: t * 100.0)
        en_cero = self._w_con_objetivos_de_prueba(lambda t: t * 0.0)
        self.assertEqual(por_cien.blend_weight, base.blend_weight)
        self.assertEqual(en_cero.blend_weight, base.blend_weight)
        # Lo reportado si cambia: la prueba interna es solo para reportar.
        self.assertNotEqual(por_cien.blend_metrics["mae"], base.blend_metrics["mae"])

    def test_el_w_optimo_en_la_prueba_puede_ser_otro(self):
        """Datos en los que el w que minimiza el sesgo en la prueba interna no
        es el elegido: la seleccion no lo mira."""
        optimos, elegidos = set(), set()
        for factor in (100.0, 0.0):
            m = self._w_con_objetivos_de_prueba(lambda t, f=factor: t * f)
            res = m.reg_test_results
            optimos.add(im.choose_blend_weight(res["real"], res["predicho"],
                                               m.reg_test_media_3, criterion="sesgo"))
            elegidos.add(m.blend_weight)
        self.assertEqual(len(optimos), 2)   # el optimo en la prueba cambia...
        self.assertEqual(len(elegidos), 1)  # ...el elegido no
        self.assertTrue(optimos - elegidos)

    def test_w_se_elige_con_la_validacion_interna_del_entrenamiento(self):
        m = self._modelo()
        capturado = {}
        real_choose = im.choose_blend_weight

        def espia(real, rf, base, *a, **kw):
            capturado.update(real=np.asarray(real, float), rf=np.asarray(rf, float),
                             base=np.asarray(base, float), kw=kw)
            return real_choose(real, rf, base, *a, **kw)

        with mock.patch.object(tk.im, "choose_blend_weight", espia):
            m.train_regressor()
        v = m.blend_validacion
        self.assertEqual(capturado["kw"].get("criterion"), "sesgo")
        np.testing.assert_allclose(capturado["real"], v["real"].to_numpy(float))
        np.testing.assert_allclose(capturado["rf"], v["rf"].to_numpy(float))
        np.testing.assert_allclose(capturado["base"], v["media_3"].to_numpy(float))

        c = m._c
        prueba = _llaves(m._df_train.loc[m._test_idx], c("id"))
        entrenamiento = _llaves(m._df_train.loc[m._train_idx], c("id"))
        validacion = set(zip(v["id"], pd.to_datetime(v["fecha"])))
        self.assertGreater(len(validacion), 0)
        self.assertTrue(validacion <= entrenamiento)
        self.assertFalse(validacion & prueba)

    def test_validacion_interna_posterior_al_entrenamiento_interno(self):
        m = self._modelo()
        p = m._particion_interna()
        c = m._c
        d = m._df_train
        self.assertEqual(sorted(p["tr"] + p["va"]), sorted(m._train_idx))
        for prod, g in d.loc[p["va"]].groupby(c("id")):
            tr = d.loc[p["tr"]]
            tr = tr[tr[c("id")] == prod]
            self.assertGreater(len(tr), 0)
            self.assertLess(tr["fecha"].max(), g["fecha"].min())

    def test_el_modelo_interno_no_ve_los_objetivos_de_validacion(self):
        base = self._modelo()
        base.train_regressor()
        m = self._modelo()
        va = m._particion_interna()["va"]
        m._df_train.loc[va, "target"] = m._df_train.loc[va, "target"] * 7.0 + 1.0
        m.train_regressor()
        np.testing.assert_allclose(m.blend_validacion["rf"].to_numpy(float),
                                   base.blend_validacion["rf"].to_numpy(float))

    def test_kmeans_interno_solo_con_entrenamiento_interno(self):
        m = self._modelo()
        p = m._particion_interna()
        esc, km = p["cluster"]
        esperado = m._cluster_features(m._df_train.loc[p["tr"]])
        np.testing.assert_allclose(esc.center_, np.median(esperado.to_numpy(float), axis=0))
        self.assertEqual(km.n_clusters, m.n_clusters)


class UmbralVentaSinPruebaTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _modelo(self):
        m = _preparado(self._tmp.name, n_months=14, drop=DROP)
        m.train_classifier()
        m.train_regressor()
        m.predict_next_month()
        m.estimar_por_promedio()
        return m

    def _con_prueba_invertida(self):
        m = self._modelo()
        real = tk.sp.sale_train_frame

        def invertida(*a, **kw):
            frame, tr, te = real(*a, **kw)
            t = frame.loc[te, "target"]
            frame.loc[te, "target"] = np.where(t > 0, 0.0, 1000.0)
            frame.loc[te, "vende"] = (frame.loc[te, "target"] > 0).astype(int)
            return frame, tr, te

        with mock.patch.object(tk.sp, "sale_train_frame", invertida):
            m.agregar_probabilidad_venta()
        return m

    def test_umbral_no_cambia_si_cambian_las_etiquetas_de_prueba(self):
        base = self._modelo()
        base.agregar_probabilidad_venta()
        inv = self._con_prueba_invertida()
        self.assertEqual(inv.sale_threshold, base.sale_threshold)
        self.assertNotEqual(inv.sale_test_accuracy, base.sale_test_accuracy)

    def test_exactitud_reportada_en_la_prueba_interna(self):
        m = self._modelo()
        capturado = {}
        real = tk.sp.sale_train_frame

        def espia(*a, **kw):
            r = real(*a, **kw)
            capturado["r"] = r
            return r

        with mock.patch.object(tk.sp, "sale_train_frame", espia):
            m.agregar_probabilidad_venta()
        frame, tr, te = capturado["r"]
        feats = m._sale_features
        p = m.sale_clf.predict_proba(frame.iloc[te][feats].fillna(0).to_numpy())[:, 1]
        y = frame.iloc[te]["vende"].to_numpy()
        self.assertAlmostEqual(m.sale_test_accuracy,
                               float(((p >= m.sale_threshold).astype(int) == y).mean()))
        # La validacion interna (seleccion) es parte del entrenamiento.
        c = m._c
        val = set(zip(m._sale_val["id"], pd.to_datetime(m._sale_val["fecha"])))
        self.assertTrue(val <= _llaves(frame.iloc[tr], c("id")))
        self.assertFalse(val & _llaves(frame.iloc[te], c("id")))


class StockSinPruebaTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _modelo(self, transformar=None):
        m = _preparado(self._tmp.name, n_months=14, drop=DROP)
        if transformar is not None:
            idx = m._test_idx
            m._df_train.loc[idx, "target"] = transformar(m._df_train.loc[idx, "target"])
        m.train_classifier()
        m.train_regressor()
        m.predict_next_month()
        m.estimar_por_promedio()
        m.agregar_probabilidad_venta()
        m.agregar_stock_sugerido()
        return m

    def test_calibracion_sin_filas_de_la_prueba_interna(self):
        m = self._modelo()
        c = m._c
        v = m.stock_validacion
        self.assertGreater(len(v), 0)
        prueba = _llaves(m._df_train.loc[m._test_idx], c("id"))
        self.assertFalse(set(zip(v["id"], pd.to_datetime(v["fecha"]))) & prueba)

    def test_cuantiles_no_cambian_si_cambian_los_objetivos_de_prueba(self):
        base = self._modelo()
        otro = self._modelo(lambda t: t * 100.0)
        a, b = base.calibracion_stock, otro.calibracion_stock
        self.assertEqual(a["edges"], b["edges"])
        self.assertEqual(a["n"], b["n"])
        self.assertEqual(a["n_total"], b["n_total"])
        # tolerancia solo por el orden de suma en paralelo de los arboles
        for alfa in a["q"]:
            np.testing.assert_allclose(a["q"][alfa], b["q"][alfa], rtol=1e-9)
            self.assertAlmostEqual(a["global"][alfa], b["global"][alfa], places=9)


class AccionSinPruebaTest(unittest.TestCase):
    """t de Reforzar, s_baja, el monto si se vende y la regla de stock se
    eligen con la validacion interna: cambiar las etiquetas y objetivos de la
    prueba interna (clasificador de 3 clases, regresor y modelo de venta) o las
    ventas del mes predicho (holdout, que el modelo nunca ve) no los cambia."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _modelo(self, perturbar=False):
        m = _preparado(self._tmp.name, n_products=8, n_months=14, drop=DROP)
        if perturbar:
            te = m._test_idx
            m._y.loc[te] = (m._y.loc[te] + 1) % 3
            m._df_train.loc[te, "target"] = m._df_train.loc[te, "target"] * 100.0
        m.train_classifier()
        m.train_regressor()
        m.predict_next_month()
        m.estimar_por_promedio()
        real = tk.sp.sale_train_frame

        def invertida(*a, **kw):
            frame, tr, te = real(*a, **kw)
            if perturbar:
                t = frame.loc[te, "target"]
                frame.loc[te, "target"] = np.where(t > 0, 0.0, 1000.0)
                frame.loc[te, "vende"] = (frame.loc[te, "target"] > 0).astype(int)
            return frame, tr, te

        with mock.patch.object(tk.sp, "sale_train_frame", invertida):
            m.agregar_probabilidad_venta()
        m.agregar_stock_sugerido()
        m.agregar_accion()
        return m

    def test_selecciones_no_cambian_con_la_prueba_interna(self):
        base = self._modelo()
        otro = self._modelo(perturbar=True)
        self.assertIsNotNone(base.umbral_reforzar)
        self.assertEqual(otro.umbral_reforzar, base.umbral_reforzar)
        self.assertEqual(otro.regla_umbral_reforzar, base.regla_umbral_reforzar)
        self.assertEqual(otro.sale_threshold, base.sale_threshold)
        self.assertEqual(otro.s_baja, base.s_baja)
        self.assertEqual(otro.monto_si_vende_clave, base.monto_si_vende_clave)
        a, b = base.regla_stock, otro.regla_stock
        self.assertEqual(a["adoptada"], b["adoptada"])
        self.assertEqual(a["n"], b["n"])
        for k in ("cobertura_nivel", "cobertura_regla", "stock_real_nivel", "stock_real_regla"):
            self.assertAlmostEqual(a[k], b[k], places=9)
        # Lo reportado si cambia: la prueba interna es solo para reportar.
        self.assertNotEqual(otro.metrics, base.metrics)

    def test_validacion_del_clasificador_sin_filas_de_prueba(self):
        m = self._modelo()
        c = m._c
        v = m._clf_validacion
        self.assertGreater(len(v), 0)
        self.assertFalse(set(v.index) & set(m._test_idx))
        self.assertTrue(set(v.index) <= set(m._train_idx))
        prueba = _llaves(m._df_train.loc[m._test_idx], c("id"))
        self.assertFalse(set(zip(v["id"], pd.to_datetime(v["fecha"]))) & prueba)


if __name__ == "__main__":
    unittest.main()
