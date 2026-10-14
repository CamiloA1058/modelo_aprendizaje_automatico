"""Elegibilidad ampliada (experimento, desactivada por defecto): ayudantes
puros de ventas_forecast.eligibility y su integracion opcional en
SalesForecastModel. Con el parametro en False el comportamiento es el de
produccion."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import eligibility as el  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402


class DecisionBaseTest(unittest.TestCase):
    def test_usa_mes_actual_si_hubo_ventas_y_media_3_si_no(self):
        b = el.decision_base([100.0, 0.0, 0.0], [50.0, 30.0, np.nan])
        self.assertEqual(b[0], 100.0)
        self.assertEqual(b[1], 30.0)
        self.assertTrue(np.isnan(b[2]))

    def test_razon_de_crecimiento_contra_la_base(self):
        r = el.growth_ratio_base([60.0, 15.0, 5.0], [100.0, 0.0, 0.0], [90.0, 30.0, 0.0])
        self.assertAlmostEqual(r[0], 0.6)        # produccion: contra el mes actual
        self.assertAlmostEqual(r[1], 0.5)        # contra media_3
        self.assertTrue(np.isnan(r[2]))          # base 0: indefinida

    def test_siguiente_mes_en_cero_da_razon_cero(self):
        r = el.growth_ratio_base([0.0], [0.0], [40.0])
        self.assertEqual(r[0], 0.0)


class MasksTest(unittest.TestCase):
    def test_entrenamiento_por_defecto_es_la_regla_actual(self):
        cur = np.array([5.0, 0.0, 0.0, 7.0])
        m3 = np.array([4.0, 2.0, 0.0, np.nan])
        tgt = np.array([1.0, 1.0, 1.0, np.nan])
        np.testing.assert_array_equal(
            el.training_mask(cur, m3, tgt, False), [True, False, False, False])

    def test_entrenamiento_ampliado_agrega_actual_cero_con_media_3_positiva(self):
        cur = np.array([5.0, 0.0, 0.0, 7.0])
        m3 = np.array([4.0, 2.0, 0.0, np.nan])
        tgt = np.array([1.0, 1.0, 1.0, np.nan])
        np.testing.assert_array_equal(
            el.training_mask(cur, m3, tgt, True), [True, True, False, False])

    def test_prediccion_por_defecto_exige_ventas_del_ultimo_mes(self):
        cur = np.array([5.0, 0.0, 0.0])
        m3 = np.array([4.0, 2.0, 0.0])
        ok = np.array([True, True, True])
        np.testing.assert_array_equal(
            el.prediction_mask(cur, m3, ok, False), [True, False, False])
        np.testing.assert_array_equal(
            el.prediction_mask(cur, m3, ok, True), [True, True, False])

    def test_prediccion_exige_historial_minimo_tambien_ampliada(self):
        cur = np.array([0.0])
        np.testing.assert_array_equal(
            el.prediction_mask(cur, np.array([2.0]), np.array([False]), True), [False])


class SplitTest(unittest.TestCase):
    def test_filas_base_se_parten_igual_y_las_extra_siguen_la_fecha(self):
        # 12 filas base y 2 extra: una en la posicion 3 y otra en la 11
        is_base = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], bool)
        train = el.temporal_train_mask(is_base, 0.8)
        # n_base=12 -> cut=9: las primeras 9 base van a entrenamiento
        self.assertEqual(int(train[is_base].sum()), 9)
        self.assertTrue(train[3])        # extra anterior al primer test base
        self.assertFalse(train[11])      # extra posterior: prueba

    def test_sin_filas_extra_equivale_al_corte_actual(self):
        is_base = np.ones(10, bool)
        train = el.temporal_train_mask(is_base, 0.8)
        np.testing.assert_array_equal(train, [True] * 8 + [False] * 2)


def _model(ampliada, drop_last_for=(0,), min_months=6, n_months=12):
    drop = {p: [n_months - 4, n_months - 1] for p in drop_last_for}
    kw = {"elegibilidad_ampliada": ampliada} if ampliada is not None else {}
    m = SalesForecastModel(filepath="unused.csv", min_months=min_months, **kw)
    m.df = synthetic_history(n_products=4, n_months=n_months, drop_months=drop)
    m._ultimo_mes = m.df["fecha"].max()
    m._mes_pred = m._ultimo_mes + pd.DateOffset(months=1)
    m.df = m._complete_calendar(m.df)
    m.build_features()
    m.cluster()
    m.prepare_split()
    return m


class ModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = _model(None)
        cls.off = _model(False)
        cls.on = _model(True)

    def test_parametro_apagado_por_defecto(self):
        self.assertFalse(SalesForecastModel(filepath="x.csv").elegibilidad_ampliada)

    def test_defecto_conserva_conjunto_elegible_etiquetas_y_particion(self):
        c = self.default._c
        # producto 0 no vendio el ultimo mes: no se predice
        self.assertEqual(set(self.default.df_predict[c("id")]), {1, 2, 3})
        pd.testing.assert_frame_equal(self.default.df_predict, self.off.df_predict)
        pd.testing.assert_series_equal(self.default._y, self.off._y)
        self.assertEqual(self.default._train_idx, self.off._train_idx)
        self.assertEqual(self.default._test_idx, self.off._test_idx)
        self.assertTrue((self.default._df_train[c("total_sold")] > 0).all())

    def test_ampliada_agrega_el_producto_sin_ventas_del_ultimo_mes(self):
        c = self.on._c
        self.assertEqual(set(self.on.df_predict[c("id")]), {0, 1, 2, 3})
        nuevo = self.on.df_predict[self.on.df_predict[c("id")] == 0].iloc[0]
        self.assertEqual(nuevo[c("total_sold")], 0)
        self.assertGreater(nuevo["media_3_actual"], 0)

    def test_ampliada_filas_de_produccion_conservan_etiqueta(self):
        c = self.on._c
        base = self.off._df_train
        amp = self.on._df_train
        llave = [c("id"), "fecha"]
        a = amp.merge(base[llave + ["target_class"]], on=llave,
                      suffixes=("", "_base"))
        self.assertEqual(len(a), len(base))
        self.assertTrue((a["target_class"] == a["target_class_base"]).all())

    def test_ampliada_etiqueta_filas_nuevas_contra_media_3(self):
        c = self.on._c
        d = self.on._df_train
        extra = d[d[c("total_sold")] == 0]
        self.assertGreater(len(extra), 0)
        ratio = extra["target"] / extra["media_3_actual"]
        esperado = np.where(ratio > 1.15, 2, np.where(ratio < 0.90, 0, 1))
        np.testing.assert_array_equal(extra["target_class"].to_numpy(), esperado)

    def test_ampliada_particion_de_filas_base_no_cambia(self):
        c = self.on._c
        off_tr = set(map(tuple, self.off._df_train.loc[self.off._train_idx, [c("id"), "fecha"]].values))
        on_d = self.on._df_train
        on_tr = on_d.loc[self.on._train_idx]
        on_tr_base = set(map(tuple, on_tr[on_tr[c("total_sold")] > 0][[c("id"), "fecha"]].values))
        self.assertEqual(off_tr, on_tr_base)


if __name__ == "__main__":
    unittest.main()
