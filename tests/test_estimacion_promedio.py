"""Estimacion por promedio (produccion, activada por defecto): los productos
sin ventas el ultimo mes pero con ventas en alguno de los 3 meses previos
(y >= min_months meses con ventas, dentro del top de productos) reciben como
monto su media de 3 meses y NO reciben decision del modelo. El entrenamiento
no cambia. Con el parametro en False la salida es la de siempre."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.model import (  # noqa: E402
    ETIQUETA_SIN_RECOMENDACION, SalesForecastModel,
)

from tests._helpers import synthetic_history  # noqa: E402

N_MONTHS = 12
# Producto 0: sin ventas el ultimo mes y en el mes 8 (vendio en 9 y 10) -> promedio
# Producto 4: sin ventas en los 3 ultimos meses (media_3 = 0)           -> excluido
# Producto 5: solo 5 meses con ventas (< min_months) y 0 el ultimo mes  -> excluido
DROP = {0: [8, 11], 4: [9, 10, 11], 5: [0, 1, 2, 3, 4, 5, 11]}


def _trained(tmp, estimacion=None, top_n_products=1900):
    kw = {} if estimacion is None else {"estimacion_por_promedio": estimacion}
    m = SalesForecastModel(filepath="unused.csv", min_months=6,
                           output_path=Path(tmp) / "pred.csv",
                           top_n_products=top_n_products, **kw)
    m.df = synthetic_history(n_products=6, n_months=N_MONTHS, drop_months=DROP)
    m._ultimo_mes = m.df["fecha"].max()
    m._mes_pred = m._ultimo_mes + pd.DateOffset(months=1)
    m.df = m._complete_calendar(m.df)
    m.build_features()
    m.cluster()
    m.prepare_split()
    m.train_classifier()
    m.train_regressor()
    m.predict_next_month()
    return m


class EstimacionPorPromedioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.m = _trained(cls._tmp.name)
        cls.antes = cls.m.results.copy()
        cls.m.estimar_por_promedio()
        cls.res = cls.m.results
        cls.c = cls.m._c

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_activada_por_defecto(self):
        self.assertTrue(SalesForecastModel(filepath="x.csv").estimacion_por_promedio)

    def test_solo_el_producto_elegible_pasa_a_promedio(self):
        c = self.c
        prom = self.res[self.res["origen_estimacion"] == "promedio"]
        self.assertEqual(set(prom[c("id")]), {0})

    def test_monto_es_media_3_sin_modelo(self):
        c = self.c
        fila = self.res[self.res[c("id")] == 0].iloc[0]
        h = self.m.df[(self.m.df[c("id")] == 0)].sort_values("fecha")
        esperado = h[c("total_sold")].tail(3).mean()
        self.assertGreater(esperado, 0)
        self.assertAlmostEqual(fila["ventas_predichas"], esperado)
        self.assertTrue(np.isnan(fila["ventas_predichas_rf"]))
        self.assertEqual(fila[c("total_sold")], 0)

    def test_sin_decision_del_modelo(self):
        fila = self.res[self.res["origen_estimacion"] == "promedio"].iloc[0]
        self.assertEqual(fila["decision"], ETIQUETA_SIN_RECOMENDACION)
        self.assertEqual(ETIQUETA_SIN_RECOMENDACION,
                         "Sin recomendación (estimación por promedio)")
        self.assertTrue(np.isnan(fila["prob_crecimiento"]))
        self.assertTrue(np.isnan(fila["prob_reduccion"]))

    def test_productos_del_modelo_no_cambian(self):
        n = len(self.antes)
        modelo = self.res[self.res["origen_estimacion"] == "modelo"]
        self.assertEqual(len(modelo), n)
        pd.testing.assert_frame_equal(
            modelo.drop(columns="origen_estimacion"), self.antes)

    def test_excluye_sin_media_3_o_sin_historial_minimo(self):
        c = self.c
        self.assertNotIn(4, set(self.res[c("id")]))
        self.assertNotIn(5, set(self.res[c("id")]))

    def test_csv_exportado_incluye_origen(self):
        csv = pd.read_csv(self.m.output_path)
        self.assertIn("origen_estimacion", csv.columns)
        self.assertEqual(set(csv["origen_estimacion"]), {"modelo", "promedio"})
        self.assertEqual(len(csv), len(self.res))

    def test_restringe_a_los_top_productos(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = _trained(tmp, top_n_products=2)
            self.assertNotIn(0, set(m._top_prods))
            m.estimar_por_promedio()
            self.assertEqual(
                (m.results["origen_estimacion"] == "promedio").sum(), 0)


class ParametroApagadoTest(unittest.TestCase):
    def test_apagado_la_salida_es_la_de_siempre(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            off = _trained(t1, estimacion=False)
            on = _trained(t2, estimacion=True)
            antes = off.results.copy()
            off.estimar_por_promedio()
            pd.testing.assert_frame_equal(off.results, antes)
            self.assertNotIn("origen_estimacion", off.results.columns)
            csv = pd.read_csv(off.output_path)
            self.assertNotIn("origen_estimacion", csv.columns)
            self.assertEqual(len(csv), len(antes))

    def test_entrenamiento_identico_con_y_sin_estimacion(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            off = _trained(t1, estimacion=False)
            on = _trained(t2, estimacion=True)
            self.assertFalse(on.elegibilidad_ampliada)
            pd.testing.assert_frame_equal(off._df_train, on._df_train)
            self.assertEqual(off._train_idx, on._train_idx)
            self.assertEqual(off._test_idx, on._test_idx)
            pd.testing.assert_frame_equal(off.df_predict, on.df_predict)
            pd.testing.assert_frame_equal(off.results, on.results)


if __name__ == "__main__":
    unittest.main()
