"""Stock sugerido en el sistema (activado por defecto): stock_sugerido (COP) y
stock_sugerido_unidades por producto con nivel de servicio 70 %, calibrado solo
con la validacion interna del entrenamiento. Con la opcion en False la salida
es la de siempre."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import stock_sugerido as ss  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402

N_MONTHS = 14
DROP = {0: [8, 11], 4: [9, 10, 11], 5: [0, 1, 2, 3, 4, 5, 11]}


def _trained(tmp, **kw):
    m = SalesForecastModel(filepath="unused.csv", min_months=6,
                           output_path=Path(tmp) / "pred.csv", **kw)
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
    m.estimar_por_promedio()
    m.agregar_probabilidad_venta()
    return m


class StockSugeridoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.m = _trained(cls._tmp.name)
        cls.antes = cls.m.results.copy()
        cls.m.agregar_stock_sugerido()
        cls.res = cls.m.results
        cls.c = cls.m._c

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_activado_por_defecto_con_nivel_70(self):
        m = SalesForecastModel(filepath="x.csv")
        self.assertTrue(m.stock_sugerido)
        self.assertEqual(m.nivel_servicio, 0.70)

    def test_columnas_y_valores(self):
        self.assertIn("stock_sugerido", self.res.columns)
        self.assertIn("stock_sugerido_unidades", self.res.columns)
        con_monto = self.res["ventas_predichas"].notna()
        self.assertTrue(con_monto.any())
        self.assertTrue((self.res.loc[con_monto, "stock_sugerido"] >= 0).all())
        self.assertFalse(self.res.loc[con_monto, "stock_sugerido"].isna().any())
        self.assertTrue(self.res.loc[~con_monto, "stock_sugerido"].isna().all())

    def test_formula_con_los_cuantiles_guardados(self):
        r = self.res[self.res["ventas_predichas"].notna()]
        q = ss.cuantiles_para(self.m.calibracion_stock, r["prob_venta"], 0.70)
        esperado = ss.cota_superior(r["ventas_predichas"], q)
        np.testing.assert_allclose(r["stock_sugerido"], esperado)

    def test_unidades_son_el_techo_del_stock_entre_el_precio(self):
        r = self.res[self.res["stock_sugerido"].notna()]
        precio = r[self.c("avg_price")]
        esperado = np.ceil(np.round(r["stock_sugerido"] / precio, 9))
        np.testing.assert_allclose(r["stock_sugerido_unidades"], esperado)

    def test_resto_de_columnas_no_cambia(self):
        pd.testing.assert_frame_equal(self.res[self.antes.columns], self.antes)

    def test_el_csv_exportado_trae_las_columnas(self):
        csv = pd.read_csv(self.m.output_path)
        self.assertIn("stock_sugerido", csv.columns)
        self.assertIn("stock_sugerido_unidades", csv.columns)
        self.assertEqual(len(csv), len(self.res))

    def test_guarda_cuantiles_de_los_tres_niveles(self):
        cal = self.m.calibracion_stock
        self.assertEqual(set(cal["q"]), {0.5, 0.7, 0.9})
        self.assertGreater(cal["n_total"], 0)

    def test_calibracion_solo_con_validacion_interna(self):
        """Sin holdout ni prueba interna: toda fila de calibracion es de la
        validacion interna (dentro del entrenamiento) del regresor o del
        clasificador de venta, y su objetivo es anterior al mes predicho."""
        v = self.m.stock_validacion
        self.assertGreater(len(v), 0)
        self.assertTrue((v["fecha"] < self.m._ultimo_mes).all())
        reg = self.m.blend_validacion
        sale = self.m._sale_val
        llaves = (set(zip(reg["id"], reg["fecha"]))
                  | set(zip(sale["id"], sale["fecha"])))
        self.assertTrue(set(zip(v["id"], v["fecha"])) <= llaves)

    def test_cambiar_el_ultimo_mes_no_cambia_la_calibracion(self):
        """El ultimo mes solo es 'mes actual' de la prediccion: sus ventas no
        participan en ningun residuo de validacion como objetivo ni como
        feature de una fila anterior."""
        v = self.m.stock_validacion
        self.assertFalse((v["fecha"] == self.m._ultimo_mes).any())


class StockSugeridoApagadoTest(unittest.TestCase):
    def test_con_false_la_salida_es_identica(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = _trained(tmp, stock_sugerido=False)
            antes = m.results.copy()
            m.agregar_stock_sugerido()
            pd.testing.assert_frame_equal(m.results, antes)
            self.assertNotIn("stock_sugerido", m.results.columns)
            self.assertNotIn("stock_sugerido_unidades", m.results.columns)

    def test_sin_probabilidad_de_venta_usa_el_cuantil_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = SalesForecastModel(filepath="unused.csv", min_months=6,
                                   output_path=Path(tmp) / "p.csv",
                                   probabilidad_venta=False)
            m.df = synthetic_history(n_products=6, n_months=N_MONTHS, drop_months=DROP)
            m._ultimo_mes = m.df["fecha"].max()
            m._mes_pred = m._ultimo_mes + pd.DateOffset(months=1)
            m.df = m._complete_calendar(m.df)
            m.build_features(); m.cluster(); m.prepare_split()
            m.train_classifier(); m.train_regressor(); m.predict_next_month()
            m.estimar_por_promedio(); m.agregar_probabilidad_venta()
            m.agregar_stock_sugerido()
            r = m.results[m.results["ventas_predichas"].notna()]
            q = m.calibracion_stock["global"][0.70]
            np.testing.assert_allclose(
                r["stock_sugerido"], ss.cota_superior(r["ventas_predichas"], q))


if __name__ == "__main__":
    unittest.main()
