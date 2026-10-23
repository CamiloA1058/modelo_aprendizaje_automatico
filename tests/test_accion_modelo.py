"""Acción única en el pipeline de producción (SalesForecastModel.agregar_accion).

Ejecutar:  python -m unittest tests.test_accion_modelo
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import accion as ac  # noqa: E402
from ventas_forecast import sale_probability as sp  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402

DROP = {0: [8, 11], 4: [9, 10, 11], 5: [0, 1, 2, 3, 4, 5, 11]}


def modelo_completo(tmp, n_products=8, n_months=14, drop=DROP, hasta="accion", **kw):
    m = SalesForecastModel(filepath="unused.csv", min_months=6,
                           output_path=Path(tmp) / "pred.csv", **kw)
    m.df = synthetic_history(n_products=n_products, n_months=n_months, drop_months=drop)
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
    m.agregar_stock_sugerido()
    if hasta == "accion":
        m.agregar_accion()
    return m


class ClasificadorVarianteTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.m = modelo_completo(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_variables_de_regularidad_y_pesos_por_arbol(self):
        m = self.m
        self.assertEqual(m._clf_features, [*m._features, *sp.NEW_FEATURES])
        self.assertEqual(m.clf.n_features_in_, len(m._clf_features))
        self.assertEqual(m.clf.class_weight, "balanced_subsample")
        self.assertEqual(m.clf.min_samples_leaf, 3)

    def test_el_regresor_sigue_con_sus_variables(self):
        self.assertEqual(self.m.reg.n_features_in_, len(self.m._features))

    def test_umbral_de_reforzar_en_la_rejilla(self):
        self.assertIn(self.m.umbral_reforzar, [float(t) for t in ac.GRID_T])
        self.assertIn(self.m.regla_umbral_reforzar, ("precision", "f05"))

    def test_resultados_con_accion_y_montos(self):
        r = self.m.results
        for col in ("accion", "accion_nombre", "monto_si_vende", "venta_esperada",
                    "monto_mostrado", "monto_rango"):
            self.assertIn(col, r.columns)
        self.assertTrue(set(r["accion"]) <= set(ac.ACCIONES))
        self.assertEqual(r["accion_nombre"].tolist(), [ac.nombre(a) for a in r["accion"]])
        csv = pd.read_csv(self.m.output_path)
        self.assertIn("accion", csv.columns)

    def test_reforzar_solo_si_comprar_mas(self):
        r = self.m.results
        self.assertEqual(((r["decision"] == "Reforzar stock") != (r["accion"] == "comprar_mas"))
                         .sum(), 0)

    def test_venta_esperada_es_probabilidad_por_monto_si_vende(self):
        r = self.m.results
        con = r["ventas_predichas"].notna()
        np.testing.assert_allclose(r.loc[con, "venta_esperada"],
                                   r.loc[con, "prob_venta"] * r.loc[con, "monto_si_vende"])
        self.assertTrue(r.loc[~con, "venta_esperada"].isna().all())

    def test_sin_combinaciones_prohibidas(self):
        m = self.m
        r = m.results.assign(base=m.results[m._c("total_sold")])
        cuenta = ac.contradicciones(r, m.sale_threshold)
        self.assertEqual(cuenta["comprar_sin_venta"], 0)
        self.assertEqual(cuenta["no_reponer_con_monto"], 0)
        self.assertEqual(cuenta["mantener_sin_stock"], 0)
        if m.regla_stock["adoptada"]:
            self.assertEqual(cuenta["no_reponer_con_stock"], 0)
        self.assertEqual(m.contradicciones, cuenta)

    def test_regla_de_stock_evaluada_con_la_validacion_interna(self):
        rs = self.m.regla_stock
        for k in ("n", "cobertura_nivel", "cobertura_regla", "stock_real_nivel",
                  "stock_real_regla", "adoptada"):
            self.assertIn(k, rs)
        self.assertGreater(rs["n"], 0)

    def test_stock_segun_la_regla(self):
        m = self.m
        r = m.results
        nr = (r["accion"] == "no_reponer") & r["ventas_predichas"].notna()
        rev = r["accion"] == "revisar"
        if m.regla_stock["adoptada"]:
            self.assertTrue((r.loc[nr, "stock_sugerido"] == 0).all())
            np.testing.assert_allclose(r.loc[rev, "stock_sugerido"],
                                       r.loc[rev, "ventas_predichas"])


class DesactivadoTest(unittest.TestCase):

    def test_sin_accion_no_agrega_columnas(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = modelo_completo(tmp, accion_unica=False)
            self.assertNotIn("accion", m.results.columns)


if __name__ == "__main__":
    unittest.main()
