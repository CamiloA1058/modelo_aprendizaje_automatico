"""Dashboard con la acción única: tarjetas por acción, total = venta esperada,
filtro por acción, detalle del producto y ninguna combinación contradictoria
en las filas que muestra la API.

Ejecutar:  python -m unittest tests.test_dashboard_accion
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))
import app as dashboard_app  # noqa: E402
from ventas_forecast import accion as ac  # noqa: E402
from ventas_forecast.model import ETIQUETA_SIN_RECOMENDACION, SalesForecastModel  # noqa: E402

from tests.test_accion_modelo import modelo_completo  # noqa: E402


def _resultados(c):
    return pd.DataFrame({
        c("id"):            ["A", "B", "C", "D", "E", "F"],
        c("description"):   ["FRENO A", "ACEITE B", "FILTRO C", "VALVULA D", "TUBO E", "TAPA F"],
        c("total_sold"):    [100.0, 200.0, 50.0, 80.0, 0.0, 30.0],
        "ventas_predichas": [130.0, 180.0, 60.0, np.nan, 40.0, 25.0],
        "decision":         ["Reforzar stock", "Reducir stock", "Mantener stock",
                             "Mantener stock", ETIQUETA_SIN_RECOMENDACION, "Mantener stock"],
        "prob_crecimiento": [0.8, 0.1, 0.4, 0.3, np.nan, 0.2],
        "prob_reduccion":   [0.1, 0.7, 0.2, 0.3, np.nan, 0.5],
        "cluster":          [0, 1, 0, 1, 0, 2],
        "origen_estimacion": ["modelo", "modelo", "modelo", "modelo", "promedio", "modelo"],
        "prob_venta":       [0.9, 0.3, 0.8, 0.7, 0.5, 0.62],
        "se_vendera":       ["Sí", "No", "Sí", "Sí", "No", "No"],
        "stock_sugerido":   [160.0, 0.0, 70.0, np.nan, 45.0, 25.0],
        "stock_sugerido_unidades": [4.0, 0.0, 2.0, np.nan, 1.0, 1.0],
        "accion":           ["comprar_mas", "no_reponer", "mantener", "mantener",
                             "por_promedio", "revisar"],
        "accion_nombre":    ["Comprar más", "No reponer este mes", "Mantener", "Mantener",
                             "Por promedio", "Revisar"],
        "monto_si_vende":   [140.0, 170.0, 65.0, np.nan, 50.0, 30.0],
        "venta_esperada":   [126.0, 51.0, 52.0, np.nan, 25.0, 18.6],
        "monto_mostrado":   [126.0, 0.0, 52.0, np.nan, 0.0, 18.6],
        "monto_rango":      [False, False, False, False, False, True],
    })


class AccionApiTest(unittest.TestCase):
    def setUp(self):
        model = SalesForecastModel(filepath="unused.csv")
        c = model._c
        model.df = pd.DataFrame({
            c("id"): ["A", "B", "C", "E", "F"], "fecha": pd.Timestamp("2026-03-01"),
            c("total_sold"): [100.0, 200.0, 50.0, 0.0, 30.0], c("description"): "x",
            c("frequency"): 1, c("avg_price"): 10.0})
        model.metrics = {"accuracy": 0.6, "f1_macro": 0.4, "balanced_accuracy": 0.45}
        model.sale_threshold = 0.63
        model.s_baja = 0.6
        model.umbral_reforzar = 0.66
        self._saved = dict(dashboard_app._state)
        dashboard_app._state.update(model=model, results=_resultados(c),
                                    mes_predicho="2026-04")
        self.client = dashboard_app.app.test_client()

    def tearDown(self):
        dashboard_app._state.clear()
        dashboard_app._state.update(self._saved)

    def test_summary_cuenta_por_accion(self):
        d = self.client.get("/api/summary").get_json()
        self.assertEqual(d["acciones"], {"comprar_mas": 1, "mantener": 2, "no_reponer": 1,
                                         "revisar": 1, "por_promedio": 1})
        self.assertEqual([a["clave"] for a in d["acciones_catalogo"]], list(ac.ACCIONES))
        self.assertEqual(d["acciones_catalogo"][2]["nombre"], "No reponer este mes")

    def test_total_es_la_venta_esperada_de_todos_los_productos_con_monto(self):
        d = self.client.get("/api/summary").get_json()
        self.assertAlmostEqual(d["pred_total"], 126 + 51 + 52 + 25 + 18.6, places=0)
        self.assertAlmostEqual(d["pred_total_modelo"], 126 + 51 + 52 + 18.6, places=0)
        self.assertAlmostEqual(d["pred_total_promedio"], 25.0)
        self.assertEqual(d["con_pred"], 5)

    def test_tablas_del_resumen_por_accion(self):
        d = self.client.get("/api/summary").get_json()
        self.assertEqual([r["CODIGO"] for r in d["top_growth"]], ["A"])
        self.assertEqual([r["CODIGO"] for r in d["top_risk"]], ["B"])
        self.assertEqual(d["top_risk"][0]["accion_nombre"], "No reponer este mes")

    def test_filtro_por_accion(self):
        d = self.client.get("/api/products?accion=mantener").get_json()
        self.assertEqual(sorted(r["CODIGO"] for r in d["data"]), ["C", "D"])
        d = self.client.get("/api/products?accion=no_reponer").get_json()
        self.assertEqual([r["CODIGO"] for r in d["data"]], ["B"])

    def test_monto_sin_dato_es_null(self):
        d = self.client.get("/api/products?accion=mantener&sort=codigo").get_json()
        fila = {r["CODIGO"]: r for r in d["data"]}["D"]
        self.assertIsNone(fila["monto_mostrado"])
        self.assertIsNone(fila["venta_esperada"])
        self.assertIsNone(fila["stock_sugerido"])

    def test_orden_por_monto_mostrado(self):
        d = self.client.get("/api/products?sort=ventas").get_json()
        self.assertEqual(d["data"][0]["CODIGO"], "A")

    def test_detalle_con_accion_y_explicacion(self):
        d = self.client.get("/api/product/B").get_json()
        p = d["prediccion"]
        self.assertEqual(p["accion"], "no_reponer")
        self.assertEqual(p["accion_nombre"], "No reponer este mes")
        self.assertEqual(p["accion_explicacion"], ac.ACCIONES["no_reponer"]["explicacion"])
        self.assertEqual(p["monto_mostrado"], 0.0)
        self.assertEqual(p["stock_sugerido"], 0.0)


class SinContradiccionesEnLaApiTest(unittest.TestCase):
    """Con un pipeline real (datos sintéticos), ninguna fila de /api/products
    trae una combinación prohibida."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.m = modelo_completo(cls._tmp.name)
        cls._saved = dict(dashboard_app._state)
        dashboard_app._state.update(model=cls.m, results=cls.m.results,
                                    mes_predicho="2025-03")
        cls.client = dashboard_app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        dashboard_app._state.clear()
        dashboard_app._state.update(cls._saved)
        cls._tmp.cleanup()

    def test_ninguna_fila_contradictoria(self):
        d = self.client.get("/api/products?size=10000").get_json()
        filas = pd.DataFrame(d["data"]).astype({"monto_mostrado": float,
                                                "stock_sugerido": float})
        self.assertEqual(len(filas), len(self.m.results))
        cuenta = ac.contradicciones(filas.rename(columns={"ventas_base": "base"}),
                                    self.m.sale_threshold)
        self.assertEqual(cuenta["comprar_sin_venta"], 0)
        self.assertEqual(cuenta["no_reponer_con_monto"], 0)
        self.assertEqual(cuenta["mantener_sin_stock"], 0)
        if self.m.regla_stock["adoptada"]:
            self.assertEqual(cuenta["no_reponer_con_stock"], 0)
        # Comprar más siempre con probabilidad de venta >= umbral ("se venderá: Sí")
        comprar = filas[filas["accion"] == "comprar_mas"]
        self.assertTrue((comprar["se_vendera"] == "Sí").all())


if __name__ == "__main__":
    unittest.main()
