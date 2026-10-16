"""Dashboard con el stock sugerido (nivel de servicio 70 %): /api/products,
/api/product, /api/summary (tops) y /api/grupos traen stock_sugerido (COP) y
stock_sugerido_unidades; la pagina trae la columna y la nota explicativa.
Con resultados sin esas columnas todo se comporta como antes."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))
import app as dashboard_app  # noqa: E402
from ventas_forecast import grupos as gr  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402


def _resultados(c, con_stock=True):
    r = pd.DataFrame({
        c("id"):          ["A", "B", "C", "E"],
        c("description"): ["pa", "pb", "pc", "pe"],
        c("total_sold"):  [100.0, 200.0, 50.0, 10.0],
        "ventas_predichas": [120.0, 180.0, np.nan, 30.0],
        "decision":       ["Reforzar stock", "Reducir stock", "Mantener stock",
                           "Sin recomendación (estimación por promedio)"],
        "prob_crecimiento": [0.8, 0.1, 0.4, None],
        "prob_reduccion": [0.1, 0.7, 0.2, None],
        "cluster": [0, 1, 0, 0],
        "origen_estimacion": ["modelo", "modelo", "modelo", "promedio"],
        "prob_venta": [0.9, 0.3, 0.8, 0.7],
        "se_vendera": ["Sí", "No", "Sí", "Sí"],
    })
    if con_stock:
        r["stock_sugerido"] = [200.0, 90.0, np.nan, 50.0]
        r["stock_sugerido_unidades"] = [4.0, 3.0, np.nan, np.nan]
    return r


class StockApiTest(unittest.TestCase):
    def setUp(self):
        model = SalesForecastModel(filepath="unused.csv")
        c = model._c
        self.c = c
        self.results = _resultados(c)
        model.mapa_grupos = gr.cargar_mapa(io.StringIO(
            "codigo;id_grupo;grupo\nA;1;FRENOS\nB;1;FRENOS\nC;2;ACEITES\nE;3;\n"))
        model.resumen_grupos = gr.resumen_grupos(
            self.results, model.mapa_grupos, c("id"), c("total_sold"))
        m = pd.to_datetime(["2026-01-01", "2026-02-01"])
        model.df = pd.DataFrame({
            c("id"): ["A", "A", "B", "B", "C", "C", "E", "E"],
            c("description"): ["pa"] * 2 + ["pb"] * 2 + ["pc"] * 2 + ["pe"] * 2,
            "fecha": list(m) * 4,
            c("total_sold"): [1.0] * 8, c("frequency"): [1] * 8,
            c("avg_price"): [10.0] * 8,
        })
        model.metrics = {}
        self._saved = dict(dashboard_app._state)
        dashboard_app._state.update(model=model, results=self.results,
                                    mes_predicho="2026-04")
        self.client = dashboard_app.app.test_client()

    def tearDown(self):
        dashboard_app._state.clear()
        dashboard_app._state.update(self._saved)

    def test_productos_traen_stock_y_unidades(self):
        d = self.client.get("/api/products").get_json()
        fila = {r["CODIGO"]: r for r in d["data"]}
        self.assertEqual(fila["A"]["stock_sugerido"], 200.0)
        self.assertEqual(fila["A"]["stock_sugerido_unidades"], 4.0)
        self.assertEqual(fila["B"]["stock_sugerido"], 90.0)

    def test_sin_stock_es_null_no_cero(self):
        d = self.client.get("/api/products").get_json()
        fila = {r["CODIGO"]: r for r in d["data"]}
        self.assertIsNone(fila["C"]["stock_sugerido"])
        self.assertIsNone(fila["C"]["stock_sugerido_unidades"])
        self.assertIsNone(fila["E"]["stock_sugerido_unidades"])

    def test_orden_por_stock(self):
        d = self.client.get("/api/products?sort=stock").get_json()
        self.assertEqual([r["CODIGO"] for r in d["data"]][:3], ["A", "B", "E"])

    def test_detalle_trae_stock(self):
        d = self.client.get("/api/product/A").get_json()
        p = d["prediccion"]
        self.assertEqual(p["stock_sugerido"], 200.0)
        self.assertEqual(p["stock_sugerido_unidades"], 4.0)

    def test_detalle_sin_stock_es_null(self):
        p = self.client.get("/api/product/C").get_json()["prediccion"]
        self.assertIsNone(p["stock_sugerido"])
        self.assertIsNone(p["stock_sugerido_unidades"])

    def test_resumen_top_trae_stock(self):
        d = self.client.get("/api/summary").get_json()
        fila = {r["CODIGO"]: r for r in d["top_growth"]}
        self.assertEqual(fila["A"]["stock_sugerido"], 200.0)

    def test_grupos_suman_el_stock(self):
        d = self.client.get("/api/grupos").get_json()
        g = {f["grupo"]: f for f in d["data"]}
        self.assertEqual(g["FRENOS"]["stock_sugerido"], 290.0)
        self.assertEqual(g["Sin grupo"]["stock_sugerido"], 50.0)

    def test_pagina_trae_columna_y_nota(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("Stock sugerido (70 %)", html)
        self.assertIn("se cubrió la demanda de 7 de cada 10 productos", html)

    def test_nota_con_la_cobertura_medida(self):
        # experimento_stock_sugerido.txt: cobertura agrupada 71,8 %
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("7 de cada 10 productos (72 % medido)", html)

    def test_sin_columnas_todo_sigue_igual(self):
        sin = _resultados(self.c, con_stock=False)
        m = dashboard_app._state["model"]
        m.resumen_grupos = gr.resumen_grupos(sin, m.mapa_grupos, self.c("id"),
                                             self.c("total_sold"))
        dashboard_app._state["results"] = sin
        d = self.client.get("/api/products").get_json()
        self.assertEqual(d["total"], 4)
        self.assertNotIn("stock_sugerido", d["data"][0])
        self.assertEqual(self.client.get("/api/summary").status_code, 200)
        self.assertEqual(self.client.get("/api/product/A").status_code, 200)
        g = self.client.get("/api/grupos").get_json()
        self.assertNotIn("stock_sugerido", g["data"][0])


class ResumenGruposStockTest(unittest.TestCase):
    def test_suma_del_stock_por_grupo_solo_con_monto(self):
        mapa = gr.cargar_mapa(io.StringIO("codigo;id_grupo;grupo\nA;1;X\nB;1;X\nC;2;Y\n"))
        r = pd.DataFrame({
            "CODIGO": ["A", "B", "C"], "TOTAL_VENDIDO": [1.0, 1.0, 1.0],
            "ventas_predichas": [5.0, 6.0, np.nan],
            "stock_sugerido": [10.0, 20.0, np.nan]})
        t = gr.resumen_grupos(r, mapa, "CODIGO", "TOTAL_VENDIDO").set_index("grupo")
        self.assertEqual(t.loc["X", "stock_sugerido"], 30.0)
        self.assertNotIn("Y", t.index)


class PipelineDelDashboardTest(unittest.TestCase):
    """_run_model ejecuta el paso de stock sugerido entre la probabilidad de
    venta y el resumen por grupo, y respeta los parametros del formulario."""

    def _correr(self, params):
        from unittest import mock
        modelo = mock.MagicMock()
        modelo._c = lambda rol: rol
        modelo.metrics = {"accuracy": .5, "precision": .5, "recall": .5, "f1": .5,
                          "f1_macro": .4, "balanced_accuracy": .5}
        modelo.sale_threshold, modelo.sale_val_accuracy = 0.5, 0.7
        modelo.sale_test_accuracy = 0.68
        modelo._mes_pred = pd.Timestamp("2026-04-01")
        modelo._ultimo_mes = pd.Timestamp("2026-03-01")
        modelo.probabilidad_venta = True
        modelo.resumen_por_grupo = True
        modelo.results = pd.DataFrame({"a": [1]})
        modelo.df = pd.DataFrame({"id": ["A"]})
        modelo.df_predict = pd.DataFrame({"a": [1]})
        clase = mock.MagicMock(return_value=modelo)
        saved = dict(dashboard_app._state)
        try:
            with mock.patch.object(dashboard_app, "SalesForecastModel", clase):
                dashboard_app._run_model("x.csv", None, params)
            return clase, modelo, dashboard_app._state["status"]
        finally:
            dashboard_app._state.clear()
            dashboard_app._state.update(saved)

    def test_ejecuta_el_paso_en_orden(self):
        clase, modelo, estado = self._correr({})
        self.assertEqual(estado, "done")
        nombres = [c[0] for c in modelo.method_calls]
        self.assertIn("agregar_stock_sugerido", nombres)
        self.assertLess(nombres.index("agregar_probabilidad_venta"),
                        nombres.index("agregar_stock_sugerido"))
        self.assertLess(nombres.index("agregar_stock_sugerido"),
                        nombres.index("agregar_resumen_grupo"))
        kw = clase.call_args.kwargs
        self.assertTrue(kw["stock_sugerido"])
        self.assertEqual(kw["nivel_servicio"], 0.70)

    def test_parametros_del_formulario(self):
        clase, modelo, _ = self._correr({"stock_sugerido": "false",
                                         "nivel_servicio": "0.8"})
        kw = clase.call_args.kwargs
        self.assertFalse(kw["stock_sugerido"])
        self.assertEqual(kw["nivel_servicio"], 0.8)


if __name__ == "__main__":
    unittest.main()
