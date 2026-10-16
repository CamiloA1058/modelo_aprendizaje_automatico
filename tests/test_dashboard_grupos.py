"""Dashboard: vista "Grupos" (/api/grupos) y filtro de /api/products por grupo."""
import io
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))
import app as dashboard_app  # noqa: E402
from ventas_forecast import grupos as gr  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402


class GruposApiTest(unittest.TestCase):
    def setUp(self):
        model = SalesForecastModel(filepath="unused.csv")
        c = model._c
        self.c = c
        self.results = pd.DataFrame({
            c("id"):          ["A", "B", "C", "E"],
            c("description"): ["pa", "pb", "pc", "pe"],
            c("total_sold"):  [100.0, 200.0, 50.0, 10.0],
            "ventas_predichas": [120.0, 180.0, 60.0, 30.0],
            "decision":       ["Reforzar stock", "Reducir stock", "Mantener stock",
                               "Sin recomendación (estimación por promedio)"],
            "prob_crecimiento": [0.8, 0.1, 0.4, None],
            "prob_reduccion": [0.1, 0.7, 0.2, None],
            "cluster": [0, 1, 0, 0],
            "origen_estimacion": ["modelo", "modelo", "modelo", "promedio"],
            "prob_venta": [0.9, 0.3, 0.8, 0.7],
            "se_vendera": ["Sí", "No", "Sí", "Sí"],
        })
        model.mapa_grupos = gr.cargar_mapa(io.StringIO(
            "codigo;id_grupo;grupo\nA;1;FRENOS\nB;1;FRENOS\nC;2;ACEITES\nE;3;\n"))
        model.resumen_grupos = gr.resumen_grupos(
            self.results, model.mapa_grupos, c("id"), c("total_sold"))
        model.df = pd.DataFrame()
        model.metrics = {}
        self._saved = dict(dashboard_app._state)
        dashboard_app._state.update(model=model, results=self.results,
                                    mes_predicho="2026-04")
        self.client = dashboard_app.app.test_client()

    def tearDown(self):
        dashboard_app._state.clear()
        dashboard_app._state.update(self._saved)

    def test_sin_resultados_404(self):
        dashboard_app._state["results"] = None
        self.assertEqual(self.client.get("/api/grupos").status_code, 404)

    def test_lista_de_grupos_ordenada_por_prediccion(self):
        d = self.client.get("/api/grupos").get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(d["mes_predicho"], "2026-04")
        self.assertEqual([g["grupo"] for g in d["data"]],
                         ["FRENOS", "ACEITES", "Sin grupo"])
        f = d["data"][0]
        self.assertEqual(f["productos"], 2)
        self.assertEqual(f["se_vendera"], 1)
        self.assertEqual(f["prediccion"], 300.0)
        self.assertEqual(f["mes_anterior"], 300.0)
        self.assertAlmostEqual(f["variacion"], 0.0)

    def test_total_igual_a_la_suma_de_las_predicciones(self):
        d = self.client.get("/api/grupos").get_json()
        self.assertEqual(d["pred_total"], 390.0)
        self.assertEqual(sum(g["prediccion"] for g in d["data"]), 390.0)
        self.assertEqual(d["total_grupos"], 3)

    def test_variacion_sin_base_es_null_en_json(self):
        r = self.results.copy()
        r.loc[r.index[2], self.c("total_sold")] = 0.0
        m = dashboard_app._state["model"]
        m.resumen_grupos = gr.resumen_grupos(r, m.mapa_grupos, self.c("id"),
                                             self.c("total_sold"))
        d = self.client.get("/api/grupos").get_json()
        aceites = [g for g in d["data"] if g["grupo"] == "ACEITES"][0]
        self.assertIsNone(aceites["variacion"])

    def test_sin_resumen_devuelve_lista_vacia(self):
        dashboard_app._state["model"].resumen_grupos = None
        d = self.client.get("/api/grupos").get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(d["data"], [])

    def test_productos_filtra_por_grupo(self):
        d = self.client.get("/api/products?grupo=FRENOS").get_json()
        self.assertEqual(d["total"], 2)
        self.assertEqual({r["CODIGO"] for r in d["data"]}, {"A", "B"})
        self.assertEqual({r["grupo"] for r in d["data"]}, {"FRENOS"})

    def test_productos_filtra_sin_grupo(self):
        d = self.client.get("/api/products?grupo=Sin grupo").get_json()
        self.assertEqual([r["CODIGO"] for r in d["data"]], ["E"])

    def test_productos_sin_filtro_no_cambia(self):
        d = self.client.get("/api/products").get_json()
        self.assertEqual(d["total"], 4)

    def test_pagina_trae_la_vista_y_la_nota(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("page-grupos", html)
        self.assertIn("/api/grupos", html)
        # Cifras del estudio de caso con el w elegido en la validación interna
        # (experimento_grupos.txt: G0 agrupado 68,9 %; 20 grupos del mes anterior 74,1 %).
        self.assertIn("acertó en promedio el 69 % del valor vendido", html)
        self.assertIn("en los 20 grupos que más vendieron el mes anterior, cerca del 74 %", html)


if __name__ == "__main__":
    unittest.main()
