"""
Dashboard con la probabilidad de venta: /api/summary cuenta los productos que
probablemente se venderan y expone el umbral; /api/products filtra por
"se_vendera" y devuelve prob_venta; el detalle y el CSV traen ambas columnas.
Con resultados antiguos (sin las columnas) todo se comporta como antes.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))
import app as dashboard_app  # noqa: E402
from ventas_forecast.model import (  # noqa: E402
    ETIQUETA_SIN_RECOMENDACION, SalesForecastModel,
)


class ProbVentaApiTest(unittest.TestCase):

    def setUp(self):
        model = SalesForecastModel(filepath="unused.csv")
        model.blend_weight = 0.3
        model.sale_threshold = 0.45
        model.sale_val_accuracy = 0.71
        c = model._c
        m = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
        model.df = pd.DataFrame({
            c("id"):          ["A"] * 3 + ["B"] * 3 + ["P"] * 3,
            c("description"): ["Producto A"] * 3 + ["Producto B"] * 3 + ["Producto P"] * 3,
            "fecha":          list(m) * 3,
            c("total_sold"):  [100.0, 100.0, 200.0, 5000.0, 6000.0, 7000.0,
                               300.0, 600.0, 0.0],
            c("frequency"):   [1] * 9,
            c("avg_price"):   [10.0] * 9,
        })
        model.metrics = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
        self.results = pd.DataFrame({
            c("id"):              ["A", "B", "P"],
            c("description"):     ["Producto A", "Producto B", "Producto P"],
            c("total_sold"):      [200.0, 7000.0, 0.0],
            "ventas_predichas":   [184.0, None, 300.0],
            "ventas_predichas_rf": [100.0, None, None],
            "decision":           ["Reducir stock", "Reforzar stock",
                                   ETIQUETA_SIN_RECOMENDACION],
            "prob_crecimiento":   [0.2, 0.7, None],
            "prob_reduccion":     [0.6, 0.1, None],
            "cluster":            [0, 1, 0],
            "origen_estimacion":  ["modelo", "modelo", "promedio"],
            "prob_venta":         [0.30, 0.90, 0.60],
            "se_vendera":         ["No", "Sí", "Sí"],
        })
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._saved = dict(dashboard_app._state)
        self._saved_out = dashboard_app.DEFAULT_OUTPUT
        dashboard_app.DEFAULT_OUTPUT = Path(self._tmp.name) / "pred.csv"
        self.results.to_csv(dashboard_app.DEFAULT_OUTPUT, index=False)
        dashboard_app._state.update(model=model, results=self.results,
                                    mes_predicho="2026-04")
        self.client = dashboard_app.app.test_client()

    def tearDown(self):
        dashboard_app.DEFAULT_OUTPUT = self._saved_out
        dashboard_app._state.clear()
        dashboard_app._state.update(self._saved)

    def test_summary_cuenta_los_que_probablemente_se_venderan(self):
        d = self.client.get("/api/summary").get_json()
        self.assertEqual(d["n_se_vendera"], 2)     # B (modelo) y P (promedio)
        self.assertEqual(d["n_con_prob_venta"], 3)
        self.assertEqual(d["sale_threshold"], 0.45)
        self.assertEqual(d["sale_val_accuracy"], 0.71)

    def test_summary_no_cambia_lo_demas(self):
        d = self.client.get("/api/summary").get_json()
        self.assertEqual(d["total"], 2)
        self.assertEqual(d["pred_total"], 484.0)
        self.assertEqual(d["n_promedio"], 1)

    def test_summary_top_incluye_prob_venta(self):
        d = self.client.get("/api/summary").get_json()
        fila = [r for r in d["top_growth"] if r["CODIGO"] == "B"][0]
        self.assertEqual(fila["prob_venta"], 0.9)
        self.assertEqual(fila["se_vendera"], "Sí")

    def test_products_devuelve_prob_venta_y_se_vendera(self):
        d = self.client.get("/api/products").get_json()
        fila = {r["CODIGO"]: r for r in d["data"]}
        self.assertEqual(fila["A"]["prob_venta"], 0.3)
        self.assertEqual(fila["A"]["se_vendera"], "No")
        self.assertEqual(fila["P"]["prob_venta"], 0.6)

    def test_products_filtra_por_se_vendera(self):
        si = self.client.get("/api/products", query_string={"venta": "Sí"}).get_json()
        no = self.client.get("/api/products", query_string={"venta": "No"}).get_json()
        self.assertEqual(sorted(r["CODIGO"] for r in si["data"]), ["B", "P"])
        self.assertEqual([r["CODIGO"] for r in no["data"]], ["A"])

    def test_products_ordena_por_probabilidad_de_venta(self):
        d = self.client.get("/api/products", query_string={"sort": "venta_prob"}).get_json()
        self.assertEqual([r["CODIGO"] for r in d["data"]], ["B", "P", "A"])

    def test_detalle_trae_prob_venta_tambien_en_promedio(self):
        p = self.client.get("/api/product/P").get_json()["prediccion"]
        self.assertEqual(p["prob_venta"], 0.6)
        self.assertEqual(p["se_vendera"], "Sí")
        self.assertIsNone(p["prob_crecimiento"])
        a = self.client.get("/api/product/A").get_json()["prediccion"]
        self.assertEqual(a["prob_venta"], 0.3)

    def test_csv_descargado_incluye_las_columnas(self):
        r = self.client.get("/api/download")
        csv = pd.read_csv(pd.io.common.BytesIO(r.data))
        self.assertIn("prob_venta", csv.columns)
        self.assertIn("se_vendera", csv.columns)

    def test_resultados_sin_prob_venta_se_comportan_como_antes(self):
        dashboard_app._state["results"] = self.results.drop(
            columns=["prob_venta", "se_vendera"])
        dashboard_app._state["model"].sale_threshold = None
        d = self.client.get("/api/summary").get_json()
        self.assertIsNone(d["n_se_vendera"])
        self.assertIsNone(d["sale_threshold"])
        self.assertEqual(d["total"], 2)
        p = self.client.get("/api/products", query_string={"venta": "Sí"}).get_json()
        self.assertEqual(p["total"], 3)       # sin la columna el filtro no aplica

    def test_pagina_ofrece_columna_kpi_y_filtro(self):
        html = self.client.get("/").get_data(as_text=True)
        for texto in ("Probablemente se venderán", "Probabilidad de venta",
                      "Se venderá: Sí", "No se venderá", 'id="filter-venta"'):
            self.assertIn(texto, html)


if __name__ == "__main__":
    unittest.main()
