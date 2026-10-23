"""Dashboard: estado de todos los productos (/api/estados, /api/estados/productos,
/api/estados/descargar) y conteo por estado en /api/summary."""
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
from ventas_forecast import estado_productos as ep  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402


def _tabla():
    return pd.DataFrame({
        "CODIGO": ["A", "B", "C", "D", "E", "F"],
        "DESCRIPCION": ["FRENO A", "ACEITE B", "FILTRO C", "VALVULA D", "TUBO E", "TAPA F"],
        "grupo": ["FRENOS", "ACEITES", "FRENOS", "Sin grupo", "FRENOS", "ACEITES"],
        "estado": ["recomendacion_monto", "solo_recomendacion", "monto_promedio",
                   "venta_ocasional", "sin_ventas_recientes", "venta_ocasional"],
        "estado_nombre": ["Con recomendación y monto", "Solo recomendación",
                          "Monto por promedio", "Venta ocasional",
                          "Sin ventas recientes", "Venta ocasional"],
        "meses_con_venta": [24, 20, 18, 3, 10, 1],
        "ultimo_mes_venta": ["2026-03", "2026-03", "2026-01", "2026-03", "2025-06", "2024-05"],
        "meses_sin_venta": [0, 0, 2, 0, 9, 22],
        "ventas_12m": [600.0, 100.0, 200.0, 50.0, 50.0, 0.0],
        "promedio_mensual": [50.0, 5.0, 11.1, 16.7, 20.0, 10.0],
        "decision": ["Reforzar stock", "Reducir stock",
                     "Sin recomendación (estimación por promedio)", None, None, None],
        "ventas_predichas": [70.0, np.nan, 15.0, np.nan, np.nan, np.nan],
        "prob_venta": [0.9, 0.4, 0.6, np.nan, np.nan, np.nan],
        "se_vendera": ["Sí", "No", "Sí", None, None, None],
        "stock_sugerido": [90.0, np.nan, 20.0, np.nan, np.nan, np.nan],
        "stock_sugerido_unidades": [3.0, np.nan, 1.0, np.nan, np.nan, np.nan],
    })


class EstadosApiTest(unittest.TestCase):
    def setUp(self):
        model = SalesForecastModel(filepath="unused.csv")
        model.estado_productos = _tabla()
        model._ultimo_mes = pd.Timestamp("2026-03-01")
        model.df = pd.DataFrame({
            "CODIGO": ["A", "B", "C"], "fecha": pd.Timestamp("2026-03-01"),
            "TOTAL_VENDIDO": [60.0, 10.0, 0.0]})
        model.metrics = {}
        results = pd.DataFrame({
            "CODIGO": ["A", "B", "C"], "DESCRIPCION": ["FRENO A", "ACEITE B", "FILTRO C"],
            "TOTAL_VENDIDO": [60.0, 10.0, 0.0],
            "ventas_predichas": [70.0, np.nan, 15.0],
            "decision": ["Reforzar stock", "Reducir stock",
                         "Sin recomendación (estimación por promedio)"],
            "prob_crecimiento": [0.8, 0.1, np.nan], "prob_reduccion": [0.1, 0.7, np.nan],
            "cluster": [0, 1, 0], "origen_estimacion": ["modelo", "modelo", "promedio"],
        })
        self._saved = dict(dashboard_app._state)
        dashboard_app._state.update(model=model, results=results, mes_predicho="2026-04")
        self.client = dashboard_app.app.test_client()

    def tearDown(self):
        dashboard_app._state.clear()
        dashboard_app._state.update(self._saved)

    # ── Resumen ──────────────────────────────────────────────────────────────
    def test_sin_resultados_404(self):
        dashboard_app._state["results"] = None
        self.assertEqual(self.client.get("/api/estados").status_code, 404)
        self.assertEqual(self.client.get("/api/estados/productos").status_code, 404)
        self.assertEqual(self.client.get("/api/estados/descargar").status_code, 404)

    def test_modelo_sin_clasificacion_404(self):
        dashboard_app._state["model"].estado_productos = None
        self.assertEqual(self.client.get("/api/estados").status_code, 404)

    def test_resumen_cuenta_y_participacion_por_estado(self):
        d = self.client.get("/api/estados").get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(d["mes_predicho"], "2026-04")
        self.assertEqual(d["ultimo_mes"], "2026-03")
        self.assertEqual(d["total_productos"], 6)
        self.assertEqual(d["con_prediccion"], 3)
        self.assertAlmostEqual(d["ventas_12m_total"], 1000.0)
        self.assertEqual([e["clave"] for e in d["estados"]], list(ep.ESTADOS))
        por = {e["clave"]: e for e in d["estados"]}
        self.assertEqual(por["venta_ocasional"]["productos"], 2)
        self.assertAlmostEqual(por["venta_ocasional"]["pct_productos"], 2 / 6)
        self.assertAlmostEqual(por["recomendacion_monto"]["ventas_12m"], 600.0)
        self.assertAlmostEqual(por["recomendacion_monto"]["pct_ventas_12m"], 0.6)
        self.assertEqual(por["solo_recomendacion"]["nombre"], "Solo recomendación")
        self.assertTrue(por["venta_ocasional"]["explicacion"])

    def test_explicacion_usa_min_months_del_modelo(self):
        dashboard_app._state["model"].min_months = 4
        d = self.client.get("/api/estados").get_json()
        por = {e["clave"]: e for e in d["estados"]}
        self.assertIn("menos de 4 meses", por["venta_ocasional"]["explicacion"])

    def test_summary_expone_conteo_por_estado(self):
        d = self.client.get("/api/summary").get_json()
        self.assertEqual(d["estados"], {
            "recomendacion_monto": 1, "solo_recomendacion": 1, "monto_promedio": 1,
            "venta_ocasional": 2, "sin_ventas_recientes": 1})

    # ── Lista ────────────────────────────────────────────────────────────────
    def _lista(self, **params):
        return self.client.get("/api/estados/productos", query_string=params).get_json()

    def test_lista_por_defecto_ordenada_por_ventas_12m(self):
        d = self._lista()
        self.assertEqual(d["total"], 6)
        self.assertEqual([r["CODIGO"] for r in d["data"]][:3], ["A", "C", "B"])
        a = d["data"][0]
        self.assertEqual(a["estado_nombre"], "Con recomendación y monto")
        self.assertTrue(a["tiene_prediccion"])
        self.assertEqual(a["grupo"], "FRENOS")

    def test_nan_es_null_en_json(self):
        d = self._lista(estado="venta_ocasional", sort="codigo")
        f = d["data"][0]
        self.assertEqual(f["CODIGO"], "D")
        self.assertIsNone(f["ventas_predichas"])
        self.assertIsNone(f["decision"])
        self.assertIsNone(f["prob_venta"])
        self.assertFalse(f["tiene_prediccion"])

    def test_filtro_por_estado(self):
        d = self._lista(estado="venta_ocasional")
        self.assertEqual({r["CODIGO"] for r in d["data"]}, {"D", "F"})

    def test_busqueda_por_codigo_o_nombre(self):
        self.assertEqual([r["CODIGO"] for r in self._lista(q="freno")["data"]], ["A"])
        self.assertEqual([r["CODIGO"] for r in self._lista(q="d")["data"]], ["D"])

    def test_filtro_por_grupo(self):
        d = self._lista(grupo="ACEITES")
        self.assertEqual({r["CODIGO"] for r in d["data"]}, {"B", "F"})

    def test_orden_por_meses_sin_venta_y_codigo(self):
        self.assertEqual([r["CODIGO"] for r in self._lista(sort="meses_sin_venta")["data"]][:2],
                         ["F", "E"])
        self.assertEqual([r["CODIGO"] for r in self._lista(sort="codigo")["data"]],
                         ["A", "B", "C", "D", "E", "F"])

    def test_paginacion(self):
        d = self._lista(sort="codigo", size=4, page=2)
        self.assertEqual(d["total"], 6)
        self.assertEqual([r["CODIGO"] for r in d["data"]], ["E", "F"])

    # ── Descarga ─────────────────────────────────────────────────────────────
    def test_descarga_csv_con_toda_la_tabla(self):
        r = self.client.get("/api/estados/descargar")
        self.assertEqual(r.status_code, 200)
        self.assertIn("attachment", r.headers["Content-Disposition"])
        self.assertIn("estado_productos_2026-04.csv", r.headers["Content-Disposition"])
        texto = r.data.decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(texto), sep=";", decimal=",")
        self.assertEqual(len(df), 6)
        self.assertIn("Código", df.columns)
        self.assertIn("Estado", df.columns)
        self.assertIn("Ventas últimos 12 meses (COP)", df.columns)
        self.assertEqual(df.loc[0, "Estado"], "Con recomendación y monto")
        self.assertEqual(df.loc[2, "Promedio mensual (COP)"], 11)
        self.assertEqual(df.loc[0, "Ventas últimos 12 meses (COP)"], 600)
        self.assertAlmostEqual(df.loc[0, "Probabilidad de venta"], 0.9)


if __name__ == "__main__":
    unittest.main()
