"""Dashboard: origen de los datos "Leer de la base de datos" (lector simulado)."""
import json
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))
import app as dashboard_app  # noqa: E402
from ventas_forecast import fuente_bd as fb  # noqa: E402

CLAVE = "Cl4ve-Secreta-123"
RUTA_BD = "C:/taller/secreta/base.fdb"


def _cfg():
    return fb.ConfigBD(base_datos=RUTA_BD, cliente="C:/fb/fbclient.dll", clave=CLAVE)


def _datos_bd():
    df = pd.DataFrame({"CODIGO": ["1", "2", "3"]})
    df.attrs["ventana"] = fb.Ventana(date(2024, 9, 1), date(2026, 8, 31), date(2026, 9, 1))
    df.attrs["fecha_max"] = date(2026, 9, 22)
    return df


def _modelo_falso():
    modelo = mock.MagicMock()
    modelo._c = lambda rol: rol
    modelo.metrics = {"accuracy": .5, "f1_macro": .4, "balanced_accuracy": .5}
    modelo.sale_threshold, modelo.sale_test_accuracy = 0.5, 0.68
    modelo._mes_pred = pd.Timestamp("2026-09-01")
    modelo._ultimo_mes = pd.Timestamp("2026-08-01")
    modelo.results = pd.DataFrame({"a": [1]})
    modelo.df = pd.DataFrame({"id": ["A"]})
    modelo.df_predict = pd.DataFrame({"a": [1]})
    return modelo


class _EstadoLimpio(unittest.TestCase):
    def setUp(self):
        self._guardado = dict(dashboard_app._state)
        dashboard_app._state.update(status="idle", log=[], error=None)
        self.cliente = dashboard_app.app.test_client()

    def tearDown(self):
        dashboard_app._state.clear()
        dashboard_app._state.update(self._guardado)


class ApiRunFuenteTest(_EstadoLimpio):
    def _post(self, datos):
        with mock.patch.object(dashboard_app.threading, "Thread") as hilo:
            r = self.cliente.post("/api/run", data=datos)
        return r, hilo

    def test_bd_no_configurada_mensaje_amable(self):
        error = fb.ErrorConfiguracionBD("la lectura de la base de datos no está configurada")
        with mock.patch.object(dashboard_app.fuente_bd, "cargar_config", side_effect=error):
            r, hilo = self._post({"fuente": "bd"})
        self.assertEqual(r.status_code, 400)
        j = r.get_json()
        self.assertFalse(j["ok"])
        self.assertIn("no está configurada", j["msg"])
        self.assertIn("Subir archivo CSV", j["msg"])
        hilo.assert_not_called()

    def test_bd_configurada_inicia_el_pipeline(self):
        cfg = _cfg()
        with mock.patch.object(dashboard_app.fuente_bd, "cargar_config", return_value=cfg):
            r, hilo = self._post({"fuente": "bd", "min_months": "6"})
        self.assertTrue(r.get_json()["ok"])
        kw = hilo.call_args.kwargs
        self.assertIs(kw["target"], dashboard_app._run_model)
        filepath, _col_map, params = kw["args"]
        self.assertIsNone(filepath)
        self.assertIs(kw["kwargs"]["config_bd"], cfg)
        self.assertNotIn("fuente", params)
        self.assertNotIn(CLAVE, r.get_data(as_text=True))

    def test_csv_sigue_igual(self):
        r, hilo = self._post({})
        self.assertTrue(r.get_json()["ok"])
        kw = hilo.call_args.kwargs
        self.assertEqual(kw["args"][0], dashboard_app.DEFAULT_DATA)
        self.assertIsNone(kw.get("kwargs", {}).get("config_bd"))


class RunModelBdTest(_EstadoLimpio):
    def _correr(self, leer_ventas):
        modelo = _modelo_falso()
        clase = mock.MagicMock(return_value=modelo)
        mapa = pd.Series(["RETENES"], index=["1"])
        with mock.patch.object(dashboard_app, "SalesForecastModel", clase), \
             mock.patch.object(dashboard_app.fuente_bd, "leer_ventas", leer_ventas), \
             mock.patch.object(dashboard_app.fuente_bd, "leer_grupos", return_value=mapa):
            dashboard_app._run_model(None, None, {}, config_bd=_cfg())
        return clase, mapa

    def test_lee_la_base_y_corre_el_mismo_pipeline(self):
        datos = _datos_bd()
        clase, mapa = self._correr(mock.Mock(return_value=datos))
        self.assertEqual(dashboard_app._state["status"], "done")
        kw = clase.call_args.kwargs
        self.assertIsNone(kw["filepath"])
        self.assertIs(kw["datos"], datos)
        self.assertIs(kw["grupos_mapa"], mapa)
        log = "\n".join(dashboard_app._state["log"])
        self.assertIn("Leyendo la base de datos…", log)
        self.assertIn("3 registros del 01/09/2024 al 31/08/2026", log)
        self.assertIn("22/09/2026", log)
        self.assertIn("Mes a predecir: 2026-09", log)
        self.assertNotIn(CLAVE, log)
        self.assertNotIn(RUTA_BD, log)

    def test_status_informa_rango_y_fuente(self):
        self._correr(mock.Mock(return_value=_datos_bd()))
        j = self.cliente.get("/api/status").get_json()
        self.assertEqual(j["fuente"], "bd")
        self.assertEqual(j["rango"], {"desde": "01/09/2024", "hasta": "31/08/2026"})
        self.assertEqual(j["mes_predicho"], "2026-09")

    def test_error_de_la_base_sin_traza_ni_secretos(self):
        error = fb.ErrorBaseDatos("no se pudo conectar a la base de datos: revise la ruta")
        self._correr(mock.Mock(side_effect=error))
        self.assertEqual(dashboard_app._state["status"], "error")
        self.assertNotIn("Traceback", dashboard_app._state["error"])
        self.assertIn("no se pudo conectar", dashboard_app._state["error"])
        ultimo = dashboard_app._state["log"][-1]
        self.assertTrue(ultimo.startswith("❌"))
        self.assertIn("Subir archivo CSV", ultimo)
        j = json.dumps(self.cliente.get("/api/status").get_json(), ensure_ascii=False)
        self.assertNotIn(CLAVE, j)
        self.assertNotIn(RUTA_BD, j)

    def test_csv_no_informa_rango(self):
        clase = mock.MagicMock(return_value=_modelo_falso())
        with mock.patch.object(dashboard_app, "SalesForecastModel", clase):
            dashboard_app._run_model("x.csv", None, {})
        self.assertEqual(clase.call_args.kwargs["filepath"], "x.csv")
        self.assertNotIn("datos", clase.call_args.kwargs)
        j = self.cliente.get("/api/status").get_json()
        self.assertEqual(j["fuente"], "csv")
        self.assertIsNone(j["rango"])


class ApiEstadoBdTest(_EstadoLimpio):
    def test_configurada(self):
        with mock.patch.object(dashboard_app.fuente_bd, "cargar_config", return_value=_cfg()):
            r = self.cliente.get("/api/bd/estado")
        j = r.get_json()
        self.assertTrue(j["configurada"])
        self.assertEqual(j["meses"], 24)
        texto = r.get_data(as_text=True)
        self.assertNotIn(CLAVE, texto)
        self.assertNotIn(RUTA_BD, texto)

    def test_no_configurada(self):
        error = fb.ErrorConfiguracionBD("falta base_datos (FIREBIRD_DB)")
        with mock.patch.object(dashboard_app.fuente_bd, "cargar_config", side_effect=error):
            j = self.cliente.get("/api/bd/estado").get_json()
        self.assertFalse(j["configurada"])
        self.assertIn("FIREBIRD_DB", j["msg"])


class HtmlFuenteTest(unittest.TestCase):
    def test_control_de_origen_en_ejecutar_modelo(self):
        html = (ROOT / "dashboard" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("Subir archivo CSV", html)
        self.assertIn("Leer de la base de datos", html)
        self.assertIn("fd.append('fuente'", html)
        self.assertIn("/api/bd/estado", html)


if __name__ == "__main__":
    unittest.main()
