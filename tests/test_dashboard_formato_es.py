"""
Formato es-CO del dashboard (coma decimal, punto de miles, '%' separado) y
tarjetas de métricas coherentes con la tesis (F1 macro, exactitud balanceada).
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))
import app as dashboard_app  # noqa: E402
from ventas_forecast.figuras import pct_es  # noqa: E402

HTML = (ROOT / "dashboard" / "dashboard.html").read_text(encoding="utf-8")


class PctEs(unittest.TestCase):
    def test_fraccion_a_porcentaje_es(self):
        self.assertEqual(pct_es(0.793, 1), "79,3 %")
        self.assertEqual(pct_es(0.5043, 1), "50,4 %")
        self.assertEqual(pct_es(0.7, 0), "70 %")


class MensajesDeEjecucion(unittest.TestCase):
    def test_registros_con_punto_de_miles(self):
        self.assertEqual(dashboard_app._msg_registros(66166, 1045),
                         "  → 66.166 registros | 1.045 productos únicos")

    def test_metricas_clasificador_principales_en_es(self):
        m = {"accuracy": 0.6684, "precision": 0.7926, "recall": 0.6684, "f1": 0.7126,
             "f1_macro": 0.4501, "balanced_accuracy": 0.5043}
        msg = dashboard_app._msg_clasificador(m)
        self.assertIn("F1 macro 0,45", msg)
        self.assertIn("exactitud balanceada 50,4 %", msg)
        self.assertNotIn(".", msg.replace("...", ""))

    def test_umbral_y_exactitud_interna(self):
        self.assertEqual(dashboard_app._msg_umbral(0.63, 0.793),
                         "  → Umbral 0,63 | exactitud de validación interna 79,3 %")


class HtmlEs(unittest.TestCase):
    def test_tarjetas_de_metricas_en_espanol_y_de_la_tesis(self):
        for etiqueta in ("F1 macro", "Exactitud balanceada", "Exactitud", "Acierto ¿se venderá?"):
            self.assertIn(f"label: '{etiqueta}'", HTML)
        for ingles in ("Accuracy", "Precision", "Recall", "F1-Score"):
            self.assertNotIn(ingles, HTML)

    def test_sin_formato_anglosajon(self):
        self.assertNotIn("en-US", HTML)
        self.assertNotIn(".toFixed(1) + '%'", HTML)
        self.assertNotIn(".toFixed(1)}%", HTML)

    def test_agrupa_miles_tambien_de_cuatro_cifras(self):
        # es-CO por defecto escribe 1045; la tesis usa 1.045
        self.assertEqual(HTML.count("toLocaleString('es-CO')"), 0)
        self.assertIn("useGrouping: 'always'", HTML)

    def test_notas_sin_fechas_explicitas(self):
        self.assertNotIn("abril a septiembre de 2026", HTML)
        self.assertEqual(HTML.lower().count("en los seis meses de prueba"), 2)


if __name__ == "__main__":
    unittest.main()
