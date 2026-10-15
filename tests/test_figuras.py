"""Formato en español de las figuras (coma decimal, punto de miles)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.figuras import fmt_es, formateador_es  # noqa: E402


class FormatoEspanol(unittest.TestCase):
    def test_miles_y_decimales(self):
        self.assertEqual(fmt_es(1234567.5, 1), "1.234.567,5")
        self.assertEqual(fmt_es(1200), "1.200")

    def test_decimal_sin_miles(self):
        self.assertEqual(fmt_es(0.4501, 2), "0,45")
        self.assertEqual(fmt_es(65.47, 1), "65,5")

    def test_formateador_con_sufijo(self):
        f = formateador_es(1, " %")
        self.assertEqual(f(12.34, 0), "12,3 %")


class EtiquetasEstudioCaso(unittest.TestCase):
    def test_septiembre_lleva_asterisco(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from figura_estudio_caso_totales import etiquetas_mes
        meses = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
        self.assertEqual(etiquetas_mes(meses),
                         ["Mes 1", "Mes 2", "Mes 3", "Mes 4", "Mes 5", "Mes 6*"])


if __name__ == "__main__":
    unittest.main()
