"""Formato de salida del extractor: debe coincidir byte a byte con el CSV V5."""
import csv
import io
import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sql"))

import extraer_ventas as ex  # noqa: E402

HEADER = [
    "CODIGO", "DESCRIPCION", "FECHA", "ANIO", "MES", "DIA", "DIA_SEMANA", "VENTAS",
    "TOTAL_VENDIDO", "PRECIO_PROMEDIO", "FRECUENCIA", "HUBO_PROMOCION", "TEMPORADA",
]


class TestFormatoNumerico(unittest.TestCase):
    def test_miles_con_punto_y_decimales_con_coma(self):
        self.assertEqual(ex.format_es(Decimal("416000")), "416.000,00")
        self.assertEqual(ex.format_es(13), "13,00")
        self.assertEqual(ex.format_es(1446000.0), "1.446.000,00")

    def test_redondeo_a_dos_decimales(self):
        self.assertEqual(ex.format_es(Decimal("144333.3333")), "144.333,33")
        self.assertEqual(ex.format_es(Decimal("0.005")), "0,01")

    def test_millones_y_valores_pequenos(self):
        self.assertEqual(ex.format_es(Decimal("1234567.891")), "1.234.567,89")
        self.assertEqual(ex.format_es(Decimal("500")), "500,00")
        self.assertEqual(ex.format_es(Decimal("0.5")), "0,50")


class TestFormatoFila(unittest.TestCase):
    def test_fila_equivale_a_v5(self):
        raw = ("1800", "RETEN SNA", date(2024, 1, 9), 2024, 1, 9, 2,
               Decimal("13"), Decimal("416000"), Decimal("32000"), 8, 0, "Baja")
        self.assertEqual(
            ex.format_row(raw),
            ["1800", "RETEN SNA", "09/01/2024", "2024", "1", "9", "2",
             "13,00", "416.000,00", "32.000,00", "8", "0", "Baja "],
        )

    def test_temporada_rellena_a_cinco(self):
        self.assertEqual(ex.pad_season("Alta"), "Alta ")
        self.assertEqual(ex.pad_season("Media"), "Media")
        self.assertEqual(ex.pad_season("Baja "), "Baja ")


class TestEscrituraCsv(unittest.TestCase):
    def test_bom_crlf_separador_y_comillas(self):
        rows = [["1722", 'RODAJA NQR 1"', "09/01/2024", "2024", "1", "9", "2",
                 "2,00", "12.400,00", "6.200,00", "1", "0", "Baja "]]
        buf = io.BytesIO()
        ex.write_csv(buf, HEADER, rows)
        data = buf.getvalue()
        self.assertTrue(data.startswith(b"\xef\xbb\xbfCODIGO;DESCRIPCION;"))
        self.assertIn(b'1722;"RODAJA NQR 1""";09/01/2024', data)
        self.assertTrue(data.endswith(b"Baja \r\n"))
        self.assertEqual(data.count(b"\r\n"), 2)

    def test_roundtrip_con_lector_csv(self):
        rows = [["1", "A;B", "01/02/2024", "2024", "2", "1", "3",
                 "1,00", "10,00", "10,00", "1", "1", "Baja "]]
        buf = io.BytesIO()
        ex.write_csv(buf, HEADER, rows)
        text = buf.getvalue().decode("utf-8-sig")
        back = list(csv.reader(io.StringIO(text, newline=""), delimiter=";"))
        self.assertEqual(back[1][1], "A;B")


class TestParametrosSql(unittest.TestCase):
    def test_reemplaza_rango_de_fechas(self):
        sql = "WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13' AND x"
        out = ex.apply_date_range(sql, "2024-02-01", "2026-03-27")
        self.assertIn("BETWEEN '2024-02-01' AND '2026-03-27'", out)

    def test_rechaza_fecha_mal_formada(self):
        with self.assertRaises(ValueError):
            ex.apply_date_range("d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'", "01/02/2024", "2026-03-27")

    def test_falla_si_no_hay_rango(self):
        with self.assertRaises(ValueError):
            ex.apply_date_range("SELECT 1", "2024-01-01", "2024-02-01")


if __name__ == "__main__":
    unittest.main()
