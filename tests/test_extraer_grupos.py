"""Pruebas del extractor del catalogo producto -> grupo."""
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sql"))

import extraer_grupos as eg  # noqa: E402


class FormatoGruposTest(unittest.TestCase):
    def test_fila_con_grupo(self):
        self.assertEqual(eg.format_row(("1001", 4, "RETENES")), ["1001", "4", "RETENES"])

    def test_fila_sin_grupo_queda_vacia(self):
        self.assertEqual(eg.format_row(("1002", None, None)), ["1002", "", ""])

    def test_csv_utf8_bom_y_separador_punto_y_coma(self):
        buf = io.BytesIO()
        eg.write_csv(buf, ["CODIGO", "ID_GRUPO", "GRUPO"], [["1", "4", "RETÉNES"]])
        datos = buf.getvalue()
        self.assertTrue(datos.startswith(b"\xef\xbb\xbf"))
        self.assertIn("1;4;RETÉNES".encode("utf-8"), datos)

    def test_el_sql_no_toca_tablas_de_clientes(self):
        lineas = (ROOT / "sql" / "grupos_productos.sql").read_text(encoding="utf-8").splitlines()
        sql = " ".join(l for l in lineas if not l.lstrip().startswith("--")).upper()
        self.assertIn("FROM PRODUCTO", sql)
        for prohibido in ("TERCERO", "CLIENTE", "DOCINV"):
            self.assertNotIn(prohibido, sql)


if __name__ == "__main__":
    unittest.main()
