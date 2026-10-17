"""Lectura directa de la base Firebird (ventas_forecast.fuente_bd).

Las pruebas unitarias usan una conexión simulada: no necesitan Firebird. La
prueba de integración solo corre si FIREBIRD_DB_PRUEBA y FIREBIRD_CLIENT_PRUEBA
apuntan a una COPIA de la base y a su fbclient.dll.
"""
import io
import os
import re
import sys
import tempfile
import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ventas_forecast import fuente_bd as fb  # noqa: E402
from ventas_forecast import extraccion  # noqa: E402
from ventas_forecast import grupos as gr  # noqa: E402

CLAVE = "Cl4ve-Secreta-123"

CABECERA_V6 = [
    "CODIGO", "DESCRIPCION", "FECHA", "ANIO", "MES", "DIA", "DIA_SEMANA", "VENTAS",
    "TOTAL_VENDIDO", "PRECIO_PROMEDIO", "FRECUENCIA", "HUBO_PROMOCION", "TEMPORADA",
]
FILAS_V6 = [
    ("1800", "RETEN SNA", date(2024, 9, 3), 2024, 9, 3, 2,
     Decimal("13"), Decimal("416000"), Decimal("32000"), 8, 0, "Baja"),
    ("1722", 'RODAJA NQR 1"', date(2025, 7, 15), 2025, 7, 15, 2,
     Decimal("2"), Decimal("12400.5"), Decimal("6200.25"), 1, 1, "Media"),
    ("T7120", "TORN ZAPT; 005", date(2026, 8, 31), 2026, 8, 31, 1,
     Decimal("4"), Decimal("8400"), Decimal("2100"), 2, 0, "Baja"),
]
FILAS_GRUPOS = [("1722", 4, "RODAJAS"), ("1800", 7, "RETENES"), ("T7120", None, None)]


def _config(**kw):
    datos = dict(base_datos="C:/taller/base.fdb", cliente="C:/fb/fbclient.dll",
                 usuario="SYSDBA", clave=CLAVE, host=None)
    datos.update(kw)
    return fb.ConfigBD(**datos)


class ConexionFalsa:
    """Imita la API de firebird-driver que usa fuente_bd: transaction_manager,
    begin, cursor, execute, description, fetchone/fetchall, rollback."""

    def __init__(self, fecha_max=date(2026, 9, 22)):
        self.fecha_max = fecha_max
        self.sql = []
        self.tpb = None
        self.commits = 0
        self.rollbacks = 0
        self.cerrada = False
        self.solo_lectura = True

    @property
    def info(self):
        return mock.Mock(is_read_only=mock.Mock(return_value=self.solo_lectura))

    # conexión
    def transaction_manager(self, default_tpb=None, default_action=None):
        self.tpb = default_tpb
        self.default_action = default_action
        return self

    def close(self):
        self.cerrada = True

    # transacción
    def begin(self):
        pass

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def cursor(self):
        return CursorFalso(self)

    def execute(self, sql, params=None):
        self.sql.append(sql)
        up = sql.upper()
        if "MAX(" in up:
            self._cab, self._filas = ["MAX"], [(self.fecha_max,)]
        elif "FROM PRODUCTO P" in up and "GRUPO" in up and "DOCINV" not in up:
            self._cab, self._filas = ["CODIGO", "ID_GRUPO", "GRUPO"], FILAS_GRUPOS
        else:
            self._cab, self._filas = CABECERA_V6, FILAS_V6

    @property
    def description(self):
        return [(c,) for c in self._cab]

    def fetchone(self):
        return self._filas[0]

    def fetchall(self):
        return list(self._filas)


class CursorFalso:
    def __init__(self, con):
        self.con = con
        con.cursores_abiertos = getattr(con, "cursores_abiertos", 0) + 1

    def execute(self, sql, params=None):
        self.con.execute(sql, params)

    @property
    def description(self):
        return self.con.description

    def fetchone(self):
        return self.con.fetchone()

    def fetchall(self):
        return self.con.fetchall()

    def close(self):
        self.con.cursores_abiertos -= 1


def _csv_esperado():
    """Lo que el pipeline lee de un CSV V6 con esas mismas filas."""
    buf = io.BytesIO()
    extraccion.write_csv(buf, CABECERA_V6, [extraccion.format_row(r) for r in FILAS_V6])
    buf.seek(0)
    return pd.read_csv(buf, sep=";", encoding="utf-8")


class CalcularVentanaTest(unittest.TestCase):
    def test_mes_incompleto_se_excluye(self):
        v = fb.calcular_ventana(date(2026, 9, 22))
        self.assertEqual(v.hasta, date(2026, 8, 31))
        self.assertEqual(v.desde, date(2024, 9, 1))
        self.assertEqual(v.mes_predicho, date(2026, 9, 1))

    def test_mes_completo_se_incluye(self):
        v = fb.calcular_ventana(date(2026, 9, 30))
        self.assertEqual(v.hasta, date(2026, 9, 30))
        self.assertEqual(v.desde, date(2024, 10, 1))
        self.assertEqual(v.mes_predicho, date(2026, 10, 1))

    def test_febrero_bisiesto_y_cambio_de_anio(self):
        v = fb.calcular_ventana(date(2028, 2, 29), meses=3)
        self.assertEqual((v.desde, v.hasta), (date(2027, 12, 1), date(2028, 2, 29)))
        v = fb.calcular_ventana(date(2026, 1, 10), meses=12)
        self.assertEqual((v.desde, v.hasta), (date(2025, 1, 1), date(2025, 12, 31)))
        self.assertEqual(v.mes_predicho, date(2026, 1, 1))

    def test_un_mes_y_datetime(self):
        v = fb.calcular_ventana(datetime(2026, 3, 31, 18, 5), meses=1)
        self.assertEqual((v.desde, v.hasta), (date(2026, 3, 1), date(2026, 3, 31)))

    def test_meses_invalidos(self):
        with self.assertRaises(ValueError):
            fb.calcular_ventana(date(2026, 3, 31), meses=0)


class CargarConfigTest(unittest.TestCase):
    def _ini(self, texto):
        d = tempfile.mkdtemp()
        ruta = Path(d) / "base_datos.ini"
        ruta.write_text(texto, encoding="utf-8")
        return ruta

    def test_lee_el_archivo(self):
        ini = self._ini("[firebird]\nbase_datos = C:/taller/b.fdb\n"
                        "cliente = C:/fb/fbclient.dll\nusuario = LECTOR\nclave = x%y\n")
        cfg = fb.cargar_config(entorno={}, archivo=ini)
        self.assertEqual(cfg.base_datos, "C:/taller/b.fdb")
        self.assertEqual(cfg.usuario, "LECTOR")
        self.assertEqual(cfg.clave, "x%y")
        self.assertIsNone(cfg.host)
        self.assertEqual(cfg.meses, 24)

    def test_el_entorno_tiene_prioridad(self):
        ini = self._ini("[firebird]\nbase_datos = C:/a.fdb\ncliente = C:/fb.dll\n"
                        "usuario = LECTOR\nmeses = 12\n")
        cfg = fb.cargar_config(entorno={"FIREBIRD_DB": "D:/b.fdb", "FIREBIRD_USER": "OTRO",
                                        "FIREBIRD_HOST": "servidor"}, archivo=ini)
        self.assertEqual(cfg.base_datos, "D:/b.fdb")
        self.assertEqual(cfg.usuario, "OTRO")
        self.assertEqual(cfg.cliente, "C:/fb.dll")
        self.assertEqual(cfg.meses, 12)
        self.assertEqual(cfg.dsn, "servidor:D:/b.fdb")

    def test_solo_entorno_sin_archivo(self):
        cfg = fb.cargar_config(entorno={"FIREBIRD_DB": "D:/b.fdb",
                                        "FIREBIRD_CLIENT": "D:/fbclient.dll"},
                               archivo=Path(tempfile.mkdtemp()) / "no_existe.ini")
        self.assertEqual(cfg.usuario, "SYSDBA")
        self.assertIsNone(cfg.clave)
        self.assertEqual(cfg.dsn, "D:/b.fdb")

    def test_falta_la_base(self):
        ini = self._ini(f"[firebird]\ncliente = C:/fb.dll\nclave = {CLAVE}\n")
        with self.assertRaises(fb.ErrorConfiguracionBD) as ctx:
            fb.cargar_config(entorno={}, archivo=ini)
        msg = str(ctx.exception)
        self.assertIn("base_datos", msg)
        self.assertIn("FIREBIRD_DB", msg)
        self.assertNotIn(CLAVE, msg)

    def test_sin_configuracion(self):
        with self.assertRaises(fb.ErrorConfiguracionBD) as ctx:
            fb.cargar_config(entorno={}, archivo=Path(tempfile.mkdtemp()) / "no.ini")
        self.assertIn("config/base_datos.ini", str(ctx.exception))

    def test_meses_no_numerico(self):
        ini = self._ini("[firebird]\nbase_datos = a.fdb\ncliente = fb.dll\nmeses = dos\n")
        with self.assertRaises(fb.ErrorConfiguracionBD):
            fb.cargar_config(entorno={}, archivo=ini)

    def test_la_clave_no_aparece_en_repr(self):
        cfg = _config()
        self.assertNotIn(CLAVE, repr(cfg))
        self.assertNotIn(CLAVE, str(cfg))


class SoloSelectTest(unittest.TestCase):
    def test_acepta_las_consultas_del_modulo(self):
        fb.validar_solo_lectura(fb.consulta_ventas(date(2024, 1, 1), date(2026, 3, 31)))
        fb.validar_solo_lectura(fb.consulta_fecha_max(date(2026, 10, 17)))
        fb.validar_solo_lectura(fb.consulta_grupos())

    def test_rechaza_escrituras(self):
        for sql in ("UPDATE PRODUCTO SET X = 1",
                    "delete from DOCINV",
                    "INSERT INTO GRUPO VALUES (1)",
                    "EXECUTE PROCEDURE P",
                    "SELECT 1 FROM RDB$DATABASE; DROP TABLE X",
                    "SELECT * FROM PRODUCTO WITH LOCK",
                    "-- SELECT\nALTER TABLE X ADD Y INT",
                    "SELECT 1 FROM RDB$DATABASE WHERE 1 = (EXECUTE BLOCK)",
                    "MERGE INTO X USING Y ON 1=1"):
            with self.subTest(sql=sql), self.assertRaises(fb.ErrorBaseDatos):
                fb.validar_solo_lectura(sql)

    def test_palabras_dentro_de_textos_o_comentarios_no_cuentan(self):
        fb.validar_solo_lectura("SELECT 'UPDATE' AS X FROM RDB$DATABASE -- DELETE\n")


class LecturaConConexionFalsaTest(unittest.TestCase):
    def setUp(self):
        self.con = ConexionFalsa()
        p = mock.patch.object(fb, "_abrir_conexion", return_value=self.con)
        self.abrir = p.start()
        self.addCleanup(p.stop)

    def test_esquema_igual_al_del_csv(self):
        df = fb.leer_ventas(_config(), meses=24, hoy=date(2026, 10, 17))
        pd.testing.assert_frame_equal(df, _csv_esperado())

    def test_ventana_y_rango_de_la_consulta(self):
        df = fb.leer_ventas(_config(), meses=24, hoy=date(2026, 10, 17))
        v = df.attrs["ventana"]
        self.assertEqual((v.desde, v.hasta), (date(2024, 9, 1), date(2026, 8, 31)))
        self.assertEqual(df.attrs["fecha_max"], date(2026, 9, 22))
        self.assertIn("BETWEEN '2024-09-01' AND '2026-08-31'", self.con.sql[-1])
        self.assertIn("'2026-10-17'", self.con.sql[0])

    def test_rango_explicito_no_consulta_la_fecha_maxima(self):
        df = fb.leer_ventas(_config(), desde=date(2024, 1, 1), hasta=date(2026, 3, 31))
        self.assertEqual(len(self.con.sql), 1)
        self.assertIn("BETWEEN '2024-01-01' AND '2026-03-31'", self.con.sql[0])
        self.assertEqual(df.attrs["ventana"].mes_predicho, date(2026, 4, 1))

    def test_solo_select_en_transaccion_de_solo_lectura(self):
        from firebird.driver import DefaultAction, TraAccessMode
        fb.leer_ventas(_config(), hoy=date(2026, 10, 17))
        fb.leer_grupos(_config())
        self.assertTrue(self.con.sql)
        for sql in self.con.sql:
            sin_comentarios = re.sub(r"--.*", "", sql).strip().upper()
            self.assertTrue(sin_comentarios.startswith("SELECT"), sin_comentarios[:40])
            self.assertNotIn(";", sin_comentarios)
        # isc_tpb_version3, isc_tpb_read, isc_tpb_concurrency, isc_tpb_wait
        self.assertEqual(list(self.con.tpb), [3, 8, 2, 6])
        self.assertIn(TraAccessMode.READ, self.con.tpb)
        self.assertNotIn(TraAccessMode.WRITE, self.con.tpb)
        self.assertEqual(self.con.default_action, DefaultAction.ROLLBACK)
        self.assertEqual(self.con.commits, 0)
        self.assertGreaterEqual(self.con.rollbacks, 1)
        self.assertTrue(self.con.cerrada)
        self.assertEqual(self.con.cursores_abiertos, 0)

    def test_grupos_igual_que_el_catalogo_csv(self):
        mapa = fb.leer_grupos(_config())
        texto = "codigo;id_grupo;grupo\r\n1722;4;RODAJAS\r\n1800;7;RETENES\r\nT7120;;\r\n"
        pd.testing.assert_series_equal(mapa, gr.cargar_mapa(io.StringIO(texto)))

    def test_rechaza_si_el_motor_no_confirma_solo_lectura(self):
        self.con.solo_lectura = False
        with self.assertRaises(fb.ErrorBaseDatos) as ctx:
            fb.leer_grupos(_config())
        self.assertIn("solo lectura", str(ctx.exception))
        self.assertEqual(self.con.sql, [])
        self.assertTrue(self.con.cerrada)

    def test_base_sin_ventas(self):
        self.con.fecha_max = None
        with self.assertRaises(fb.ErrorBaseDatos) as ctx:
            fb.leer_ventas(_config(), hoy=date(2026, 10, 17))
        self.assertIn("no tiene facturas de venta", str(ctx.exception))


class ErroresSinSecretosTest(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        (d / "base.fdb").write_bytes(b"")
        (d / "fbclient.dll").write_bytes(b"")
        self.cfg = _config(base_datos=str(d / "base.fdb"), cliente=str(d / "fbclient.dll"))

    def test_fallo_de_conexion(self):
        cfg = self.cfg
        falla = RuntimeError(f"login failed for SYSDBA password {CLAVE} at {cfg.base_datos}")
        with mock.patch.object(fb, "_conectar_driver", side_effect=falla):
            with self.assertRaises(fb.ErrorBaseDatos) as ctx:
                fb.leer_ventas(cfg, hoy=date(2026, 10, 17))
        exc = ctx.exception
        self.assertIn("no se pudo conectar a la base de datos", str(exc))
        self.assertIn("config/base_datos.ini", str(exc))
        self.assertNotIn(CLAVE, str(exc))
        self.assertIsNone(exc.__cause__)
        self.assertTrue(exc.__suppress_context__)

    def test_fallo_del_cliente_fbclient(self):
        with mock.patch.object(fb, "_conectar_driver", side_effect=OSError("dll")):
            with self.assertRaises(fb.ErrorBaseDatos) as ctx:
                fb.leer_grupos(self.cfg)
        self.assertIn("fbclient.dll", str(ctx.exception))

    def test_archivos_inexistentes(self):
        with self.assertRaises(fb.ErrorConfiguracionBD) as ctx:
            fb.leer_grupos(_config(cliente=self.cfg.cliente))
        self.assertIn("archivo de la base de datos", str(ctx.exception))
        self.assertNotIn("C:/taller", str(ctx.exception))
        with self.assertRaises(fb.ErrorConfiguracionBD) as ctx:
            fb.leer_grupos(_config(host="servidor"))
        self.assertIn("fbclient.dll", str(ctx.exception))

    def test_fallo_durante_la_consulta(self):
        con = ConexionFalsa()
        con.execute = mock.Mock(side_effect=RuntimeError(f"error {CLAVE}"))
        with mock.patch.object(fb, "_abrir_conexion", return_value=con):
            with self.assertRaises(fb.ErrorBaseDatos) as ctx:
                fb.leer_grupos(_config())
        self.assertIn("no se pudo leer", str(ctx.exception))
        self.assertNotIn(CLAVE, str(ctx.exception))
        self.assertTrue(con.cerrada)


class ModeloConDatosEnMemoriaTest(unittest.TestCase):
    """El pipeline acepta el DataFrame leído de la base en lugar de un CSV."""

    def test_load_and_clean_igual_que_desde_archivo(self):
        from ventas_forecast.model import SalesForecastModel
        buf = io.BytesIO()
        extraccion.write_csv(buf, CABECERA_V6, [extraccion.format_row(r) for r in FILAS_V6])
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "v6.csv"
            ruta.write_bytes(buf.getvalue())
            desde_csv = SalesForecastModel(filepath=ruta, output_path=Path(d) / "o.csv")
            desde_csv.load_and_clean()
        datos = _csv_esperado()
        en_memoria = SalesForecastModel(filepath=None, datos=datos)
        en_memoria.load_and_clean()
        pd.testing.assert_frame_equal(en_memoria.df, desde_csv.df)
        pd.testing.assert_frame_equal(datos, _csv_esperado())  # no se modifica

    def test_mapa_de_grupos_en_memoria(self):
        from ventas_forecast.model import SalesForecastModel
        mapa = pd.Series(["RODAJAS"], index=["1722"])
        m = SalesForecastModel(filepath=None, datos=_csv_esperado(), grupos_mapa=mapa,
                               grupos_path=Path(tempfile.mkdtemp()) / "no_existe.csv")
        m.results = pd.DataFrame({"CODIGO": ["1722"], "TOTAL_VENDIDO": [1.0],
                                  "ventas_predichas": [2.0]})
        m.agregar_resumen_grupo()
        self.assertIs(m.mapa_grupos, mapa)
        self.assertIn("RODAJAS", list(m.resumen_grupos["grupo"]))

    def test_sin_archivo_ni_datos(self):
        from ventas_forecast.model import SalesForecastModel
        with self.assertRaises(ValueError):
            SalesForecastModel(filepath=None)


_DB_PRUEBA = os.environ.get("FIREBIRD_DB_PRUEBA", "")
_CLIENTE_PRUEBA = os.environ.get("FIREBIRD_CLIENT_PRUEBA", "")


@unittest.skipUnless(_DB_PRUEBA and _CLIENTE_PRUEBA and Path(_DB_PRUEBA).is_file()
                     and Path(_CLIENTE_PRUEBA).is_file(),
                     "defina FIREBIRD_DB_PRUEBA (COPIA de la base) y FIREBIRD_CLIENT_PRUEBA")
class IntegracionCopiaTest(unittest.TestCase):
    """Contra una COPIA de la base: mismo resultado que el CSV V6 versionado."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = fb.ConfigBD(base_datos=_DB_PRUEBA, cliente=_CLIENTE_PRUEBA,
                              usuario=os.environ.get("FIREBIRD_USER", "SYSDBA"),
                              clave=os.environ.get("FIREBIRD_PASSWORD") or None)

    def test_equivale_al_csv_v6(self):
        df = fb.leer_ventas(self.cfg, desde=date(2024, 1, 1), hasta=date(2026, 3, 31))
        csv = pd.read_csv(ROOT / "data" / "raw" / "Query_Result_V6.csv", sep=";",
                          encoding="utf-8")
        self.assertEqual(len(df), len(csv))
        pd.testing.assert_frame_equal(df, csv)

    def test_ventana_por_defecto(self):
        df = fb.leer_ventas(self.cfg, meses=24)
        v, fmax = df.attrs["ventana"], df.attrs["fecha_max"]
        self.assertLessEqual(v.hasta, fmax)
        self.assertEqual(v.desde.day, 1)
        self.assertEqual((v.hasta.year - v.desde.year) * 12 + v.hasta.month - v.desde.month, 23)
        fechas = pd.to_datetime(df["FECHA"], format="%d/%m/%Y")
        self.assertGreaterEqual(fechas.min().date(), v.desde)
        self.assertLessEqual(fechas.max().date(), v.hasta)

    def test_grupos_igual_al_catalogo(self):
        mapa = fb.leer_grupos(self.cfg)
        pd.testing.assert_series_equal(
            mapa, gr.cargar_mapa(ROOT / "data" / "raw" / "grupos_productos.csv"))


if __name__ == "__main__":
    unittest.main()
