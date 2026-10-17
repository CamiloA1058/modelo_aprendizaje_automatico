"""Lectura directa de la base Firebird del taller (solo lectura).

El dashboard puede generar la predicción leyendo la base en lugar de un CSV
subido. Se usa la misma consulta V6 (sql/extraccion_ventas_v6.sql) y el mismo
formato que el extractor de línea de comandos, así que el DataFrame que
devuelve leer_ventas() es idéntico al que el pipeline obtiene al leer
data/raw/Query_Result_V6.csv con el mismo rango de fechas.

Seguridad:
- Solo lectura: cada lectura abre una transacción SNAPSHOT de solo lectura
  (TraAccessMode.READ), que el motor Firebird rechaza si algo intenta escribir;
  cada sentencia pasa antes por validar_solo_lectura() (solo SELECT) y la
  transacción termina siempre con rollback, nunca con commit.
- Credenciales solo por configuración (variables FIREBIRD_* o el archivo local
  config/base_datos.ini, ignorado por git). Ningún mensaje de error incluye la
  clave, la ruta de la base ni el texto del error del motor.
"""
import calendar
import configparser
import io
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from ventas_forecast import extraccion
from ventas_forecast import grupos as grupos_mod
from ventas_forecast.paths import PROJECT_ROOT

ARCHIVO_CONFIG = PROJECT_ROOT / "config" / "base_datos.ini"
SECCION = "firebird"
MESES_POR_DEFECTO = 24
USUARIO_POR_DEFECTO = "SYSDBA"

# clave del archivo -> variable de entorno (el entorno tiene prioridad)
VARIABLES = {
    "base_datos": "FIREBIRD_DB",
    "host": "FIREBIRD_HOST",
    "usuario": "FIREBIRD_USER",
    "clave": "FIREBIRD_PASSWORD",
    "cliente": "FIREBIRD_CLIENT",
    "meses": "FIREBIRD_MESES",
}
_DONDE = "config/base_datos.ini (o en las variables FIREBIRD_*)"

# Lo mismo que el pipeline usa al leer un CSV (SalesForecastModel: sep, encoding)
_SEP, _ENCODING = ";", "utf-8"


class ErrorBaseDatos(RuntimeError):
    """Fallo al conectar o leer la base. El mensaje es apto para el usuario:
    no incluye claves, rutas ni el texto original del motor."""


class ErrorConfiguracionBD(ErrorBaseDatos):
    """La lectura de la base no está configurada o la configuración es inválida."""


@dataclass(frozen=True)
class ConfigBD:
    base_datos: str
    cliente: str
    usuario: str = USUARIO_POR_DEFECTO
    clave: str | None = field(default=None, repr=False)
    host: str | None = None
    meses: int = MESES_POR_DEFECTO

    @property
    def dsn(self):
        """'host:ruta' con servidor; la ruta sola con el cliente embebido."""
        return f"{self.host}:{self.base_datos}" if self.host else self.base_datos


@dataclass(frozen=True)
class Ventana:
    desde: date
    hasta: date
    mes_predicho: date


# ── Configuración ────────────────────────────────────────────────────────────
def cargar_config(entorno=None, archivo=ARCHIVO_CONFIG):
    """Configuración de la base: primero las variables de entorno, luego el
    archivo local (sección [firebird]). Exige base_datos y cliente."""
    entorno = os.environ if entorno is None else entorno
    valores = {}
    if archivo is not None and os.path.isfile(archivo):
        parser = configparser.RawConfigParser()  # sin interpolación: claves con '%'
        try:
            parser.read(archivo, encoding="utf-8-sig")
        except configparser.Error:
            raise ErrorConfiguracionBD(
                "el archivo config/base_datos.ini no tiene un formato válido: "
                "compárelo con config/base_datos.ejemplo.ini") from None
        if parser.has_section(SECCION):
            valores = {k: v.strip() for k, v in parser.items(SECCION)}

    def valor(clave):
        v = (entorno.get(VARIABLES[clave]) or "").strip()
        return v or valores.get(clave) or None

    faltan = [c for c in ("base_datos", "cliente") if not valor(c)]
    if faltan:
        detalle = ", ".join(f"{c} ({VARIABLES[c]})" for c in faltan)
        raise ErrorConfiguracionBD(
            f"la lectura de la base de datos no está configurada: falta {detalle} "
            f"en {_DONDE}")
    meses = valor("meses") or MESES_POR_DEFECTO
    try:
        meses = int(meses)
        if meses < 1:
            raise ValueError
    except ValueError:
        raise ErrorConfiguracionBD(
            f"'meses' debe ser un número entero mayor que cero en {_DONDE}") from None
    return ConfigBD(base_datos=valor("base_datos"), cliente=valor("cliente"),
                    usuario=valor("usuario") or USUARIO_POR_DEFECTO,
                    clave=valor("clave"), host=valor("host"), meses=meses)


# ── Ventana de datos ─────────────────────────────────────────────────────────
def calcular_ventana(fecha_max, meses=MESES_POR_DEFECTO):
    """Últimos `meses` meses COMPLETOS hasta fecha_max. Si fecha_max no es el
    último día de su mes, ese mes está incompleto y se excluye; la predicción
    es para el mes siguiente a `hasta`."""
    if meses < 1:
        raise ValueError("meses debe ser >= 1")
    if isinstance(fecha_max, datetime):
        fecha_max = fecha_max.date()
    ultimo_dia = calendar.monthrange(fecha_max.year, fecha_max.month)[1]
    if fecha_max.day == ultimo_dia:
        hasta = fecha_max
    else:
        hasta = fecha_max.replace(day=1) - timedelta(days=1)
    indice = hasta.year * 12 + hasta.month - 1 - (meses - 1)
    desde = date(indice // 12, indice % 12 + 1, 1)
    mes_predicho = hasta + timedelta(days=1)
    return Ventana(desde=desde, hasta=hasta, mes_predicho=mes_predicho)


# ── Consultas (solo SELECT) ──────────────────────────────────────────────────
_PROHIBIDAS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|UPSERT|EXECUTE|EXEC|ALTER|CREATE|DROP|RECREATE|"
    r"GRANT|REVOKE|COMMIT|ROLLBACK|SET|DECLARE|LOCK)\b")


def _sin_comentarios_ni_textos(sql):
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validar_solo_lectura(sql):
    """Acepta una sola sentencia SELECT sin palabras de escritura; si no,
    ErrorBaseDatos. Es una defensa adicional a la transacción de solo lectura."""
    limpio = _sin_comentarios_ni_textos(sql).strip().rstrip(";").strip().upper()
    if not limpio.startswith("SELECT") or ";" in limpio or _PROHIBIDAS.search(limpio):
        raise ErrorBaseDatos("consulta rechazada: solo se permiten lecturas (SELECT)")
    return sql


def _iso(d):
    return d.isoformat() if isinstance(d, date) else str(d)


def consulta_ventas(desde, hasta):
    """Consulta V6 con el rango [desde, hasta] (fechas incluidas)."""
    sql = extraccion.SQL_VENTAS_V6.read_text(encoding="utf-8")
    return extraccion.sin_punto_y_coma(extraccion.apply_date_range(sql, _iso(desde), _iso(hasta)))


def consulta_fecha_max(hoy):
    """Fecha de la última venta válida: MAX(FECHA) sobre la misma consulta V6
    (mismos filtros, GROUP BY y HAVING), sin fechas posteriores a `hoy`."""
    sql = consulta_ventas(date(1900, 1, 1), hoy)
    corte = sql.upper().rindex("ORDER BY")
    return f"SELECT MAX(FECHA) AS FECHA_MAX FROM ({sql[:corte].rstrip()})"


def consulta_grupos():
    return extraccion.sin_punto_y_coma(extraccion.SQL_GRUPOS.read_text(encoding="utf-8"))


# ── Conexión de solo lectura ─────────────────────────────────────────────────
def tpb_solo_lectura():
    """TPB de la transacción: versión 3, READ (solo lectura), SNAPSHOT
    (concurrency) y WAIT. Son los mismos bytes que
    firebird.driver.tpb(Isolation.SNAPSHOT, access_mode=TraAccessMode.READ),
    pero sin cargar fbclient para construirlos."""
    from firebird.driver import Isolation, TraAccessMode
    from firebird.driver.types import TraLockResolution

    return bytes([3, TraAccessMode.READ, Isolation.SNAPSHOT, TraLockResolution.WAIT])


def _conectar_driver(config):
    from firebird.driver import connect, driver_config

    driver_config.fb_client_library.value = config.cliente
    kwargs = {"user": config.usuario, "charset": extraccion.CHARSET}
    if config.clave:
        kwargs["password"] = config.clave
    return connect(config.dsn, **kwargs)


def _abrir_conexion(config):
    if not os.path.isfile(config.cliente):
        raise ErrorConfiguracionBD(
            f"no se encontró el cliente de Firebird (fbclient.dll) indicado en {_DONDE}")
    if not config.host and not os.path.isfile(config.base_datos):
        raise ErrorConfiguracionBD(
            f"no se encontró el archivo de la base de datos indicado en {_DONDE}")
    try:
        return _conectar_driver(config)
    except OSError:
        raise ErrorBaseDatos(
            "no se pudo cargar el cliente de Firebird (fbclient.dll): revise la ruta "
            f"'cliente' en {_DONDE}") from None
    except Exception:
        raise ErrorBaseDatos(
            "no se pudo conectar a la base de datos: revise la ruta y el usuario en "
            f"{_DONDE}") from None


class _SesionLectura:
    """Conexión + transacción SNAPSHOT de solo lectura. Todas las consultas de
    una lectura ven la misma foto de la base; al salir se hace rollback."""

    def __init__(self, config):
        self.config = config

    def __enter__(self):
        from firebird.driver import DefaultAction

        self._con = _abrir_conexion(self.config)
        try:
            self._tm = self._con.transaction_manager(
                default_tpb=tpb_solo_lectura(), default_action=DefaultAction.ROLLBACK)
            self._tm.begin()
            solo_lectura = self._tm.info.is_read_only()
        except Exception:
            self._cerrar()
            raise ErrorBaseDatos(
                "no se pudo abrir una transacción de solo lectura en la base de datos"
            ) from None
        if not solo_lectura:  # el motor confirma el modo; si no, no se lee nada
            self._cerrar()
            raise ErrorBaseDatos("la base de datos no aceptó una transacción de solo lectura")
        return self

    def consultar(self, sql):
        validar_solo_lectura(sql)
        try:
            cur = self._tm.cursor()
            try:
                cur.execute(sql)
                return [d[0] for d in cur.description], cur.fetchall()
            finally:
                cur.close()
        except Exception:
            raise ErrorBaseDatos(
                "no se pudo leer la base de datos: verifique que sea la base de ventas "
                "del taller y que el usuario tenga permiso de lectura") from None

    def _cerrar(self):
        try:
            tm = getattr(self, "_tm", None)
            if tm is not None:
                tm.rollback()
        except Exception:
            pass
        try:
            self._con.close()
        except Exception:
            pass

    def __exit__(self, *exc):
        self._cerrar()
        return False


# ── Lecturas ─────────────────────────────────────────────────────────────────
def leer_ventas(config, meses=None, desde=None, hasta=None, hoy=None):
    """Ventas V6 como DataFrame con el mismo esquema y tipos que el pipeline
    obtiene al leer el CSV V6 (pd.read_csv con sep=';' y utf-8).

    Sin desde/hasta, la ventana son los últimos `meses` meses completos hasta
    la última venta válida (calcular_ventana). En df.attrs quedan 'ventana'
    (Ventana) y 'fecha_max' (última venta; None con rango explícito)."""
    meses = meses or config.meses
    hoy = hoy or date.today()
    with _SesionLectura(config) as sesion:
        if desde is None or hasta is None:
            _, filas = sesion.consultar(consulta_fecha_max(hoy))
            fecha_max = filas[0][0] if filas else None
            if fecha_max is None:
                raise ErrorBaseDatos("la base de datos no tiene facturas de venta válidas")
            ventana = calcular_ventana(fecha_max, meses)
        else:
            fecha_max = None
            siguiente = (hasta.replace(day=1) + timedelta(days=32)).replace(day=1)
            ventana = Ventana(desde=desde, hasta=hasta, mes_predicho=siguiente)
        cabecera, filas = sesion.consultar(consulta_ventas(ventana.desde, ventana.hasta))
    if not filas:
        raise ErrorBaseDatos("la base de datos no tiene ventas en el rango de fechas elegido")
    texto = extraccion.csv_texto(cabecera, [extraccion.format_row(f) for f in filas])
    df = pd.read_csv(io.BytesIO(texto.encode(_ENCODING)), sep=_SEP, encoding=_ENCODING)
    df.attrs["ventana"] = ventana
    df.attrs["fecha_max"] = fecha_max
    return df


def leer_grupos(config):
    """Mapa codigo -> grupo, igual a grupos.cargar_mapa(data/raw/grupos_productos.csv)."""
    with _SesionLectura(config) as sesion:
        _, filas = sesion.consultar(consulta_grupos())
    texto = extraccion.csv_texto(extraccion.CABECERA_GRUPOS,
                                 [extraccion.format_row_grupo(f) for f in filas])
    return grupos_mod.cargar_mapa(io.StringIO(texto))
