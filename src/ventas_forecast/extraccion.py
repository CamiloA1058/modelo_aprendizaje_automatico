"""Formato de la extracción de ventas V6 y del catálogo de grupos.

Lo comparten el extractor de línea de comandos (sql/extraer_ventas.py,
sql/extraer_grupos.py) y la lectura directa de la base desde el dashboard
(ventas_forecast.fuente_bd), para que ambos caminos produzcan exactamente el
mismo CSV que data/raw/Query_Result_V6.csv.

Formato: separador ';', UTF-8 con BOM, CRLF, fechas dd/mm/aaaa, VENTAS /
TOTAL_VENDIDO / PRECIO_PROMEDIO con punto de miles y coma decimal (2
decimales) y TEMPORADA rellenada a 5 caracteres (columna CHAR(5)).
"""
import csv
import io
import re
from decimal import ROUND_HALF_UP, Decimal

from ventas_forecast.paths import PROJECT_ROOT

CHARSET = "ISO8859_1"
SQL_VENTAS_V6 = PROJECT_ROOT / "sql" / "extraccion_ventas_v6.sql"
SQL_GRUPOS = PROJECT_ROOT / "sql" / "grupos_productos.sql"
CABECERA_GRUPOS = ["codigo", "id_grupo", "grupo"]

_RANGE_RE = re.compile(
    r"(BETWEEN\s+)'\d{4}-\d{2}-\d{2}'(\s+AND\s+)'\d{4}-\d{2}-\d{2}'", re.IGNORECASE
)
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def format_es(value):
    """Formatea un numero como '1.234.567,89' (miles con punto, 2 decimales)."""
    dec = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    entero, decimales = f"{abs(dec):.2f}".split(".")
    miles = f"{int(entero):,}".replace(",", ".")
    signo = "-" if dec < 0 else ""
    return f"{signo}{miles},{decimales}"


def pad_season(value):
    """TEMPORADA sale de un CASE CHAR(5): 'Alta ', 'Media', 'Baja '."""
    return str(value).ljust(5)


def format_row(row):
    """Convierte una fila de la consulta (13 columnas) a las cadenas del CSV V5."""
    (codigo, descripcion, fecha, anio, mes, dia, dia_semana,
     ventas, total, precio, frecuencia, promo, temporada) = row
    return [
        str(codigo),
        str(descripcion),
        fecha.strftime("%d/%m/%Y"),
        str(int(anio)), str(int(mes)), str(int(dia)), str(int(dia_semana)),
        format_es(ventas), format_es(total), format_es(precio),
        str(int(frecuencia)), str(int(promo)),
        pad_season(temporada),
    ]


def format_row_grupo(row):
    """(codigo, id_grupo, nombre) -> cadenas; None queda como cadena vacia."""
    return ["" if v is None else str(v).strip() for v in row]


def csv_texto(header, rows):
    """Texto CSV con separador ';', CRLF y comillas minimas (sin BOM)."""
    text = io.StringIO(newline="")
    writer = csv.writer(text, delimiter=";", lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return text.getvalue()


def write_csv(binary_stream, header, rows):
    """Escribe UTF-8 con BOM, separador ';' y CRLF, con comillas minimas."""
    binary_stream.write(b"\xef\xbb\xbf" + csv_texto(header, rows).encode("utf-8"))


def apply_date_range(sql, desde, hasta):
    """Sustituye el primer 'BETWEEN <fecha> AND <fecha>' por el rango pedido."""
    for valor in (desde, hasta):
        if not _ISO_RE.match(valor):
            raise ValueError(f"Fecha debe ser AAAA-MM-DD: {valor!r}")
    nuevo, n = _RANGE_RE.subn(rf"\g<1>'{desde}'\g<2>'{hasta}'", sql, count=1)
    if n == 0:
        raise ValueError("El SQL no contiene un rango 'BETWEEN <fecha> AND <fecha>'")
    return nuevo


def sin_punto_y_coma(sql):
    """Quita el ';' final: el driver ejecuta una sola sentencia sin terminador."""
    return sql.rstrip().rstrip(";")
