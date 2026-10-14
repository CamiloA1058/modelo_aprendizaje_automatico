"""Extractor offline: ejecuta un archivo .sql contra una base Firebird y escribe
un CSV con EXACTAMENTE el formato de data/raw/Query_Result_V5.csv.

Formato de salida: separador ';', UTF-8 con BOM, CRLF, fechas dd/mm/aaaa,
VENTAS / TOTAL_VENDIDO / PRECIO_PROMEDIO con punto de miles y coma decimal
(2 decimales) y TEMPORADA rellenada a 5 caracteres (columna CHAR(5)).

Uso (la base se abre con el cliente Firebird embebido, sin servidor):

    set FIREBIRD_CLIENT=C:\ruta\fbclient.dll
    python sql/extraer_ventas.py --db copia.fdb --sql sql/extraccion_ventas_v6.sql \
        --out data/raw/Query_Result_V6.csv --desde 2024-01-01 --hasta 2026-03-27

Variables de entorno: FIREBIRD_CLIENT (o --fbclient), FIREBIRD_USER (por defecto
SYSDBA), FIREBIRD_PASSWORD (opcional con el cliente embebido). Nunca se escriben
credenciales en el repositorio. Trabajar siempre sobre una COPIA de la base.
"""
import argparse
import csv
import io
import os
import re
import sys
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

CHARSET = "ISO8859_1"
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


def write_csv(binary_stream, header, rows):
    """Escribe UTF-8 con BOM, separador ';' y CRLF, con comillas minimas."""
    text = io.StringIO(newline="")
    writer = csv.writer(text, delimiter=";", lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    binary_stream.write(b"\xef\xbb\xbf" + text.getvalue().encode("utf-8"))


def apply_date_range(sql, desde, hasta):
    """Sustituye el primer 'BETWEEN <fecha> AND <fecha>' por el rango pedido."""
    for valor in (desde, hasta):
        if not _ISO_RE.match(valor):
            raise ValueError(f"Fecha debe ser AAAA-MM-DD: {valor!r}")
    nuevo, n = _RANGE_RE.subn(rf"\g<1>'{desde}'\g<2>'{hasta}'", sql, count=1)
    if n == 0:
        raise ValueError("El SQL no contiene un rango 'BETWEEN <fecha> AND <fecha>'")
    return nuevo


def run_query(db, sql, fbclient, user, password):
    from firebird.driver import connect, driver_config

    driver_config.fb_client_library.value = fbclient
    kwargs = {"user": user, "charset": CHARSET}
    if password:
        kwargs["password"] = password
    with connect(db, **kwargs) as con:
        cur = con.cursor()
        cur.execute(sql)
        header = [d[0] for d in cur.description]
        return header, cur.fetchall()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="ruta de la COPIA de la base .fdb")
    ap.add_argument("--sql", required=True, help="archivo .sql con la consulta")
    ap.add_argument("--out", required=True, help="CSV de salida")
    ap.add_argument("--desde", required=True, help="AAAA-MM-DD")
    ap.add_argument("--hasta", required=True, help="AAAA-MM-DD")
    ap.add_argument("--fbclient", default=os.environ.get("FIREBIRD_CLIENT"))
    args = ap.parse_args(argv)
    if not args.fbclient:
        ap.error("indique --fbclient o la variable FIREBIRD_CLIENT")

    sql = apply_date_range(Path(args.sql).read_text(encoding="utf-8"), args.desde, args.hasta)
    sql = sql.rstrip().rstrip(";")
    header, rows = run_query(
        args.db, sql, args.fbclient,
        os.environ.get("FIREBIRD_USER", "SYSDBA"), os.environ.get("FIREBIRD_PASSWORD"),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        write_csv(fh, header, [format_row(r) for r in rows])
    print(f"[OK] {len(rows):,} filas -> {out}")


if __name__ == "__main__":
    sys.exit(main())
