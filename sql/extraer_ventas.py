"""Extractor offline: ejecuta un archivo .sql contra una base Firebird y escribe
un CSV con EXACTAMENTE el formato de data/raw/Query_Result_V5.csv.

Formato de salida: separador ';', UTF-8 con BOM, CRLF, fechas dd/mm/aaaa,
VENTAS / TOTAL_VENDIDO / PRECIO_PROMEDIO con punto de miles y coma decimal
(2 decimales) y TEMPORADA rellenada a 5 caracteres (columna CHAR(5)).

Uso (la base se abre con el cliente Firebird embebido, sin servidor):

    set FIREBIRD_CLIENT=C:\ruta\fbclient.dll
    python sql/extraer_ventas.py --db copia.fdb --sql sql/extraccion_ventas_v6.sql \
        --out data/raw/Query_Result_V6.csv --desde 2024-01-01 --hasta 2026-03-31

Variables de entorno: FIREBIRD_CLIENT (o --fbclient), FIREBIRD_USER (por defecto
SYSDBA), FIREBIRD_PASSWORD (opcional con el cliente embebido). Nunca se escriben
credenciales en el repositorio. Trabajar siempre sobre una COPIA de la base.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# El formato del CSV vive en el paquete: lo comparte la lectura directa de la
# base desde el dashboard (ventas_forecast.fuente_bd).
from ventas_forecast.extraccion import (  # noqa: E402,F401
    CHARSET, apply_date_range, format_es, format_row, pad_season, write_csv,
)


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
