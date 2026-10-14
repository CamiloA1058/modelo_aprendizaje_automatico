"""Extractor offline del catalogo producto -> grupo (sql/grupos_productos.sql).

Escribe data/raw/grupos_productos.csv (separador ';', UTF-8 con BOM, CRLF) con
las columnas codigo;id_grupo;grupo. Son datos de catalogo: no hay clientes.

    set FIREBIRD_CLIENT=C:\ruta\fbclient.dll
    python sql/extraer_grupos.py --db copia.fdb --out data/raw/grupos_productos.csv

Usa siempre una COPIA de la base. Variables: FIREBIRD_CLIENT (o --fbclient),
FIREBIRD_USER (SYSDBA por defecto), FIREBIRD_PASSWORD (opcional).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraer_ventas as ev  # noqa: E402

SQL = Path(__file__).resolve().parent / "grupos_productos.sql"
CABECERA = ["codigo", "id_grupo", "grupo"]


def format_row(row):
    """(codigo, id_grupo, nombre) -> cadenas; None queda como cadena vacia."""
    return ["" if v is None else str(v).strip() for v in row]


def write_csv(binary_stream, header, rows):
    ev.write_csv(binary_stream, header, rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="ruta de la COPIA de la base .fdb")
    ap.add_argument("--out", default="data/raw/grupos_productos.csv")
    ap.add_argument("--fbclient", default=os.environ.get("FIREBIRD_CLIENT"))
    args = ap.parse_args(argv)
    if not args.fbclient:
        ap.error("indique --fbclient o la variable FIREBIRD_CLIENT")
    sql = SQL.read_text(encoding="utf-8").rstrip().rstrip(";")
    _, rows = ev.run_query(args.db, sql, args.fbclient,
                           os.environ.get("FIREBIRD_USER", "SYSDBA"),
                           os.environ.get("FIREBIRD_PASSWORD"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        write_csv(fh, CABECERA, [format_row(r) for r in rows])
    print(f"[OK] {len(rows):,} productos -> {out}")


if __name__ == "__main__":
    sys.exit(main())
