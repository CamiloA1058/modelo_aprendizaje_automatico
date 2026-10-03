"""
explorar_bd_v2.py
=================
Versión 2 — con filtros completos contra datos corruptos.

Problemas que resuelve:
  1. VALORU = 1     → productos sin precio configurado
  2. Overflow en TOTALITEM → valores de billones de COP
  3. CANTREAL absurda → 0.25 unidades de banda, 10000 unidades por factura
  4. Items con total cero o negativo

Uso:
    pip install fdb
    # definir FIREBIRD_HOST, FIREBIRD_DATABASE, FIREBIRD_USER y FIREBIRD_PASSWORD
    # (ver .env.example en esta carpeta)
    python sql/exploracion/explorar_bd_v2.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

def _env(nombre, obligatoria=True, defecto=None):
    """Lee una variable de entorno; si es obligatoria y falta, termina con un mensaje claro."""
    valor = os.environ.get(nombre, defecto)
    if obligatoria and not valor:
        print(f"Falta la variable de entorno {nombre}.")
        print("Define FIREBIRD_HOST, FIREBIRD_DATABASE, FIREBIRD_USER y FIREBIRD_PASSWORD")
        print("(ver sql/exploracion/.env.example) antes de ejecutar este script.")
        sys.exit(1)
    return valor


# ── Configuración de conexión (desde variables de entorno) ────────────────────
DB_PATH  = _env("FIREBIRD_DATABASE")
USER     = _env("FIREBIRD_USER")
PASSWORD = _env("FIREBIRD_PASSWORD")
HOST     = _env("FIREBIRD_HOST")

# Carpeta donde se guardan los CSV (opcional)
OUTPUT_DIR = _env("FIREBIRD_OUTPUT_DIR", obligatoria=False, defecto="exploracion_output")

# ── Driver ────────────────────────────────────────────────────────────────────
try:
    import fdb
    DRIVER = "fdb"
except ImportError:
    try:
        import firebird.driver as fdb
        DRIVER = "firebird-driver"
    except ImportError:
        print("❌ Instala el driver: pip install fdb")
        sys.exit(1)

print(f"✅ Driver: {DRIVER}")


def conectar():
    try:
        if DRIVER == "fdb":
            con = fdb.connect(
                host=HOST, database=DB_PATH,
                user=USER, password=PASSWORD, charset="UTF8",
            )
        else:
            con = fdb.connect(
                database=f"{HOST}:{DB_PATH}",
                user=USER, password=PASSWORD,
            )
        print("✅ Conexión exitosa")
        return con
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        sys.exit(1)


def exportar(con, nombre, query):
    try:
        df = pd.read_sql(query, con)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ruta = os.path.join(OUTPUT_DIR, f"{nombre}.csv")
        df.to_csv(ruta, index=False, sep=";", encoding="utf-8-sig")
        print(f"  ✅ {nombre}.csv  →  {len(df):,} filas  →  {ruta}")
        return df
    except Exception as e:
        print(f"  ❌ Error en {nombre}: {e}")
        return None


# ── Queries ───────────────────────────────────────────────────────────────────
QUERIES = {

    # 1. Dataset principal limpio — V5 con todos los filtros
    "01_ventas_v5": """
        SELECT
            p.CODIGO,
            p.DESCRIPCION,
            d.FECHA,
            EXTRACT(YEAR    FROM d.FECHA) AS ANIO,
            EXTRACT(MONTH   FROM d.FECHA) AS MES,
            EXTRACT(DAY     FROM d.FECHA) AS DIA,
            EXTRACT(WEEKDAY FROM d.FECHA) AS DIA_SEMANA,
            SUM(i.CANTREAL)              AS VENTAS,
            SUM(i.TOTALITEM)             AS TOTAL_VENDIDO,
            AVG(i.VALORU)                AS PRECIO_PROMEDIO,
            COUNT(DISTINCT d.IDDOCINV)   AS FRECUENCIA,
            CASE WHEN SUM(i.TOTALDESCUENTO) > 0 THEN 1 ELSE 0 END AS HUBO_PROMOCION,
            CASE
                WHEN EXTRACT(MONTH FROM d.FECHA) IN (11,12) THEN 'Alta'
                WHEN EXTRACT(MONTH FROM d.FECHA) IN (6,7)   THEN 'Media'
                ELSE 'Baja'
            END AS TEMPORADA
        FROM DOCINV d
        INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
        INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO
        WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'
          AND i.VALORU    >  10
          AND i.VALORU    <  20000000
          AND i.CANTREAL  >  0
          AND i.CANTREAL  <  10000
          AND i.TOTALITEM >  0
          AND i.TOTALITEM <  50000000
        GROUP BY p.CODIGO, p.DESCRIPCION, d.FECHA
        HAVING AVG(i.VALORU)    >  10
           AND AVG(i.VALORU)    <  20000000
           AND SUM(i.TOTALITEM) >  0
           AND SUM(i.TOTALITEM) <  50000000
        ORDER BY d.FECHA
    """,

    # 2. Resumen mensual (para validar que dic-2024 ya no es anómalo)
    "02_resumen_mensual_v5": """
        SELECT
            EXTRACT(YEAR  FROM d.FECHA)  AS ANIO,
            EXTRACT(MONTH FROM d.FECHA)  AS MES,
            COUNT(DISTINCT p.CODIGO)     AS PRODUCTOS_DISTINTOS,
            COUNT(DISTINCT d.IDDOCINV)   AS NUM_FACTURAS,
            SUM(i.CANTREAL)              AS UNIDADES_VENDIDAS,
            SUM(i.TOTALITEM)             AS TOTAL_VENDIDO
        FROM DOCINV d
        INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
        INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO
        WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'
          AND i.VALORU    >  10
          AND i.VALORU    <  20000000
          AND i.CANTREAL  >  0
          AND i.CANTREAL  <  10000
          AND i.TOTALITEM >  0
          AND i.TOTALITEM <  50000000
        GROUP BY EXTRACT(YEAR FROM d.FECHA), EXTRACT(MONTH FROM d.FECHA)
        ORDER BY ANIO, MES
    """,

    # 3. Último mes detallado (para verificar que ya no hay precios de $56)
    "03_ultimo_mes_v5": """
        SELECT
            p.CODIGO,
            p.DESCRIPCION,
            d.FECHA,
            i.VALORU        AS PRECIO_UNITARIO,
            i.CANTREAL      AS CANTIDAD,
            i.TOTALITEM     AS TOTAL_ITEM,
            i.TOTALDESCUENTO AS DESCUENTO
        FROM DOCINV d
        INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
        INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO
        WHERE d.FECHA BETWEEN '2026-03-01' AND '2026-03-31'
          AND i.VALORU    >  10
          AND i.VALORU    <  20000000
          AND i.CANTREAL  >  0
          AND i.CANTREAL  <  10000
          AND i.TOTALITEM >  0
          AND i.TOTALITEM <  50000000
        ORDER BY p.CODIGO, d.FECHA
    """,

    # 4. Registros que se están eliminando (para auditoría)
    "04_registros_eliminados": """
        SELECT
            p.CODIGO,
            p.DESCRIPCION,
            d.FECHA,
            i.VALORU       AS PRECIO_UNITARIO,
            i.CANTREAL     AS CANTIDAD,
            i.TOTALITEM    AS TOTAL_ITEM,
            CASE
                WHEN i.VALORU <= 10              THEN 'Precio <= 10'
                WHEN i.VALORU >= 20000000        THEN 'Precio >= 20M'
                WHEN i.CANTREAL <= 0             THEN 'Cantidad <= 0'
                WHEN i.CANTREAL >= 10000         THEN 'Cantidad >= 10000'
                WHEN i.TOTALITEM <= 0            THEN 'Total <= 0'
                WHEN i.TOTALITEM >= 50000000     THEN 'Total >= 50M'
                ELSE 'Otro'
            END AS RAZON_EXCLUSION
        FROM DOCINV d
        INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
        INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO
        WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'
          AND (
              i.VALORU    <= 10           OR
              i.VALORU    >= 20000000     OR
              i.CANTREAL  <= 0            OR
              i.CANTREAL  >= 10000        OR
              i.TOTALITEM <= 0            OR
              i.TOTALITEM >= 50000000
          )
        ORDER BY i.TOTALITEM DESC
        ROWS 500
    """,

    # 5. Precios históricos por producto (versión limpia)
    "05_precios_por_producto_v5": """
        SELECT
            p.CODIGO,
            p.DESCRIPCION,
            EXTRACT(YEAR  FROM d.FECHA) AS ANIO,
            EXTRACT(MONTH FROM d.FECHA) AS MES,
            MIN(i.VALORU)               AS PRECIO_MIN,
            MAX(i.VALORU)               AS PRECIO_MAX,
            AVG(i.VALORU)               AS PRECIO_PROMEDIO,
            COUNT(*)                    AS NUM_TRANSACCIONES
        FROM DOCINV d
        INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
        INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO
        WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'
          AND i.VALORU    >  10
          AND i.VALORU    <  20000000
          AND i.CANTREAL  >  0
          AND i.CANTREAL  <  10000
          AND i.TOTALITEM >  0
          AND i.TOTALITEM <  50000000
        GROUP BY p.CODIGO, p.DESCRIPCION,
                 EXTRACT(YEAR FROM d.FECHA),
                 EXTRACT(MONTH FROM d.FECHA)
        ORDER BY p.CODIGO, ANIO, MES
    """,
}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  EXPLORADOR BD V2 — Filtros completos contra corrupción")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    print("Filtros aplicados:")
    print("  ✅ VALORU        > 10 y < 20.000.000")
    print("  ✅ CANTREAL      > 0  y < 10.000")
    print("  ✅ TOTALITEM     > 0  y < 50.000.000")
    print()

    con = conectar()

    print(f"\nExportando {len(QUERIES)} consultas a '{OUTPUT_DIR}'...\n")
    for nombre, query in QUERIES.items():
        exportar(con, nombre, query)

    con.close()
    print(f"\n✅ Listo. Sube la carpeta '{OUTPUT_DIR}' al chat.")
    print("   El archivo más importante es '01_ventas_v5.csv'")
    print("   — ese es el dataset final para el modelo.")
