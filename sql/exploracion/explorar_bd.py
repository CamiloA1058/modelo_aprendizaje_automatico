"""
explorar_bd.py
==============
Conecta a la base de datos Firebird local y exporta
los datos necesarios para análisis del modelo de ventas.

Uso:
    pip install fdb
    # definir FIREBIRD_HOST, FIREBIRD_DATABASE, FIREBIRD_USER y FIREBIRD_PASSWORD
    # (ver .env.example en esta carpeta)
    python sql/exploracion/explorar_bd.py
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

# ── Intentar importar driver ──────────────────────────────────────────────────
try:
    import fdb
    DRIVER = "fdb"
except ImportError:
    try:
        import firebird.driver as fdb
        DRIVER = "firebird-driver"
    except ImportError:
        print("❌ No se encontró driver de Firebird.")
        print("   Instala uno de estos:")
        print("   pip install fdb")
        print("   pip install firebird-driver")
        sys.exit(1)

print(f"✅ Driver cargado: {DRIVER}")


def conectar():
    """Retorna una conexión a la base de datos."""
    try:
        if DRIVER == "fdb":
            con = fdb.connect(
                host=HOST,
                database=DB_PATH,
                user=USER,
                password=PASSWORD,
                charset="UTF8",
            )
        else:
            con = fdb.connect(
                database=f"{HOST}:{DB_PATH}",
                user=USER,
                password=PASSWORD,
            )
        print("✅ Conexión exitosa a la base de datos")
        return con
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        sys.exit(1)


def exportar(con, nombre, query):
    """Ejecuta una query y guarda el resultado como CSV."""
    try:
        df = pd.read_sql(query, con)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ruta = os.path.join(OUTPUT_DIR, f"{nombre}.csv")
        df.to_csv(ruta, index=False, sep=";", encoding="utf-8-sig")
        print(f"  ✅ {nombre}.csv → {len(df):,} filas → {ruta}")
        return df
    except Exception as e:
        print(f"  ❌ Error en {nombre}: {e}")
        return None


# ── Queries de análisis ───────────────────────────────────────────────────────
QUERIES = {

    # 1. Vista general de tablas del esquema
    "00_tablas": """
        SELECT RDB$RELATION_NAME AS tabla
        FROM RDB$RELATIONS
        WHERE RDB$SYSTEM_FLAG = 0
          AND RDB$VIEW_BLR IS NULL
        ORDER BY RDB$RELATION_NAME
    """,

    # 2. Ventas completas para el modelo (con filtro de precio real)
    "01_ventas_limpias": """
        SELECT
            p.CODIGO,
            p.DESCRIPCION,
            d.FECHA,
            EXTRACT(YEAR  FROM d.FECHA) AS ANIO,
            EXTRACT(MONTH FROM d.FECHA) AS MES,
            EXTRACT(DAY   FROM d.FECHA) AS DIA,
            EXTRACT(WEEKDAY FROM d.FECHA) AS DIA_SEMANA,
            SUM(i.CANTREAL)            AS VENTAS,
            SUM(i.TOTALITEM)           AS TOTAL_VENDIDO,
            AVG(i.VALORU)              AS PRECIO_PROMEDIO,
            COUNT(DISTINCT d.IDDOCINV) AS FRECUENCIA,
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
          AND i.VALORU > 10
        GROUP BY p.CODIGO, p.DESCRIPCION, d.FECHA
        HAVING AVG(i.VALORU) > 10
        ORDER BY d.FECHA
    """,

    # 3. Precios históricos por producto para detectar outliers
    "02_precios_por_producto": """
        SELECT
            p.CODIGO,
            p.DESCRIPCION,
            EXTRACT(YEAR  FROM d.FECHA) AS ANIO,
            EXTRACT(MONTH FROM d.FECHA) AS MES,
            MIN(i.VALORU)  AS precio_min,
            MAX(i.VALORU)  AS precio_max,
            AVG(i.VALORU)  AS precio_promedio,
            COUNT(*)       AS num_transacciones
        FROM DOCINV d
        INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
        INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO
        WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'
          AND i.VALORU > 10
        GROUP BY p.CODIGO, p.DESCRIPCION,
                 EXTRACT(YEAR FROM d.FECHA),
                 EXTRACT(MONTH FROM d.FECHA)
        ORDER BY p.CODIGO, ANIO, MES
    """,

    # 4. Ventas del último mes (marzo 2026) — base para predicción
    "03_ultimo_mes": """
        SELECT
            p.CODIGO,
            p.DESCRIPCION,
            d.FECHA,
            i.VALORU        AS precio_unitario,
            i.CANTREAL      AS cantidad,
            i.TOTALITEM     AS total_item,
            i.TOTALDESCUENTO AS descuento
        FROM DOCINV d
        INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
        INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO
        WHERE d.FECHA BETWEEN '2026-03-01' AND '2026-03-31'
          AND i.VALORU > 10
        ORDER BY p.CODIGO, d.FECHA
    """,

    # 5. Productos con precio muy variable (posibles outliers)
    "04_outliers_precio": """
        SELECT
            p.CODIGO,
            p.DESCRIPCION,
            COUNT(*)            AS registros,
            MIN(i.VALORU)       AS precio_min,
            MAX(i.VALORU)       AS precio_max,
            AVG(i.VALORU)       AS precio_promedio,
            MAX(i.VALORU) / NULLIF(MIN(i.VALORU), 0) AS ratio_max_min
        FROM DOCINV d
        INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
        INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO
        WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'
          AND i.VALORU > 10
        GROUP BY p.CODIGO, p.DESCRIPCION
        HAVING MAX(i.VALORU) / NULLIF(MIN(i.VALORU), 0) > 10
        ORDER BY ratio_max_min DESC
    """,

    # 6. Resumen general del negocio
    "05_resumen_general": """
        SELECT
            EXTRACT(YEAR  FROM d.FECHA) AS ANIO,
            EXTRACT(MONTH FROM d.FECHA) AS MES,
            COUNT(DISTINCT p.CODIGO)    AS productos_distintos,
            COUNT(DISTINCT d.IDDOCINV)  AS num_facturas,
            SUM(i.CANTREAL)             AS unidades_vendidas,
            SUM(i.TOTALITEM)            AS total_vendido
        FROM DOCINV d
        INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
        INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO
        WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'
          AND i.VALORU > 10
        GROUP BY EXTRACT(YEAR FROM d.FECHA), EXTRACT(MONTH FROM d.FECHA)
        ORDER BY ANIO, MES
    """,
}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  EXPLORADOR BD FIREBIRD — Modelo de Ventas")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    con = conectar()

    print(f"\nExportando {len(QUERIES)} análisis a '{OUTPUT_DIR}'...\n")
    for nombre, query in QUERIES.items():
        exportar(con, nombre, query)

    con.close()
    print(f"\n✅ Listo. Sube la carpeta '{OUTPUT_DIR}' completa al chat.")
    print("   (o al menos los archivos que te pida Claude)")
