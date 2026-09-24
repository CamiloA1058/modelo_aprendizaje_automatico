"""
eda.py
======
Funciones puras de análisis exploratorio de datos (EDA), usadas por
scripts/eda.py para la fase 2 (comprensión de datos) y fase 3 (preparación
de datos) de CRISP-DM.

Todas las funciones son deliberadamente pequeñas y sin efectos secundarios
(no imprimen, no grafican, no leen archivos): reciben un DataFrame ya en
memoria y devuelven datos listos para imprimir/graficar. Así se pueden
probar de forma aislada (ver tests/test_eda.py) sin necesidad del CSV real
ni del pipeline completo de SalesForecastModel.
"""

import pandas as pd


# ── 1. CALIDAD DE DATOS (crudo, antes de cualquier agregación) ──────────────
def data_quality_summary(raw_df, col_map):
    """
    Resumen de calidad sobre el DataFrame CRUDO (antes de agregar a
    producto-mes ni de completar el calendario).

    col_map : dict con roles de columna, al menos {"id": ..., "year": ...,
        "month": ...}. Opcionalmente "date" (columna con la fecha completa
        del registro, formato dd/mm/aaaa) para un rango de fechas exacto;
        si no se da, el rango se aproxima con el primer día del mes de
        (year, month) mínimo y máximo.

    Devuelve un dict:
      n_rows, n_cols            : dimensiones del DataFrame.
      nulls                     : dict {columna: nulos}.
      n_duplicates              : filas EXACTAMENTE duplicadas (todas las
                                   columnas iguales; pandas.duplicated()
                                   solo marca la segunda ocurrencia en
                                   adelante, así que esto es "duplicados de
                                   más", no "registros involucrados").
      n_null_id                 : filas con id de producto nulo.
      n_products                : productos distintos (sin contar el id nulo).
      date_min, date_max        : pd.Timestamp del rango de fechas.
      n_distinct_months         : combinaciones (year, month) distintas.
    """
    id_col = col_map["id"]
    year_col = col_map["year"]
    month_col = col_map["month"]
    date_col = col_map.get("date")

    nulls = {col: int(n) for col, n in raw_df.isna().sum().items()}
    n_duplicates = int(raw_df.duplicated().sum())
    n_null_id = int(raw_df[id_col].isna().sum())
    n_products = int(raw_df[id_col].dropna().nunique())

    if date_col and date_col in raw_df.columns:
        fechas = pd.to_datetime(raw_df[date_col], format="%d/%m/%Y")
        date_min, date_max = fechas.min(), fechas.max()
    else:
        combo = raw_df[year_col].astype(int) * 100 + raw_df[month_col].astype(int)
        ym_min, ym_max = int(combo.min()), int(combo.max())
        date_min = pd.Timestamp(year=ym_min // 100, month=ym_min % 100, day=1)
        date_max = pd.Timestamp(year=ym_max // 100, month=ym_max % 100, day=1)

    n_distinct_months = int(
        raw_df[[year_col, month_col]].drop_duplicates().shape[0]
    )

    return {
        "n_rows": int(len(raw_df)),
        "n_cols": int(raw_df.shape[1]),
        "nulls": nulls,
        "n_duplicates": n_duplicates,
        "n_null_id": n_null_id,
        "n_products": n_products,
        "date_min": date_min,
        "date_max": date_max,
        "n_distinct_months": n_distinct_months,
    }


# ── 2. TOTALES MENSUALES Y ESTACIONALIDAD ───────────────────────────────────
def monthly_totals(df, date_col, value_col):
    """
    Total de value_col por mes calendario (suma de TODOS los productos).

    Devuelve una Serie indexada por fecha (inicio de mes), ordenada
    cronológicamente. Solo incluye meses con al menos una fila en df (no
    rellena huecos con cero): con el calendario ya completo de
    SalesForecastModel (una fila por producto-mes desde el primer producto
    activo hasta el último mes global), esto no introduce huecos reales.
    """
    return (
        df.groupby(date_col)[value_col]
        .sum()
        .sort_index()
    )


def seasonal_profile(monthly):
    """
    A partir de una Serie mensual (índice = fecha, ver monthly_totals),
    calcula el promedio de venta por número de mes calendario (1-12) y
    cuántos años distintos aportaron una observación a ese mes — con solo
    ~2 años de historial esta estimación es débil y ese conteo se muestra
    junto al promedio para que quede visible (ver scripts/eda.py).

    Devuelve un DataFrame indexado 1-12 (los 12 meses SIEMPRE presentes,
    aunque falten datos) con columnas "promedio" y "n_anios".
    """
    tmp = pd.DataFrame({
        "mes": monthly.index.month,
        "anio": monthly.index.year,
        "valor": monthly.to_numpy(),
    })
    resumen = tmp.groupby("mes").agg(
        promedio=("valor", "mean"),
        n_anios=("anio", "nunique"),
    )
    resumen = resumen.reindex(range(1, 13))
    resumen["n_anios"] = resumen["n_anios"].fillna(0).astype(int)
    return resumen


# ── 3. CLASIFICACIÓN ABC ─────────────────────────────────────────────────────
def abc_classification(df, id_col, value_col, a=0.80, b=0.95):
    """
    Clasificación ABC por producto: total de value_col por id_col, ordenado
    descendente, con participación acumulada y clase A/B/C.

    Regla de frontera (DECIDIDA para este módulo, ver tests/test_eda.py):
    la clase de cada producto se define por su participación ACUMULADA
    incluyéndolo a él mismo — A si acumulado <= a, B si acumulado <= b, C en
    otro caso. Un producto que por sí solo hace que el acumulado SUPERE el
    corte (por ejemplo, uno que concentra el 82 % de las ventas) queda
    clasificado en la clase siguiente (B), no en A: la regla describe el
    CORTE acumulado de participación, no la importancia individual del
    producto.

    Devuelve (tabla, resumen):
      tabla   : DataFrame [id_col, "total", "share", "cum_share", "clase"],
                ordenado descendente por "total".
      resumen : dict {"A": {"n_products": int, "pct_sales": float}, "B": ...,
                "C": ...} — número de productos y % de ventas de cada clase.
    """
    totales = (
        df.groupby(id_col)[value_col].sum()
        .sort_values(ascending=False)
        .rename("total")
        .reset_index()
    )

    total_general = totales["total"].sum()
    totales["share"] = totales["total"] / total_general
    totales["cum_share"] = totales["share"].cumsum()

    def _clase(cum_share):
        if cum_share <= a:
            return "A"
        if cum_share <= b:
            return "B"
        return "C"

    totales["clase"] = totales["cum_share"].apply(_clase)

    resumen = {}
    for clase in ("A", "B", "C"):
        subset = totales[totales["clase"] == clase]
        resumen[clase] = {
            "n_products": int(len(subset)),
            "pct_sales": float(subset["total"].sum() / total_general * 100),
        }

    return totales, resumen


# ── 4. INTERMITENCIA ──────────────────────────────────────────────────────────
def intermittency_summary(df, id_col, value_col):
    """
    Sobre un DataFrame producto-mes YA con calendario completo (ver
    SalesForecastModel._complete_calendar), calcula la fracción de filas
    con venta cero y la distribución (por producto) de esa fracción.

    Devuelve un dict:
      share_zero_overall     : fracción global de filas producto-mes con
                                value_col == 0.
      share_zero_per_product : Serie indexada por id_col con la fracción de
                                meses en cero de CADA producto.
      median, q1, q3          : mediana y cuartiles de esa distribución.
    """
    es_cero = (df[value_col] == 0).astype(float)
    share_zero_overall = float(es_cero.mean())

    share_zero_per_product = (
        df.assign(_cero=es_cero).groupby(id_col)["_cero"].mean()
    )

    return {
        "share_zero_overall": share_zero_overall,
        "share_zero_per_product": share_zero_per_product,
        "median": float(share_zero_per_product.median()),
        "q1": float(share_zero_per_product.quantile(0.25)),
        "q3": float(share_zero_per_product.quantile(0.75)),
    }
