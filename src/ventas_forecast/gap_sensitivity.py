"""Ayudantes puros del experimento de sensibilidad al tramo de baja facturacion.

Entre 2024-02 y 2025-04 las facturas de venta caen de ~1.500 a ~450 por mes
(tambien en la cuenta contable de ingresos) y se recuperan en 2025-05. Estos
ayudantes permiten probar, solo en el experimento, dos tratamientos del tramo:
reescalar las ventas existentes (datos ajustados, imputacion experimental) o
darles menos peso al entrenar. No modifican los datos oficiales ni el modelo
de produccion.
"""
import numpy as np
import pandas as pd

GAP = ("2024-02", "2025-04")
REF = ("2025-05", "2025-10")
MAX_FACTOR = 5.0
MIN_ACTIVE_MONTHS = 3


def _n_months(window):
    return len(pd.period_range(window[0], window[1], freq="M"))


def gap_scale_factors(monthly, gap=GAP, ref=REF, cap=MAX_FACTOR,
                      min_active=MIN_ACTIVE_MONTHS):
    """Factores de reescalado por producto.

    monthly : DataFrame largo con columnas id, month ('AAAA-MM') y amount
              (ventas del producto en ese mes; solo meses con ventas).
    Factor del producto = (ventas medias por mes en la ventana de referencia)
    / (ventas medias por mes en el tramo), calculadas sobre los meses
    calendario de cada ventana (los meses sin venta cuentan como cero), y
    acotado a [1, cap]. Solo se usa si el producto tiene al menos min_active
    meses con ventas en ambas ventanas; si no, se usa el factor global:
    razon de las ventas totales mensuales medias fuera / dentro del tramo,
    acotada igual. Nunca baja ventas (factor >= 1).

    Devuelve (DataFrame indexado por id con 'factor' y 'source', factor global,
    resumen)."""
    n_gap, n_ref = _n_months(gap), _n_months(ref)
    in_gap = monthly["month"].between(*gap)
    in_ref = monthly["month"].between(*ref)

    tot_gap = monthly.loc[in_gap, "amount"].sum() / n_gap
    tot_ref = monthly.loc[in_ref, "amount"].sum() / n_ref
    g = float(np.clip(tot_ref / tot_gap, 1.0, cap)) if tot_gap > 0 else 1.0

    def agg(mask):
        d = monthly[mask & (monthly["amount"] > 0)]
        return d.groupby("id")["amount"].agg(["sum", "count"])

    a_gap, a_ref = agg(in_gap), agg(in_ref)
    ids = monthly["id"].drop_duplicates().to_numpy()
    out = pd.DataFrame(index=pd.Index(ids, name="id"))
    out["factor"] = g
    out["source"] = "global"
    both = a_gap.join(a_ref, how="inner", lsuffix="_gap", rsuffix="_ref")
    ok = both[(both["count_gap"] >= min_active) & (both["count_ref"] >= min_active)]
    ratio = (ok["sum_ref"] / n_ref) / (ok["sum_gap"] / n_gap)
    out.loc[ratio.index, "factor"] = ratio.clip(1.0, cap).to_numpy()
    out.loc[ratio.index, "source"] = "producto"

    summary = {
        "n_gap_months": n_gap, "n_ref_months": n_ref, "n_products": len(out),
        "n_by_product": int((out["source"] == "producto").sum()),
        "n_by_global": int((out["source"] == "global").sum()),
        "global_factor": g, "cap": cap, "min_active": min_active,
    }
    return out, g, summary


def scale_gap_rows(df, factors, global_factor, id_col, date_col, gap, scale_cols):
    """Copia de df con las columnas scale_cols multiplicadas por el factor del
    producto en las filas cuya fecha cae en el tramo. No crea ni borra filas;
    un producto sin factor usa el global. Devuelve (copia, mascara de filas
    reescaladas)."""
    out = df.copy()
    month = pd.to_datetime(out[date_col]).dt.strftime("%Y-%m")
    mask = month.between(*gap).to_numpy()
    f = (out[id_col].map(factors["factor"]).fillna(global_factor)
         .to_numpy(dtype=float))
    for col in scale_cols:
        out[col] = np.where(mask, out[col].to_numpy(dtype=float) * f,
                            out[col].to_numpy(dtype=float))
    return out, mask


def gap_sample_weights(dates, gap, weight):
    """Vector de pesos de muestra: 'weight' para las filas de entrenamiento
    cuyo mes actual o cuyo mes objetivo (el siguiente) cae en el tramo; 1 para
    las demas. 'dates' son los meses actuales de las filas."""
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    cur = d.dt.strftime("%Y-%m")
    nxt = (d + pd.DateOffset(months=1)).dt.strftime("%Y-%m")
    touches = cur.between(*gap) | nxt.between(*gap)
    return np.where(touches.to_numpy(), float(weight), 1.0)


def format_es(values):
    """Serie de numeros -> texto con el formato del CSV (miles '.', decimal ',',
    dos decimales)."""
    return pd.Series(values).map(
        lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))


def monthly_long(df, id_col, date_col, amount_col):
    """DataFrame largo (id, month, amount) con la suma mensual por producto."""
    month = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m")
    g = (df.assign(_m=month).groupby([id_col, "_m"])[amount_col].sum()
         .reset_index())
    g.columns = ["id", "month", "amount"]
    return g


def class_share(classes, cls):
    """Fraccion de elementos iguales a cls (NaN si no hay elementos)."""
    c = np.asarray(classes)
    return float((c == cls).mean()) if len(c) else float("nan")


def label_counts(df_train, months):
    """Conteo [Reducir, Mantener, Reforzar] de 'target_class' por mes de origen
    ('fecha') indicado."""
    ym = pd.to_datetime(df_train["fecha"]).dt.strftime("%Y-%m")
    out = {}
    for m in months:
        vc = df_train.loc[ym == m, "target_class"].value_counts()
        out[m] = [int(vc.get(k, 0)) for k in (0, 1, 2)]
    return out
