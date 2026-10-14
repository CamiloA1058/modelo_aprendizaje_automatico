"""Ayudantes puros del experimento "acierto del monto por producto": objetivo
a varios meses, combinacion en dos pasos (hurdle), regresor Tweedie, correccion
multiplicativa por tamano y metricas de acierto por producto. No modifican el
modelo de produccion."""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

CUBETAS = 10
RECORTE_FACTOR = (0.25, 4.0)


def multi_month_target(df, id_col, total_col, horizon):
    """Suma de las ventas de los `horizon` meses siguientes por producto
    (df con calendario mensual completo y la columna 'fecha'). NaN si la
    ventana no esta completa. Devuelve una Serie con el indice de df."""
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1")
    d = df.sort_values([id_col, "fecha"])
    g = d.groupby(id_col, sort=False)[total_col]
    total = None
    for j in range(1, horizon + 1):
        s = g.shift(-j)
        total = s if total is None else total + s
    return total.reindex(df.index)


def hurdle_combine(proba, cond, threshold=None):
    """Dos pasos: P(venta) * monto condicional; con umbral, 0 si P < umbral y
    el monto condicional si P >= umbral."""
    p = np.asarray(proba, float)
    c = np.asarray(cond, float)
    if threshold is None:
        return p * c
    return np.where(p >= threshold, c, 0.0)


def product_accuracy(real, pred):
    """Acierto por producto 1 - |real - pred| / real recortado a [0, 1];
    NaN donde el real no es positivo."""
    r = np.asarray(real, float)
    p = np.asarray(pred, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.clip(1.0 - np.abs(r - p) / r, 0.0, 1.0)
    return np.where(r > 0, acc, np.nan)


def accuracy_metrics(real, pred):
    """Metricas de monto. Las de acierto por producto (mean_acc, within30,
    within50) usan solo productos con real > 0; acc_wape (1 - WAPE), sum_ratio
    y mae usan todos los productos."""
    r = np.asarray(real, float)
    p = np.asarray(pred, float)
    pos = r > 0
    nan = float("nan")
    out = {"n": int(len(r)), "n_pos": int(pos.sum()), "mean_acc": nan,
           "within30": nan, "within50": nan, "acc_wape": nan,
           "sum_ratio": nan, "mae": nan}
    if len(r):
        out["mae"] = float(np.mean(np.abs(r - p)))
    if pos.any():
        err = np.abs(r[pos] - p[pos]) / r[pos]
        out["mean_acc"] = float(np.nanmean(product_accuracy(r, p)))
        out["within30"] = float(np.mean(err <= 0.30 + 1e-12))
        out["within50"] = float(np.mean(err <= 0.50 + 1e-12))
    if r.sum() > 0:
        out["acc_wape"] = float(1.0 - np.abs(r - p).sum() / r.sum())
        out["sum_ratio"] = float(p.sum() / r.sum())
    return out


def sells_accuracy(real, pred):
    """Exactitud de "vende o no": (real > 0) == (pred > 0)."""
    return float(np.mean((np.asarray(real, float) > 0) == (np.asarray(pred, float) > 0)))


def segment_masks(window, top_n=100):
    """Segmentos de una ventana de ventas reales (filas = productos, columnas
    = meses): "top" (top_n por ventas totales), "todos_los_meses" (venden en
    todos los meses de la ventana) y "resto" (ninguno de los dos)."""
    w = np.asarray(window, float)
    if w.ndim == 1:
        w = w[:, None]
    total = w.sum(axis=1)
    top = np.zeros(len(w), bool)
    top[np.argsort(-total, kind="stable")[:top_n]] = True
    top &= total > 0
    todos = (w > 0).all(axis=1)
    return {"top": top, "todos_los_meses": todos, "resto": ~(top | todos)}


def fit_tweedie(X, y, power, n_estimators=300, max_depth=4, learning_rate=0.05,
                random_state=42):
    """XGBoost con objetivo reg:tweedie sobre pesos crudos (admite ceros)."""
    m = XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=power,
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=-1)
    m.fit(np.asarray(X, float), np.asarray(y, float))
    return m


def predict_nonneg(model, X):
    p = np.asarray(model.predict(np.asarray(X, float)), float)
    return np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def choose_tweedie(Xtr, ytr, Xva, yva, powers=(1.2, 1.5, 1.8), **kw):
    """Ajusta un Tweedie por potencia de varianza con entrenamiento y elige la
    de menor MAE en la validacion interna. Devuelve (potencia, modelo,
    {potencia: mae})."""
    maes, modelos = {}, {}
    yva = np.asarray(yva, float)
    for pw in powers:
        modelos[pw] = fit_tweedie(Xtr, ytr, pw, **kw)
        maes[pw] = float(np.mean(np.abs(yva - predict_nonneg(modelos[pw], Xva))))
    mejor = min(maes, key=maes.get)
    return mejor, modelos[mejor], maes


def _mejor_factor_acierto(p, r, clip, pasos=60):
    """Factor c de una rejilla geometrica dentro de `clip` que maximiza el
    acierto por producto recortado, media de clip(1 - |r - c*p| / r, 0, 1),
    sobre las filas con r > 0 y p > 0. Empates: el c mas cercano a 1."""
    grilla = np.geomspace(clip[0], clip[1], pasos)
    r_, p_ = r[:, None], p[:, None]
    acc = np.clip(1.0 - np.abs(r_ - grilla[None, :] * p_) / r_, 0.0, 1.0).mean(axis=0)
    mejores = grilla[acc >= acc.max() - 1e-12]
    return float(mejores[np.argmin(np.abs(np.log(mejores)))])


def fit_bucket_correction(pred, real, n_buckets=CUBETAS, mode="acc",
                          clip=RECORTE_FACTOR, min_n=5):
    """Factor multiplicativo por cubeta de monto predicho (cuantiles de pred),
    aprendido con datos de validacion interna.
    mode "sum": suma(real) / suma(pred) de la cubeta (corrige el total).
    mode "acc": factor que maximiza el acierto por producto recortado a [0, 1]
    (la metrica del experimento) sobre los productos con real > 0 y pred > 0.
    Cubetas con menos de min_n filas (o sin dato) quedan en 1. Devuelve
    {"edges", "factors"}."""
    if mode not in ("sum", "acc"):
        raise ValueError(f"modo desconocido: {mode}")
    p = np.asarray(pred, float)
    r = np.asarray(real, float)
    edges = np.unique(np.quantile(p, np.arange(1, n_buckets) / n_buckets))
    b = np.searchsorted(edges, p, side="right")
    factors = np.ones(len(edges) + 1)
    for k in range(len(factors)):
        m = b == k
        if m.sum() < min_n:
            continue
        if mode == "sum":
            if p[m].sum() > 0:
                f = r[m].sum() / p[m].sum()
            else:
                continue
        else:
            mm = m & (r > 0) & (p > 0)
            if not mm.any():
                continue
            f = _mejor_factor_acierto(p[mm], r[mm], clip)
        factors[k] = float(np.clip(f, clip[0], clip[1]))
    return {"edges": edges, "factors": factors}


def apply_bucket_correction(pred, corr):
    p = np.asarray(pred, float)
    b = np.searchsorted(corr["edges"], p, side="right")
    return p * corr["factors"][b]
