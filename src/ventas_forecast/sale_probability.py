"""Ayudantes puros del experimento "probabilidad de venta": predecir, por
producto, si habra ventas el mes siguiente (si/no).

Todas las variables se calculan por producto sobre el calendario mensual
completo (una fila por mes, ver SalesForecastModel._complete_calendar) y la
fila del mes t solo usa informacion hasta t (sin mirar al futuro). Las
ventanas "ultimos k meses" incluyen el mes t: son los k meses previos al mes
que se predice (t+1).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    roc_auc_score,
)

# Variables nuevas (ademas de las de SalesForecastModel.prepare_split)
NEW_FEATURES = [
    "meses_desde_venta", "ventas_3m", "ventas_6m", "ventas_12m", "lag_12",
    "frec_lag_1", "frec_lag_2", "frec_lag_3", "frec_mean_3",
    "edad_meses", "frac_hist",
]
VENTANA_ENTRENAMIENTO = 6  # meses (t-5..t) con al menos una venta


def add_probability_features(df, id_col, total_col, freq_col):
    """Copia ordenada (id, fecha) de df con NEW_FEATURES y el mes ciclico del
    mes que se predice (mes_sig_sin / mes_sig_cos). df debe tener calendario
    completo por producto y la columna 'fecha'."""
    d = df.sort_values([id_col, "fecha"]).copy()
    g = d.groupby(id_col, sort=False)
    vende = (d[total_col] > 0).astype(float)
    d["_vende"] = vende
    gv = d.groupby(id_col, sort=False)["_vende"]

    pos = g.cumcount()
    d["edad_meses"] = pos
    ultima = pos.where(vende > 0)
    ultima = ultima.groupby(d[id_col], sort=False).ffill()
    d["meses_desde_venta"] = (pos - ultima).fillna(pos + 1).astype(int)

    for k, nombre in ((3, "ventas_3m"), (6, "ventas_6m"), (12, "ventas_12m")):
        d[nombre] = gv.transform(
            lambda x, k=k: x.rolling(k, min_periods=1).sum()).astype(int)

    d["lag_12"] = g[total_col].shift(12)
    for lag in (1, 2, 3):
        d[f"frec_lag_{lag}"] = g[freq_col].shift(lag)
    d["frec_mean_3"] = d[["frec_lag_1", "frec_lag_2", "frec_lag_3"]].mean(axis=1, skipna=False)

    d["frac_hist"] = gv.cumsum() / (pos + 1)

    m = (d["fecha"].dt.month % 12) + 1
    d["mes_sig_sin"] = np.sin(2 * np.pi * m / 12)
    d["mes_sig_cos"] = np.cos(2 * np.pi * m / 12)
    return d.drop(columns="_vende")


def sale_train_frame(df, id_col, total_col, train_ratio=0.8,
                     window=VENTANA_ENTRENAMIENTO):
    """Marco de entrenamiento del objetivo binario vende = target > 0.

    Incluye las filas con target conocido y al menos una venta en los ultimos
    `window` meses (mes actual incluido), de modo que entran tambien las de
    mes actual en cero si el producto vendio poco antes. df debe traer
    'target' (ventas del mes siguiente) y 'ventas_6m' (add_probability_features;
    si window != 6 se recalcula desde total_col). Particion temporal por
    producto: las primeras train_ratio filas (por fecha) van a entrenamiento,
    el resto a validacion interna, igual que _build_train_frame.

    Devuelve (frame, train_idx, test_idx) con indices posicionales 0..n-1 del
    frame (reset_index)."""
    d = df.sort_values([id_col, "fecha"])
    if window == 6 and "ventas_6m" in d.columns:
        recientes = d["ventas_6m"] > 0
    else:
        v = (d[total_col] > 0).astype(float)
        recientes = v.groupby(d[id_col], sort=False).transform(
            lambda x: x.rolling(window, min_periods=1).sum()) > 0
    frame = d[d["target"].notna() & recientes].copy().reset_index(drop=True)
    frame["vende"] = (frame["target"] > 0).astype(int)

    train_idx, test_idx = [], []
    for _, g in frame.groupby(id_col):
        g = g.sort_values("fecha")
        cut = int(len(g) * train_ratio)
        train_idx += g.index[:cut].tolist()
        test_idx += g.index[cut:].tolist()
    return frame, train_idx, test_idx


def choose_threshold(y_true, proba, grid=None):
    """Umbral de probabilidad que maximiza la exactitud sobre (y_true, proba),
    que deben ser de la validacion interna del entrenamiento (nunca del mes
    evaluado). Empates: el umbral mas cercano a 0,5."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, float)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else np.asarray(grid)
    acc = np.array([((p >= t).astype(int) == y).mean() for t in grid])
    mejores = grid[acc >= acc.max() - 1e-12]
    return float(mejores[np.argmin(np.abs(mejores - 0.5))])


def classification_metrics(y_true, y_pred, proba=None):
    """Exactitud, exactitud balanceada, precision/recall de 'vende' y de 'no
    vende' y AUC (NaN sin probabilidad o con una sola clase real)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    auc = float("nan")
    if proba is not None and len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, proba))
    return {
        "n": int(len(y)),
        "tasa_vende": float(y.mean()) if len(y) else float("nan"),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision_vende": float(precision_score(y, p, pos_label=1, zero_division=0)),
        "recall_vende": float(recall_score(y, p, pos_label=1, zero_division=0)),
        "precision_no": float(precision_score(y, p, pos_label=0, zero_division=0)),
        "recall_no": float(recall_score(y, p, pos_label=0, zero_division=0)),
        "auc": auc,
    }


def calibration_table(y_true, proba, n_bins=10):
    """Calibracion por deciles de probabilidad (cuantiles): n, probabilidad
    media predicha y tasa real de venta de cada decil."""
    d = pd.DataFrame({"y": np.asarray(y_true).astype(int),
                      "p": np.asarray(proba, float)})
    d["decil"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False) + 1
    t = d.groupby("decil").agg(n=("y", "size"), prob_media=("p", "mean"),
                               tasa_real=("y", "mean")).reset_index()
    return t


def rule_predictions(frame, total_col):
    """Reglas simples sin entrenamiento (booleanos 'vendera el mes siguiente')."""
    return {
        "vendio_mes_anterior": (frame[total_col] > 0).to_numpy(),
        "2_de_3_meses": (frame["ventas_3m"] >= 2).to_numpy(),
        "fraccion_historica": (frame["frac_hist"] > 0.5).to_numpy(),
    }
