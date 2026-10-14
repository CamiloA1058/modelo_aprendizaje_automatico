"""Ayudantes puros del experimento de mejoras de la prediccion en pesos:
reconciliacion de arriba hacia abajo, mezcla con media_3 e intervalos.
No modifican el modelo de produccion."""
import math

import numpy as np

BLEND_WEIGHTS = [round(i / 10, 1) for i in range(11)]


def wape(real, pred):
    """Error porcentual absoluto ponderado: sum|real-pred| / sum(real)."""
    real = np.asarray(real, float)
    pred = np.asarray(pred, float)
    total = real.sum()
    return float(np.abs(real - pred).sum() / total) if total else float("nan")


def topdown_reconcile(total_hat, rf_pred):
    """Reparte total_hat entre productos en proporcion a rf_pred."""
    rf = np.asarray(rf_pred, float)
    s = rf.sum()
    if s <= 0:
        return np.full(len(rf), total_hat / len(rf))
    return total_hat * rf / s


def blend(w, rf_pred, base_pred):
    """w * RF + (1 - w) * base."""
    return w * np.asarray(rf_pred, float) + (1 - w) * np.asarray(base_pred, float)


def choose_blend_weight(real, rf_pred, base_pred, weights=None, criterion="mae"):
    """w de la rejilla que minimiza el criterio sobre (real, rf, base); en
    empate, el menor w. criterion: "mae" (error absoluto medio) o "sesgo"
    (|suma pred - suma real|). Debe llamarse solo con datos de validacion de
    entrenamiento, nunca con el mes objetivo."""
    if criterion not in ("mae", "sesgo"):
        raise ValueError(f"criterio desconocido: {criterion}")
    real = np.asarray(real, float)
    best_w, best = None, float("inf")
    for w in (BLEND_WEIGHTS if weights is None else weights):
        pred = blend(w, rf_pred, base_pred)
        if criterion == "mae":
            score = float(np.mean(np.abs(real - pred)))
        else:
            score = abs(float(pred.sum() - real.sum()))
        if score < best - 1e-12:
            best_w, best = w, score
    return best_w


def interval_coverage(real, lo, hi):
    """Fraccion de valores reales dentro de [lo, hi] (extremos incluidos)."""
    real = np.asarray(real, float)
    inside = (real >= np.asarray(lo, float)) & (real <= np.asarray(hi, float))
    return float(inside.mean())


def interval_width(lo, hi):
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


def conformal_quantile(residuals, alpha):
    """Cuantil conformal dividido: el ceil((n+1)(1-alpha))-esimo residuo
    absoluto menor (el maximo si n es chico)."""
    r = np.sort(np.abs(np.asarray(residuals, float)))
    k = min(math.ceil((len(r) + 1) * (1 - alpha)), len(r))
    return float(r[k - 1])


def tree_quantiles(tree_preds, q_lo, q_hi):
    """Cuantiles por columna de una matriz (arboles x productos)."""
    t = np.asarray(tree_preds, float)
    return np.quantile(t, q_lo, axis=0), np.quantile(t, q_hi, axis=0)
