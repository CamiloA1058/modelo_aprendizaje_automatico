"""Elegibilidad ampliada (experimento, desactivada por defecto).

Produccion solo entrena y predice productos con ventas en el mes actual
(ultimo mes) porque la razon de crecimiento (siguiente / actual) es
indefinida con un mes actual en cero. La variante amplia tambien incluye
productos sin ventas en el mes actual pero con media_3 > 0 (media de los
ultimos 3 meses, incluido el actual), usando media_3 como base de la
decision. Con el mes actual > 0 la base sigue siendo el mes actual, asi que
las filas de produccion no cambian.
"""
import numpy as np


def decision_base(current, media_3):
    """Base de la decision: ventas del mes actual si hubo; si no, media_3."""
    current = np.asarray(current, float)
    return np.where(current > 0, current, np.asarray(media_3, float))


def growth_ratio_base(target, current, media_3):
    """target / base; NaN si la base es NaN o no positiva. Un mes siguiente
    en cero da razon 0 (Reducir)."""
    base = decision_base(current, media_3)
    target = np.asarray(target, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = target / base
    return np.where(np.isnan(base) | (base <= 0), np.nan, ratio)


def _extra(current, media_3):
    return (np.asarray(current, float) <= 0) & (np.asarray(media_3, float) > 0)


def training_mask(current, media_3, target, ampliada):
    """Filas de entrenamiento: target conocido y (mes actual > 0, o, si
    ampliada, mes actual 0 con media_3 > 0)."""
    current = np.asarray(current, float)
    ok = ~np.isnan(np.asarray(target, float))
    base = current > 0
    if ampliada:
        base = base | _extra(current, media_3)
    return ok & base


def prediction_mask(current, media_3, enough_history, ampliada):
    """Filas a predecir en el ultimo mes (ya con el historial minimo)."""
    current = np.asarray(current, float)
    ok = current > 0
    if ampliada:
        ok = ok | _extra(current, media_3)
    return ok & np.asarray(enough_history, bool)


def temporal_train_mask(is_base, train_ratio):
    """Pertenencia a entrenamiento para las filas (en orden de fecha) de un
    producto. Las filas base se parten como en produccion: las primeras
    int(n_base * train_ratio) van a entrenamiento. Una fila extra va a
    entrenamiento si es anterior a la primera fila base de prueba."""
    is_base = np.asarray(is_base, bool)
    n_base = int(is_base.sum())
    cut = int(n_base * train_ratio)
    train = np.zeros(len(is_base), bool)
    pos_base = np.flatnonzero(is_base)
    train[pos_base[:cut]] = True
    first_test = pos_base[cut] if cut < n_base else len(is_base)
    extra = np.flatnonzero(~is_base)
    train[extra[extra < first_test]] = True
    return train
