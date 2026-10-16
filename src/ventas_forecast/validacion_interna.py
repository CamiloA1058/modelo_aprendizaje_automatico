"""Particion temporal por producto para la validacion interna.

Toda seleccion del sistema (peso de la mezcla, umbral de la probabilidad de
venta, cuantiles del stock sugerido) se hace con una validacion interna
DENTRO de la particion de entrenamiento: las filas de entrenamiento de cada
producto se ordenan por fecha y las primeras int(n * ratio) quedan en el
entrenamiento interno; el resto, en la validacion interna. Es la misma regla
de SalesForecastModel._build_train_frame (entrenamiento / prueba interna),
aplicada una segunda vez sobre el entrenamiento. La prueba interna y el
holdout nunca participan en una seleccion: solo se reportan.
"""
import numpy as np
import pandas as pd


def particion_temporal(ids, fechas, ratio):
    """Mascara booleana (alineada con ids/fechas) de las filas que van al
    entrenamiento interno: por producto, las primeras int(n * ratio) filas en
    orden de fecha. El resto es la validacion interna."""
    d = pd.DataFrame({"id": np.asarray(ids, dtype=object),
                      "fecha": pd.to_datetime(np.asarray(fechas))})
    d["pos"] = np.arange(len(d))
    mask = np.zeros(len(d), bool)
    for _, g in d.groupby("id", sort=False):
        g = g.sort_values("fecha", kind="stable")
        cut = int(len(g) * ratio)
        mask[g["pos"].to_numpy()[:cut]] = True
    return mask
