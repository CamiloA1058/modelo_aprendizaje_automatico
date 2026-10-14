"""Ayudantes puros del estudio de caso fuera de muestra (origen rodante).

El estudio entrena el procedimiento de produccion con datos hasta el mes M-1
y compara su prediccion con las ventas reales de M. Aqui viven el etiquetado
de la clase real, las metricas (mismas que evaluate_model.py) y la lectura y
combinacion de los CSV con el formato de Query_Result_V5/V6.
"""
import csv

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
)

CLASS_LABELS = [0, 1, 2]  # 0 Reducir, 1 Mantener, 2 Reforzar


def _read(source):
    kwargs = {"sep": ";", "dtype": str, "keep_default_na": False}
    if hasattr(source, "read"):
        return pd.read_csv(source, **kwargs)
    return pd.read_csv(source, encoding="utf-8-sig", **kwargs)


def _parse_es(series):
    return (
        series.str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False).astype(float)
    )


def _month_key(fechas):
    d = pd.to_datetime(fechas, format="%d/%m/%Y")
    return d.dt.strftime("%Y-%m")


def real_class(next_sales, current_sales, growth, reduce):
    """Clase real con la misma regla que las etiquetas de entrenamiento
    (prepare_split): razon = ventas del mes siguiente / ventas del mes actual;
    > growth -> 2 (Reforzar), < reduce -> 0 (Reducir), en otro caso 1.
    Un mes siguiente sin ventas da razon 0, es decir Reducir."""
    ratio = np.asarray(next_sales, float) / np.asarray(current_sales, float)
    return np.where(ratio > growth, 2, np.where(ratio < reduce, 0, 1))


def classification_summary(y_true, y_pred):
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=CLASS_LABELS,
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
        "n": int(len(y_true)),
    }


def majority_baseline_summary(y_true, majority_class=0):
    """Linea base: siempre predice la clase mayoritaria del entrenamiento."""
    y_true = list(y_true)
    return classification_summary(y_true, [majority_class] * len(y_true))


def regression_summary(real, pred, scale):
    real = np.asarray(real, float)
    pred = np.asarray(pred, float)
    mae = float(np.mean(np.abs(real - pred)))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean((real - pred) ** 2))),
        "mase": mae / scale if scale else float("nan"),
        "sum_ratio": float(pred.sum() / real.sum()) if real.sum() else float("nan"),
        "n": int(len(real)),
    }


def naive_scale(df_train, id_col, total_col, top_ids):
    """Escala de MASE: error absoluto medio del pronostico ingenuo (mes actual
    predice el siguiente) sobre las filas de entrenamiento de los top
    productos, igual que train_regressor."""
    d = df_train[df_train[id_col].isin(top_ids)]
    return float((d["target"] - d[total_col]).abs().mean())


def pooled_scale(scales_and_counts):
    """Escala agrupada: promedio de las escalas mensuales ponderado por filas."""
    total = sum(n for _, n in scales_and_counts)
    return sum(s * n for s, n in scales_and_counts) / total


def pool(frames, true_col, pred_col):
    df = pd.concat(frames, ignore_index=True)
    return df[true_col].to_numpy(), df[pred_col].to_numpy()


def monthly_sales(source, month):
    """Ventas (TOTAL_VENDIDO) por codigo de producto en el mes 'AAAA-MM'."""
    df = _read(source)
    df = df[_month_key(df["FECHA"]) == month]
    df = df.assign(TOTAL_VENDIDO=_parse_es(df["TOTAL_VENDIDO"]))
    return df.groupby("CODIGO")["TOTAL_VENDIDO"].sum()


def months_in(source):
    df = _read(source)
    return sorted(_month_key(df["FECHA"]).unique())


def combine_training(base, holdout, before_month, out):
    """Escribe en 'out' (texto) el CSV base mas las filas del holdout con fecha
    estrictamente anterior al mes 'AAAA-MM'. Devuelve las filas de datos."""
    b = _read(base)
    h = _read(holdout)
    h = h[pd.to_datetime(h["FECHA"], format="%d/%m/%Y") < pd.Timestamp(before_month + "-01")]
    allrows = pd.concat([b, h], ignore_index=True)
    writer = csv.writer(out, delimiter=";", lineterminator="\r\n")
    writer.writerow(allrows.columns)
    writer.writerows(allrows.itertuples(index=False, name=None))
    return len(allrows)
