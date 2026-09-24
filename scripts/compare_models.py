"""
compare_models.py
==================
Comparación justa de modelos sobre EXACTAMENTE los mismos datos, partición y
filas de prueba que el modelo de producción (SalesForecastModel):

  1. Clasificador: Random Forest (modelo de producción) vs XGBoost (con
     ajuste de hiperparámetros, mismos pliegues mensuales que
     tune_classifier()) vs línea base de clase mayoritaria.
  2. Regresión A (conjunto de prueba completo, top productos): Random
     Forest vs XGBoost vs ingenuo vs media_3.
  3. Regresión B (muestra semillada de productos): los mismos cuatro
     modelos anteriores + Prophet y ARIMA(1,1,1), con pronóstico a un paso
     re-entrenado mes a mes (ver src/ventas_forecast/benchmarks.py).

Los scripts legacy scripts/run_xgboost.py y scripts/run_prophet.py corrían
sobre datasets antiguos (no comparables con Query_Result_V5.csv) y quedan
SUPERADOS por este script para la comparación de la tesis.

Genera:
  outputs/reports/comparacion_modelos.txt
  outputs/reports/comparacion_modelos.csv
  outputs/figures/comparacion_modelos.png

Uso
---
    PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/compare_models.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.paths import DATA_RAW, FIGURES, REPORTS, ensure_output_dirs  # noqa: E402
from ventas_forecast.benchmarks import (  # noqa: E402
    arima_fit_predict,
    build_eval_frame,
    prophet_fit_predict,
    rolling_one_step_forecast,
    sample_products_with_test_rows,
)

from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402

CLASS_NAMES = ["Reducir", "Mantener", "Reforzar"]
N_SAMPLE_PRODUCTS = 150
RANDOM_STATE = 42

GRID_XGB_CLF = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}


# ── Clasificación ─────────────────────────────────────────────────────────
def _metricas_clasificacion(y_true, y_pred):
    """Mismas métricas para todos los modelos comparados: F1 macro y
    accuracy balanceada (principales, ver CHANGELOG 2026-09-23 "métrica
    macro"), accuracy y F1 ponderado (referencia), y F1 por clase."""
    f1_por_clase = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_reducir": round(float(f1_por_clase[0]), 4),
        "f1_mantener": round(float(f1_por_clase[1]), 4),
        "f1_reforzar": round(float(f1_por_clase[2]), 4),
    }


def compare_classification(model):
    """
    RF (ya entrenado por model.train_classifier(), self.clf) vs XGBoost
    (ajustado con GridSearchCV sobre los MISMOS pliegues mensuales que
    tune_classifier(), ver _monthly_time_series_folds) vs línea base de
    clase mayoritaria, sobre las MISMAS filas de entrenamiento/prueba y las
    MISMAS features SIN escalar (model._X — a XGBoost, como a RF, no le
    afecta la escala de las features, pero se usa la versión sin escalar
    para que el pipeline de comparación no dependa del escalador ajustado
    para RF).
    """
    print("\n===== COMPARACIÓN — CLASIFICADOR =====")
    inicio = time.time()

    X_tr = model._X.loc[model._train_idx]
    y_tr = model._y.loc[model._train_idx]
    X_te = model._X.loc[model._test_idx]
    y_te = model._y.loc[model._test_idx]
    fechas_tr = model._df_train.loc[model._train_idx, "fecha"]

    resultados = {}

    # ── Random Forest (modelo de producción) ─────────────────────────────
    pred_rf = model.clf.predict(model._X_scaled[model._test_idx])
    resultados["Random Forest"] = _metricas_clasificacion(y_te, pred_rf)

    # ── Línea base: clase mayoritaria de entrenamiento ────────────────────
    clase_mayoritaria = y_tr.value_counts().idxmax()
    pred_base = np.full(len(y_te), clase_mayoritaria)
    resultados["Clase mayoritaria"] = _metricas_clasificacion(y_te, pred_base)

    # ── XGBoost, con el MISMO peso balanceado que class_weight="balanced" ──
    pesos_tr = compute_sample_weight("balanced", y_tr)
    pliegues = model._monthly_time_series_folds(fechas_tr, n_splits=5)

    xgb_base = XGBClassifier(
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    grid = GridSearchCV(
        xgb_base, param_grid=GRID_XGB_CLF, cv=pliegues, scoring="f1_macro",
        n_jobs=1, refit=False,
    )
    grid.fit(X_tr, y_tr, sample_weight=pesos_tr)
    mejores_params_xgb = grid.best_params_

    xgb_clf = XGBClassifier(
        **mejores_params_xgb, random_state=42, n_jobs=-1, tree_method="hist",
    )
    xgb_clf.fit(X_tr, y_tr, sample_weight=pesos_tr)
    pred_xgb = xgb_clf.predict(X_te)
    resultados["XGBoost"] = _metricas_clasificacion(y_te, pred_xgb)

    elapsed = time.time() - inicio
    print(f"[✓] Comparación de clasificadores completada ({elapsed:.1f} s)")
    print(f"     Mejores hiperparámetros XGBoost: {mejores_params_xgb}")
    print(f"     n_test = {len(y_te):,} (idéntico para los tres modelos)")

    return resultados, mejores_params_xgb, len(y_te), elapsed


# ── Regresión A: conjunto de prueba completo ────────────────────────────
def compare_regression_full(model):
    """
    RF (ya entrenado por model.train_regressor(), self.reg_metrics/
    self.baseline_metrics) vs XGBoost vs ingenuo vs media_3, sobre EXACTAMENTE
    las mismas filas de prueba de los top productos que usa train_regressor()
    (mismo filtro es_train/es_test/es_top/tiene_target).

    XGBoost es un modelo de árboles: es invariante a transformaciones
    monótonas por feature, así que entrenarlo con las features SIN escalar
    (model._X) da el mismo resultado que con las escaladas — a diferencia
    de RF, que en train_regressor() sí se entrena sobre model.scaler.
    transform(...) simplemente porque ese es el pipeline ya definido, no
    porque el escalado le aporte algo. Sin ajuste de hiperparámetros para
    XGBoost en esta sección (se declara explícitamente): los valores usados
    (n_estimators=400, max_depth=6, learning_rate=0.05) son los mismos que
    en la Regresión B, para que ambas tablas usen el mismo XGBoost.
    """
    print("\n===== COMPARACIÓN — REGRESIÓN (conjunto de prueba completo) =====")
    inicio = time.time()

    c = model._c
    es_train = model._df_train.index.isin(model._train_idx)
    es_test = model._df_train.index.isin(model._test_idx)
    es_top = model._df_train[c("id")].isin(model._top_prods)
    tiene_target = model._df_train["target"].notna()

    df_reg_train = model._df_train[es_train & es_top & tiene_target]
    df_reg_test = model._df_train[es_test & es_top & tiene_target]

    X_reg_train = model._X.loc[df_reg_train.index]
    X_reg_test = model._X.loc[df_reg_test.index]
    y_reg_train = np.log1p(df_reg_train["target"])
    y_true = df_reg_test["target"].to_numpy()

    xgb_reg = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    xgb_reg.fit(X_reg_train, y_reg_train)
    pred_xgb = np.expm1(xgb_reg.predict(X_reg_test))

    # Misma escala de MASE que train_regressor(): MAE del pronóstico
    # ingenuo sobre la partición de ENTRENAMIENTO de los top productos.
    naive_err_train = (df_reg_train["target"] - df_reg_train[c("total_sold")]).abs()
    escala = naive_err_train.mean()

    resultados = {
        "Random Forest": dict(model.reg_metrics),
        "XGBoost": model._regression_metrics(y_true, pred_xgb, escala),
        "Ingenuo": dict(model.baseline_metrics["ingenuo"]),
        "Media_3": dict(model.baseline_metrics["media_3"]),
    }

    elapsed = time.time() - inicio
    print(f"[✓] Comparación de regresión (prueba completa) completada ({elapsed:.1f} s)")
    print(f"     n_test = {resultados['Random Forest']['n_test']:,} (idéntico en las 4 filas)")

    return resultados, xgb_reg, elapsed


# ── Regresión B: muestra de productos (+ Prophet/ARIMA) ────────────────────
def compare_regression_sample(model, xgb_reg, sample_size=N_SAMPLE_PRODUCTS, random_state=RANDOM_STATE):
    """
    Los mismos cuatro modelos de compare_regression_full() (reutilizando el
    XGBoost ya entrenado allí) + Prophet y ARIMA(1,1,1), evaluados sobre una
    muestra semillada de productos (con al menos 1 fila de prueba) tomada
    de model._top_prods, EXACTAMENTE en las mismas filas para los 6 modelos
    (ver build_eval_frame/sample_products_with_test_rows).

    Prophet/ARIMA no pueden usar model._X (son modelos de serie de tiempo
    univariados, no de features tabulares): se les da el historial mensual
    COMPLETO de cada producto (model.df, calendario ya completado con
    ceros) hasta el mes de cada fila de prueba, y se re-entrenan mes a mes
    (rolling_one_step_forecast) — el mismo criterio de "sin fuga hacia el
    futuro" que la partición temporal del resto del pipeline.
    """
    print(f"\n===== COMPARACIÓN — REGRESIÓN (muestra de {sample_size} productos) =====")
    inicio = time.time()

    c = model._c
    id_col = c("id")

    ids_muestra = sample_products_with_test_rows(
        model._df_train, model._test_idx, id_col, model._top_prods,
        n=sample_size, random_state=random_state,
    )
    eval_frame = build_eval_frame(model._df_train, model._test_idx, id_col, ids_muestra)
    y_true = eval_frame["target"].to_numpy()

    # ── RF y XGBoost: mismas features que en la Regresión A ────────────────
    X_eval = model._X.loc[eval_frame.index]
    pred_rf = np.expm1(model.reg.predict(model.scaler.transform(X_eval)))
    pred_xgb = np.expm1(xgb_reg.predict(X_eval))

    # ── Ingenuo / media_3 (mismo criterio que train_regressor()) ───────────
    pred_ingenuo = eval_frame[c("total_sold")].to_numpy()
    pred_media_3 = eval_frame["media_3_actual"].to_numpy()
    pred_media_3 = np.where(np.isnan(pred_media_3), pred_ingenuo, pred_media_3)

    # ── Escala de MASE: la MISMA definición que train_regressor() (naive
    # MAE de entrenamiento), pero recalculada sobre los top productos
    # completos (no solo la muestra) para que sea comparable con la
    # Regresión A — es la escala que ya usa model.reg_metrics.
    es_train = model._df_train.index.isin(model._train_idx)
    es_top = model._df_train[id_col].isin(model._top_prods)
    df_reg_train = model._df_train[es_train & es_top]
    escala = (df_reg_train["target"] - df_reg_train[c("total_sold")]).abs().mean()

    # ── ARIMA y Prophet: pronóstico a un paso, re-entrenado mes a mes ──────
    pred_arima = np.full(len(eval_frame), np.nan)
    pred_prophet = np.full(len(eval_frame), np.nan)
    respaldos_arima = []
    respaldos_prophet = []

    inicio_ts = time.time()
    for prod_id in ids_muestra:
        serie_producto = (
            model.df[model.df[id_col] == prod_id]
            .sort_values("fecha")
            .reset_index(drop=True)
        )
        serie = serie_producto[c("total_sold")].to_numpy()
        pos_por_fecha = {fecha: pos for pos, fecha in enumerate(serie_producto["fecha"])}

        filas_producto = eval_frame[eval_frame[id_col] == prod_id]
        posiciones = [pos_por_fecha[f] for f in filas_producto["fecha"]]

        preds_arima_prod = rolling_one_step_forecast(
            serie, posiciones,
            fit_predict=lambda h: arima_fit_predict(h, fallback_counter=respaldos_arima),
        )
        preds_prophet_prod = rolling_one_step_forecast(
            serie, posiciones,
            fit_predict=lambda h: prophet_fit_predict(h, fallback_counter=respaldos_prophet),
        )

        for idx_fila, pos in zip(filas_producto.index, posiciones):
            loc = eval_frame.index.get_loc(idx_fila)
            pred_arima[loc] = preds_arima_prod[pos]
            pred_prophet[loc] = preds_prophet_prod[pos]
    elapsed_ts = time.time() - inicio_ts

    resultados = {
        "Random Forest": model._regression_metrics(y_true, pred_rf, escala),
        "XGBoost": model._regression_metrics(y_true, pred_xgb, escala),
        "Ingenuo": model._regression_metrics(y_true, pred_ingenuo, escala),
        "Media_3": model._regression_metrics(y_true, pred_media_3, escala),
        "ARIMA(1,1,1)": model._regression_metrics(y_true, pred_arima, escala),
        "Prophet": model._regression_metrics(y_true, pred_prophet, escala),
    }
    fallbacks = {"ARIMA(1,1,1)": len(respaldos_arima), "Prophet": len(respaldos_prophet)}

    elapsed = time.time() - inicio
    print(f"[✓] Comparación de regresión (muestra) completada ({elapsed:.1f} s)")
    print(f"     Productos muestreados: {len(ids_muestra)} | filas de prueba: {len(eval_frame):,}")
    print(f"     Tiempo Prophet+ARIMA (rolling): {elapsed_ts:.1f} s")
    print(f"     Respaldos ingenuos — ARIMA: {fallbacks['ARIMA(1,1,1)']} | Prophet: {fallbacks['Prophet']}")

    return resultados, fallbacks, len(eval_frame), elapsed


# ── Reporte de texto / CSV / figura ─────────────────────────────────────────
def _tabla_texto(resultados, metricas, etiquetas, ancho_modelo=20):
    encabezado = f"{'Métrica':<22}" + "".join(f"{m:<{ancho_modelo}}" for m in resultados)
    lineas = [encabezado]
    for k, etiqueta in zip(metricas, etiquetas):
        fila = f"{etiqueta:<22}"
        for modelo in resultados:
            valor = resultados[modelo].get(k, float("nan"))
            fmt = ",.0f" if k in ("mae", "mse", "rmse") else ",.4f"
            fila += f"{valor:<{ancho_modelo}{fmt}}"
        lineas.append(fila)
    return "\n".join(lineas)


def build_report(res_clf, xgb_clf_params, n_test_clf,
                  res_reg_a, res_reg_b, fallbacks_b, n_test_b, elapsed_total):
    lineas = []
    lineas.append("COMPARACIÓN DE MODELOS — RF vs XGBoost vs Prophet vs ARIMA")
    lineas.append("=" * 70)
    lineas.append("")
    lineas.append(
        "Condiciones de equidad: los tres bloques usan EXACTAMENTE la misma "
        "partición temporal 80/20 por producto (SalesForecastModel."
        "prepare_split), las mismas filas de prueba y, cuando aplica, las "
        "mismas features sin escalar (model._X). El clasificador XGBoost se "
        "ajustó con GridSearchCV sobre los MISMOS pliegues mensuales "
        "(_monthly_time_series_folds) que usa tune_classifier(), con "
        "sample_weight balanceado (compute_sample_weight('balanced', ...)) "
        "para igualar class_weight='balanced' de RF. El regresor XGBoost NO "
        "se ajustó (n_estimators=400, max_depth=6, learning_rate=0.05, sin "
        "búsqueda de grilla) — se indica explícitamente para que la "
        "comparación no oculte esa diferencia. Prophet/ARIMA(1,1,1) son "
        "modelos de serie de tiempo univariados: reciben el historial "
        "mensual completo de cada producto y se re-entrenan mes a mes "
        "(pronóstico a un paso, sin ver nunca el mes que pronostican)."
    )
    lineas.append("")

    lineas.append("1) CLASIFICADOR — prueba (n=%s)" % f"{n_test_clf:,}")
    lineas.append("-" * 70)
    lineas.append(f"Mejores hiperparámetros XGBoost (GridSearchCV): {xgb_clf_params}")
    lineas.append("")
    metricas_clf = ["f1_macro", "balanced_accuracy", "accuracy", "f1_weighted",
                    "f1_reducir", "f1_mantener", "f1_reforzar"]
    etiquetas_clf = ["F1 macro", "Acc. balanceada", "Accuracy", "F1 ponderado",
                     "F1 Reducir", "F1 Mantener", "F1 Reforzar"]
    lineas.append(_tabla_texto(res_clf, metricas_clf, etiquetas_clf))
    lineas.append("")

    lineas.append("2) REGRESIÓN A — conjunto de prueba completo (top productos, n=%s)"
                   % f"{res_reg_a['Random Forest']['n_test']:,}")
    lineas.append("-" * 70)
    metricas_reg = ["mae", "mse", "rmse", "r2", "mase"]
    etiquetas_reg = ["MAE", "MSE", "RMSE", "R2", "MASE"]
    lineas.append(_tabla_texto(res_reg_a, metricas_reg, etiquetas_reg))
    lineas.append("")

    lineas.append("3) REGRESIÓN B — muestra de productos (n=%s filas, %s respaldos ARIMA, "
                   "%s respaldos Prophet)"
                   % (f"{n_test_b:,}", fallbacks_b["ARIMA(1,1,1)"], fallbacks_b["Prophet"]))
    lineas.append("-" * 70)
    lineas.append(_tabla_texto(res_reg_b, metricas_reg, etiquetas_reg))
    lineas.append("")

    lineas.append(f"Tiempo total: {elapsed_total:.1f} s")
    return "\n".join(lineas)


def build_csv_rows(res_clf, res_reg_a, res_reg_b):
    filas = []
    for modelo, metricas in res_clf.items():
        for metrica, valor in metricas.items():
            filas.append({"seccion": "clasificacion", "modelo": modelo,
                          "metrica": metrica, "valor": valor})
    for modelo, metricas in res_reg_a.items():
        for metrica, valor in metricas.items():
            filas.append({"seccion": "regresion_a_completa", "modelo": modelo,
                          "metrica": metrica, "valor": valor})
    for modelo, metricas in res_reg_b.items():
        for metrica, valor in metricas.items():
            filas.append({"seccion": "regresion_b_muestra", "modelo": modelo,
                          "metrica": metrica, "valor": valor})
    return pd.DataFrame(filas)


def build_figure(res_clf, res_reg_b, salida):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel izquierdo: F1 macro y accuracy balanceada por modelo ─────────
    modelos_clf = list(res_clf.keys())
    x = np.arange(len(modelos_clf))
    ancho = 0.35
    color_f1 = "#2a78d6"   # paleta categórica de referencia, posición 1
    color_bal = "#eb6834"  # posición 2

    f1_macro = [res_clf[m]["f1_macro"] for m in modelos_clf]
    bal_acc = [res_clf[m]["balanced_accuracy"] for m in modelos_clf]

    ax1.bar(x - ancho / 2, f1_macro, width=ancho, color=color_f1, label="F1 macro")
    ax1.bar(x + ancho / 2, bal_acc, width=ancho, color=color_bal, label="Accuracy balanceada")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modelos_clf, rotation=15)
    ax1.set_ylabel("Valor de la métrica")
    ax1.set_title("Clasificador — F1 macro y accuracy balanceada")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=.3)

    # ── Panel derecho: MASE por modelo (Regresión B), ordenado ─────────────
    modelos_mase = sorted(res_reg_b.keys(), key=lambda m: res_reg_b[m]["mase"])
    valores_mase = [res_reg_b[m]["mase"] for m in modelos_mase]

    ax2.barh(modelos_mase, valores_mase, color="#2a78d6")
    # Línea de referencia en tinta neutra (el rojo queda reservado para
    # estados); su etiqueta va arriba del panel para no chocar con el eje X.
    ax2.axvline(1.0, color="#52514e", linestyle="--", linewidth=1.2)
    ax2.set_ylim(-0.6, len(modelos_mase) - 0.1)
    ax2.text(1.0, len(modelos_mase) - 0.2, " MASE = 1: pronóstico ingenuo\n (entrenamiento)",
             color="#52514e", ha="left", va="top", fontsize=8.5)
    ax2.set_xlabel("MASE (muestra de productos)")
    ax2.set_title("Regresión — MASE por modelo (menor es mejor)")
    ax2.grid(True, axis="x", alpha=.3)

    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    plt.close(fig)


def main():
    inicio = time.time()
    ensure_output_dirs()

    model = SalesForecastModel(filepath=DATA_RAW / "Query_Result_V5.csv")
    model.load_and_clean()
    model.build_features()
    model.cluster()
    model.prepare_split()
    model.train_classifier()
    model.train_regressor()

    res_clf, xgb_clf_params, n_test_clf, t_clf = compare_classification(model)
    res_reg_a, xgb_reg, t_reg_a = compare_regression_full(model)
    res_reg_b, fallbacks_b, n_test_b, t_reg_b = compare_regression_sample(model, xgb_reg)

    elapsed_total = time.time() - inicio

    resumen = build_report(
        res_clf, xgb_clf_params, n_test_clf,
        res_reg_a, res_reg_b, fallbacks_b, n_test_b, elapsed_total,
    )
    print("\n" + resumen)

    salida_txt = REPORTS / "comparacion_modelos.txt"
    salida_txt.write_text(resumen, encoding="utf-8")
    print(f"\n[✓] Reporte exportado → {salida_txt}")

    tabla_csv = build_csv_rows(res_clf, res_reg_a, res_reg_b)
    salida_csv = REPORTS / "comparacion_modelos.csv"
    tabla_csv.to_csv(salida_csv, index=False)
    print(f"[✓] CSV exportado → {salida_csv}")

    salida_fig = FIGURES / "comparacion_modelos.png"
    build_figure(res_clf, res_reg_b, salida_fig)
    print(f"[✓] Figura exportada → {salida_fig}")

    print(f"[✓] Tiempo total: {elapsed_total:.1f} s "
          f"(clasificación {t_clf:.1f} s, regresión A {t_reg_a:.1f} s, "
          f"regresión B {t_reg_b:.1f} s)")


if __name__ == "__main__":
    main()
