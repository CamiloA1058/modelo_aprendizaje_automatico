"""
evaluate_model.py
==================
Evaluación completa del modelo de producción (clasificador + regresor) sobre
la partición de PRUEBA, con líneas base de comparación:

  - Clasificador: métricas de model.metrics, reporte de clasificación,
    matriz de confusión, y una línea base de clase mayoritaria (siempre
    predice la clase más frecuente de ENTRENAMIENTO).
  - Regresor: tabla MAE/MSE/RMSE/R2/MASE del modelo contra los baselines
    "ingenuo" y "media_3" (ver SalesForecastModel.train_regressor).

Genera:
  outputs/reports/evaluacion_modelo.txt
  outputs/figures/matriz_confusion.png
  outputs/figures/real_vs_predicho.png

Uso
---
    PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/evaluate_model.py
"""

import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.paths import DATA_RAW, FIGURES, REPORTS, ensure_output_dirs  # noqa: E402

from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402

CLASS_NAMES = ["Reducir", "Mantener", "Reforzar"]


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

    # ── Clasificador: predicciones de prueba y línea base de mayoría ────────
    y_tr = model._y.iloc[model._train_idx]
    y_te = model._y.iloc[model._test_idx]
    X_te = model._X_scaled[model._test_idx]
    pred = model.clf.predict(X_te)

    reporte_clf = classification_report(
        y_te, pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0
    )
    matriz = confusion_matrix(y_te, pred, labels=[0, 1, 2])

    # Línea base de clase mayoritaria: siempre predice la clase más frecuente
    # de la partición de ENTRENAMIENTO, evaluada sobre la partición de prueba.
    clase_mayoritaria = y_tr.value_counts().idxmax()
    pred_mayoritaria = np.full(len(y_te), clase_mayoritaria)
    baseline_clf = {
        "accuracy": round(accuracy_score(y_te, pred_mayoritaria), 4),
        "f1": round(f1_score(y_te, pred_mayoritaria, average="weighted", zero_division=0), 4),
        # f1_macro y balanced_accuracy de la línea base: con una sola clase
        # predicha, ambas colapsan al F1/recall de esa clase promediado con
        # 0 en las demás (ver nota de desbalance más abajo).
        "f1_macro": round(f1_score(y_te, pred_mayoritaria, average="macro", zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_te, pred_mayoritaria), 4),
    }

    # Distribución de clases de la prueba: sustenta la nota de desbalance
    # (accuracy y f1_weighted están dominados por la clase mayoritaria).
    distribucion_test = y_te.value_counts().reindex([0, 1, 2], fill_value=0)

    # ── Regresor: métricas ya calculadas en train_regressor() ───────────────
    reg_metrics = model.reg_metrics
    baseline_reg = model.baseline_metrics

    # ── Reporte de texto ─────────────────────────────────────────────────────
    lineas = []
    lineas.append("EVALUACIÓN DEL MODELO (clasificador + regresor)")
    lineas.append("=" * 60)

    lineas.append("")
    lineas.append("CLASIFICADOR — métricas de prueba")
    lineas.append("-" * 60)
    lineas.append("Métricas principales (robustas al desbalance de clases):")
    lineas.append(f"  F1 macro          : {model.metrics['f1_macro']:.4f}")
    lineas.append(f"  Accuracy balanceada: {model.metrics['balanced_accuracy']:.4f}")
    lineas.append("")
    lineas.append("Métricas ponderadas por soporte de clase (referencia):")
    for k in ("accuracy", "precision", "recall", "f1"):
        lineas.append(f"  {k.capitalize():<10}: {model.metrics[k]:.4f}")
    lineas.append("")
    lineas.append(
        f"Distribución de clases en la prueba (n={len(y_te):,}): "
        + ", ".join(
            f"{CLASS_NAMES[cls]} {cnt:,} ({cnt / len(y_te):.2%})"
            for cls, cnt in distribucion_test.items()
        )
    )
    lineas.append(
        "Nota: la accuracy (y el F1 ponderado) por sí solos engañan bajo "
        "este desbalance — un modelo que SIEMPRE prediga la clase "
        "mayoritaria ('Reducir') ya obtiene una accuracy alta (ver línea "
        "base abajo); por eso F1 macro y accuracy balanceada son las "
        "métricas principales de este reporte."
    )
    lineas.append("")
    lineas.append(
        f"Línea base de clase mayoritaria (siempre predice "
        f"'{CLASS_NAMES[clase_mayoritaria]}', clase más frecuente en "
        f"entrenamiento):"
    )
    lineas.append(f"  F1 macro           : {baseline_clf['f1_macro']:.4f}")
    lineas.append(f"  Accuracy balanceada: {baseline_clf['balanced_accuracy']:.4f}")
    lineas.append(f"  Accuracy           : {baseline_clf['accuracy']:.4f}")
    lineas.append(f"  F1 (ponderado)     : {baseline_clf['f1']:.4f}")

    lineas.append("")
    lineas.append("Reporte de clasificación (prueba):")
    lineas.append(reporte_clf)

    lineas.append("Matriz de confusión (filas = clase real, columnas = clase predicha):")
    lineas.append(f"  Orden de clases: {CLASS_NAMES}")
    for fila, nombre in zip(matriz, CLASS_NAMES):
        lineas.append(f"  {nombre:<10}: {fila.tolist()}")

    lineas.append("")
    lineas.append("REGRESOR — métricas de prueba (top productos)")
    lineas.append("-" * 60)
    lineas.append(f"{'Métrica':<10}{'Modelo':>20}{'Ingenuo':>20}{'Media_3':>20}")
    for k in ("mae", "mse", "rmse", "r2", "mase"):
        # MSE se muestra en notación científica porque supera los 10^12
        fmt = ".4e" if k == "mse" else ",.4f"
        lineas.append(
            f"{k.upper():<10}"
            f"{reg_metrics[k]:>20{fmt}}"
            f"{baseline_reg['ingenuo'][k]:>20{fmt}}"
            f"{baseline_reg['media_3'][k]:>20{fmt}}"
        )
    lineas.append(f"n_test: {reg_metrics['n_test']}")

    elapsed = time.time() - inicio
    lineas.append("")
    lineas.append(f"Tiempo total: {elapsed:.1f} s")

    resumen = "\n".join(lineas)
    print(resumen)

    salida_txt = REPORTS / "evaluacion_modelo.txt"
    salida_txt.write_text(resumen, encoding="utf-8")
    print(f"\n[✓] Reporte exportado -> {salida_txt}")

    # ── Figura: matriz de confusión ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title("Matriz de confusión — clasificador")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    umbral = matriz.max() / 2 if matriz.max() > 0 else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            color = "white" if matriz[i, j] > umbral else "black"
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    salida_matriz = FIGURES / "matriz_confusion.png"
    fig.savefig(salida_matriz, dpi=200)
    plt.close(fig)
    print(f"[✓] Figura exportada -> {salida_matriz}")

    # ── Figura: real vs predicho (regresor, escala log-log) ──────────────────
    tabla = model.reg_test_results
    real      = tabla["real"].to_numpy()
    predicho  = tabla["predicho"].to_numpy()

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(real, predicho, alpha=.6, color="#1D4ED8", s=30)

    minimo = max(min(real.min(), predicho.min()), 1e-6)
    maximo = max(real.max(), predicho.max())
    ax2.plot([minimo, maximo], [minimo, maximo],
             linestyle="--", color="#DC2626", label="y = x")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Ventas reales (COP)")
    ax2.set_ylabel("Ventas predichas (COP)")
    ax2.set_title("Regresor — real vs. predicho (prueba, top productos)")
    ax2.legend()
    ax2.grid(True, alpha=.3, which="both")

    fig2.tight_layout()
    salida_scatter = FIGURES / "real_vs_predicho.png"
    fig2.savefig(salida_scatter, dpi=200)
    plt.close(fig2)
    print(f"[✓] Figura exportada -> {salida_scatter}")

    print(f"[✓] Tiempo total: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
