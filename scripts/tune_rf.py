"""
tune_rf.py
==========
Ajuste de hiperparámetros del clasificador (RandomForest) mediante
GridSearchCV con validación cruzada temporal por mes (TimeSeriesSplit sobre
la partición de entrenamiento), ver SalesForecastModel.tune_classifier.

Compara las métricas de validación (holdout de prueba) del clasificador con
los hiperparámetros por defecto contra los hiperparámetros encontrados por
la búsqueda en grilla.

Uso
---
    PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/tune_rf.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.paths import DATA_RAW, REPORTS, ensure_output_dirs  # noqa: E402

from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402


def main():
    inicio = time.time()
    ensure_output_dirs()

    model = SalesForecastModel(filepath=DATA_RAW / "Query_Result_V5.csv")
    model.load_and_clean()
    model.build_features()
    model.cluster()
    model.prepare_split()

    # ── Métricas de referencia con hiperparámetros por defecto ───────────────
    print("\n===== Entrenando con hiperparámetros por defecto =====")
    model.train_classifier()
    metricas_default = dict(model.metrics)
    params_default = dict(model.clf_params)

    # ── Ajuste de hiperparámetros (GridSearchCV + TimeSeriesSplit mensual) ───
    print("\n===== Ajustando hiperparámetros (GridSearchCV) =====")
    best_params = model.tune_classifier()

    scoring = "f1_macro"  # DECIDIDO por el usuario: ver tune_classifier()
    fila_mejor = model.tuning_results.iloc[0]
    mejor_cv_score_mean = fila_mejor["mean_test_score"]
    mejor_cv_score_std = fila_mejor["std_test_score"]

    # ── Reentrenar con los mejores hiperparámetros ───────────────────────────
    print("\n===== Entrenando con hiperparámetros ajustados =====")
    model.clf_params = {**model.clf_params, **best_params}
    model.train_classifier()
    metricas_tuned = dict(model.metrics)

    # ── Exportar tabla completa de la búsqueda ───────────────────────────────
    salida_csv = REPORTS / "tuning_rf_clasificador.csv"
    model.tuning_results.to_csv(salida_csv, index=False)

    # ── Resumen ───────────────────────────────────────────────────────────────
    elapsed = time.time() - inicio
    lineas = []
    lineas.append("AJUSTE DE HIPERPARÁMETROS DEL CLASIFICADOR (RandomForest)")
    lineas.append("=" * 60)
    lineas.append(f"Hiperparámetros por defecto: {params_default}")
    lineas.append(f"Mejores hiperparámetros:     {best_params}")
    lineas.append(f"Scoring usado en GridSearchCV: {scoring}")
    lineas.append(
        f"{scoring} en validación cruzada (media ± desv.): "
        f"{mejor_cv_score_mean:.4f} ± {mejor_cv_score_std:.4f}"
    )
    lineas.append("")
    lineas.append("Métricas de prueba (holdout) — principales (desbalance de clases)")
    lineas.append(f"{'Métrica':<20}{'Por defecto':<15}{'Ajustado':<15}")
    for k, etiqueta in (
        ("f1_macro", "F1 macro"),
        ("balanced_accuracy", "Acc. balanceada"),
    ):
        lineas.append(
            f"{etiqueta:<20}{metricas_default[k]:<15.4f}{metricas_tuned[k]:<15.4f}"
        )
    lineas.append("")
    lineas.append("Métricas de prueba (holdout) — ponderadas por soporte (referencia)")
    lineas.append(f"{'Métrica':<20}{'Por defecto':<15}{'Ajustado':<15}")
    for k in ("accuracy", "precision", "recall", "f1"):
        lineas.append(
            f"{k.capitalize():<20}{metricas_default[k]:<15.4f}{metricas_tuned[k]:<15.4f}"
        )
    lineas.append("")
    lineas.append(f"Tiempo total: {elapsed:.1f} s")

    resumen = "\n".join(lineas)
    print("\n" + resumen)

    salida_txt = REPORTS / "tuning_rf_resumen.txt"
    salida_txt.write_text(resumen, encoding="utf-8")

    print(f"\n[✓] Tabla completa exportada  → {salida_csv}")
    print(f"[✓] Resumen exportado         → {salida_txt}")
    print(f"[✓] Tiempo total: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
