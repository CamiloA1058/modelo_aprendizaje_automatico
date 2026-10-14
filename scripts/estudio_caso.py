"""
estudio_caso.py
===============
Estudio de caso fuera de muestra con origen rodante. Para cada mes objetivo M
(abril a septiembre de 2026) entrena el procedimiento de produccion
(SalesForecastModel con sus valores por defecto: crecimiento 1,15, reduccion
0,90, k=3, min_months=6, top_n_products=1900) con la V6 mas el holdout
estrictamente ANTERIOR a M, predice M exactamente como entrenar.py y lo compara
con las ventas reales de M (data/holdout/ventas_2026_04_09.csv).

Los meses del holdout nunca se usaron para elegir k ni hiperparametros.

Genera:
  outputs/reports/estudio_caso.txt
  outputs/reports/estudio_caso.csv
  outputs/figures/estudio_caso.png

Uso:  PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/estudio_caso.py
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import case_study as cs  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402
from ventas_forecast.paths import (  # noqa: E402
    DEFAULT_DATASET, FIGURES, PROJECT_ROOT, REPORTS, ensure_output_dirs,
)

HOLDOUT = PROJECT_ROOT / "data" / "holdout" / "ventas_2026_04_09.csv"
MESES = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
NOMBRES = ["Reducir", "Mantener", "Reforzar"]
DECISION_A_CLASE = {"Reducir stock": 0, "Mantener stock": 1, "Reforzar stock": 2}

# Paleta de las figuras anteriores (compare_models.py, eda.py)
COLOR_MODELO = "#1baf7a"
COLOR_BASE_A = "#eb6834"
COLOR_BASE_B = "#4a3aa7"
COLOR_GRID = "#e1e0d9"
COLOR_INK = "#52514e"
COLOR_SURFACE = "#fcfcfb"


def evaluar_mes(mes, tmp_dir):
    """Entrena con datos < mes, predice mes y devuelve (filas, escala, mayoritaria)."""
    ruta = Path(tmp_dir) / f"entrenamiento_{mes}.csv"
    with open(ruta, "w", encoding="utf-8-sig", newline="") as fh:
        n = cs.combine_training(DEFAULT_DATASET, HOLDOUT, mes, fh)
    print(f"\n=== Mes objetivo {mes}: {n:,} filas de entrenamiento ===")

    model = SalesForecastModel(
        filepath=ruta, output_path=Path(tmp_dir) / f"pred_{mes}.csv", min_months=6,
    )
    model.load_and_clean()
    if model._mes_pred.strftime("%Y-%m") != mes:
        raise RuntimeError(f"El modelo predice {model._mes_pred:%Y-%m}, se esperaba {mes}")
    model.build_features()
    model.cluster()
    model.prepare_split()
    model.train_classifier()
    model.train_regressor()
    model.predict_next_month()

    c = model._c
    res = model.results.copy()
    res["id"] = res[c("id")].astype(str)
    res["actual"] = res[c("total_sold")].astype(float)
    res["media_3"] = model.df_predict.loc[res.index, "media_3_actual"].to_numpy()
    res["media_3"] = res["media_3"].fillna(res["actual"])
    res["real"] = cs.monthly_sales(HOLDOUT, mes).reindex(res["id"]).fillna(0.0).to_numpy()
    res["clase_real"] = cs.real_class(res["real"], res["actual"], model.growth_threshold,
                                      model.reduce_threshold)
    res["clase_pred"] = res["decision"].map(DECISION_A_CLASE)
    res["mes"] = mes

    escala = cs.naive_scale(model._df_train.loc[model._train_idx], c("id"),
                            c("total_sold"), model._top_prods)
    mayoritaria = int(model._y.iloc[model._train_idx].value_counts().idxmax())
    return res, escala, mayoritaria


def resumen_mes(res, escala, mayoritaria):
    dec = cs.classification_summary(res["clase_real"], res["clase_pred"])
    base = cs.majority_baseline_summary(res["clase_real"], mayoritaria)
    m = res[res["ventas_predichas"].notna()]
    reg = {
        "modelo": cs.regression_summary(m["real"], m["ventas_predichas"], escala),
        "ingenuo": cs.regression_summary(m["real"], m["actual"], escala),
        "media_3": cs.regression_summary(m["real"], m["media_3"], escala),
    }
    return dec, base, reg


def fila_csv(mes, dec, base, reg):
    fila = {
        "mes": mes, "n_decisiones": dec["n"],
        "f1_macro": dec["f1_macro"], "acc_balanceada": dec["balanced_accuracy"],
        "accuracy": dec["accuracy"],
        "base_f1_macro": base["f1_macro"], "base_acc_balanceada": base["balanced_accuracy"],
        "base_accuracy": base["accuracy"], "n_montos": reg["modelo"]["n"],
    }
    for nombre, r in reg.items():
        for k in ("mae", "rmse", "mase", "sum_ratio"):
            fila[f"{k}_{nombre}"] = r[k]
    return fila


def bloque_texto(titulo, dec, base, reg):
    L = [titulo, "-" * 70]
    L.append(f"Decisiones (n={dec['n']:,}): F1 macro {dec['f1_macro']:.4f} | "
             f"acc. balanceada {dec['balanced_accuracy']:.4f} | accuracy {dec['accuracy']:.4f}")
    L.append(f"  Linea base mayoritaria: F1 macro {base['f1_macro']:.4f} | "
             f"acc. balanceada {base['balanced_accuracy']:.4f} | accuracy {base['accuracy']:.4f}")
    L.append(f"  Matriz de confusion (filas = real, columnas = predicha; {NOMBRES}):")
    for nombre, fila in zip(NOMBRES, dec["confusion"]):
        L.append(f"    {nombre:<9}: {fila}")
    L.append(f"Montos (n={reg['modelo']['n']:,} productos con prediccion de monto):")
    L.append(f"  {'':<10}{'MAE':>14}{'RMSE':>14}{'MASE':>9}{'suma pred/real':>16}")
    for nombre, r in reg.items():
        L.append(f"  {nombre:<10}{r['mae']:>14,.0f}{r['rmse']:>14,.0f}"
                 f"{r['mase']:>9.4f}{r['sum_ratio']:>16.4f}")
    L.append("")
    return L


def figura(filas, salida):
    meses = [f["mes"] for f in filas]
    x = np.arange(len(meses))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(COLOR_SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(COLOR_SURFACE)
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=COLOR_INK)
        for s in ax.spines.values():
            s.set_color(COLOR_GRID)
        ax.set_xticks(x)
        ax.set_xticklabels(meses)

    ancho = 0.38
    ax1.bar(x - ancho / 2, [f["f1_macro"] for f in filas], ancho, color=COLOR_MODELO,
            label="Modelo (Random Forest)")
    ax1.bar(x + ancho / 2, [f["base_f1_macro"] for f in filas], ancho, color=COLOR_BASE_A,
            label="Clase mayoritaria")
    ax1.set_ylabel("F1 macro de las decisiones", color=COLOR_INK)
    ax1.set_xlabel("Mes predicho", color=COLOR_INK)
    ax1.set_ylim(0, max(f["f1_macro"] for f in filas) * 1.3)
    ax1.legend(frameon=False, loc="upper left")

    ancho = 0.27
    ax2.bar(x - ancho, [f["mae_modelo"] / 1e3 for f in filas], ancho, color=COLOR_MODELO,
            label="Modelo (Random Forest)")
    ax2.bar(x, [f["mae_ingenuo"] / 1e3 for f in filas], ancho, color=COLOR_BASE_A,
            label="Ingenuo (mes anterior)")
    ax2.bar(x + ancho, [f["mae_media_3"] / 1e3 for f in filas], ancho, color=COLOR_BASE_B,
            label="Media de 3 meses")
    ax2.set_ylabel("MAE por producto (miles de COP)", color=COLOR_INK)
    ax2.set_xlabel("Mes predicho", color=COLOR_INK)
    ax2.set_ylim(0, max(f["mae_ingenuo"] for f in filas) / 1e3 * 1.3)
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(salida, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    inicio = time.time()
    ensure_output_dirs()
    todos, filas, lineas_mes, escalas, mayoritarias = [], [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for mes in MESES:
            res, escala, mayoritaria = evaluar_mes(mes, tmp)
            dec, base, reg = resumen_mes(res, escala, mayoritaria)
            filas.append(fila_csv(mes, dec, base, reg))
            lineas_mes += bloque_texto(f"MES {mes}", dec, base, reg)
            todos.append(res)
            escalas.append((escala, int(res["ventas_predichas"].notna().sum())))
            mayoritarias.append(mayoritaria)

    pool = pd.concat(todos, ignore_index=True)
    escala_p = cs.pooled_scale(escalas)
    dec_p, base_p, reg_p = resumen_mes(pool, escala_p, 0)
    filas.append(fila_csv("agrupado", dec_p, base_p, reg_p))

    L = ["ESTUDIO DE CASO FUERA DE MUESTRA (origen rodante)", "=" * 70, "",
         "Para cada mes M se entrena con la V6 mas el holdout anterior a M (nunca M) y se "
         "predice M con los valores por defecto del dashboard (crecimiento 1,15; reduccion "
         "0,90; k=3; min_months=6; top_n_products=1900).",
         "Los meses del holdout nunca se usaron para elegir k ni los hiperparametros.",
         "Clase real: razon ventas reales de M / ventas del mes anterior con la misma regla "
         "de las etiquetas de entrenamiento; un mes sin ventas implica Reducir.",
         "Linea base: siempre predice la clase mayoritaria del entrenamiento "
         f"({', '.join(sorted({NOMBRES[m] for m in mayoritarias}))}); en el agrupado se usa Reducir.",
         "MASE: escala = error del ingenuo en el entrenamiento de cada mes; el agrupado usa "
         "el promedio de esas escalas ponderado por filas.",
         "Nota: el holdout llega hasta el 22 de septiembre, asi que las ventas reales de "
         "septiembre estan incompletas (favorece a Reducir y baja la razon de sumas).", ""]
    L += lineas_mes
    L += bloque_texto("AGRUPADO (todos los meses)", dec_p, base_p, reg_p)
    L.append(f"Tiempo total: {time.time() - inicio:.1f} s")
    texto = "\n".join(L)
    print("\n" + texto)
    (REPORTS / "estudio_caso.txt").write_text(texto, encoding="utf-8")
    pd.DataFrame(filas).to_csv(REPORTS / "estudio_caso.csv", index=False)
    figura(filas[:-1], FIGURES / "estudio_caso.png")
    print(f"[OK] {REPORTS / 'estudio_caso.txt'}, estudio_caso.csv y {FIGURES / 'estudio_caso.png'}")


if __name__ == "__main__":
    main()
