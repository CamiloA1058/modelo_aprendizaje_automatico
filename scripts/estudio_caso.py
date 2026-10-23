"""
estudio_caso.py
===============
Estudio de caso fuera de muestra con origen rodante. Para cada mes objetivo M
(abril a septiembre de 2026) entrena el procedimiento de produccion
(SalesForecastModel con sus valores por defecto: crecimiento 1,15, reduccion
0,90, k=3, min_months=6, top_n_products=1900) con la V6 mas el holdout
estrictamente ANTERIOR a M, predice M exactamente como entrenar.py y lo compara
con las ventas reales de M (data/holdout/ventas_2026_04_09.csv).

Las decisiones que se evaluan son las finales del sistema (con el umbral t de
Reforzar y la coherencia con la probabilidad de venta de agregar_accion). Ademas
reporta la accion unica de todos los productos del dashboard (modelo y
promedio): productos y acierto de Comprar más y de No reponer este mes,
incoherencias de monto, venta esperada frente a lo real y cobertura del stock
sugerido por accion.

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
from ventas_forecast import eligibility as el  # noqa: E402
from ventas_forecast import prediccion_vs_real as pr  # noqa: E402
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


def evaluar_mes(mes, tmp_dir, con_modelo=False, dataset=None, sample_weight_fn=None,
                elegibilidad_ampliada=False):
    """Entrena con datos < mes, predice mes y devuelve (filas, escala, mayoritaria);
    con con_modelo=True devuelve ademas el modelo entrenado como cuarto valor.
    dataset (ruta) y sample_weight_fn solo los usan los experimentos; por
    defecto se entrena con DEFAULT_DATASET y sin pesos. elegibilidad_ampliada
    (experimento) tambien predice productos sin ventas el ultimo mes; su clase
    real se mide contra media_3 (ver ventas_forecast.eligibility)."""
    ruta = Path(tmp_dir) / f"entrenamiento_{mes}.csv"
    with open(ruta, "w", encoding="utf-8-sig", newline="") as fh:
        n = cs.combine_training(dataset or DEFAULT_DATASET, HOLDOUT, mes, fh)
    print(f"\n=== Mes objetivo {mes}: {n:,} filas de entrenamiento ===")

    model = SalesForecastModel(
        filepath=ruta, output_path=Path(tmp_dir) / f"pred_{mes}.csv", min_months=6,
        elegibilidad_ampliada=elegibilidad_ampliada,
    )
    model.sample_weight_fn = sample_weight_fn
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
    # Base de la decision: mes actual (produccion, siempre > 0) o media_3 si el
    # producto no vendio el ultimo mes (solo con elegibilidad ampliada).
    res["base_decision"] = el.decision_base(res["actual"], res["media_3"])
    res["nuevo"] = res["actual"] <= 0
    res["clase_real"] = cs.real_class(res["real"], res["base_decision"],
                                      model.growth_threshold, model.reduce_threshold)
    res["clase_pred"] = res["decision"].map(DECISION_A_CLASE)
    res["mes"] = mes

    escala = cs.naive_scale(model._df_train.loc[model._train_idx], c("id"),
                            c("total_sold"), model._top_prods)
    mayoritaria = int(model._y.iloc[model._train_idx].value_counts().idxmax())
    if con_modelo:
        return res, escala, mayoritaria, model
    return res, escala, mayoritaria


def completar_accion(model, res, mes):
    """Corre el resto del pipeline de produccion (promedio, probabilidad de
    venta, stock y accion), deja en res la decision final de cada producto
    del modelo y devuelve la tabla por producto de prediccion_vs_real (todos
    los productos del dashboard) con lo real de mes."""
    model.estimar_por_promedio()
    model.agregar_probabilidad_venta()
    model.agregar_stock_sugerido()
    model.agregar_accion()
    final = model.results.loc[res.index, "decision"]
    res["decision"] = final.to_numpy()
    res["clase_pred"] = res["decision"].map(DECISION_A_CLASE)
    c = model._c
    return pr.tabla_producto(model.results, cs.monthly_sales(HOLDOUT, mes), c("id"),
                             c("description"), c("total_sold"), model.growth_threshold,
                             model.reduce_threshold,
                             real_unid=cs.monthly_units(HOLDOUT, mes))


def umbrales_accion(model):
    """Lo elegido con la validacion interna del mes (para el reporte)."""
    rs = model.regla_stock or {}
    return {"t_reforzar": model.umbral_reforzar, "umbral_venta": model.sale_threshold,
            "s_baja": model.s_baja, "monto_si_vende": model.monto_si_vende_clave,
            "stock_val_cob_70": rs.get("cobertura_nivel"),
            "stock_val_cob_accion": rs.get("cobertura_regla"),
            "stock_val_razon_70": rs.get("stock_real_nivel"),
            "stock_val_razon_accion": rs.get("stock_real_regla"),
            "stock_por_accion": rs.get("adoptada")}


COLS_ACCION = ["Comprar más (productos)", "% acierto Comprar más",
               "Comprar más con «se venderá: No»", "Mantener (productos)",
               "No reponer (productos)", "% acierto No reponer", "Revisar (productos)",
               "Por promedio (productos)", "Incoherencias de monto",
               "Venta esperada (COP)", "Razón venta esperada / real",
               "Cobertura del stock sugerido", "Stock total / real"]


def bloque_accion(titulo, r, umbrales=None):
    L = [titulo, "-" * 70]
    if umbrales:
        L.append(f"Elegido con la validacion interna: t Reforzar {umbrales['t_reforzar']} | "
                 f"umbral de venta {umbrales['umbral_venta']:.2f} | s_baja "
                 f"{umbrales['s_baja']:.2f} | monto si se vende {umbrales['monto_si_vende']}")
        if umbrales["stock_por_accion"] is not None:
            L.append(f"  Stock por accion en la validacion interna: cobertura "
                     f"{100 * umbrales['stock_val_cob_70']:.1f} % -> "
                     f"{100 * umbrales['stock_val_cob_accion']:.1f} %, stock/real "
                     f"{umbrales['stock_val_razon_70']:.3f} -> "
                     f"{umbrales['stock_val_razon_accion']:.3f} -> "
                     f"{'adoptado' if umbrales['stock_por_accion'] else 'NO adoptado'}")
    L.append(f"Comprar más {r['Comprar más (productos)']:,} (acierto "
             f"{100 * r['% acierto Comprar más']:.1f} %; con «se venderá: No» "
             f"{r['Comprar más con «se venderá: No»']}) | No reponer "
             f"{r['No reponer (productos)']:,} (acierto {100 * r['% acierto No reponer']:.1f} %)"
             f" | Mantener {r['Mantener (productos)']:,} | Revisar "
             f"{r['Revisar (productos)']:,} | Por promedio {r['Por promedio (productos)']:,}")
    L.append(f"Incoherencias de monto {r['Incoherencias de monto']} | venta esperada / real "
             f"{r['Razón venta esperada / real']:.3f} | stock sugerido: cobertura "
             f"{100 * r['Cobertura del stock sugerido']:.1f} %, stock / real "
             f"{r['Stock total / real']:.3f}")
    L.append("")
    return L


def resumen_mes(res, escala, mayoritaria):
    dec = cs.classification_summary(res["clase_real"], res["clase_pred"])
    base = cs.majority_baseline_summary(res["clase_real"], mayoritaria)
    m = res[res["ventas_predichas_rf"].notna()]
    # "modelo" = Random Forest puro; "mezcla" = prediccion del sistema
    # (w*RF + (1-w)*media_3, w elegido con la validacion interna de cada mes).
    reg = {
        "modelo": cs.regression_summary(m["real"], m["ventas_predichas_rf"], escala),
        "mezcla": cs.regression_summary(m["real"], m["ventas_predichas"], escala),
        "ingenuo": cs.regression_summary(m["real"], m["actual"], escala),
        "media_3": cs.regression_summary(m["real"], m["media_3"], escala),
    }
    return dec, base, reg


def sumas_mes(res, ventas_mes):
    """Sumas en COP sobre los productos con prediccion de monto y venta real total del mes."""
    m = res[res["ventas_predichas_rf"].notna()]
    return {
        "suma_pred_modelo": float(m["ventas_predichas_rf"].sum()),
        "suma_pred_mezcla": float(m["ventas_predichas"].sum()),
        "suma_real_con_pred": float(m["real"].sum()),
        "suma_ingenuo": float(m["actual"].sum()),
        "suma_media_3": float(m["media_3"].sum()),
        "ventas_reales_mes": float(ventas_mes),
    }


def fila_csv(mes, dec, base, reg, sumas, accion=None, umbrales=None):
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
    fila.update(sumas)
    if accion is not None:
        fila.update({k: accion[k] for k in COLS_ACCION})
    if umbrales is not None:
        fila.update(umbrales)
    return fila


def bloque_texto(titulo, dec, base, reg, sumas):
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
    L.append("Sumas en COP (productos con prediccion de monto):")
    L.append(f"  modelo {sumas['suma_pred_modelo']:>15,.0f} | mezcla {sumas['suma_pred_mezcla']:>15,.0f} | real {sumas['suma_real_con_pred']:>15,.0f} | "
             f"ingenuo {sumas['suma_ingenuo']:>15,.0f} | media_3 {sumas['suma_media_3']:>15,.0f}")
    L.append(f"  ventas reales totales del mes (todos los productos): {sumas['ventas_reales_mes']:,.0f}")
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
    todos, filas, lineas_mes, escalas, mayoritarias, sumas_todas = [], [], [], [], [], []
    tablas, lineas_acc = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for mes in MESES:
            res, escala, mayoritaria, model = evaluar_mes(mes, tmp, con_modelo=True)
            tabla = completar_accion(model, res, mes)
            tablas.append(tabla)
            acc_mes = pr.resumen_mes(mes, tabla)
            umbrales = umbrales_accion(model)
            lineas_acc += bloque_accion(f"MES {mes}", acc_mes, umbrales)
            dec, base, reg = resumen_mes(res, escala, mayoritaria)
            ventas_mes = cs.monthly_sales(HOLDOUT, mes).sum()
            sumas = sumas_mes(res, ventas_mes)
            sumas_todas.append(sumas)
            filas.append(fila_csv(mes, dec, base, reg, sumas, acc_mes, umbrales))
            lineas_mes += bloque_texto(f"MES {mes}", dec, base, reg, sumas)
            todos.append(res)
            escalas.append((escala, int(res["ventas_predichas_rf"].notna().sum())))
            mayoritarias.append(mayoritaria)

    pool = pd.concat(todos, ignore_index=True)
    escala_p = cs.pooled_scale(escalas)
    dec_p, base_p, reg_p = resumen_mes(pool, escala_p, 0)
    sumas_p = {k: sum(x[k] for x in sumas_todas) for k in sumas_todas[0]}
    acc_p = pr.resumen_mes("agrupado", pd.concat(tablas, ignore_index=True))
    filas.append(fila_csv("agrupado", dec_p, base_p, reg_p, sumas_p, acc_p))

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
    L += bloque_texto("AGRUPADO (todos los meses)", dec_p, base_p, reg_p, sumas_p)
    L += ["ACCION UNICA (todos los productos del dashboard: modelo y promedio)", "=" * 70,
          "Acierto de Comprar más = crecio mas de 15 % (clase real Reforzar); de No reponer "
          "= no vendio o cayo mas de 10 % (clase real Reducir). Incoherencias de monto: "
          "«se venderá: No» con un monto puntual > 0, No reponer con monto > 0 o Comprar más "
          "con monto menor al mes anterior (este ultimo caso es permitido: el monto es la "
          "venta esperada, que descuenta la probabilidad de venta).", ""]
    L += lineas_acc
    L += bloque_accion("AGRUPADO (todos los meses)", acc_p)
    L += ["RESUMEN DE SUMAS EN PESOS POR MES (millones de COP)", "-" * 70,
          f"{'Mes':<10}{'Pred. modelo':>14}{'Pred. mezcla':>14}{'Real (con pred.)':>18}{'Ingenuo':>10}"
          f"{'Media_3':>10}{'Ventas reales mes':>19}"]
    for f in filas:
        L.append(f"{f['mes']:<10}{f['suma_pred_modelo'] / 1e6:>14,.1f}"
                 f"{f['suma_pred_mezcla'] / 1e6:>14,.1f}"
                 f"{f['suma_real_con_pred'] / 1e6:>18,.1f}{f['suma_ingenuo'] / 1e6:>10,.1f}"
                 f"{f['suma_media_3'] / 1e6:>10,.1f}{f['ventas_reales_mes'] / 1e6:>19,.1f}")
    L.append("")
    L.append(f"Tiempo total: {time.time() - inicio:.1f} s")
    texto = "\n".join(L)
    print("\n" + texto)
    (REPORTS / "estudio_caso.txt").write_text(texto, encoding="utf-8")
    pd.DataFrame(filas).to_csv(REPORTS / "estudio_caso.csv", index=False)
    figura(filas[:-1], FIGURES / "estudio_caso.png")
    print(f"[OK] {REPORTS / 'estudio_caso.txt'}, estudio_caso.csv y {FIGURES / 'estudio_caso.png'}")


if __name__ == "__main__":
    main()
