"""
experimento_elegibilidad.py
===========================
Experimento (NO cambia el modelo de produccion): "elegibilidad ampliada".
Produccion solo predice productos con ventas en el ultimo mes. La variante
tambien predice productos sin ventas el ultimo mes pero con media_3 > 0
(media de los ultimos 3 meses, incluido el actual) y >= min_months meses con
ventas. Su decision usa media_3 como base del mes actual (etiquetas de
entrenamiento y clase real: razon = ventas del mes siguiente / media_3, mismos
umbrales 1,15 / 0,90; mes siguiente en cero => Reducir). Las filas de
produccion conservan su etiqueta; solo se agregan filas.

Mismo estudio de caso de origen rodante que estudio_caso.py (meses objetivo
2026-04..2026-09, entrenamiento estrictamente anterior a cada mes, evaluacion
contra las ventas REALES del holdout). Nada se elige con el holdout.
El MASE de ambas variantes usa la escala de la Base.

Genera:
  outputs/reports/experimento_elegibilidad.txt / .csv
  outputs/figures/experimento_elegibilidad.png

Uso:  MPLBACKEND=Agg PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/experimento_elegibilidad.py
Los resultados por mes se guardan en odd/elig_cache/ (sin versionar); borrar
esa carpeta para recalcular.
"""
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ventas_forecast import case_study as cs  # noqa: E402
from ventas_forecast import gap_sensitivity as gs  # noqa: E402
from ventas_forecast import improvements as im  # noqa: E402
from ventas_forecast.paths import FIGURES, REPORTS, ensure_output_dirs  # noqa: E402
import estudio_caso as ec  # noqa: E402

CACHE = ROOT / "odd" / "elig_cache"
VARIANTES = [("base", False, "Base"), ("ampliada", True, "Ampliada")]
SUBCONJUNTOS = ["todos", "previos", "nuevos"]
COLOR_REAL = "#52514e"
COLOR_BASE = "#eb6834"
COLOR_AMPL = "#1baf7a"


def _f(v, d=4):
    """Decimal con coma."""
    return f"{v:.{d}f}".replace(".", ",")


def _m(v):
    """Millones de COP con una cifra decimal y formato espanol."""
    return f"{v / 1e6:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _n(v):
    return f"{v:,.0f}".replace(",", ".")


def correr_mes(mes, tmp):
    """Base y Ampliada para un mes; devuelve un dict serializable."""
    ventas = cs.monthly_sales(ec.HOLDOUT, mes)
    out = {"mes": mes, "ventas_tienda": float(ventas.sum())}
    for clave, flag, _ in VARIANTES:
        res, escala, mayoritaria, model = ec.evaluar_mes(
            mes, tmp, con_modelo=True, elegibilidad_ampliada=flag)
        top = [str(i) for i in model._top_prods]
        out[clave] = {
            "res": res, "escala": escala, "mayoritaria": mayoritaria,
            "n_top": len(top),
            "real_top": float(ventas.reindex(top).fillna(0.0).sum()),
        }
    return out


def cargar_mes(mes):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{mes}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        d = correr_mes(mes, tmp)
    f.write_bytes(pickle.dumps(d))
    return d


def subconjunto(res, nombre):
    if nombre == "nuevos":
        return res[res["nuevo"]]
    if nombre == "previos":
        return res[~res["nuevo"]]
    return res


def metricas(sub, escala, mayoritaria):
    """Decisiones y montos de un subconjunto de filas de resultados."""
    fila = {"n_decisiones": len(sub)}
    if len(sub):
        dec = cs.classification_summary(sub["clase_real"], sub["clase_pred"])
        may = cs.majority_baseline_summary(sub["clase_real"], mayoritaria)
        fila.update({
            "f1_macro": dec["f1_macro"], "acc_balanceada": dec["balanced_accuracy"],
            "accuracy": dec["accuracy"], "accuracy_mayoritaria": may["accuracy"],
            "pct_reforzar_pred": gs.class_share(sub["clase_pred"], 2),
            "pct_reforzar_real": gs.class_share(sub["clase_real"], 2),
        })
    m = sub[sub["ventas_predichas_rf"].notna()]
    fila["n_montos"] = len(m)
    fila["suma_real"] = float(m["real"].sum())
    for nombre, col in (("rf", "ventas_predichas_rf"), ("mezcla", "ventas_predichas")):
        fila[f"suma_pred_{nombre}"] = float(m[col].sum())
        if len(m):
            r = cs.regression_summary(m["real"], m[col], escala)
            fila[f"mae_{nombre}"] = r["mae"]
            fila[f"rmse_{nombre}"] = r["rmse"]
            fila[f"mase_{nombre}"] = r["mase"]
            fila[f"ratio_{nombre}"] = r["sum_ratio"]
            fila[f"wape_{nombre}"] = im.wape(m["real"], m[col])
    return fila


def construir_filas(datos):
    """Filas largas (variante, mes, subconjunto) por mes y agrupadas. La escala
    de MASE es la de la Base de cada mes (agrupado: promedio ponderado de la Base)."""
    filas = []
    escalas = [(d["base"]["escala"], int(d["base"]["res"]["ventas_predichas_rf"].notna().sum()))
               for d in datos]
    escala_pool = cs.pooled_scale(escalas)
    for clave, _, etiqueta in VARIANTES:
        for d in datos:
            v, esc = d[clave], d["base"]["escala"]
            for sc in SUBCONJUNTOS:
                f = {"variante": etiqueta, "mes": d["mes"], "subconjunto": sc}
                f.update(metricas(subconjunto(v["res"], sc), esc, v["mayoritaria"]))
                if sc == "todos":
                    f["n_top"] = v["n_top"]
                    f["real_top1900"] = v["real_top"]
                    f["real_tienda"] = d["ventas_tienda"]
                filas.append(f)
        pool = pd.concat([d[clave]["res"] for d in datos], ignore_index=True)
        for sc in SUBCONJUNTOS:
            f = {"variante": etiqueta, "mes": "agrupado", "subconjunto": sc}
            f.update(metricas(subconjunto(pool, sc), escala_pool, 0))
            if sc == "todos":
                f["n_top"] = sum(d[clave]["n_top"] for d in datos)
                f["real_top1900"] = sum(d[clave]["real_top"] for d in datos)
                f["real_tienda"] = sum(d["ventas_tienda"] for d in datos)
            filas.append(f)
    return pd.DataFrame(filas)


def _g(df, var, mes, sc, col):
    r = df[(df["variante"] == var) & (df["mes"] == mes) & (df["subconjunto"] == sc)]
    return r[col].iloc[0] if len(r) else np.nan


def texto(df, meses):
    L = ["EXPERIMENTO: ELEGIBILIDAD AMPLIADA (origen rodante)", "=" * 70, "",
         "Regla: ademas de los productos con ventas el ultimo mes, se entrenan y predicen "
         "los que NO vendieron el ultimo mes pero tienen media_3 > 0 (media de los ultimos "
         "3 meses, incluido el actual) y >= 6 meses con ventas.",
         "Decision de esos productos: la base del 'mes actual' es media_3. Etiquetas de "
         "entrenamiento y clase real: razon = ventas del mes siguiente / media_3 con los "
         "mismos umbrales (1,15 / 0,90); mes siguiente en cero => Reducir. Las filas de "
         "produccion conservan su etiqueta; solo se agregan filas. Monto: RF y mezcla "
         "w*RF + (1-w)*media_3 como siempre (w de la validacion interna de cada mes).",
         "Evaluacion: mismos meses y datos que estudio_caso.py; MASE con la escala de la "
         "Base. 'previos' = productos que ya se predecian en la Base (ventas el ultimo "
         "mes); 'nuevos' = agregados por la variante. Septiembre del holdout esta incompleto.",
         "Nada se eligio con el holdout.", ""]
    cols = meses + ["agrupado"]
    L += ["COBERTURA (millones de COP)", "-" * 70,
          f"{'Mes':<10}{'Var.':<10}{'Decis.':>7}{'Montos':>7}{'Real c/monto':>13}"
          f"{'Real top1900':>13}{'Cobertura':>10}{'Real tienda':>12}"]
    for mes in cols:
        for _, _, var in VARIANTES:
            rt = _g(df, var, mes, "todos", "real_top1900")
            rc = _g(df, var, mes, "todos", "suma_real")
            L.append(f"{mes:<10}{var:<10}{_n(_g(df, var, mes, 'todos', 'n_decisiones')):>7}"
                     f"{_n(_g(df, var, mes, 'todos', 'n_montos')):>7}{_m(rc):>13}{_m(rt):>13}"
                     f"{_f(rc / rt, 3):>10}{_m(_g(df, var, mes, 'todos', 'real_tienda')):>12}")
    L += ["", "TOTALES PREDICHOS VS REAL DE TODOS LOS PRODUCTOS TOP 1900 (millones de COP)",
          "-" * 70,
          f"{'Mes':<10}{'Real top1900':>13}{'Base mezcla':>13}{'Ampl. mezcla':>13}"
          f"{'Base RF':>10}{'Ampl. RF':>10}{'B/real':>8}{'A/real':>8}"]
    for mes in cols:
        rb = _g(df, "Base", mes, "todos", "real_top1900")
        ra = _g(df, "Ampliada", mes, "todos", "real_top1900")
        pb = _g(df, "Base", mes, "todos", "suma_pred_mezcla")
        pa = _g(df, "Ampliada", mes, "todos", "suma_pred_mezcla")
        L.append(f"{mes:<10}{_m(rb):>13}{_m(pb):>13}{_m(pa):>13}"
                 f"{_m(_g(df, 'Base', mes, 'todos', 'suma_pred_rf')):>10}"
                 f"{_m(_g(df, 'Ampliada', mes, 'todos', 'suma_pred_rf')):>10}"
                 f"{_f(pb / rb, 3):>8}{_f(pa / ra, 3):>8}")
    L.append("(El real de cada variante es el de SU top 1900, elegido con su entrenamiento; "
             "se muestra el de la Base. B/real y A/real usan el real de cada variante.)")
    for sc, titulo in (("todos", "TODOS LOS PRODUCTOS PREDICHOS"),
                       ("previos", "SOLO PRODUCTOS PREVIOS (ya predichos en la Base)"),
                       ("nuevos", "SOLO PRODUCTOS NUEVOS (sin ventas el ultimo mes)")):
        L += ["", f"DECISIONES - {titulo}", "-" * 70,
              f"{'Mes':<10}{'Var.':<10}{'n':>6}{'F1 macro':>10}{'Acc.bal.':>10}{'Accuracy':>10}"
              f"{'Mayorit.':>10}{'%Ref.pred':>10}{'%Ref.real':>10}"]
        for mes in cols:
            for _, _, var in VARIANTES:
                n = _g(df, var, mes, sc, "n_decisiones")
                if not n:
                    continue
                L.append(f"{mes:<10}{var:<10}{_n(n):>6}"
                         f"{_f(_g(df, var, mes, sc, 'f1_macro')):>10}"
                         f"{_f(_g(df, var, mes, sc, 'acc_balanceada')):>10}"
                         f"{_f(_g(df, var, mes, sc, 'accuracy')):>10}"
                         f"{_f(_g(df, var, mes, sc, 'accuracy_mayoritaria')):>10}"
                         f"{_f(100 * _g(df, var, mes, sc, 'pct_reforzar_pred'), 1):>10}"
                         f"{_f(100 * _g(df, var, mes, sc, 'pct_reforzar_real'), 1):>10}")
        L += ["", f"MONTOS - {titulo}", "-" * 70,
              f"{'Mes':<10}{'Var.':<10}{'Pred.':<7}{'n':>6}{'MAE':>10}{'RMSE':>12}{'MASE':>8}"
              f"{'WAPE':>8}{'suma p/r':>10}{'pred M':>10}{'real M':>10}"]
        for mes in cols:
            for _, _, var in VARIANTES:
                n = _g(df, var, mes, sc, "n_montos")
                if not n:
                    continue
                for nombre in ("rf", "mezcla"):
                    def g(c):
                        return _g(df, var, mes, sc, f"{c}_{nombre}")
                    L.append(f"{mes:<10}{var:<10}{nombre:<7}{_n(n):>6}{_n(g('mae')):>10}"
                             f"{_n(g('rmse')):>12}{_f(g('mase')):>8}{_f(g('wape')):>8}"
                             f"{_f(g('ratio')):>10}{_m(g('suma_pred')):>10}"
                             f"{_m(_g(df, var, mes, sc, 'suma_real')):>10}")
    return L


def figura(df, meses, salida):
    x = np.arange(len(meses))
    real = [_g(df, "Base", m, "todos", "real_top1900") / 1e6 for m in meses]
    base = [_g(df, "Base", m, "todos", "suma_pred_mezcla") / 1e6 for m in meses]
    ampl = [_g(df, "Ampliada", m, "todos", "suma_pred_mezcla") / 1e6 for m in meses]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(x, real, color=COLOR_REAL, marker="o", linewidth=2.4,
            label="Ventas reales de los productos top 1900")
    ax.plot(x, base, color=COLOR_BASE, marker="s", linewidth=2,
            label="Predicción de la Base (mezcla)")
    ax.plot(x, ampl, color=COLOR_AMPL, marker="^", linewidth=2,
            label="Predicción con elegibilidad ampliada (mezcla)")
    ax.set_xticks(x)
    ax.set_xticklabels(meses)
    ax.set_ylim(0, max(real + base + ampl) * 1.2)
    ax.set_ylabel("Millones de COP por mes", color="#52514e")
    ax.set_xlabel("Mes predicho", color="#52514e")
    ax.tick_params(colors="#52514e")
    for s in ax.spines.values():
        s.set_color("#e1e0d9")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(salida, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    inicio = time.time()
    ensure_output_dirs()
    datos = [cargar_mes(m) for m in ec.MESES]
    df = construir_filas(datos)
    L = texto(df, ec.MESES)
    L.append("")
    L.append(f"Tiempo total: {time.time() - inicio:.1f} s")
    t = "\n".join(L)
    print(t)
    (REPORTS / "experimento_elegibilidad.txt").write_text(t, encoding="utf-8")
    df.to_csv(REPORTS / "experimento_elegibilidad.csv", index=False)
    figura(df, ec.MESES, FIGURES / "experimento_elegibilidad.png")


if __name__ == "__main__":
    main()
