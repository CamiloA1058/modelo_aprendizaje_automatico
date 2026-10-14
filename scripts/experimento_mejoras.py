"""
experimento_mejoras.py
======================
Experimento (NO cambia el modelo de produccion): mide en el estudio de caso
fuera de muestra (mismos meses, filas y valores reales que estudio_caso.py)
si cuatro opciones acercan la prediccion en pesos a la realidad, sobre todo la
suma mensual.

  Base     : Random Forest regresor (log1p), el de produccion.
  Opcion 1 : reconciliacion de arriba hacia abajo. Total del mes = suma de la
             media_3 de los productos con prediccion; se reparte en proporcion
             a las predicciones del RF. Variante b: el total sale de la media
             de los ultimos 3 meses de la serie agregada de esos productos.
  Opcion 2 : otra perdida sobre pesos crudos, mismas variables y misma
             particion que train_regressor: (a) HistGradientBoostingRegressor
             con loss="poisson", (b) RandomForestRegressor sin log.
  Opcion 3 : mezcla w*RF + (1-w)*media_3; w en {0,0.1,...,1} elegido SOLO con la
             particion de prueba interna del entrenamiento de cada mes (la del
             propio modelo): (3) minimizando el MAE; (3b) minimizando el sesgo de
             la suma |suma pred - suma real|. Nunca con el mes objetivo.
  Opcion 4 : intervalos 80 % del RF: (a) cuantiles 10/90 de los arboles en
             escala log, (b) conformal dividido con los residuos de la
             validacion interna (escala log).

Nada se ajusto con el mes objetivo (holdout). Hiperparametros fijos y simples.

Genera:
  outputs/reports/experimento_mejoras.txt
  outputs/reports/experimento_mejoras.csv
  outputs/figures/experimento_mejoras.png

Uso:  MPLBACKEND=Agg PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/experimento_mejoras.py
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

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ventas_forecast import case_study as cs  # noqa: E402
from ventas_forecast import improvements as im  # noqa: E402
from ventas_forecast.paths import FIGURES, REPORTS, ensure_output_dirs  # noqa: E402
import estudio_caso as ec  # noqa: E402

ALPHA = 0.20  # intervalos de 80 %

VARIANTES = [
    ("base", "Base: RF (log1p)"),
    ("media_3", "Media de 3 meses"),
    ("op1", "Op.1 Arriba-abajo (total media_3)"),
    ("op1b", "Op.1b Arriba-abajo (total serie agregada)"),
    ("op2a", "Op.2a HistGB Poisson (pesos)"),
    ("op2b", "Op.2b RF sin log (pesos)"),
    ("op3", "Op.3 Mezcla, w por MAE (valid.)"),
    ("op3b", "Op.3b Mezcla, w por sesgo de la suma"),
]
INTERVALOS = [("op4a", "Op.4a Cuantiles de arboles 10-90"),
              ("op4b", "Op.4b Conformal 80 %")]


def procesar_mes(mes, tmp):
    res, escala, _, model = ec.evaluar_mes(mes, tmp, con_modelo=True)
    c = model._c
    feats = model._features
    m = res[res["ventas_predichas_rf"].notna()].copy()

    # Entrenamiento y validacion interna (igual que train_regressor)
    d = model._df_train
    es_top = d[c("id")].isin(model._top_prods)
    ok = d["target"].notna()
    d_tr = d[d.index.isin(model._train_idx) & es_top & ok]
    d_va = d[d.index.isin(model._test_idx) & es_top & ok]
    Xtr = model.scaler.transform(model._X.loc[d_tr.index])
    Xva = model.scaler.transform(model._X.loc[d_va.index])
    Xte = model.scaler.transform(model.df_predict.loc[m.index, feats].fillna(0))
    y_tr = d_tr["target"].to_numpy()

    # Predicciones puntuales sobre el mes objetivo
    # "base" = RF puro: ventas_predichas ya es la mezcla del sistema.
    pred = {"base": m["ventas_predichas_rf"].to_numpy(), "media_3": m["media_3"].to_numpy()}
    pred["op1"] = im.topdown_reconcile(m["media_3"].sum(), pred["base"])

    # Op.1b: media de los ultimos 3 meses de la serie agregada de los mismos productos
    ids = set(m["id"])
    dfm = model.df[model.df[c("id")].astype(str).isin(ids)]
    serie = dfm.groupby("fecha")[c("total_sold")].sum().sort_index()
    total_b = float(serie.iloc[-3:].mean())
    pred["op1b"] = im.topdown_reconcile(total_b, pred["base"])
    dif_1_1b = total_b - float(m["media_3"].sum())

    hgb = HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, random_state=42)
    hgb.fit(Xtr, np.clip(y_tr, 0, None))
    pred["op2a"] = np.clip(hgb.predict(Xte), 0, None)
    rfc = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rfc.fit(Xtr, y_tr)
    pred["op2b"] = rfc.predict(Xte)

    # Opcion 3: w con la validacion interna (nunca el mes objetivo)
    real_va = d_va["target"].to_numpy()
    rf_va = np.expm1(model.reg.predict(Xva))
    m3_va = d_va["media_3_actual"].to_numpy()
    m3_va = np.where(np.isnan(m3_va), d_va[c("total_sold")].to_numpy(), m3_va)
    w = im.choose_blend_weight(real_va, rf_va, m3_va)
    pred["op3"] = im.blend(w, pred["base"], pred["media_3"])
    w_b = im.choose_blend_weight(real_va, rf_va, m3_va, criterion="sesgo")
    pred["op3b"] = im.blend(w_b, pred["base"], pred["media_3"])

    # Opcion 4: intervalos del RF
    arboles = np.array([t.predict(Xte) for t in model.reg.estimators_])
    lo_a, hi_a = im.tree_quantiles(arboles, ALPHA / 2, 1 - ALPHA / 2)
    q = im.conformal_quantile(np.log1p(real_va) - model.reg.predict(Xva), ALPHA)
    centro = np.log1p(pred["base"])
    ints = {
        "op4a": (np.expm1(lo_a), np.expm1(hi_a)),
        "op4b": (np.clip(np.expm1(centro - q), 0, None), np.expm1(centro + q)),
    }

    out = pd.DataFrame({"mes": mes, "id": m["id"].to_numpy(), "real": m["real"].to_numpy()})
    for k, v in pred.items():
        out[k] = v
    for k, (lo, hi) in ints.items():
        out[k + "_lo"], out[k + "_hi"] = lo, hi
    info = {"mes": mes, "w": w, "w_b": w_b, "q_conformal": q, "n_val": len(d_va),
            "dif_op1_op1b": dif_1_1b, "total_op1b": total_b}
    return out, escala, info


def metricas(real, pred, escala):
    r = cs.regression_summary(real, pred, escala)
    real = np.asarray(real, float)
    pred = np.asarray(pred, float)
    return {"mae": r["mae"], "rmse": r["rmse"], "mase": r["mase"],
            "wape": im.wape(real, pred), "sum_ratio": r["sum_ratio"],
            "suma_pred": float(pred.sum()), "suma_real": float(real.sum()), "n": r["n"]}


def main():
    inicio = time.time()
    ensure_output_dirs()
    frames, escalas, infos = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for mes in ec.MESES:
            out, escala, info = procesar_mes(mes, tmp)
            frames.append(out)
            escalas.append((escala, len(out)))
            infos.append(info)
            print(f"[{mes}] n={len(out)} w={info['w']} q_conformal={info['q_conformal']:.4f}")

    escala_p = cs.pooled_scale(escalas)
    pool = pd.concat(frames, ignore_index=True)
    escala_de = {mes: e for mes, (e, _) in zip(ec.MESES, escalas)}
    etiquetas = dict(VARIANTES)

    filas = []
    for clave, _ in VARIANTES:
        for mes, f in zip(ec.MESES, frames):
            filas.append({"mes": mes, "variante": clave,
                          **metricas(f["real"], f[clave], escala_de[mes])})
        filas.append({"mes": "agrupado", "variante": clave,
                      **metricas(pool["real"], pool[clave], escala_p)})
    tabla = pd.DataFrame(filas)

    filas_int = []
    for clave, _ in INTERVALOS:
        for mes, f in zip(ec.MESES + ["agrupado"], frames + [pool]):
            lo, hi = f[clave + "_lo"], f[clave + "_hi"]
            sr = float(f["real"].sum())
            slo, shi = float(lo.sum()), float(hi.sum())
            filas_int.append({
                "mes": mes, "variante": clave,
                "cobertura": im.interval_coverage(f["real"], lo, hi),
                "ancho_medio": im.interval_width(lo, hi),
                "suma_real": sr, "suma_inferior": slo, "suma_superior": shi,
                "total_cubierto": bool(slo <= sr <= shi), "n": len(f)})
    tint = pd.DataFrame(filas_int)

    # Chequeo contra estudio_caso.csv: la base debe reproducirlo
    ref = pd.read_csv(REPORTS / "estudio_caso.csv").set_index("mes")
    chk = []
    for mes in ec.MESES + ["agrupado"]:
        b = tabla[(tabla.mes == mes) & (tabla.variante == "base")].iloc[0]
        m3 = tabla[(tabla.mes == mes) & (tabla.variante == "media_3")].iloc[0]
        chk.append(f"  {mes:<9} MAE base {b['mae']:>10,.0f} vs estudio_caso {ref.loc[mes, 'mae_modelo']:>10,.0f}"
                   f" | MAE media_3 {m3['mae']:>10,.0f} vs {ref.loc[mes, 'mae_media_3']:>10,.0f}")
        if abs(b["mae"] - ref.loc[mes, "mae_modelo"]) >= 1.0:
            raise RuntimeError(f"La base no reproduce estudio_caso en {mes}")

    L = ["EXPERIMENTO DE MEJORAS DE LA PREDICCION EN PESOS (origen rodante)", "=" * 100, "",
         "Mismos meses (2026-04 a 2026-09), filas y valores reales que estudio_caso.py; el "
         "entrenamiento de cada mes usa solo datos anteriores al mes.",
         "Nada se ajusto ni se eligio con el mes objetivo (holdout): los hiperparametros de las "
         "opciones 2 son fijos; el peso w (opcion 3) y el cuantil conformal (opcion 4) salen de la "
         "particion de prueba interna del entrenamiento de cada mes.",
         "Hiperparametros: HistGB Poisson (max_iter 300, learning_rate 0,05, max_leaf_nodes 31, "
         "min_samples_leaf 20); RF sin log (200 arboles, max_depth 12), igual que el de produccion "
         "salvo el objetivo.",
         "Clasificador: no cambia en ninguna opcion (decisiones y metricas como en estudio_caso).",
         "Productos con prediccion de monto. MASE con la escala de estudio_caso (agrupado: escala "
         "ponderada). WAPE = suma|real-pred| / suma(real). Razon = suma pred / suma real.",
         "Nota: el holdout llega hasta el 22 de septiembre; las ventas reales de septiembre estan "
         "incompletas.", "",
         "Chequeo contra estudio_caso.csv (la base debe reproducirlo):"] + chk + [""]

    def tabla_texto(titulo, sub):
        T = [titulo, "-" * 100,
             f"{'Variante':<44}{'MAE':>10}{'RMSE':>10}{'MASE':>8}{'WAPE':>8}{'Razon':>8}"
             f"{'Pred (M)':>11}{'Real (M)':>11}"]
        for clave, _ in VARIANTES:
            r = sub[sub.variante == clave].iloc[0]
            T.append(f"{etiquetas[clave]:<44}{r['mae']:>10,.0f}{r['rmse']:>10,.0f}{r['mase']:>8.4f}"
                     f"{r['wape']:>8.4f}{r['sum_ratio']:>8.4f}{r['suma_pred'] / 1e6:>11,.1f}"
                     f"{r['suma_real'] / 1e6:>11,.1f}")
        T.append("")
        return T

    L += tabla_texto("AGRUPADO (todos los meses)", tabla[tabla.mes == "agrupado"])
    L += ["Peso w elegido por mes con la validacion interna (opcion 3: criterio MAE; 3b: criterio sesgo de la suma):"]
    for i in infos:
        L.append(f"  {i['mes']}: w(3) = {i['w']:.1f} | w(3b) = {i['w_b']:.1f} (n validacion {i['n_val']:,})")
    L += ["", "Opcion 1 vs 1b, total de la serie agregada menos suma de media_3 por mes (COP): "
          + ", ".join(f"{i['mes']}: {i['dif_op1_op1b']:,.0f}" for i in infos), ""]

    L += ["SUMA PREDICHA VS REAL POR MES (millones de COP)", "-" * 100,
          f"{'Mes':<10}{'Real':>9}" + "".join(f"{k:>10}" for k, _ in VARIANTES)]
    for mes in ec.MESES + ["agrupado"]:
        sub = tabla[tabla.mes == mes]
        fila = f"{mes:<10}{sub.iloc[0]['suma_real'] / 1e6:>9,.1f}"
        for k, _ in VARIANTES:
            fila += f"{sub[sub.variante == k].iloc[0]['suma_pred'] / 1e6:>10,.1f}"
        L.append(fila)
    L.append("")
    for mes in ec.MESES:
        L += tabla_texto(f"MES {mes}", tabla[tabla.mes == mes])

    L += ["INTERVALOS DE 80 % DEL RF (opcion 4)", "-" * 100,
          "Cobertura = % de productos cuyo real cae en [inferior, superior]; ancho medio en COP.",
          "Cobertura del TOTAL: la suma de limites inferiores y la de superiores es una cota BURDA "
          "(suma de cuantiles, no cuantil de la suma); es demasiado ancha y solo indica si el total "
          "real cabe.",
          f"{'Variante':<36}{'Mes':<10}{'Cobertura':>10}{'Ancho medio':>13}{'Suma inf (M)':>14}"
          f"{'Suma sup (M)':>14}{'Real (M)':>10}{'Total en cota':>15}"]
    for clave, et in INTERVALOS:
        for _, r in tint[tint.variante == clave].iterrows():
            L.append(f"{et:<36}{r['mes']:<10}{r['cobertura'] * 100:>9.1f}%{r['ancho_medio']:>13,.0f}"
                     f"{r['suma_inferior'] / 1e6:>14,.1f}{r['suma_superior'] / 1e6:>14,.1f}"
                     f"{r['suma_real'] / 1e6:>10,.1f}{'si' if r['total_cubierto'] else 'no':>15}")
    L += ["", f"Tiempo total: {time.time() - inicio:.1f} s"]
    texto = "\n".join(L)
    print("\n" + texto)
    (REPORTS / "experimento_mejoras.txt").write_text(texto, encoding="utf-8")

    t1 = tabla.copy()
    t1["w_elegido"] = np.nan
    for var, clave in (("op3", "w"), ("op3b", "w_b")):
        pesos = {i["mes"]: i[clave] for i in infos}
        mask = t1.variante == var
        t1.loc[mask, "w_elegido"] = t1.loc[mask, "mes"].map(pesos)
    t1["tipo"] = "puntual"
    tint["tipo"] = "intervalo"
    pd.concat([t1, tint], ignore_index=True).to_csv(REPORTS / "experimento_mejoras.csv", index=False)

    figura(tabla, ec.MESES, FIGURES / "experimento_mejoras.png")
    print(f"[OK] experimento_mejoras.txt/.csv y {FIGURES / 'experimento_mejoras.png'}")


def figura(tabla, meses, salida):
    """Suma mensual predicha vs real: base, media_3 y las opciones 1, 2 (la de
    menor MAE agrupado) y 3b (la 3 por MAE coincide con la base)."""
    agr = tabla[tabla.mes == "agrupado"].set_index("variante")
    mejor2 = min(("op2a", "op2b"), key=lambda k: agr.loc[k, "mae"])
    series = [("base", "Base (Random Forest)", "#1baf7a", "o"),
              ("media_3", "Media de 3 meses", "#4a3aa7", "s"),
              ("op1", "Arriba-abajo (total media_3)", "#eb6834", "^"),
              (mejor2, "Opcion 2: " + ("HistGB Poisson" if mejor2 == "op2a" else "RF sin log"),
               "#2a78d6", "D"),
              ("op3b", "Mezcla RF + media_3 (w por sesgo)", "#c0a000", "v")]
    x = np.arange(len(meses))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.patch.set_facecolor(ec.COLOR_SURFACE)
    ax.set_facecolor(ec.COLOR_SURFACE)
    ax.grid(axis="y", color=ec.COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    real = [tabla[(tabla.mes == m) & (tabla.variante == "base")].iloc[0]["suma_real"] / 1e6
            for m in meses]
    ax.plot(x, real, color="#111111", linewidth=2.6, marker="o", label="Real")
    for clave, et, color, marcador in series:
        y = [tabla[(tabla.mes == m) & (tabla.variante == clave)].iloc[0]["suma_pred"] / 1e6
             for m in meses]
        ax.plot(x, y, color=color, linewidth=1.8, marker=marcador, label=et)
    ax.set_xticks(x)
    ax.set_xticklabels(meses)
    ax.set_xlabel("Mes predicho", color=ec.COLOR_INK)
    ax.set_ylabel("Suma mensual en millones de COP", color=ec.COLOR_INK)
    ax.tick_params(colors=ec.COLOR_INK)
    for s in ax.spines.values():
        s.set_color(ec.COLOR_GRID)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(salida, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
