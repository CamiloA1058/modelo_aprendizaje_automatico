"""
experimento_grupos.py
=====================
Experimento (NO cambia el modelo de produccion): prediccion mensual por grupo
de producto (PRODUCTO.IDGRUPO) para apoyar compras. Meta del usuario: ~70 %.

Mismo estudio de caso de origen rodante que estudio_caso.py (meses objetivo
2026-04 a 2026-09; septiembre llega al 22). Para cada origen se entrena solo con
datos estrictamente anteriores al mes objetivo; nada se ajusta con el holdout:
el peso de la mezcla del modelo directo se elige con una validacion interna
(ultimos 3 meses conocidos) y el de las mezclas G3 con los meses ya cerrados.

Universo (objetivo "universo"): los productos que muestra el dashboard de cada
mes (modelo con monto + estimacion por promedio), sumados por grupo. Objetivo
"todos": ventas reales de todos los productos del grupo, incluidos los que el
sistema no predice (muestra la cobertura).

Variantes:
  G0   suma de las predicciones por producto de produccion
  G1   media de los ultimos 3 meses del grupo
  G2   modelo directo (RF) sobre la serie mensual del grupo (log1p) mezclado con
       G1; G2x lo mismo con XGBoost
  G3   mezcla de G0 y G2 (G3b: de G0 y G1) con el peso que minimiza el error en
       los meses ya cerrados (0,5 sin historia); G3f: 50/50 fijo
  G4   G0 reconciliada al total del grupo de G2 (G4b: de G1)

Genera: outputs/reports/experimento_grupos.txt / .csv y
outputs/figures/experimento_grupos.png
Uso:  MPLBACKEND=Agg PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/experimento_grupos.py
Reutiliza la cache de estados de odd/monto_cache/ (la crea si falta).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ventas_forecast import grupos as gr  # noqa: E402
from ventas_forecast.paths import (  # noqa: E402
    DATA_RAW, DEFAULT_DATASET, FIGURES, REPORTS, ensure_output_dirs,
)
import estudio_caso as ec  # noqa: E402
import experimento_monto_producto as emp  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

META = 0.70
MAPA = DATA_RAW / "grupos_productos.csv"
VARIANTES_UNI = ["G0", "G1", "G2", "G2x", "G3", "G3b", "G3f", "G4", "G4b"]
VARIANTES_TODOS = ["G0", "G1", "G2", "G2x"]
ETIQ = {
    "G0": "G0: suma de productos (producción)", "G1": "G1: media de 3 meses del grupo",
    "G2": "G2: modelo directo (RF)", "G2x": "G2x: modelo directo (XGBoost)",
    "G3": "G3: mezcla G0 y G2", "G3b": "G3b: mezcla G0 y G1",
    "G3f": "G3f: mezcla 50/50 G0 y G2", "G4": "G4: G0 reconciliada a G2",
    "G4b": "G4b: G0 reconciliada a G1",
}
COLORES = {
    "G0": "#52514e", "G1": "#8b5e3c", "G2": "#1baf7a", "G2x": "#2a78d6",
    "G3": "#4a3aa7", "G3b": "#a79be0", "G3f": "#d55181", "G4": "#eb6834",
    "G4b": "#f5b08f",
}
FIGURA = ["G0", "G1", "G2", "G3", "G3f"]
NOMBRE_MES = {m: m for m in ec.MESES}


def _p(v, d=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else \
        f"{100 * v:.{d}f} %".replace(".", ",")


def _n(v):
    return f"{v:,.0f}".replace(",", ".")


def xgb_grupo():
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, n_jobs=-1)


def _dir(panel, estimador):
    return gr.predecir_directo(panel, estimador=estimador)


def evaluar_origen(mes, ventas, mapa, estado, real_mes):
    """Predicciones por grupo de G0, G1, G2, G2x para un origen, con los dos
    objetivos, y la tabla por producto del universo."""
    r = estado["results"]
    r = r[r["ventas_predichas"].notna()].copy()
    cid = estado["id"]
    r["codigo"] = r[cid].astype(str)
    r["grupo"] = gr.asignar_grupo(r["codigo"], mapa)
    r["real"] = real_mes.reindex(r["codigo"]).fillna(0.0).to_numpy()
    r["pred"] = r["ventas_predichas"].astype(float)

    # objetivo universo
    g_uni = sorted(r["grupo"].unique())
    p_uni = gr.panel_grupo(ventas, mapa, mes, productos=r["codigo"])
    d_uni = _dir(p_uni, gr.rf_grupo)
    d_x = _dir(p_uni, xgb_grupo)
    uni = pd.DataFrame(index=g_uni)
    uni["real"] = r.groupby("grupo")["real"].sum().reindex(g_uni)
    uni["G0"] = r.groupby("grupo")["pred"].sum().reindex(g_uni)
    uni["G1"] = gr.media_3(p_uni).reindex(g_uni).fillna(0.0)
    uni["G2"] = d_uni["pred"].reindex(g_uni).fillna(0.0)
    uni["G2x"] = d_x["pred"].reindex(g_uni).fillna(0.0)

    # objetivo todos los productos del grupo
    mapa_real = pd.DataFrame({"codigo": real_mes.index.astype(str), "real": real_mes.to_numpy()})
    mapa_real["grupo"] = gr.asignar_grupo(mapa_real["codigo"], mapa)
    real_todos = mapa_real.groupby("grupo")["real"].sum()
    g_todos = sorted(set(g_uni) | set(real_todos[real_todos > 0].index))
    p_todos = gr.panel_grupo(ventas, mapa, mes)
    d_t = _dir(p_todos, gr.rf_grupo)
    d_tx = _dir(p_todos, xgb_grupo)
    todos = pd.DataFrame(index=g_todos)
    todos["real"] = real_todos.reindex(g_todos).fillna(0.0)
    todos["G0"] = uni["G0"].reindex(g_todos).fillna(0.0)
    todos["G1"] = gr.media_3(p_todos).reindex(g_todos).fillna(0.0)
    todos["G2"] = d_t["pred"].reindex(g_todos).fillna(0.0)
    todos["G2x"] = d_tx["pred"].reindex(g_todos).fillna(0.0)

    diag = {
        "peso_rf": d_uni["peso"], "peso_xgb": d_x["peso"], "peso_rf_todos": d_t["peso"],
        "ultimo_mes_panel": float(p_uni["total"].iloc[-1].sum()),
        "ultimo_mes_modelo": float(r[estado["tot"]].sum()),
        "n_productos": int(len(r)), "n_grupos": len(g_uni),
        "n_sin_grupo": int((r["grupo"] == gr.SIN_GRUPO).sum()),
    }
    return uni, todos, r, diag


def metricas_filas(real, pred, top=20):
    return gr.resumen(real, pred, top=top)


def _mascara_top(real, top=20):
    m = np.zeros(len(real), bool)
    m[np.argsort(-np.asarray(real, float), kind="stable")[:top]] = True
    return m


def metricas_producto(real, pred):
    r, p = np.asarray(real, float), np.asarray(pred, float)
    e = gr.exactitud_grupo(r, p)
    return {"ponderado": gr.acierto_ponderado(r, p), "media": float(np.nanmean(e)),
            "pct_70": float(np.mean(e[r > 0] >= 0.7)), "razon": float(p.sum() / r.sum())}


def construir(mapa, ventas, estados, reales):
    """Recorre los origenes en orden y arma las variantes dependientes de la
    historia (G3) y de la reconciliacion (G4)."""
    por_mes_uni, por_mes_todos, prod, diags = {}, {}, {}, {}
    hist = {"real": [], "g0": [], "g1": [], "g2": []}
    for mes in ec.MESES:
        uni, todos, r, diag = evaluar_origen(mes, ventas, mapa, estados[mes], reales[mes])
        w3 = gr.peso_online(hist["real"], hist["g0"], hist["g2"])
        w3b = gr.peso_online(hist["real"], hist["g0"], hist["g1"])
        uni["G3"] = gr.mezcla(w3, uni["G0"], uni["G2"])
        uni["G3b"] = gr.mezcla(w3b, uni["G0"], uni["G1"])
        uni["G3f"] = gr.mezcla(0.5, uni["G0"], uni["G2"])
        r["G0"] = r["pred"]
        r["G4"] = gr.reconciliar(r["pred"], r["grupo"], uni["G2"]).to_numpy()
        r["G4b"] = gr.reconciliar(r["pred"], r["grupo"], uni["G1"]).to_numpy()
        uni["G4"] = r.groupby("grupo")["G4"].sum().reindex(uni.index).fillna(0.0)
        uni["G4b"] = r.groupby("grupo")["G4b"].sum().reindex(uni.index).fillna(0.0)
        diag.update({"w_g3": w3, "w_g3b": w3b})
        hist["real"].append(uni["real"].to_numpy())
        hist["g0"].append(uni["G0"].to_numpy())
        hist["g1"].append(uni["G1"].to_numpy())
        hist["g2"].append(uni["G2"].to_numpy())
        por_mes_uni[mes], por_mes_todos[mes], prod[mes], diags[mes] = uni, todos, r, diag
        print(f"[{mes}] grupos {diag['n_grupos']} productos {diag['n_productos']} "
              f"pesos RF {diag['peso_rf']:.1f} XGB {diag['peso_xgb']:.1f} G3 {w3:.1f} G3b {w3b:.1f}",
              flush=True)
    return por_mes_uni, por_mes_todos, prod, diags


def tabla_metricas(por_mes, variantes, objetivo):
    filas = []
    for v in variantes:
        reales, preds, tops = [], [], []
        for mes, df in por_mes.items():
            m = gr.resumen(df["real"], df[v])
            filas.append({"objetivo": objetivo, "variante": v, "mes": mes, **m})
            reales.append(df["real"].to_numpy())
            preds.append(df[v].to_numpy())
            tops.append(_mascara_top(df["real"].to_numpy()))
        R, P, T = np.concatenate(reales), np.concatenate(preds), np.concatenate(tops)
        pool = gr.resumen(R, P)
        pool["top"] = gr.acierto_ponderado(R[T], P[T])
        filas.append({"objetivo": objetivo, "variante": v, "mes": "agrupado", **pool})
    return pd.DataFrame(filas)


def ic_bootstrap(por_mes, v, n=2000, semilla=7):
    """IC 95 % del 1-WAPE agrupado remuestreando grupo-meses."""
    R = np.concatenate([d["real"].to_numpy() for d in por_mes.values()])
    P = np.concatenate([d[v].to_numpy() for d in por_mes.values()])
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(R), size=(n, len(R)))
    a = 1 - np.abs(R[idx] - P[idx]).sum(1) / R[idx].sum(1)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def g(df, objetivo, v, mes, col):
    return float(df[(df["objetivo"] == objetivo) & (df["variante"] == v) & (df["mes"] == mes)][col].iloc[0])


def tabla_texto(df, objetivo, variantes):
    cab = f"{'Variante':<38}" + "".join(f"{m[5:]:>8}" for m in ec.MESES) + f"{'Agrupado':>10}{'Meses>=70':>11}"
    L = [cab]
    for v in variantes:
        vals = [g(df, objetivo, v, m, "ponderado") for m in ec.MESES]
        n70 = sum(x >= META for x in vals)
        L.append(f"{ETIQ[v]:<38}" + "".join(f"{_p(x):>8}" for x in vals)
                 + f"{_p(g(df, objetivo, v, 'agrupado', 'ponderado')):>10}{n70:>8} / 6")
    return L


def tabla_otras(df, objetivo, variantes):
    L = [f"{'Variante':<38}{'Media grupo':>12}{'Grupos>=70%':>13}{'Top-20':>9}{'Suma P/R':>10}"]
    for v in variantes:
        L.append(f"{ETIQ[v]:<38}{_p(g(df, objetivo, v, 'agrupado', 'media')):>12}"
                 f"{_p(g(df, objetivo, v, 'agrupado', 'pct_70')):>13}"
                 f"{_p(g(df, objetivo, v, 'agrupado', 'top')):>9}"
                 f"{g(df, objetivo, v, 'agrupado', 'razon'):>10.3f}".replace(".", ","))
    return L


def texto(df, uni, todos, prod, diags):
    L = ["EXPERIMENTO: PREDICCION MENSUAL POR GRUPO DE PRODUCTO", "=" * 60,
         "Origen rodante, meses objetivo 2026-04 a 2026-09 (septiembre incompleto, al 22).",
         "Acierto ponderado = 1 - WAPE entre grupos (sum|R-P| / sum R). Meta del usuario: 70 %.", ""]
    L += ["Diagnostico por origen:"]
    for mes, d in diags.items():
        L.append(f"  {mes}: {d['n_productos']} productos, {d['n_grupos']} grupos "
                 f"({d['n_sin_grupo']} productos sin grupo); peso RF {d['peso_rf']:.1f}, "
                 f"XGB {d['peso_xgb']:.1f}, G3 {d['w_g3']:.1f}, G3b {d['w_g3b']:.1f}; "
                 f"ventas del ultimo mes: panel {_n(d['ultimo_mes_panel'])} vs modelo "
                 f"{_n(d['ultimo_mes_modelo'])} COP")
    L += ["", "A. UNIVERSO (productos del dashboard de cada mes, sumados por grupo)",
          "Acierto ponderado por valor (1 - WAPE) por mes y agrupado:"]
    L += tabla_texto(df, "universo", VARIANTES_UNI)
    L += ["", "Otras metricas agrupadas:"] + tabla_otras(df, "universo", VARIANTES_UNI)
    L += ["", "IC 95 % del acierto ponderado agrupado (remuestreo de grupo-meses):"]
    for v in VARIANTES_UNI:
        lo, hi = ic_bootstrap(uni, v)
        L.append(f"  {ETIQ[v]:<38}{_p(lo)} a {_p(hi)}")
    L += ["", "Acierto ponderado de los 20 grupos con mas ventas, por mes (universo):"]
    cab = f"{'Variante':<38}" + "".join(f"{m[5:]:>8}" for m in ec.MESES)
    L.append(cab)
    for v in VARIANTES_UNI:
        L.append(f"{ETIQ[v]:<38}" + "".join(f"{_p(g(df, 'universo', v, m, 'top')):>8}" for m in ec.MESES))
    L += ["", "Acierto medio por grupo, por mes (universo):", cab]
    for v in VARIANTES_UNI:
        L.append(f"{ETIQ[v]:<38}" + "".join(f"{_p(g(df, 'universo', v, m, 'media')):>8}" for m in ec.MESES))

    L += ["", "B. TODOS LOS PRODUCTOS DEL GRUPO (real incluye productos sin prediccion)",
          "Acierto ponderado por valor (1 - WAPE):"]
    L += tabla_texto(df, "todos", VARIANTES_TODOS)
    L += ["", "Otras metricas agrupadas:"] + tabla_otras(df, "todos", VARIANTES_TODOS)
    cob = [(uni[m]["real"].sum(), todos[m]["real"].sum()) for m in ec.MESES]
    L.append("Cobertura del real del universo sobre el real de todos los productos, por mes: "
             + ", ".join(f"{m[5:]} {_p(a / b)}" for m, (a, b) in zip(ec.MESES, cob)))

    L += ["", "C. EFECTO EN LOS PRODUCTOS (G0 frente a su reconciliacion al total del grupo)"]
    L.append(f"{'Variante':<38}{'1-WAPE prod.':>14}{'Acierto medio':>15}{'Prod.>=70%':>12}{'Suma P/R':>10}")
    for v, col in (("G0", "G0"), ("G4", "G4"), ("G4b", "G4b")):
        R = np.concatenate([prod[m]["real"].to_numpy() for m in ec.MESES])
        P = np.concatenate([prod[m][col].to_numpy() for m in ec.MESES])
        m_ = metricas_producto(R, P)
        L.append(f"{ETIQ[v]:<38}{_p(m_['ponderado']):>14}{_p(m_['media']):>15}{_p(m_['pct_70']):>12}"
                 f"{m_['razon']:>10.3f}".replace(".", ","))

    # Descriptivo y a posteriori (no se usa para elegir nada): junio cayo a 263 M
    # frente a 492 M de mayo, un salto que ninguna variante anticipa.
    L += ["", "Agrupado sin junio (descriptivo, a posteriori; no se uso para elegir variantes):"]
    for v in VARIANTES_UNI:
        R = np.concatenate([uni[m]["real"].to_numpy() for m in ec.MESES if m != "2026-06"])
        P = np.concatenate([uni[m][v].to_numpy() for m in ec.MESES if m != "2026-06"])
        L.append(f"  {ETIQ[v]:<38}{_p(gr.acierto_ponderado(R, P))}")

    # veredicto
    L += ["", "VEREDICTO (universo, acierto ponderado agrupado):"]
    ranking = sorted(VARIANTES_UNI, key=lambda v: -g(df, "universo", v, "agrupado", "ponderado"))
    for v in ranking:
        val = g(df, "universo", v, "agrupado", "ponderado")
        n70 = sum(g(df, "universo", v, m, "ponderado") >= META for m in ec.MESES)
        L.append(f"  {ETIQ[v]:<38}{_p(val)}  {'>= 70 %' if val >= META else '< 70 %'}; "
                 f"meses >= 70 %: {n70} de 6")
    return L


def figura(df, salida):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    x = np.arange(len(ec.MESES))
    for v in FIGURA:
        y = [100 * g(df, "universo", v, m, "ponderado") for m in ec.MESES]
        ax.plot(x, y, marker="o", linewidth=2.2, color=COLORES[v], label=ETIQ[v])
    ax.axhline(100 * META, color="#e34948", linewidth=1.6, linestyle=":", label="Meta 70 %")
    ax.set_xticks(x)
    ax.set_xticklabels(["Abr", "May", "Jun", "Jul", "Ago", "Sep*"])
    ax.set_ylabel("Acierto ponderado por valor, 1−WAPE (%)", color="#52514e")
    ax.set_xlabel("Mes predicho de 2026 (*septiembre incompleto)", color="#52514e")
    ax.set_ylim(25, 85)
    for s in ax.spines.values():
        s.set_color("#e1e0d9")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(salida, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    inicio = time.time()
    ensure_output_dirs()
    mapa = gr.cargar_mapa(MAPA)
    ventas = gr.leer_ventas([DEFAULT_DATASET, ec.HOLDOUT])
    estados = {m: emp.estado_origen(m) for m in ec.MESES}
    reales = emp.ventas_reales()
    print(f"[estados listos] {time.time() - inicio:.0f} s", flush=True)
    uni, todos, prod, diags = construir(mapa, ventas, estados, reales)
    df = pd.concat([tabla_metricas(uni, VARIANTES_UNI, "universo"),
                    tabla_metricas(todos, VARIANTES_TODOS, "todos")], ignore_index=True)
    t = "\n".join(texto(df, uni, todos, prod, diags) + ["", f"Tiempo total: {time.time() - inicio:.1f} s"])
    print(t)
    (REPORTS / "experimento_grupos.txt").write_text(t, encoding="utf-8")
    df.to_csv(REPORTS / "experimento_grupos.csv", index=False)
    figura(df, FIGURES / "experimento_grupos.png")


if __name__ == "__main__":
    main()
