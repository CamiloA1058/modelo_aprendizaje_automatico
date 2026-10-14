"""
experimento_stock_sugerido.py
=============================
Experimento: stock sugerido por producto con nivel de servicio 70 %.

Meta del usuario: predicciones mensuales por producto confiables "alrededor del
70 %". El monto puntual por producto no llega (mejor acierto medio ~38 %), asi
que se mide otra cosa: un nivel "order-up-to" X tal que la demanda del mes queda
cubierta (real <= X) en ~70 % de los productos.

Metodo (el que va a produccion, SalesForecastModel.agregar_stock_sugerido):
  cota = max(expm1(log1p(pred) + q70(r)), 0), con r = log1p(real) - log1p(pred)
  medido en la validacion INTERNA del entrenamiento de cada origen (particion
  temporal de prueba del regresor y del clasificador de venta), por cubeta de
  prob_venta [0-0,4), [0,4-0,6), [0,6-0,8), [0,8-1] (cubetas con < 50 filas se
  fusionan). Nada se calibra con el mes evaluado ni con el holdout.

Evaluacion: origen rodante 2026-04..2026-09 (como estudio_caso.py: cada mes se
entrena solo con datos estrictamente anteriores). Universo = el del dashboard
(productos del modelo con monto + productos de estimacion por promedio). Lo
real sale de data/holdout/ventas_2026_04_09.csv (septiembre incompleto).
Criterio de integracion: cobertura agrupada entre 65 y 75 % y la mayoria de
los meses dentro de esa franja.

Genera: outputs/reports/experimento_stock_sugerido.txt / .csv y
outputs/figures/experimento_stock_sugerido.png
Uso:  MPLBACKEND=Agg PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/experimento_stock_sugerido.py
Cache en odd/stock_cache/ (sin versionar); borrarla para recalcular.
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
from ventas_forecast import stock_sugerido as ss  # noqa: E402
from ventas_forecast.paths import FIGURES, REPORTS, ensure_output_dirs  # noqa: E402
import estudio_caso as ec  # noqa: E402

CACHE = ROOT / "odd" / "stock_cache"
META = 0.70
BANDA = (0.65, 0.75)
NIVELES = (0.5, 0.7, 0.9)
COLOR_STOCK, COLOR_VENDIERON, COLOR_RATIO, COLOR_META = "#1baf7a", "#4a3aa7", "#eb6834", "#52514e"
COLOR_GRID, COLOR_INK, COLOR_SURFACE = "#e1e0d9", "#52514e", "#fcfcfb"


def _p(v, d=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else \
        f"{100 * v:.{d}f} %".replace(".", ",")


def _pp(v):
    """Diferencia en puntos porcentuales con signo y coma decimal."""
    return f"{100 * v:+.1f}p".replace(".", ",")


def _r(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}".replace(".", ",")


def _n(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else \
        f"{v:,.0f}".replace(",", ".")


def tabla_mes(mes):
    """Entrena como produccion con datos < mes y devuelve (tabla por producto,
    calibracion). La tabla trae lo predicho y lo real; el real solo se usa
    para evaluar, nunca para calibrar."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"tabla_{mes}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        _, _, _, model = ec.evaluar_mes(mes, tmp, con_modelo=True)
        model.estimar_por_promedio()
        model.agregar_probabilidad_venta()
        model.agregar_stock_sugerido()
    c = model._c
    r = model.results
    ids = r[c("id")].astype(str).to_numpy()
    real = cs.monthly_sales(ec.HOLDOUT, mes).reindex(ids).fillna(0.0).to_numpy()
    real_u = cs.monthly_units(ec.HOLDOUT, mes).reindex(ids).fillna(0.0).to_numpy()
    t = pd.DataFrame({
        "mes": mes, "id": ids, "origen": r["origen_estimacion"].to_numpy(),
        "prob": r["prob_venta"].to_numpy(float),
        "anterior": r[c("total_sold")].to_numpy(float),
        "pred": r["ventas_predichas"].to_numpy(float),
        "stock": r["stock_sugerido"].to_numpy(float),
        "stock_u": r["stock_sugerido_unidades"].to_numpy(float),
        "precio": r[c("avg_price")].to_numpy(float),
        "real": real, "real_u": real_u,
    })
    t = t[t["stock"].notna()].reset_index(drop=True)
    out = (t, model.calibracion_stock)
    f.write_bytes(pickle.dumps(out))
    return out


def segmentos(t):
    """{nombre: mascara} de los segmentos del informe."""
    seg = {"Todos": np.ones(len(t), bool),
           "Modelo": (t["origen"] == "modelo").to_numpy(),
           "Promedio": (t["origen"] == "promedio").to_numpy()}
    top = t["pred"].rank(method="first", ascending=False) <= 100
    seg["Top 100 por predicción"] = top.to_numpy()
    seg["Resto (fuera del top 100)"] = ~top.to_numpy()
    edges = ss.EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t["prob"] >= lo) & ((t["prob"] < hi) | ((hi == edges[-1]) & (t["prob"] <= hi)))
        rango = f"{lo:.1f}-{hi:.1f}".replace(".", ",")
        seg[f"Prob. de venta [{rango}" + ("]" if hi == edges[-1] else ")")] = m.to_numpy()
    return seg


def metricas(t, mask=None):
    if mask is not None:
        t = t[mask]
    return ss.cobertura(t["real"], t["stock"], t["real_u"], t["stock_u"])


def cobertura_nivel(t, cal, alfa):
    q = ss.cuantiles_para(cal, t["prob"], alfa)
    stock = ss.cota_superior(t["pred"], q)
    return float((t["real"].to_numpy() <= stock + 1e-6).mean())


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
        ax.set_xlabel("Mes predicho", color=COLOR_INK)
    ancho = 0.38
    ax1.bar(x - ancho / 2, [100 * f["cobertura"] for f in filas], ancho,
            color=COLOR_STOCK, label="Todos los productos")
    ax1.bar(x + ancho / 2, [100 * f["cobertura_vendieron"] for f in filas], ancho,
            color=COLOR_VENDIERON, label="Solo los que vendieron")
    ax1.axhline(100 * META, color=COLOR_META, linestyle="--", linewidth=1.5,
                label="Meta: 70 %")
    ax1.set_ylabel("Productos con demanda cubierta (%)", color=COLOR_INK)
    ax1.set_ylim(0, 100)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    ax2.bar(x - ancho / 2, [f["total_stock_real"] for f in filas], ancho,
            color=COLOR_RATIO, label="Stock total / ventas reales totales")
    ax2.bar(x + ancho / 2, [f["mediana_stock_real"] for f in filas], ancho,
            color=COLOR_VENDIERON, label="Mediana por producto (los que vendieron)")
    ax2.axhline(1.0, color=COLOR_META, linestyle="--", linewidth=1.5,
                label="Stock = ventas reales")
    ax2.set_ylabel("Stock sugerido / ventas reales (veces)", color=COLOR_INK)
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    fig.tight_layout()
    fig.savefig(salida, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    t0 = time.time()
    ensure_output_dirs()
    por_mes = {m: tabla_mes(m) for m in ec.MESES}
    todas = pd.concat([t for t, _ in por_mes.values()], ignore_index=True)

    filas_csv, filas_fig, L = [], [], []
    L += ["EXPERIMENTO STOCK SUGERIDO (nivel de servicio 70 %)", "=" * 70,
          "Origen rodante 2026-04..2026-09; entrenamiento estrictamente anterior a cada mes;",
          "calibracion solo con la validacion interna de cada entrenamiento (sin holdout).",
          "Universo: productos del dashboard (modelo con monto + estimacion por promedio).",
          "Septiembre incompleto (registro hasta el 22). Meta de cobertura: 70 % (franja "
          f"{_p(BANDA[0], 0)} a {_p(BANDA[1], 0)}).", ""]

    def fila(ambito, seg, m):
        filas_csv.append({"ambito": ambito, "segmento": seg, **m})

    L += ["1. COBERTURA POR MES (todos los productos del universo)", "-" * 70,
          f"{'Mes':<10}{'n':>6}{'cubre %':>9}{'desv.':>8}{'cubre vend.':>12}"
          f"{'med st/real':>12}{'media':>8}{'tot st/real':>12}{'agotados':>9}{'cubre unid.':>12}"]
    for mes, (t, cal) in por_mes.items():
        m = metricas(t)
        fila(mes, "Todos", m)
        filas_fig.append({"mes": mes, **m})
        L.append(f"{mes:<10}{m['n']:>6}{_p(m['cobertura']):>9}"
                 f"{_pp(m['cobertura'] - META):>8}{_p(m['cobertura_vendieron']):>12}"
                 f"{_r(m['mediana_stock_real']):>12}{_r(m['media_stock_real']):>8}"
                 f"{_r(m['total_stock_real']):>12}{_p(m['agotados']):>9}{_p(m['cobertura_unid']):>12}")
    mp = metricas(todas)
    fila("Agrupado", "Todos", mp)
    L.append(f"{'Agrupado':<10}{mp['n']:>6}{_p(mp['cobertura']):>9}"
             f"{_pp(mp['cobertura'] - META):>8}{_p(mp['cobertura_vendieron']):>12}"
             f"{_r(mp['mediana_stock_real']):>12}{_r(mp['media_stock_real']):>8}"
             f"{_r(mp['total_stock_real']):>12}{_p(mp['agotados']):>9}{_p(mp['cobertura_unid']):>12}")
    L += ["", "  cubre % = productos con real <= stock sugerido (incluye los que no vendieron);",
          "  cubre vend. = lo mismo entre los que vendieron; st/real = stock / real entre los que",
          "  vendieron (mediana y media); tot st/real = suma de stock / suma de ventas reales",
          "  (sobrestock); agotados = productos con stock 0 que si vendieron; cubre unid. =",
          f"  unidades reales <= stock en unidades (n con precio: {mp['n_unid']:,}).", ""]

    L += ["2. SEGMENTOS (agrupado y por mes: cobertura todos / cobertura vendieron / tot st/real)",
          "-" * 70,
          f"{'Segmento':<30}{'n':>6}{'cubre':>8}{'c.vend':>8}{'med':>6}{'tot':>6}{'agot.':>7}"
          + "".join(f"{mes[-2:]:>7}" for mes in ec.MESES)]
    segs = segmentos(todas)
    seg_mes = {mes: segmentos(t) for mes, (t, _) in por_mes.items()}
    for nombre, mask in segs.items():
        m = metricas(todas, mask)
        fila("Agrupado", nombre, m)
        por = []
        for mes, (t, _) in por_mes.items():
            mm = metricas(t, seg_mes[mes][nombre])
            fila(mes, nombre, mm)
            por.append(_p(mm["cobertura"], 0))
        L.append(f"{nombre:<30}{m['n']:>6}{_p(m['cobertura']):>8}{_p(m['cobertura_vendieron']):>8}"
                 f"{_r(m['mediana_stock_real']):>6}{_r(m['total_stock_real']):>6}"
                 f"{_p(m['agotados']):>7}" + "".join(f"{v:>7}" for v in por))
    L.append("  (el universo ya es el top 1.900 por construccion, por eso 'top 1.900 vs resto' "
             "no aplica; se reporta el top 100 por prediccion)")
    L.append("")

    L += ["3. OTROS NIVELES DE SERVICIO (cobertura de todos los productos)", "-" * 70,
          f"{'Mes':<10}" + "".join(f"{_p(a, 0):>10}" for a in NIVELES)]
    acum = {a: [] for a in NIVELES}
    for mes, (t, cal) in por_mes.items():
        vals = [cobertura_nivel(t, cal, a) for a in NIVELES]
        for a, v in zip(NIVELES, vals):
            acum[a].append(v * len(t))
            fila(mes, f"Nivel {int(100 * a)} %", {"cobertura": v, "n": len(t)})
        L.append(f"{mes:<10}" + "".join(f"{_p(v):>10}" for v in vals))
    n_tot = len(todas)
    L.append(f"{'Agrupado':<10}" + "".join(f"{_p(sum(acum[a]) / n_tot):>10}" for a in NIVELES))
    L.append("")

    L += ["4. CALIBRACION POR ORIGEN (cuantiles q70 de log1p(real) - log1p(pred), validacion interna)",
          "-" * 70]
    for mes, (_, cal) in por_mes.items():
        cub = ", ".join(f"[{cal['edges'][i]:.1f}-{cal['edges'][i + 1]:.1f}) n={cal['n'][i]} "
                        f"q70={cal['q'][0.7][i]:+.2f}" for i in range(len(cal["n"])))
        L.append(f"{mes}: global q50 {cal['global'][0.5]:+.2f} / q70 {cal['global'][0.7]:+.2f} / "
                 f"q90 {cal['global'][0.9]:+.2f} (n={cal['n_total']:,}); cubetas: {cub}")
    L.append("")

    dentro = sum(BANDA[0] <= f["cobertura"] <= BANDA[1] for f in filas_fig)
    ok = BANDA[0] <= mp["cobertura"] <= BANDA[1] and dentro >= 4
    L += ["5. VEREDICTO", "-" * 70,
          f"Cobertura agrupada {_p(mp['cobertura'])} (meta 70 %, desviacion "
          f"{_pp(mp['cobertura'] - META)[:-1]} pp); meses dentro de la franja "
          f"{_p(BANDA[0], 0)}-{_p(BANDA[1], 0)}: {dentro} de {len(filas_fig)}.",
          f"Sobrestock agrupado: stock total {_r(mp['total_stock_real'])} veces lo vendido; "
          f"mediana por producto {_r(mp['mediana_stock_real'])} veces (los que vendieron).",
          "INTEGRAR en el sistema." if ok else "NO integrar (fuera de la franja)."]
    texto = "\n".join(L)
    (REPORTS / "experimento_stock_sugerido.txt").write_text(texto, encoding="utf-8")
    pd.DataFrame(filas_csv).to_csv(REPORTS / "experimento_stock_sugerido.csv", index=False,
                                   sep=";", decimal=",", encoding="utf-8-sig")
    figura(filas_fig, FIGURES / "experimento_stock_sugerido.png")
    print(texto)
    print(f"\n[OK] {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
