"""
experimento_sensibilidad_gap.py
===============================
Experimento (NO cambia el modelo ni los datos oficiales): entre 2024-02 y
2025-04 las facturas de venta caen de ~1.500 a ~450 por mes (tambien en la
cuenta contable 4135.06) y se recuperan en 2025-05. Pregunta: tratar ese tramo
mejora la prediccion en meses reales no vistos?

Mismo estudio de caso de origen rodante que estudio_caso.py (meses objetivo
2026-04..2026-09, entrenamiento estrictamente anterior a cada mes, evaluacion
SIEMPRE contra las ventas REALES del holdout) con cuatro variantes:

  Base : datos oficiales (V6), sin pesos.
  V1   : datos ajustados (imputacion experimental). Cada producto reescala sus
         ventas del tramo (filas diarias con fecha 2024-02..2025-04) por un
         factor = ventas medias mensuales de 2025-05..2025-10 / ventas medias
         mensuales del tramo (meses sin venta cuentan como cero), acotado a
         [1, 5]; productos con < 3 meses con ventas en alguna de las dos
         ventanas usan el factor global (razon de totales mensuales). Se
         multiplican VENTAS y TOTAL_VENDIDO; PRECIO_PROMEDIO (precio unitario)
         y FRECUENCIA (numero de lineas) NO se tocan. No se crean filas ni
         ventas donde no habia; filas posteriores al tramo y holdout intactos.
         El CSV ajustado vive solo en un directorio temporal.
  V2   : sin datos nuevos, peso de muestra 0,3 (y 0,0 = excluir) para las filas
         de entrenamiento cuyo mes actual o mes objetivo cae en el tramo.

El MASE de todas las variantes usa la escala de la Base (comparable).

Genera:
  outputs/reports/experimento_sensibilidad_gap.txt / .csv
  outputs/figures/experimento_sensibilidad_gap.png

Uso:  MPLBACKEND=Agg PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/experimento_sensibilidad_gap.py
Los resultados por variante se guardan en odd/gap_cache/ (sin versionar); borrar
esa carpeta para recalcular.
"""
import csv
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
from ventas_forecast.paths import (  # noqa: E402
    DEFAULT_DATASET, FIGURES, REPORTS, ensure_output_dirs,
)
import estudio_caso as ec  # noqa: E402

CACHE = ROOT / "odd" / "gap_cache"
ETIQ_V1 = "datos ajustados (imputación experimental)"
VARIANTES = [
    ("base", "Base (datos oficiales)", "#52514e"),
    ("v1", f"V1: {ETIQ_V1}", "#eb6834"),
    ("v2_03", "V2: peso 0,3 al tramo", "#4a3aa7"),
    ("v2_00", "V2: tramo excluido (peso 0)", "#1baf7a"),
]
PESOS = {"v2_03": 0.3, "v2_00": 0.0}
LIMITES = ["2024-01", "2025-04"]  # meses de origen de las etiquetas en los limites
COLOR_RF, COLOR_MEZ = "#4a3aa7", "#eb6834"


def construir_ajustado(destino):
    """Escribe en destino la V6 con el tramo reescalado (formato V6). Devuelve
    el resumen de factores y estadisticas de filas."""
    raw = pd.read_csv(DEFAULT_DATASET, sep=";", dtype=str, encoding="utf-8-sig",
                      keep_default_na=False)
    num = raw.copy()
    for col in ("VENTAS", "TOTAL_VENDIDO"):
        num[col] = cs._parse_es(raw[col])
    num["_f"] = pd.to_datetime(raw["FECHA"], format="%d/%m/%Y")
    long = gs.monthly_long(num, "CODIGO", "_f", "TOTAL_VENDIDO")
    fac, g, resumen = gs.gap_scale_factors(long)
    adj, mask = gs.scale_gap_rows(num, fac, g, "CODIGO", "_f", gs.GAP,
                                  ["VENTAS", "TOTAL_VENDIDO"])
    out = raw.copy()
    for col in ("VENTAS", "TOTAL_VENDIDO"):
        out[col] = gs.format_es(adj[col].to_numpy()).to_numpy()
    with open(destino, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";", lineterminator="\r\n")
        w.writerow(out.columns)
        w.writerows(out.itertuples(index=False, name=None))
    f_fila = num["CODIGO"].map(fac["factor"]).fillna(g).to_numpy()
    escaladas = mask & (f_fila > 1.0)
    return {
        **resumen,
        "factor_mediana": float(fac["factor"].median()),
        "factor_max": float(fac["factor"].max()),
        "factor_min": float(fac["factor"].min()),
        "pct_filas_en_tramo": float(mask.mean() * 100),
        "pct_filas_escaladas": float(escaladas.mean() * 100),
        "n_filas": int(len(num)),
        "monto_tramo_antes": float(num.loc[mask, "TOTAL_VENDIDO"].sum()),
        "monto_tramo_despues": float(adj.loc[mask, "TOTAL_VENDIDO"].sum()),
        "meses_tramo": resumen["n_gap_months"],
        "monto_ref_mensual": float(long[long["month"].between(*gs.REF)]["amount"].sum()
                                   / resumen["n_ref_months"]),
        "valores_modificados_fuera_del_tramo": int(
            (num.loc[~mask, ["VENTAS", "TOTAL_VENDIDO"]].to_numpy()
             != adj.loc[~mask, ["VENTAS", "TOTAL_VENDIDO"]].to_numpy()).sum()),
    }


def correr_variante(clave, tmp, dataset=None):
    """Corre los 6 meses de una variante (con cache) -> dict mes -> resultados."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ruta = CACHE / f"{clave}.pkl"
    if ruta.exists():
        print(f"[cache] {clave}")
        return pickle.loads(ruta.read_bytes())
    fn = None
    if clave in PESOS:
        peso = PESOS[clave]

        def fn(fechas, p=peso):
            return gs.gap_sample_weights(fechas, gs.GAP, p)
    out = {}
    for mes in ec.MESES:
        res, escala, may, model = ec.evaluar_mes(
            mes, tmp, con_modelo=True, dataset=dataset, sample_weight_fn=fn)
        out[mes] = {
            "res": res, "escala": escala, "mayoritaria": may,
            "etiquetas": gs.label_counts(model._df_train, LIMITES),
            "w": model.blend_weight,
        }
    ruta.write_bytes(pickle.dumps(out))
    return out


def metricas(res, escala_base, mayoritaria):
    """Metricas de un conjunto de filas (un mes o el agrupado)."""
    dec = cs.classification_summary(res["clase_real"], res["clase_pred"])
    base = cs.majority_baseline_summary(res["clase_real"], mayoritaria)
    m = res[res["ventas_predichas_rf"].notna()]
    fila = {
        "n_decisiones": dec["n"], "f1_macro": dec["f1_macro"],
        "acc_balanceada": dec["balanced_accuracy"], "accuracy": dec["accuracy"],
        "base_accuracy": base["accuracy"], "base_f1_macro": base["f1_macro"],
        "pct_reforzar_pred": gs.class_share(res["clase_pred"], 2) * 100,
        "pct_reforzar_real": gs.class_share(res["clase_real"], 2) * 100,
        "n_montos": int(len(m)), "suma_real": float(m["real"].sum()),
    }
    for nombre, col in (("rf", "ventas_predichas_rf"), ("mezcla", "ventas_predichas")):
        r = cs.regression_summary(m["real"], m[col], escala_base)
        fila[f"mae_{nombre}"] = r["mae"]
        fila[f"rmse_{nombre}"] = r["rmse"]
        fila[f"mase_{nombre}"] = r["mase"]
        fila[f"wape_{nombre}"] = im.wape(m["real"], m[col])
        fila[f"razon_{nombre}"] = r["sum_ratio"]
        fila[f"suma_pred_{nombre}"] = float(m[col].sum())
    return fila


def evaluar_variante(datos, escalas_base):
    filas = []
    for mes in ec.MESES:
        d = datos[mes]
        filas.append({"mes": mes, **metricas(d["res"], escalas_base[mes], d["mayoritaria"])})
    pool = pd.concat([datos[m]["res"] for m in ec.MESES], ignore_index=True)
    escala_p = cs.pooled_scale(
        [(escalas_base[m], int(datos[m]["res"]["ventas_predichas_rf"].notna().sum()))
         for m in ec.MESES])
    filas.append({"mes": "agrupado", **metricas(pool, escala_p, 0)})
    return filas


def figura(tabla, salida):
    pooled = {k: next(f for f in v if f["mes"] == "agrupado") for k, v in tabla.items()}
    claves = [v[0] for v in VARIANTES]
    nombres = ["Base\n(datos\noficiales)",
               "V1: datos\najustados\n(imputación\nexperimental)",
               "V2: peso 0,3\nal tramo", "V2: tramo\nexcluido\n(peso 0)"]
    x = np.arange(len(claves))
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.patch.set_facecolor(ec.COLOR_SURFACE)
    for ax in axs:
        ax.set_facecolor(ec.COLOR_SURFACE)
        ax.grid(axis="y", color=ec.COLOR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ec.COLOR_INK)
        for s in ax.spines.values():
            s.set_color(ec.COLOR_GRID)
        ax.set_xticks(x)
        ax.set_xticklabels(nombres, fontsize=8)
    ax1, ax2, ax3 = axs
    ax1.bar(x, [pooled[k]["f1_macro"] for k in claves], 0.55,
            color=[v[2] for v in VARIANTES])
    ax1.axhline(pooled["base"]["base_f1_macro"], color="#c0392b", linestyle="--",
                label="Clase mayoritaria")
    ax1.set_ylabel("F1 macro de las decisiones (agrupado)", color=ec.COLOR_INK)
    ax1.set_ylim(0, max(pooled[k]["f1_macro"] for k in claves) * 1.3)
    ax1.legend(frameon=False, loc="upper right")
    ancho = 0.38
    ax2.bar(x - ancho / 2, [pooled[k]["mae_rf"] / 1e3 for k in claves], ancho,
            color=COLOR_RF, label="Solo Random Forest")
    ax2.bar(x + ancho / 2, [pooled[k]["mae_mezcla"] / 1e3 for k in claves], ancho,
            color=COLOR_MEZ, label="Mezcla (sistema)")
    ax2.set_ylim(0, max(pooled[k]["mae_mezcla"] for k in claves) / 1e3 * 1.35)
    ax2.set_ylabel("MAE por producto (miles de COP)", color=ec.COLOR_INK)
    ax2.legend(frameon=False, loc="upper right")
    ax3.bar(x - ancho / 2, [pooled[k]["razon_rf"] for k in claves], ancho,
            color=COLOR_RF, label="Solo Random Forest")
    ax3.bar(x + ancho / 2, [pooled[k]["razon_mezcla"] for k in claves], ancho,
            color=COLOR_MEZ, label="Mezcla (sistema)")
    ax3.axhline(1.0, color="#c0392b", linestyle="--", label="Suma real (razón 1)")
    ax3.set_ylabel("Suma predicha / suma real (agrupado)", color=ec.COLOR_INK)
    ax3.set_ylim(0, max(1.2, max(pooled[k]["razon_mezcla"] for k in claves) * 1.5))
    ax3.legend(frameon=False, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(salida, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def tabla_texto(tabla, titulo, mes_clave):
    T = [titulo, "-" * 78,
         f"{'Variante':<44}{'F1m':>7}{'AccB':>7}{'Acc':>7}{'%Ref pred/real':>16}"]
    for k, nombre, _ in VARIANTES:
        f = next(x for x in tabla[k] if x["mes"] == mes_clave)
        T.append(f"{nombre[:43]:<44}{f['f1_macro']:>7.4f}{f['acc_balanceada']:>7.4f}"
                 f"{f['accuracy']:>7.4f}{f['pct_reforzar_pred']:>9.1f}/"
                 f"{f['pct_reforzar_real']:<5.1f}")
    f = next(x for x in tabla["base"] if x["mes"] == mes_clave)
    T.append(f"{'Linea base mayoritaria':<44}{f['base_f1_macro']:>7.4f}{'':>7}"
             f"{f['base_accuracy']:>7.4f}")
    T.append("")
    T.append(f"{'Variante / prediccion':<48}{'MAE':>11}{'RMSE':>11}{'MASE':>8}"
             f"{'WAPE':>8}{'pred/real':>10}")
    for k, nombre, _ in VARIANTES:
        f = next(x for x in tabla[k] if x["mes"] == mes_clave)
        for sufijo, lab in (("rf", "RF solo"), ("mezcla", "mezcla")):
            T.append(f"{(nombre[:30] + ' / ' + lab):<48}{f['mae_' + sufijo]:>11,.0f}"
                     f"{f['rmse_' + sufijo]:>11,.0f}{f['mase_' + sufijo]:>8.4f}"
                     f"{f['wape_' + sufijo]:>8.4f}{f['razon_' + sufijo]:>10.4f}")
    T.append("")
    return T


def main():
    inicio = time.time()
    ensure_output_dirs()
    datos = {}
    with tempfile.TemporaryDirectory() as tmp:
        ajustado = Path(tmp) / "V6_ajustado_experimental.csv"
        stats = construir_ajustado(ajustado)
        print("[factores]", stats)
        for clave, _, _ in VARIANTES:
            t = time.time()
            datos[clave] = correr_variante(
                clave, tmp, dataset=ajustado if clave == "v1" else None)
            print(f"[{clave}] listo en {time.time() - t:.0f} s")
    escalas_base = {m: datos["base"][m]["escala"] for m in ec.MESES}
    tabla = {k: evaluar_variante(datos[k], escalas_base) for k, _, _ in VARIANTES}

    L = ["EXPERIMENTO DE SENSIBILIDAD AL TRAMO DE BAJA FACTURACION", "=" * 78, "",
         "Tramo: 2024-02..2025-04 (facturas ~1.500/mes -> ~450/mes; la cuenta 4135.06 cae igual).",
         "Evaluacion SIEMPRE contra las ventas REALES de los meses del holdout (2026-04..2026-09).",
         f"V1 = {ETIQ_V1}: solo para el experimento; el dataset oficial no se modifica.",
         "MASE de todas las variantes con la escala de la Base (comparable).",
         "El holdout llega al 22 de septiembre: las ventas reales de septiembre estan incompletas.",
         "",
         "FACTORES DE REESCALADO DE V1", "-" * 78,
         f"Factor por producto = (ventas medias/mes {gs.REF[0]}..{gs.REF[1]}) / (ventas "
         f"medias/mes {gs.GAP[0]}..{gs.GAP[1]}), meses sin venta = 0, acotado a "
         f"[1, {stats['cap']:.0f}]; si el producto tiene < {stats['min_active']} meses con "
         "ventas en una ventana, factor global.",
         f"Productos: {stats['n_products']:,} (factor propio {stats['n_by_product']:,}, "
         f"global {stats['n_by_global']:,}); factor global {stats['global_factor']:.3f}",
         f"Factor por producto: mediana {stats['factor_mediana']:.3f}, minimo "
         f"{stats['factor_min']:.3f}, maximo {stats['factor_max']:.3f}",
         f"Filas en el tramo: {stats['pct_filas_en_tramo']:.1f} % de {stats['n_filas']:,}; "
         f"escaladas (factor > 1): {stats['pct_filas_escaladas']:.1f} %; valores "
         f"modificados fuera del tramo: {stats['valores_modificados_fuera_del_tramo']}",
         f"Monto del tramo: {stats['monto_tramo_antes'] / 1e6:,.1f} M -> "
         f"{stats['monto_tramo_despues'] / 1e6:,.1f} M COP "
         f"({stats['monto_tramo_antes'] / stats['meses_tramo'] / 1e6:,.1f} -> "
         f"{stats['monto_tramo_despues'] / stats['meses_tramo'] / 1e6:,.1f} M/mes; "
         f"referencia {gs.REF[0]}..{gs.REF[1]}: {stats['monto_ref_mensual'] / 1e6:,.1f} M/mes)",
         "PRECIO_PROMEDIO y FRECUENCIA no se modifican (precio unitario y numero de lineas).",
         ""]
    L += tabla_texto(tabla, "AGRUPADO (todos los meses)", "agrupado")
    for mes in ec.MESES:
        L += tabla_texto(tabla, f"MES {mes}", mes)

    L += ["SUMAS EN MILLONES DE COP (productos con prediccion de monto)", "-" * 78,
          f"{'Mes':<10}{'Real':>10}"
          + "".join(f"{k + ' RF':>11}{k + ' mez':>11}" for k, _, _ in VARIANTES)]
    for mes in ec.MESES + ["agrupado"]:
        fila = {k: next(x for x in tabla[k] if x["mes"] == mes) for k, _, _ in VARIANTES}
        L.append(f"{mes:<10}{fila['base']['suma_real'] / 1e6:>10,.1f}" + "".join(
            f"{fila[k]['suma_pred_rf'] / 1e6:>11,.1f}"
            f"{fila[k]['suma_pred_mezcla'] / 1e6:>11,.1f}"
            for k, _, _ in VARIANTES))
    L.append("")
    L += ["ETIQUETAS DE ENTRENAMIENTO EN LOS LIMITES DEL TRAMO",
          "-" * 78, "Conteo [Reducir, Mantener, Reforzar] por mes de origen; modelo del mes "
          "objetivo 2026-04.",
          "2024-01 -> 2024-02 es la caida; 2025-04 -> 2025-05 es la recuperacion."]
    for k, nombre, _ in VARIANTES:
        et = datos[k]["2026-04"]["etiquetas"]
        L.append(f"{nombre:<52} 2024-01: {et['2024-01']} | 2025-04: {et['2025-04']}")
    L.append("")
    L.append("Peso w de la mezcla por mes (Base): "
             + ", ".join(f"{m}={datos['base'][m]['w']}" for m in ec.MESES))
    L.append(f"Tiempo total: {time.time() - inicio:.1f} s")
    texto = "\n".join(L)
    print("\n" + texto)
    (REPORTS / "experimento_sensibilidad_gap.txt").write_text(texto, encoding="utf-8")
    filas_csv = []
    for k, nombre, _ in VARIANTES:
        for f in tabla[k]:
            filas_csv.append({"variante": k, "etiqueta": nombre, **f})
    pd.DataFrame(filas_csv).to_csv(REPORTS / "experimento_sensibilidad_gap.csv", index=False)
    figura(tabla, FIGURES / "experimento_sensibilidad_gap.png")
    print("[OK] experimento_sensibilidad_gap.txt/.csv/.png")


if __name__ == "__main__":
    main()
