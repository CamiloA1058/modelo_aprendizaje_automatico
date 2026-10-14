"""
experimento_probabilidad_venta.py
=================================
Experimento (NO cambia el modelo de produccion): predecir, por producto, si
tendra ventas el mes siguiente (si/no) con un clasificador binario dedicado.
Meta del usuario: acertar >= 70 % en meses reales que el modelo no vio.

Mismo estudio de caso de origen rodante que estudio_caso.py (meses objetivo
2026-04..2026-09; entrenamiento con la V6 mas el holdout estrictamente anterior
a cada mes; evaluacion contra las ventas REALES del holdout). Reutiliza
estudio_caso.evaluar_mes para fijar el universo del dashboard de cada mes
(productos del modelo con monto + productos de "estimacion por promedio").

Objetivo: vende = ventas del mes siguiente > 0, sobre el calendario mensual
completo. El marco de entrenamiento incluye filas con mes actual en cero si el
producto vendio en los ultimos 6 meses y usa la particion temporal 80/20 por
producto de produccion; el 20 % es la validacion interna con la que se elige el
umbral de probabilidad (maximiza la exactitud). Nada se elige con el holdout.

Modelos: RandomForest (sin pesos y balanceado), XGBoost (binary:logistic) y
regresion logistica; reglas: "vendio el mes anterior", ">= 2 de los ultimos 3
meses", "fraccion historica de meses con venta > 0,5".

Genera:
  outputs/reports/experimento_probabilidad_venta.txt / .csv
  outputs/reports/experimento_probabilidad_venta_calibracion.csv
  outputs/figures/experimento_probabilidad_venta.png

Uso:  MPLBACKEND=Agg PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/experimento_probabilidad_venta.py 
Los datos por mes se guardan en odd/prob_cache/ (sin versionar); borrar esa
carpeta para recalcular.
"""
import pickle
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ventas_forecast import case_study as cs  # noqa: E402
from ventas_forecast import sale_probability as sp  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402
from ventas_forecast.paths import FIGURES, REPORTS, ensure_output_dirs  # noqa: E402
import estudio_caso as ec  # noqa: E402

CACHE = ROOT / "odd" / "prob_cache"
META = 0.70
UNIVERSOS = {"a": "Universo del dashboard (modelo + estimacion por promedio)",
             "b": "Productos con >= 1 venta en los 6 meses previos"}
SEGMENTOS = ["todos", "top1900", "resto", "modelo", "promedio"]
MODELOS = ["rf", "rf_balanceado", "xgboost", "regresion_logistica"]
REGLAS = ["vendio_mes_anterior", "2_de_3_meses", "fraccion_historica"]
ETIQUETAS = {
    "rf": "Random Forest", "rf_balanceado": "RF balanceado", "xgboost": "XGBoost",
    "regresion_logistica": "Regresión logística",
    "vendio_mes_anterior": "Regla: vendió el mes anterior",
    "2_de_3_meses": "Regla: ≥2 de los últimos 3",
    "fraccion_historica": "Regla: fracción histórica > 0,5",
}
COLORES = {
    "rf": "#1baf7a", "rf_balanceado": "#4a3aa7", "xgboost": "#eb6834",
    "regresion_logistica": "#d55181", "vendio_mes_anterior": "#52514e",
    "2_de_3_meses": "#e3a008", "fraccion_historica": "#8b5e3c",
}


def _f(v, d=4):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else \
        f"{v:.{d}f}".replace(".", ",")


def _p(v, d=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else \
        f"{100 * v:.{d}f} %".replace(".", ",")


def _pp(v):
    return f"{100 * v:.2f}".replace(".", ",")


def _n(v):
    return f"{v:,.0f}".replace(",", ".")


# ── Datos de cada mes ────────────────────────────────────────────────────────
def columnas_modelo(model):
    c = model._c
    lags = [f"lag_{l}" for l in model.lags]
    base = ["VENTAS_LOG", "TOTAL_LOG", "PRECIO_LOG", c("frequency"), *lags,
            "mean_3", "mean_6", "mes_sig_sin", "mes_sig_cos", "cluster"]
    return base + sp.NEW_FEATURES


def datos_mes(mes, tmp):
    """Entrena el sistema del mes (para fijar el universo), estima por promedio
    y arma el marco de entrenamiento y las filas de prediccion del clasificador
    binario. Todo con informacion estrictamente anterior a 'mes'."""
    res, _, _, model = ec.evaluar_mes(mes, tmp, con_modelo=True)
    model.estimar_por_promedio()
    c = model._c
    d = sp.add_probability_features(model.df, c("id"), c("total_sold"), c("frequency"))
    frame, tr, te = sp.sale_train_frame(d, c("id"), c("total_sold"), model.train_ratio)
    feats = columnas_modelo(model)

    pred = d[d["fecha"] == model._ultimo_mes].copy()
    pred["id"] = pred[c("id")].astype(str)
    ventas = cs.monthly_sales(ec.HOLDOUT, mes)
    pred["real_vende"] = (ventas.reindex(pred["id"]).fillna(0.0).to_numpy() > 0).astype(int)
    res_ids = set(res["id"])
    ids_monto = set(res.loc[res["ventas_predichas_rf"].notna(), "id"])
    out = model.results
    ids_prom = set(out.loc[out["origen_estimacion"] == "promedio", c("id")].astype(str))
    pred["origen"] = np.where(pred["id"].isin(ids_prom), "promedio",
                              np.where(pred["id"].isin(res_ids), "modelo", "otro"))
    pred["top1900"] = pred["id"].isin(set(map(str, model._top_prods)))
    pred["univ_a"] = pred["id"].isin(ids_monto | ids_prom)
    pred["univ_b"] = pred[c("id")].isin(pred.loc[pred["ventas_6m"] > 0, c("id")])
    return {
        "mes": mes, "feats": feats, "total_col": c("total_sold"),
        "frame": frame, "train_idx": tr, "test_idx": te,
        "pred": pred, "n_a": int(pred["univ_a"].sum()), "n_b": int(pred["univ_b"].sum()),
    }


def cargar_datos(mes):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"datos_{mes}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        d = datos_mes(mes, tmp)
    f.write_bytes(pickle.dumps(d))
    return d


# ── Modelos ──────────────────────────────────────────────────────────────────
def _x_lineal(X, columnas_monto):
    """Para la regresion logistica: log1p de las variables en pesos (muy
    asimetricas); el resto queda igual. El escalado se ajusta solo con
    entrenamiento."""
    X = X.copy()
    for col in columnas_monto:
        X[col] = np.log1p(X[col].clip(lower=0))
    return X


def construir(nombre):
    if nombre == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1)
    if nombre == "rf_balanceado":
        return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    if nombre == "xgboost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             objective="binary:logistic", eval_metric="logloss",
                             random_state=42, n_jobs=-1)
    if nombre == "regresion_logistica":
        return LogisticRegression(max_iter=2000, C=1.0)
    raise ValueError(nombre)


def importancia(modelo, feats):
    if hasattr(modelo, "feature_importances_"):
        v = np.asarray(modelo.feature_importances_, float)
    else:
        v = np.abs(np.ravel(modelo.coef_))
    return dict(zip(feats, v / v.sum() if v.sum() else v))


def ajustar_mes(datos, nombres):
    """Ajusta cada modelo con la particion de entrenamiento, elige el umbral con
    la validacion interna y predice las filas del mes siguiente."""
    feats, frame = datos["feats"], datos["frame"]
    tr, te = datos["train_idx"], datos["test_idx"]
    X = frame[feats].fillna(0)
    y = frame["vende"].to_numpy()
    pred = datos["pred"]
    Xp = pred[feats].fillna(0)
    monto = [col for col in feats
             if col.startswith("lag_") or col in ("mean_3", "mean_6", "FRECUENCIA")
             or col.startswith("frec_")]
    filas = pred[["id", "real_vende", "origen", "top1900", "univ_a", "univ_b"]].copy()
    filas["mes"] = datos["mes"]
    filas["actual"] = pred[datos["total_col"]].to_numpy()
    meta = {"umbral": {}, "val_accuracy": {}, "importancia": {}, "n_train": len(tr),
            "n_val": len(te), "tasa_val": float(y[te].mean())}
    for nombre in nombres:
        m = construir(nombre)
        if nombre == "regresion_logistica":
            esc = StandardScaler().fit(_x_lineal(X.iloc[tr], monto))
            ft = lambda Z: esc.transform(_x_lineal(Z, monto))  # noqa: E731
        else:
            ft = lambda Z: Z.to_numpy()  # noqa: E731
        m.fit(ft(X.iloc[tr]), y[tr])
        p_val = m.predict_proba(ft(X.iloc[te]))[:, 1]
        umbral = sp.choose_threshold(y[te], p_val)
        p = m.predict_proba(ft(Xp))[:, 1]
        filas[f"proba_{nombre}"] = p
        filas[f"pred_{nombre}"] = (p >= umbral).astype(int)
        meta["umbral"][nombre] = umbral
        meta["val_accuracy"][nombre] = float(((p_val >= umbral).astype(int) == y[te]).mean())
        meta["importancia"][nombre] = importancia(m, feats)
    for regla, v in sp.rule_predictions(pred, datos["total_col"]).items():
        filas[f"pred_{regla}"] = v.astype(int)
    return filas, meta


def cargar_resultado(mes, nombres, etiqueta):
    f = CACHE / f"resultado_{etiqueta}_{mes}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    r = ajustar_mes(cargar_datos(mes), nombres)
    f.write_bytes(pickle.dumps(r))
    return r


# ── Evaluacion ───────────────────────────────────────────────────────────────
def seleccion(filas, universo, segmento):
    m = filas["univ_a" if universo == "a" else "univ_b"]
    if segmento == "top1900":
        m &= filas["top1900"]
    elif segmento == "resto":
        m &= ~filas["top1900"]
    elif segmento in ("modelo", "promedio"):
        m &= filas["origen"] == segmento
    return filas[m]


def tabla_metricas(filas_por_mes, meta_por_mes, nombres_modelo):
    """Filas largas (universo, segmento, mes, modelo) por mes y agrupado."""
    todas = pd.concat(filas_por_mes.values(), ignore_index=True)
    meses = list(filas_por_mes)
    salida = []
    for u in UNIVERSOS:
        for seg in SEGMENTOS:
            for mes in meses + ["agrupado"]:
                fuente = todas if mes == "agrupado" else filas_por_mes[mes]
                sel = seleccion(fuente, u, seg)
                if len(sel) == 0:
                    continue
                for nombre in nombres_modelo + REGLAS:
                    es_modelo = nombre in nombres_modelo
                    p = sel[f"proba_{nombre}"] if es_modelo else None
                    m = sp.classification_metrics(sel["real_vende"], sel[f"pred_{nombre}"], p)
                    m.update({"universo": u, "segmento": seg, "mes": mes, "modelo": nombre,
                              "tipo": "modelo" if es_modelo else "regla",
                              "umbral": meta_por_mes[mes]["umbral"].get(nombre, np.nan)
                              if mes != "agrupado" and es_modelo else np.nan})
                    salida.append(m)
    return pd.DataFrame(salida)


def calibracion(filas_por_mes, nombres_modelo):
    todas = pd.concat(filas_por_mes.values(), ignore_index=True)
    out = []
    for u in UNIVERSOS:
        sel = seleccion(todas, u, "todos")
        for nombre in nombres_modelo:
            t = sp.calibration_table(sel["real_vende"], sel[f"proba_{nombre}"], 10)
            t.insert(0, "modelo", nombre)
            t.insert(0, "universo", u)
            out.append(t)
    return pd.concat(out, ignore_index=True)


def _g(df, u, seg, mes, nombre, col):
    r = df[(df["universo"] == u) & (df["segmento"] == seg) & (df["mes"] == mes)
           & (df["modelo"] == nombre)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def mejor_modelo(df, nombres_modelo):
    pooled = {n: _g(df, "a", "todos", "agrupado", n, "accuracy") for n in nombres_modelo}
    return max(pooled, key=pooled.get)


# ── Reporte ──────────────────────────────────────────────────────────────────
def texto(df, cal, filas_por_mes, meta_por_mes, nombres_modelo):
    meses = list(filas_por_mes)
    cols = meses + ["agrupado"]
    todas_n = nombres_modelo + REGLAS
    mejor = mejor_modelo(df, nombres_modelo)
    L = ["EXPERIMENTO: PROBABILIDAD DE VENTA (origen rodante)", "=" * 72, "",
         "Pregunta: el producto tendra ventas el mes siguiente (si/no). Meta: exactitud >= 70 % "
         "en meses reales no vistos (2026-04..2026-09).",
         "Para cada mes M se entrena con la V6 mas el holdout anterior a M (nunca M). Objetivo "
         "vende = ventas de M > 0. Marco de entrenamiento: filas con mes actual > 0 o con ventas "
         "en los ultimos 6 meses; particion temporal 80/20 por producto; el 20 % es la "
         "validacion interna que fija el umbral de probabilidad (maximiza la exactitud). "
         "El escalador (solo regresion logistica) y KMeans se ajustan solo con entrenamiento. "
         "Nada se eligio con el holdout. Septiembre del holdout llega al 22 y esta incompleto "
         "(baja la tasa real de venta).",
         "Variables: logs de ventas/precio, FRECUENCIA, rezagos 1/2/3/6, mean_3/mean_6, mes "
         "siguiente (seno/coseno), cluster, y nuevas: meses desde la ultima venta, meses con "
         "venta en los ultimos 3/6/12, rezago 12, rezagos 1-3 de FRECUENCIA y su media, edad "
         "del producto y fraccion historica de meses con venta.",
         "Hiperparametros (simples, fijos): RF 300 arboles, profundidad 12, min_samples_leaf 5; "
         "XGBoost 300 arboles, profundidad 4, tasa 0,05, submuestreo 0,8; regresion logistica "
         "C=1 con log1p de las variables en pesos y estandarizacion.",
         f"Universos: (a) {UNIVERSOS['a']} = productos con prediccion de monto del sistema "
         "de ese mes; (b) " + UNIVERSOS["b"] + ".",
         "Reglas (sin entrenamiento): vendio el mes anterior; >= 2 de los ultimos 3 meses con "
         "venta; fraccion historica de meses con venta > 0,5.", ""]

    L += ["VERIFICACION DEL UNIVERSO DEL DASHBOARD (a)", "-" * 72,
          f"{'Mes':<10}{'n (a)':>8}{'% venden':>10}{'Regla mes ant.':>16}{'n (b)':>8}"
          f"{'% venden (b)':>14}{'Regla (b)':>11}"]
    for mes in cols:
        L.append(f"{mes:<10}{_n(_g(df, 'a', 'todos', mes, 'vendio_mes_anterior', 'n')):>8}"
                 f"{_p(_g(df, 'a', 'todos', mes, 'vendio_mes_anterior', 'tasa_vende')):>10}"
                 f"{_p(_g(df, 'a', 'todos', mes, 'vendio_mes_anterior', 'accuracy')):>16}"
                 f"{_n(_g(df, 'b', 'todos', mes, 'vendio_mes_anterior', 'n')):>8}"
                 f"{_p(_g(df, 'b', 'todos', mes, 'vendio_mes_anterior', 'tasa_vende')):>14}"
                 f"{_p(_g(df, 'b', 'todos', mes, 'vendio_mes_anterior', 'accuracy')):>11}")
    L.append("")

    for u in UNIVERSOS:
        L += [f"AGRUPADO - UNIVERSO ({u}) {UNIVERSOS[u]}", "-" * 72,
              f"{'Modelo / regla':<34}{'Exact.':>8}{'Ex.bal.':>9}{'Prec.V':>8}{'Rec.V':>8}"
              f"{'Prec.NV':>9}{'Rec.NV':>8}{'AUC':>8}"]
        for n in todas_n:
            g = lambda c, n=n: _g(df, u, "todos", "agrupado", n, c)  # noqa: E731
            L.append(f"{ETIQUETAS.get(n, n):<34}{_p(g('accuracy')):>8}{_p(g('balanced_accuracy')):>9}"
                     f"{_p(g('precision_vende')):>8}{_p(g('recall_vende')):>8}"
                     f"{_p(g('precision_no')):>9}{_p(g('recall_no')):>8}{_f(g('auc')):>8}")
        L.append("(V = vende, NV = no vende)")
        for col, titulo in (("accuracy", "EXACTITUD"), ("balanced_accuracy", "EXACTITUD BALANCEADA"),
                            ("auc", "AUC")):
            L += ["", f"{titulo} POR MES - universo ({u})", "-" * 72,
                  f"{'Modelo / regla':<34}" + "".join(f"{m:>10}" for m in cols)]
            for n in todas_n:
                if col == "auc" and n in REGLAS:
                    continue
                fmt = _f if col == "auc" else _p
                L.append(f"{ETIQUETAS.get(n, n):<34}" + "".join(
                    f"{fmt(_g(df, u, 'todos', m, n, col)):>10}" for m in cols))
            if col == "accuracy":
                L.append(f"{'% real que vende':<34}" + "".join(
                    f"{_p(_g(df, u, 'todos', m, REGLAS[0], 'tasa_vende')):>10}" for m in cols))
        L.append("")

    L += ["UMBRALES ELEGIDOS CON LA VALIDACION INTERNA (exactitud interna entre parentesis)",
          "-" * 72, f"{'Modelo':<24}" + "".join(f"{m:>16}" for m in meses)]
    for n in nombres_modelo:
        L.append(f"{ETIQUETAS.get(n, n):<24}" + "".join(
            f"{_f(meta_por_mes[m]['umbral'][n], 2):>8} ({_p(meta_por_mes[m]['val_accuracy'][n], 0):>5})"
            for m in meses))
    L.append(f"{'filas ent./val.':<24}" + "".join(
        f"{_n(meta_por_mes[m]['n_train']) + '/' + _n(meta_por_mes[m]['n_val']):>16}"
        for m in meses))
    L.append("")

    L += ["CUMPLIMIENTO DE LA META (70 %)", "-" * 72]
    for u in UNIVERSOS:
        L.append(f"Universo ({u}):")
        for n in todas_n:
            acc = [_g(df, u, "todos", m, n, "accuracy") for m in meses]
            ok = sum(1 for a in acc if a >= META)
            pool = _g(df, u, "todos", "agrupado", n, "accuracy")
            L.append(f"  {ETIQUETAS.get(n, n):<34} agrupado {_p(pool):>8} "
                     f"{'CUMPLE' if pool >= META else 'no cumple':<10} meses >= 70 %: {ok} de {len(meses)}")
    pool_m = _g(df, "a", "todos", "agrupado", mejor, "accuracy")
    pool_r = max(_g(df, "a", "todos", "agrupado", r, "accuracy") for r in REGLAS)
    ok_m = sum(1 for m in meses if _g(df, "a", "todos", m, mejor, "accuracy") >= META)
    L += ["",
          f"Mejor modelo por exactitud agrupada en (a): {ETIQUETAS.get(mejor, mejor)} = "
          f"{_p(pool_m)}; meses >= 70 %: {ok_m} de {len(meses)}; mejor regla simple: {_p(pool_r)}.",
          f"META {'CUMPLIDA' if pool_m >= META else 'NO cumplida'} en el agrupado del universo (a) "
          f"(la eleccion del mejor modelo mira el resultado agrupado; es solo para reportar).", ""]

    for u in UNIVERSOS:
        for seg in SEGMENTOS:
            if seg == "todos" or _g(df, u, seg, "agrupado", todas_n[0], "n") != _g(df, u, seg, "agrupado", todas_n[0], "n"):
                continue
            L += [f"SEGMENTO {seg} - universo ({u}) (agrupado; n = "
                  f"{_n(_g(df, u, seg, 'agrupado', todas_n[0], 'n'))}, "
                  f"% venden = {_p(_g(df, u, seg, 'agrupado', todas_n[0], 'tasa_vende'))})",
                  "-" * 72,
                  f"{'Modelo / regla':<34}{'Exact.':>8}{'Ex.bal.':>9}{'Rec.V':>8}{'Rec.NV':>8}{'AUC':>8}"]
            for n in todas_n:
                g = lambda c, n=n: _g(df, u, seg, "agrupado", n, c)  # noqa: E731
                L.append(f"{ETIQUETAS.get(n, n):<34}{_p(g('accuracy')):>8}{_p(g('balanced_accuracy')):>9}"
                         f"{_p(g('recall_vende')):>8}{_p(g('recall_no')):>8}{_f(g('auc')):>8}")
            L.append("")

    for u in UNIVERSOS:
        L += [f"CALIBRACION POR DECILES - {ETIQUETAS.get(mejor, mejor)}, universo ({u}), agrupado",
              "-" * 72, f"{'Decil':>6}{'n':>8}{'Prob. media':>13}{'Tasa real':>11}"]
        t = cal[(cal["universo"] == u) & (cal["modelo"] == mejor)]
        for _, r in t.iterrows():
            L.append(f"{int(r['decil']):>6}{_n(r['n']):>8}{_p(r['prob_media']):>13}{_p(r['tasa_real']):>11}")
        L.append("")

    imp = pd.DataFrame([meta_por_mes[m]["importancia"][mejor] for m in meses]).mean()
    L += [f"IMPORTANCIA DE VARIABLES - {ETIQUETAS.get(mejor, mejor)} (media de los {len(meses)} meses)",
          "-" * 72]
    for k, v in imp.sort_values(ascending=False).head(10).items():
        L.append(f"  {k:<22}{_p(v, 2):>9}")
    L.append("")
    L += intervalo_confianza(filas_por_mes, df, mejor, nombres_modelo)
    return L


def intervalo_confianza(filas_por_mes, df, mejor, nombres_modelo):
    """Incertidumbre del agrupado en (a): IC 95 % normal de la exactitud del mejor
    modelo y de su diferencia (pareada) con la mejor regla simple. Las filas de un
    mismo producto en meses distintos no son independientes: el IC es orientativo."""
    todas = pd.concat(filas_por_mes.values(), ignore_index=True)
    a = seleccion(todas, "a", "todos")
    regla = max(REGLAS, key=lambda r: _g(df, "a", "todos", "agrupado", r, "accuracy"))
    ok_m = (a[f"pred_{mejor}"] == a["real_vende"]).astype(float).to_numpy()
    ok_r = (a[f"pred_{regla}"] == a["real_vende"]).astype(float).to_numpy()
    n = len(a)
    se = ok_m.std(ddof=1) / np.sqrt(n)
    d = ok_m - ok_r
    se_d = d.std(ddof=1) / np.sqrt(n)
    return ["INCERTIDUMBRE DEL AGRUPADO EN EL UNIVERSO (a) (orientativa: filas repetidas por producto)",
            "-" * 72,
            f"Exactitud de {ETIQUETAS.get(mejor, mejor)}: {_p(ok_m.mean(), 2)} "
            f"(IC 95 %: {_p(ok_m.mean() - 1.96 * se, 2)} a {_p(ok_m.mean() + 1.96 * se, 2)}; n = {_n(n)}).",
            f"Diferencia con la mejor regla ({ETIQUETAS.get(regla, regla)}): "
            f"{_pp(d.mean())} puntos porcentuales (IC 95 %: {_pp(d.mean() - 1.96 * se_d)} a "
            f"{_pp(d.mean() + 1.96 * se_d)}).", ""]


def figura(df, meses, nombres_modelo, salida):
    cols = meses + ["agrupado"]
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for n in nombres_modelo + REGLAS:
        y = [100 * _g(df, "a", "todos", m, n, "accuracy") for m in cols]
        ax.plot(x, y, color=COLORES.get(n, "#2a78d6"), marker="o",
                linestyle="--" if n in REGLAS else "-", linewidth=2.2 if n in nombres_modelo else 1.6,
                label=ETIQUETAS.get(n, n))
    ax.axhline(100 * META, color="#e34948", linewidth=1.6, linestyle=":", label="Meta 70 %")
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_ylabel("Exactitud sobre el universo del dashboard (%)", color="#52514e")
    ax.set_xlabel("Mes predicho", color="#52514e")
    ax.tick_params(colors="#52514e")
    ax.set_ylim(45, 85)
    for s in ax.spines.values():
        s.set_color("#e1e0d9")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(salida, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    inicio = time.time()
    ensure_output_dirs()
    nombres = list(MODELOS)
    filas_por_mes, meta_por_mes = {}, {}
    for mes in ec.MESES:
        f, m = cargar_resultado(mes, MODELOS, "base")
        filas_por_mes[mes], meta_por_mes[mes] = f, m
    df = tabla_metricas(filas_por_mes, meta_por_mes, nombres)
    cal = calibracion(filas_por_mes, nombres)
    L = texto(df, cal, filas_por_mes, meta_por_mes, nombres)
    L += ["", f"Tiempo total: {time.time() - inicio:.1f} s"]
    t = "\n".join(L)
    print(t)
    (REPORTS / "experimento_probabilidad_venta.txt").write_text(t, encoding="utf-8")
    df.to_csv(REPORTS / "experimento_probabilidad_venta.csv", index=False)
    cal.to_csv(REPORTS / "experimento_probabilidad_venta_calibracion.csv", index=False)
    figura(df, ec.MESES, nombres, FIGURES / "experimento_probabilidad_venta.png")


if __name__ == "__main__":
    main()
