"""
experimento_monto_producto.py
=============================
Experimento (NO cambia el modelo de produccion): mejorar el acierto del monto
predicho frente al real por producto. Meta del usuario: ~70 %.

Mismo estudio de caso de origen rodante que estudio_caso.py. Universo = el del
dashboard de cada origen (productos del modelo con monto + productos de
"estimacion por promedio"). Para cada horizonte (1, 2 y 3 meses) se ENTRENA el
modelo con el objetivo a ese horizonte (suma de los h meses siguientes) usando
solo datos estrictamente anteriores al primer mes del bloque, y se evalua con
bloques sin traslape: mensual abr..sep; bimestral abr-may, jun-jul, ago-sep;
trimestral abr-jun, jul-sep. Septiembre del holdout llega al 22 (incompleto).

Marcos de entrenamiento. Control (c0) y C: el de produccion (objetivo conocido,
ventas del mes actual > 0, top productos, 80/20 por producto), RF log1p y mezcla
w*RF+(1-w)*media_3 (w por sesgo en la validacion); los productos por promedio
llevan media_3. B y A: filas con ventas en los ultimos 6 meses (mes actual
incluido) y objetivo conocido (incluye meses actuales en cero), top productos;
el clasificador P(venta) usa todos los productos. En todos los casos el 20 %
final de cada producto es la validacion interna con la que se eligen potencia
Tweedie, modelo condicional, umbral de P(venta), pesos de mezcla y factores de
correccion. Nada se elige con el holdout.

Variantes:
  base   prediccion de produccion (mezcla w*RF+(1-w)*media_3) x h
  roll   suma de las predicciones mensuales rodantes de produccion (referencia;
         usa origenes posteriores al del bloque)
  m3     media_3 x h
  c0     control: pipeline de produccion reentrenado con el objetivo a h meses
  c      C: igual con variables enriquecidas (probabilidad de venta)
  b      B: XGBoost reg:tweedie en pesos, potencia {1,2; 1,5; 1,8}
  a0     A0: monto condicional (RF o XGB log1p, solo filas con venta)
  a1     A1: P(venta) del RF balanceado x monto condicional
  a2     A2: monto condicional si P >= umbral, si no 0
  bm,a1m mezcla de B / A1 con media_3 (w por sesgo, como produccion)
  *_dacc / *_dsum   D: correccion por cubetas (deciles del monto predicho) que
         maximiza el acierto por producto / corrige la suma; sobre c, a0, a1.
         A (A0/A1/A2) ya usa las variables enriquecidas, asi que A+D es A+C+D.

Genera: outputs/reports/experimento_monto_producto.txt / .csv y
outputs/figures/experimento_monto_producto.png
Uso:  MPLBACKEND=Agg PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/experimento_monto_producto.py
Cache en odd/monto_cache/ (sin versionar); borrarla para recalcular.
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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ventas_forecast import case_study as cs  # noqa: E402
from ventas_forecast import improvements as im  # noqa: E402
from ventas_forecast import monto_producto as mp  # noqa: E402
from ventas_forecast import sale_probability as sp  # noqa: E402
from ventas_forecast.paths import FIGURES, REPORTS, ensure_output_dirs  # noqa: E402
import estudio_caso as ec  # noqa: E402

CACHE = ROOT / "odd" / "monto_cache"
META = 0.70
BLOQUES = {
    1: [[m] for m in ec.MESES],
    2: [ec.MESES[0:2], ec.MESES[2:4], ec.MESES[4:6]],
    3: [ec.MESES[0:3], ec.MESES[3:6]],
}
HORIZONTES = {1: "Mensual", 2: "Bimestral", 3: "Trimestral"}
POTENCIAS = (1.2, 1.5, 1.8)
BASES = ["c0", "c", "b", "a0", "a1", "a2", "bm", "a1m"]
CORREGIBLES = ["c", "a0", "a1"]
CORREGIDAS = [f"{v}_d{m}" for v in CORREGIBLES for m in ("acc", "sum")]
VARIANTES = ["base", "roll", "m3"] + BASES + CORREGIDAS
CORTO = {"c": "C", "a0": "A0", "a1": "A1"}
ETIQ = {
    "base": "Base (producción)", "roll": "Suma mensual rodante (ref.)",
    "m3": "Media de 3 meses", "c0": "Control: RF de producción",
    "c": "C: RF enriquecido", "b": "B: XGBoost Tweedie",
    "a0": "A0: monto condicional", "a1": "A1: P(venta) × monto",
    "a2": "A2: monto si P ≥ umbral", "bm": "B+M: Tweedie y media 3",
    "a1m": "A1+M: A1 y media 3",
}
for _v in CORREGIBLES:
    ETIQ[f"{_v}_dacc"] = f"{CORTO[_v]}+D (acierto)"
    ETIQ[f"{_v}_dsum"] = f"{CORTO[_v]}+D (suma)"
COLORES = {
    "base": "#52514e", "roll": "#9a9892", "m3": "#8b5e3c", "c0": "#a8c34a",
    "c": "#1baf7a", "b": "#eb6834", "a0": "#2a78d6", "a1": "#4a3aa7", "a2": "#d55181",
    "bm": "#f5b08f", "a1m": "#a79be0",
    "c_dacc": "#0f7a55", "c_dsum": "#7fd6b4", "a0_dacc": "#124a8a", "a0_dsum": "#8fbdf0",
    "a1_dacc": "#2c2170", "a1_dsum": "#cfc8f2",
}
SEGMENTOS = {"todos": "Todos", "top": "Top 100 del real",
             "todos_los_meses": "Venden todos los meses", "resto": "Resto"}


def _p(v, d=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else \
        f"{100 * v:.{d}f} %".replace(".", ",")


def _r(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}".replace(".", ",")


def _n(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else \
        f"{v:,.0f}".replace(",", ".")


def _cache(nombre, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / nombre
    if f.exists():
        return pickle.loads(f.read_bytes())
    r = fn()
    f.write_bytes(pickle.dumps(r))
    return r


# ── Estado de produccion de cada origen ──────────────────────────────────────
def estado_origen(mes):
    def calcular():
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, model = ec.evaluar_mes(mes, tmp, con_modelo=True)
        model.estimar_por_promedio()
        c = model._c
        return {
            "df": model.df, "results": model.results, "top": [str(x) for x in model._top_prods],
            "ultimo": model._ultimo_mes, "id": c("id"), "tot": c("total_sold"),
            "freq": c("frequency"), "train_ratio": model.train_ratio,
            "lags": list(model.lags), "w": model.blend_weight,
        }
    return _cache(f"estado_{mes}.pkl", calcular)


def ventas_reales():
    return _cache("reales.pkl", lambda: {m: cs.monthly_sales(ec.HOLDOUT, m) for m in ec.MESES})


# ── Modelos de un (origen, horizonte) ────────────────────────────────────────
def columnas(est):
    base = ["VENTAS_LOG", "TOTAL_LOG", "PRECIO_LOG", est["freq"],
            *[f"lag_{l}" for l in est["lags"]], "mean_3", "mean_6",
            "mes_sig_sin", "mes_sig_cos", "cluster"]
    return base, base + sp.NEW_FEATURES


def _rf():
    return RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)


def _xgb_log():
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, n_jobs=-1)


def _particion_produccion(d, cid, tot, train_ratio, top):
    """Marco del regresor de produccion: filas con objetivo conocido y ventas
    del mes actual > 0, top productos, primeras train_ratio filas (por fecha)
    de cada producto a entrenamiento y el resto a validacion interna. Devuelve
    (indices de entrenamiento, indices de validacion) como etiquetas de d."""
    f = d[d["target"].notna() & (d[tot] > 0) & d["_id"].isin(top)].sort_values([cid, "fecha"])
    tr, te = [], []
    for _, gr in f.groupby(cid, sort=False):
        cut = int(len(gr) * train_ratio)
        tr += gr.index[:cut].tolist()
        te += gr.index[cut:].tolist()
    return tr, te


def entrenar(est, h):
    """Entrena todas las variantes para el origen y el horizonte h con datos
    hasta el ultimo mes del origen. Devuelve (ids, ventas del ultimo mes,
    predicciones por variante sobre el universo, meta)."""
    cid, tot = est["id"], est["tot"]
    d = sp.add_probability_features(est["df"], cid, tot, est["freq"])
    d["target"] = mp.multi_month_target(d, cid, tot, h)
    d["_id"] = d[cid].astype(str)
    frame, tr, te = sp.sale_train_frame(d, cid, tot, est["train_ratio"])
    f_ref, f_all = columnas(est)
    es_top = frame["_id"].isin(set(est["top"])).to_numpy()
    a_tr = np.array([i for i in tr if es_top[i]])
    a_te = np.array([i for i in te if es_top[i]])
    y = frame["target"].to_numpy(float)
    Xr, Xa = frame[f_ref].fillna(0), frame[f_all].fillna(0)

    res = est["results"]
    ids_u = set(res.loc[res["ventas_predichas"].notna(), cid].astype(str))
    ult = d[(d["fecha"] == est["ultimo"]) & d["_id"].isin(ids_u)].copy()
    Pr, Pa = ult[f_ref].fillna(0), ult[f_all].fillna(0)

    val, pred = {}, {}
    m3_u = ult["media_3_actual"].fillna(ult[tot]).to_numpy(float) * h
    m3_f = frame["media_3_actual"].fillna(frame[tot]).to_numpy(float) * h
    prom = ult[tot].to_numpy(float) <= 0  # productos por promedio: monto = media_3

    # c0 / c: regresor de produccion (marco con mes actual > 0, RF log1p,
    # mezcla con media_3 con w por sesgo en la validacion), con las variables
    # de produccion (control) o enriquecidas (C).
    p_tr, p_te = _particion_produccion(d, cid, tot, est["train_ratio"], set(est["top"]))
    pf = d.loc[p_tr + p_te]
    yp = pf["target"].to_numpy(float)
    ntr = len(p_tr)
    m3_p = pf["media_3_actual"].fillna(pf[tot]).to_numpy(float) * h
    w_prod = {}
    for nom, cols in (("c0", f_ref), ("c", f_all)):
        m = _rf().fit(pf[cols].fillna(0).iloc[:ntr], np.log1p(yp[:ntr]))
        rf_val = np.expm1(m.predict(pf[cols].fillna(0).iloc[ntr:]))
        w = im.choose_blend_weight(yp[ntr:], rf_val, m3_p[ntr:], criterion="sesgo")
        w_prod[nom] = w
        val[nom] = im.blend(w, rf_val, m3_p[ntr:])
        rf_u = np.expm1(m.predict(ult[cols].fillna(0)))
        pred[nom] = np.where(prom, m3_u, im.blend(w, rf_u, m3_u))
        val[nom + "_y"] = yp[ntr:]

    # b: Tweedie sobre pesos crudos (marco ampliado)
    pot, mt, _ = mp.choose_tweedie(Xa.iloc[a_tr], y[a_tr], Xa.iloc[a_te], y[a_te], POTENCIAS)
    val["b"] = mp.predict_nonneg(mt, Xa.iloc[a_te])
    pred["b"] = mp.predict_nonneg(mt, Pa)

    # a: P(venta) x monto condicional
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xa.iloc[tr], (y[tr] > 0).astype(int))
    p_val = clf.predict_proba(Xa.iloc[a_te])[:, 1]
    p_u = clf.predict_proba(Pa)[:, 1]
    umbral = sp.choose_threshold((y[a_te] > 0).astype(int), p_val)
    pos_tr = a_tr[y[a_tr] > 0]
    cond = {"RF": _rf().fit(Xa.iloc[pos_tr], np.log1p(y[pos_tr])),
            "XGB": _xgb_log().fit(Xa.iloc[pos_tr], np.log1p(y[pos_tr]))}
    pos_te = y[a_te] > 0
    maes_c = {k: float(np.mean(np.abs(y[a_te][pos_te]
                                      - np.expm1(m.predict(Xa.iloc[a_te][pos_te])))))
              for k, m in cond.items()}
    cond_nom = min(maes_c, key=maes_c.get)
    c_val = np.expm1(cond[cond_nom].predict(Xa.iloc[a_te]))
    c_u = np.expm1(cond[cond_nom].predict(Pa))
    val["a0"], pred["a0"] = c_val, c_u
    val["a1"], pred["a1"] = mp.hurdle_combine(p_val, c_val), mp.hurdle_combine(p_u, c_u)
    val["a2"] = mp.hurdle_combine(p_val, c_val, umbral)
    pred["a2"] = mp.hurdle_combine(p_u, c_u, umbral)

    # M: mezcla con media_3, w por sesgo en la validacion (como produccion)
    for nom, base in (("bm", "b"), ("a1m", "a1")):
        w = im.choose_blend_weight(y[a_te], val[base], m3_f[a_te], criterion="sesgo")
        w_prod[nom] = w
        pred[nom] = im.blend(w, pred[base], m3_u)

    # D: correccion por cubetas de monto predicho, aprendida con la validacion interna
    for v in CORREGIBLES:
        y_v = val["c_y"] if v == "c" else y[a_te]
        for modo in ("acc", "sum"):
            corr = mp.fit_bucket_correction(val[v], y_v, mode=modo)
            pred[f"{v}_d{modo}"] = mp.apply_bucket_correction(pred[v], corr)
    pred["p_vende"] = p_u
    pred["m3"] = m3_u
    meta = {"potencia": pot, "umbral": umbral, "cond": cond_nom, "n_train": len(a_tr),
            "n_val": len(a_te), "w": w_prod, "vende_val": float((y[a_te] > 0).mean()),
            "vende_acc_val": float(((p_val >= umbral) == (y[a_te] > 0)).mean())}
    return ult["_id"].to_numpy(), ult[tot].to_numpy(float), pred, meta


# ── Bloques ──────────────────────────────────────────────────────────────────
def _serie(df, idc, col, fecha=None):
    s = pd.Series(df[col].to_numpy(float), index=df[idc].astype(str).to_numpy())
    return s[~s.index.duplicated()]


def filas_bloque(h, meses, reales, estados):
    est = estados[meses[0]]
    ids, actual, pred, meta = _cache(f"modelo_h{h}_{meses[0]}.pkl", lambda: entrenar(est, h))
    idc = est["id"]
    b = _serie(est["results"], idc, "ventas_predichas").reindex(ids).to_numpy()
    ultimo = est["df"][est["df"]["fecha"] == est["ultimo"]]
    m3 = _serie(ultimo, idc, "media_3_actual").reindex(ids)
    m3 = m3.fillna(pd.Series(actual, index=ids)).to_numpy()
    window = np.column_stack([reales[m].reindex(ids).fillna(0.0).to_numpy(float) for m in meses])
    bloque = f"{meses[0]}..{meses[-1]}" if len(meses) > 1 else meses[0]
    out = pd.DataFrame({"h": h, "bloque": bloque, "id": ids, "real": window.sum(axis=1),
                        "actual": actual})
    for k, v in mp.segment_masks(window, top_n=100).items():
        out[f"seg_{k}"] = v
    out["p_vende"], out["umbral"] = pred["p_vende"], meta["umbral"]
    out["v_base"] = b * h
    out["v_m3"] = m3 * h
    roll = b.copy()
    for m in meses[1:]:
        s = _serie(estados[m]["results"], estados[m]["id"], "ventas_predichas")
        roll = roll + s.reindex(ids).fillna(pd.Series(b, index=ids)).to_numpy()
    out["v_roll"] = roll
    for v in BASES + CORREGIDAS:
        out[f"v_{v}"] = pred[v]
    return out, meta


# ── Metricas ─────────────────────────────────────────────────────────────────
def variantes_h(h):
    return [v for v in VARIANTES if not (v == "roll" and h == 1)]


def tabla_metricas(filas):
    rows = []
    for h, fh in filas.groupby("h"):
        grupos = [("agrupado", fh)] + list(fh.groupby("bloque"))
        for bloque, gr in grupos:
            for seg in SEGMENTOS:
                s = gr if seg == "todos" else gr[gr[f"seg_{seg}"]]
                if len(s) == 0:
                    continue
                for v in variantes_h(h):
                    m = mp.accuracy_metrics(s["real"], s[f"v_{v}"])
                    m.update({"horizonte": h, "bloque": bloque, "segmento": seg, "variante": v})
                    rows.append(m)
    return pd.DataFrame(rows)


def g(df, h, bloque, seg, v, col):
    r = df[(df["horizonte"] == h) & (df["bloque"] == bloque) & (df["segmento"] == seg)
           & (df["variante"] == v)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def mejores(df, h):
    vs = [v for v in variantes_h(h) if v not in ("roll", "m3")]
    ma = max(vs, key=lambda v: g(df, h, "agrupado", "todos", v, "mean_acc"))
    mw = max(vs, key=lambda v: g(df, h, "agrupado", "todos", v, "acc_wape"))
    return ma, mw


def tabla_texto(df, h, bloque, seg, vs):
    L = [f"{'Variante':<30}{'Acierto':>9}{'±30 %':>8}{'±50 %':>8}{'1−WAPE':>9}"
         f"{'Pred/real':>11}{'MAE':>10}{'n':>7}"]
    for v in vs:
        q = lambda c, v=v: g(df, h, bloque, seg, v, c)  # noqa: E731
        L.append(f"{ETIQ[v]:<30}{_p(q('mean_acc')):>9}{_p(q('within30')):>8}{_p(q('within50')):>8}"
                 f"{_p(q('acc_wape')):>9}{_r(q('sum_ratio')):>11}{_n(q('mae')):>10}{_n(q('n')):>7}")
    return L


def texto(df, filas, metas):
    L = ["EXPERIMENTO: ACIERTO DEL MONTO POR PRODUCTO (origen rodante)", "=" * 78, "",
         "Meta del usuario: ~70 % de acierto del monto predicho vs real por producto y mes.",
         "Acierto medio = media de 1 - |real - pred| / real (recortado a [0, 1]) sobre productos con "
         "real > 0; ±30 % / ±50 %: % de esos productos con error relativo <= 30 / 50 %; 1−WAPE = "
         "1 - suma|real - pred| / suma(real) sobre todo el universo (incluye productos sin venta); "
         "Pred/real = razón de sumas; MAE en COP sobre todo el universo.",
         "Universo = el del dashboard de cada origen (productos del modelo con monto + productos por "
         "promedio). Cada bloque se entrena con datos estrictamente anteriores a su primer mes; "
         "potencia Tweedie, umbral, modelo condicional y factores de corrección se eligieron solo "
         "con la validación interna (último 20 % de cada producto). Septiembre del holdout llega al "
         "22 (incompleto: baja el real y favorece a las predicciones bajas).",
         "ATENCIÓN: las variantes 'mejor' se señalan mirando el holdout, así que sus cifras son "
         "optimistas frente a elegir de antemano; hay pocos meses/bloques.",
         "Marcos: control y C usan el marco y la mezcla de producción (reentrenados con el objetivo a h "
         "meses; productos por promedio = media_3 x h); B y A usan filas con ventas en los últimos 6 "
         "meses (incluye meses actuales en cero). Base = mezcla de producción x h; 'rodante' suma "
         "las predicciones mensuales de cada mes (origen posterior, no comparable de forma "
         "estricta).", ""]
    for h, nom in HORIZONTES.items():
        L += [f"{nom.upper()} - AGRUPADO (n = {_n(g(df, h, 'agrupado', 'todos', 'base', 'n'))} "
              f"producto-bloque, {_n(g(df, h, 'agrupado', 'todos', 'base', 'n_pos'))} con real > 0)",
              "-" * 78] + tabla_texto(df, h, "agrupado", "todos", variantes_h(h)) + [""]
    L += ["PARÁMETROS ELEGIDOS CON LA VALIDACIÓN INTERNA", "-" * 78,
          f"{'Horizonte':<12}{'Bloque':<20}{'Potencia':>9}{'Umbral':>8}{'Condicional':>13}"
          f"{'Vende val.':>12}{'Exact. val.':>13}"]
    for (h, bloque), m in metas.items():
        L.append(f"{HORIZONTES[h]:<12}{bloque:<20}{str(m['potencia']).replace('.', ','):>9}"
                 f"{str(m['umbral']).replace('.', ','):>8}{m['cond']:>13}{_p(m['vende_val']):>12}"
                 f"{_p(m['vende_acc_val']):>13}")
    L.append("")
    for h, nom in HORIZONTES.items():
        ma, mw = mejores(df, h)
        vs = list(dict.fromkeys(["base", ma, mw]))
        L += [f"{nom.upper()} - POR BLOQUE (Base; mejor por acierto medio: {ETIQ[ma]}; "
              f"mejor por 1−WAPE: {ETIQ[mw]})", "-" * 78]
        for bl in sorted(set(df[df["horizonte"] == h]["bloque"]) - {"agrupado"}):
            L += [f"Bloque {bl}"] + tabla_texto(df, h, bl, "todos", vs)
        L.append("")
    for h, nom in HORIZONTES.items():
        ma, mw = mejores(df, h)
        vs = list(dict.fromkeys(["base", ma, mw]))
        for seg in ("top", "todos_los_meses", "resto"):
            L += [f"{nom.upper()} - SEGMENTO {SEGMENTOS[seg].upper()} (agrupado; n = "
                  f"{_n(g(df, h, 'agrupado', seg, 'base', 'n'))}, con real > 0: "
                  f"{_n(g(df, h, 'agrupado', seg, 'base', 'n_pos'))})", "-" * 78]
            L += tabla_texto(df, h, "agrupado", seg, vs) + [""]
    L += ["EXACTITUD DE 'VENDE O NO' (variantes en dos pasos: vende = P >= umbral de cada bloque; "
          "regla = el producto vendió el último mes)", "-" * 78,
          f"{'Horizonte':<12}{'% real vende':>14}{'Exactitud P':>14}{'Regla':>10}"]
    for h, nom in HORIZONTES.items():
        fh = filas[filas["h"] == h]
        real_v = fh["real"] > 0
        ex = float(((fh["p_vende"] >= fh["umbral"]) == real_v).mean())
        L.append(f"{nom:<12}{_p(float(real_v.mean())):>14}{_p(ex):>14}"
                 f"{_p(float(((fh['actual'] > 0) == real_v).mean())):>10}")
    L.append("")
    L += ["DIFERENCIA DEL MEJOR CONTRA LA BASE EN ACIERTO MEDIO (pareada por producto-bloque; IC 95 % "
          "normal, orientativo: filas repetidas por producto)", "-" * 78]
    for h, nom in HORIZONTES.items():
        ma, _ = mejores(df, h)
        fh = filas[filas["h"] == h]
        fh = fh[fh["real"] > 0]
        d_ = (mp.product_accuracy(fh["real"], fh[f"v_{ma}"])
              - mp.product_accuracy(fh["real"], fh["v_base"]))
        se = d_.std(ddof=1) / np.sqrt(len(d_))
        L.append(f"{nom:<11}{ETIQ[ma]:<28} {100 * d_.mean():+.2f} puntos (IC 95 %: "
                 f"{100 * (d_.mean() - 1.96 * se):+.2f} a {100 * (d_.mean() + 1.96 * se):+.2f}; "
                 f"n = ".replace(".", ",") + f"{_n(len(d_))})")
    L.append("")
    L += ["VARIANTES QUE ALCANZAN >= 70 % (agrupado, todo el universo)", "-" * 78]
    for h, nom in HORIZONTES.items():
        for col, etq in (("mean_acc", "acierto medio"), ("within30", "±30 %"),
                         ("within50", "±50 %"), ("acc_wape", "1−WAPE")):
            ok = [f"{ETIQ[v]} ({_p(g(df, h, 'agrupado', 'todos', v, col))})" for v in variantes_h(h)
                  if g(df, h, "agrupado", "todos", v, col) >= META]
            L.append(f"{nom:<10} {etq:<14}: " + ("; ".join(ok) if ok else "ninguna"))
    L += ["", "MÁXIMO ALCANZADO EN CUALQUIER BLOQUE Y SEGMENTO (sin la referencia rodante)", "-" * 78]
    sub = df[df["variante"] != "roll"]
    for col, etq in (("mean_acc", "acierto medio"), ("within30", "±30 %"), ("within50", "±50 %"),
                     ("acc_wape", "1−WAPE")):
        k = sub[col].idxmax()
        r = sub.loc[k]
        n70 = int((sub[col] >= META).sum())
        L.append(f"{etq:<14}: máximo {_p(r[col])} = {ETIQ[r['variante']]}, "
                 f"{HORIZONTES[int(r['horizonte'])]}, bloque {r['bloque']}, segmento "
                 f"{SEGMENTOS[r['segmento']]} (n = {_n(r['n'])}); combinaciones >= 70 %: {n70}")
    L += ["", "Nota: en el horizonte mensual 'Venden todos los meses' coincide con los productos con "
          "venta real y 'Resto' solo tiene productos sin venta (sin acierto por producto)."]
    return L


def figura(df, salida):
    vs = [v for v in VARIANTES if v != "roll"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (h, nom) in zip(axes, HORIZONTES.items()):
        y = [100 * g(df, h, "agrupado", "todos", v, "acc_wape") for v in vs]
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.bar(range(len(vs)), y, color=[COLORES[v] for v in vs])
        ax.axhline(100 * META, color="#e34948", linewidth=1.6, linestyle=":")
        ax.set_xticks([])
        ax.set_xlabel(nom, color="#52514e")
        for s in ax.spines.values():
            s.set_color("#e1e0d9")
        ax.tick_params(colors="#52514e")
    axes[0].set_ylabel("Acierto ponderado por valor, 1−WAPE (%)", color="#52514e")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORES[v]) for v in vs]
    handles.append(plt.Line2D([0], [0], color="#e34948", linestyle=":", linewidth=1.6))
    fig.legend(handles, [ETIQ[v] for v in vs] + ["Meta 70 %"], frameon=False,
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(salida, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    inicio = time.time()
    ensure_output_dirs()
    estados = {m: estado_origen(m) for m in ec.MESES}
    print(f"[estados listos] {time.time() - inicio:.0f} s", flush=True)
    reales = ventas_reales()
    partes, metas = [], {}
    for h, bloques in BLOQUES.items():
        for meses in bloques:
            f, meta = filas_bloque(h, meses, reales, estados)
            partes.append(f)
            metas[(h, f["bloque"].iloc[0])] = meta
            print(f"[h={h} {f['bloque'].iloc[0]}] n={len(f)} {time.time() - inicio:.0f} s", flush=True)
    filas = pd.concat(partes, ignore_index=True)
    df = tabla_metricas(filas)
    L = texto(df, filas, metas) + ["", f"Tiempo total: {time.time() - inicio:.1f} s"]
    t = "\n".join(L)
    print(t)
    (REPORTS / "experimento_monto_producto.txt").write_text(t, encoding="utf-8")
    df.to_csv(REPORTS / "experimento_monto_producto.csv", index=False)
    figura(df, FIGURES / "experimento_monto_producto.png")


if __name__ == "__main__":
    main()
