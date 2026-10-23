"""Tabla de prediccion vs realidad por producto y resumen mensual.

Toma los resultados del modelo (con la columna origen_estimacion, ver
SalesForecastModel.estimar_por_promedio) y las ventas reales del mes, y arma
las columnas del reporte. La clase real usa la misma regla de las etiquetas
de entrenamiento (ventas reales / ventas del mes anterior): mas de
growth -> Reforzar, menos de reduce o sin ventas -> Reducir, en otro caso
Mantener. Los productos por promedio no tienen decision, asi que no tienen
clase real ni acierto de decision. La probabilidad de venta (prob_venta /
se_vendera de los resultados) existe para todos los productos: "¿Se vendió?"
es real > 0 y "¿Acertó venta?" compara esa realidad con la prediccion.

Con la accion unica (columnas accion, monto_mostrado, venta_esperada, ... de
SalesForecastModel.agregar_accion) se agregan la accion, el monto que muestra
el dashboard, la venta esperada y el acierto de la accion: Comprar más acierta
si el producto crecio mas de growth (clase real Reforzar) y No reponer acierta
si no vendio o cayo mas de reduce (clase real Reducir); las demas acciones no
tienen una clase que acertar.
"""
import numpy as np
import pandas as pd

from ventas_forecast import case_study as cs
from ventas_forecast import stock_sugerido as ss

COLUMNAS = [
    "Código", "Descripción", "Origen", "Acción", "Decisión recomendada",
    "Prob. crecimiento", "Ventas mes anterior (COP)",
    "Monto esperado (COP)", "Rango del monto", "Venta esperada (COP)",
    "Predicción combinada (COP)", "Solo modelo (COP)", "Real del mes (COP)",
    "Diferencia (real − predicción)", "Error %", "Clase real",
    "¿Acertó la acción?", "¿Acertó la decisión?",
    "Probabilidad de venta", "¿Se venderá? (predicción)", "¿Se vendió?",
    "¿Acertó venta?",
    "Stock sugerido", "Stock sugerido (unidades)", "Unidades reales",
    "¿Cubrió la demanda?",
]
# Accion -> clase real que la hace acertada (las demas no se evaluan).
_CLASE_DE_ACCION = {"comprar_mas": "Reforzar", "no_reponer": "Reducir"}


def _miles(v):
    """Entero con punto de miles (es-CO), para textos del reporte."""
    return f"{v:,.0f}".replace(",", ".")
_NOMBRE_CLASE = {0: "Reducir", 1: "Mantener", 2: "Reforzar"}
_DECISION_A_NOMBRE = {"Reducir stock": "Reducir", "Mantener stock": "Mantener",
                      "Reforzar stock": "Reforzar"}


def tabla_producto(results, real, id_col, desc_col, total_col, growth, reduce,
                   real_unid=None):
    """DataFrame con COLUMNAS, una fila por producto de results. real es una
    Serie de ventas reales indexada por codigo (texto); sin dato = 0.
    real_unid (opcional) son las unidades reales por codigo; sin ella
    'Unidades reales' queda vacia. El stock sugerido sale de las columnas
    stock_sugerido / stock_sugerido_unidades de results (vacio si no existen)."""
    r = results
    if "origen_estimacion" in r.columns:
        origen = r["origen_estimacion"].to_numpy()
    else:
        origen = np.array(["modelo"] * len(r))
    es_prom = origen == "promedio"
    ids = r[id_col].astype(str)
    real_v = real.reindex(ids.to_numpy()).fillna(0.0).to_numpy(float)
    base = r[total_col].to_numpy(float)
    pred = r["ventas_predichas"].to_numpy(float)

    clase = np.full(len(r), "", dtype=object)
    acierto = np.full(len(r), "", dtype=object)
    mod = ~es_prom
    clase_n = cs.real_class(real_v[mod], base[mod], growth, reduce)
    clase[mod] = [_NOMBRE_CLASE[int(k)] for k in clase_n]
    dec = r["decision"].map(_DECISION_A_NOMBRE).to_numpy(object)
    acierto[mod] = np.where(dec[mod] == clase[mod], "Sí", "No")

    vendio = real_v > 0
    if "prob_venta" in r.columns and "se_vendera" in r.columns:
        prob_venta = r["prob_venta"].to_numpy(float)
        se_vendera = r["se_vendera"].to_numpy(object)
        acierto_venta = np.where((se_vendera == "Sí") == vendio, "Sí", "No")
    else:
        prob_venta = np.full(len(r), np.nan)
        se_vendera = np.full(len(r), "", dtype=object)
        acierto_venta = np.full(len(r), "", dtype=object)

    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.where(real_v > 0, np.abs(real_v - pred) / real_v, np.nan)

    if "stock_sugerido" in r.columns:
        stock = r["stock_sugerido"].to_numpy(float)
        stock_u = (r["stock_sugerido_unidades"].to_numpy(float)
                   if "stock_sugerido_unidades" in r.columns
                   else np.full(len(r), np.nan))
        cubrio = np.where(np.isnan(stock), "",
                          np.where(real_v <= np.nan_to_num(stock) + 1e-6, "Sí", "No"))
    else:
        stock = np.full(len(r), np.nan)
        stock_u = np.full(len(r), np.nan)
        cubrio = np.full(len(r), "", dtype=object)
    real_u = (real_unid.reindex(ids.to_numpy()).fillna(0.0).to_numpy(float)
              if real_unid is not None else np.full(len(r), np.nan))

    vacio = np.full(len(r), "", dtype=object)
    nan = np.full(len(r), np.nan)
    if "accion" in r.columns:
        acc = r["accion"].fillna("").to_numpy(object)
        acc_nombre = r["accion_nombre"].fillna("").to_numpy(object)
        monto = r["monto_mostrado"].to_numpy(float)
        ev = r["venta_esperada"].to_numpy(float)
        si_vende = r["monto_si_vende"].to_numpy(float)
        en_rango = r["monto_rango"].fillna(False).to_numpy(bool)
        rango = np.where(en_rango, [f"Entre 0 y {_miles(v)} si se vende" if v == v else ""
                                    for v in si_vende], "").astype(object)
        objetivo = np.array([_CLASE_DE_ACCION.get(a, "") for a in acc], dtype=object)
        clase_todos = np.array([_NOMBRE_CLASE[int(k)] for k in
                                cs.real_class(real_v, base, growth, reduce)], dtype=object)
        acierto_acc = np.where(objetivo == "", "",
                               np.where(objetivo == clase_todos, "Sí", "No")).astype(object)
    else:
        acc_nombre, rango, acierto_acc = vacio, vacio.copy(), vacio.copy()
        monto, ev = nan, nan.copy()

    out = pd.DataFrame({
        "Código": ids.to_numpy(),
        "Descripción": r[desc_col].to_numpy(),
        "Origen": origen,
        "Acción": acc_nombre,
        "Decisión recomendada": r["decision"].to_numpy(),
        "Prob. crecimiento": np.where(es_prom, np.nan,
                                      r["prob_crecimiento"].to_numpy(float)),
        "Ventas mes anterior (COP)": base,
        "Monto esperado (COP)": monto,
        "Rango del monto": rango,
        "Venta esperada (COP)": ev,
        "Predicción combinada (COP)": pred,
        "Solo modelo (COP)": (r["ventas_predichas_rf"].to_numpy(float)
                              if "ventas_predichas_rf" in r.columns
                              else np.full(len(r), np.nan)),
        "Real del mes (COP)": real_v,
        "Diferencia (real − predicción)": real_v - pred,
        "Error %": error,
        "Clase real": clase,
        "¿Acertó la acción?": acierto_acc,
        "¿Acertó la decisión?": acierto,
        "Probabilidad de venta": prob_venta,
        "¿Se venderá? (predicción)": se_vendera,
        "¿Se vendió?": np.where(vendio, "Sí", "No"),
        "¿Acertó venta?": acierto_venta,
        "Stock sugerido": stock,
        "Stock sugerido (unidades)": stock_u,
        "Unidades reales": real_u,
        "¿Cubrió la demanda?": cubrio,
    })
    return out[COLUMNAS]


def _cobertura_tabla(tabla):
    """Metricas de ventas_forecast.stock_sugerido.cobertura sobre las filas de
    la tabla con stock sugerido (el universo del dashboard)."""
    t = tabla[tabla["Stock sugerido"].notna()]
    return ss.cobertura(t["Real del mes (COP)"], t["Stock sugerido"],
                        t["Unidades reales"], t["Stock sugerido (unidades)"])


def cobertura_agrupada(tablas):
    """Cobertura del stock sugerido de todos los meses apilados; 'tablas' es
    {mes: tabla_producto}."""
    return _cobertura_tabla(pd.concat(list(tablas.values()), ignore_index=True))


def resumen_mes(mes, tabla, real_top_total=None):
    """Totales del mes: predicho (modelo / promedio / total) frente al real,
    cobertura del real de los top productos, aciertos de las decisiones y MAE,
    y el resumen por accion (ver _resumen_acciones). Las sumas del modelo usan
    solo los productos con monto."""
    pred = "Predicción combinada (COP)"
    real = "Real del mes (COP)"
    mod = tabla[tabla["Origen"] == "modelo"]
    prom = tabla[tabla["Origen"] == "promedio"]
    mod_m = mod[mod[pred].notna()]
    evaluadas = tabla[tabla["¿Acertó la decisión?"] != ""]
    aciertos = int((evaluadas["¿Acertó la decisión?"] == "Sí").sum())
    # Acierto de venta: universo del experimento "Probabilidad de venta"
    # (productos del modelo con monto + estimacion por promedio) y todos.
    con_venta = tabla[tabla["¿Acertó venta?"] != ""]
    univ = tabla[(tabla["Origen"] == "promedio") | tabla[pred].notna()]
    univ = univ[univ["¿Acertó venta?"] != ""]
    ac_univ = int((univ["¿Acertó venta?"] == "Sí").sum())
    ac_todos = int((con_venta["¿Acertó venta?"] == "Sí").sum())
    cob = _cobertura_tabla(tabla)
    pred_mod, pred_prom = float(mod_m[pred].sum()), float(prom[pred].sum())
    real_mod, real_prom = float(mod_m[real].sum()), float(prom[real].sum())
    real_total = real_mod + real_prom
    out = {
        "Mes": mes,
        "Productos modelo": int(len(mod)),
        "Productos promedio": int(len(prom)),
        "Predicción modelo (COP)": pred_mod,
        "Predicción promedio (COP)": pred_prom,
        "Predicción total (COP)": pred_mod + pred_prom,
        "Real modelo (COP)": real_mod,
        "Real promedio (COP)": real_prom,
        "Real total (COP)": real_total,
        "Razón predicción / real": ((pred_mod + pred_prom) / real_total
                                    if real_total else float("nan")),
        "Real top 1900 (COP)": (float("nan") if real_top_total is None
                                else float(real_top_total)),
        "Cobertura del real top 1900": (
            float("nan") if not real_top_total
            else (pred_mod + pred_prom) / float(real_top_total)),
        "Decisiones evaluadas": int(len(evaluadas)),
        "Decisiones acertadas": aciertos,
        "% decisiones acertadas": (aciertos / len(evaluadas)
                                   if len(evaluadas) else float("nan")),
        "Productos evaluados (venta)": int(len(univ)),
        "Ventas acertadas": ac_univ,
        "% acierto de venta": (ac_univ / len(univ) if len(univ)
                               else float("nan")),
        "% acierto de venta (todos)": (ac_todos / len(con_venta)
                                       if len(con_venta) else float("nan")),
        "MAE modelo (COP)": (float((mod_m[real] - mod_m[pred]).abs().mean())
                             if len(mod_m) else float("nan")),
        "MAE promedio (COP)": (float((prom[real] - prom[pred]).abs().mean())
                               if len(prom) else float("nan")),
        "Productos evaluados (stock)": cob["n"],
        "Cobertura del stock sugerido": cob["cobertura"],
        "Cobertura (los que vendieron)": cob["cobertura_vendieron"],
        "Cobertura (unidades)": cob["cobertura_unid"],
        "Stock total / real": cob["total_stock_real"],
    }
    out.update(_resumen_acciones(tabla, real_total))
    return out


def _resumen_acciones(tabla, real_total):
    """Productos y acierto por accion, Comprar más con "se venderá: No",
    incoherencias de monto (se venderá: No con un monto puntual > 0, No reponer
    con monto > 0 o Comprar más con monto menor al mes anterior) y venta
    esperada de los productos con monto frente a lo real."""
    acc = tabla["Acción"]
    con = tabla["Predicción combinada (COP)"].notna()
    monto = tabla["Monto esperado (COP)"]
    base = tabla["Ventas mes anterior (COP)"]
    nan = float("nan")

    def _prec(nombre):
        s = tabla[acc == nombre]["¿Acertó la acción?"]
        return float((s == "Sí").mean()) if len(s) else nan

    comprar = acc == "Comprar más"
    inc = (((tabla["¿Se venderá? (predicción)"] == "No") & (tabla["Rango del monto"] == "")
            & (monto > 0))
           | ((acc == "No reponer este mes") & (monto > 0))
           | (comprar & (monto < base)))
    ev = float(tabla.loc[con, "Venta esperada (COP)"].sum())
    return {
        "Comprar más (productos)": int(comprar.sum()),
        "% acierto Comprar más": _prec("Comprar más"),
        "Comprar más con «se venderá: No»": int(
            (comprar & (tabla["¿Se venderá? (predicción)"] == "No")).sum()),
        "Mantener (productos)": int((acc == "Mantener").sum()),
        "No reponer (productos)": int((acc == "No reponer este mes").sum()),
        "% acierto No reponer": _prec("No reponer este mes"),
        "Revisar (productos)": int((acc == "Revisar").sum()),
        "Por promedio (productos)": int((acc == "Por promedio").sum()),
        "Incoherencias de monto": int((inc & con).sum()),
        "Venta esperada (COP)": ev,
        "Razón venta esperada / real": ev / real_total if real_total else nan,
    }



# ── Vista por grupo de producto ──────────────────────────────────────────────
COLUMNAS_GRUPO = ["Grupo", "Productos", "Predicción (COP)", "Real (COP)",
                  "Diferencia (real − predicción)", "Acierto %"]


def tabla_grupos_mes(tabla, mapa):
    """Una fila por grupo con los productos que muestra el dashboard (los que
    tienen monto predicho, modelo y promedio): prediccion = suma de la venta
    esperada (o de las predicciones combinadas sin accion unica), real = suma de lo vendido por esos mismos
    productos, diferencia y acierto 1 - |R-P|/R recortado a [0, 1] (vacio si
    el real es 0). Ordenada por la prediccion del grupo."""
    from ventas_forecast import grupos as gr
    pred_col, real_col = "Predicción combinada (COP)", "Real del mes (COP)"
    t = tabla[tabla[pred_col].notna()]
    # Con la accion unica la prediccion del grupo es la venta esperada.
    if "Venta esperada (COP)" in t.columns and t["Venta esperada (COP)"].notna().any():
        pred_col = "Venta esperada (COP)"
    grupo = pd.Series(gr.asignar_grupo(t["Código"], mapa), index=t.index)
    grupo = grupo.where(grupo != gr.SIN_GRUPO, gr.ETIQUETA_SIN_GRUPO)
    g = pd.DataFrame({"grupo": grupo, "n": 1, "pred": t[pred_col].astype(float),
                      "real": t[real_col].astype(float)}).groupby("grupo").sum()
    out = pd.DataFrame({
        "Grupo": g.index,
        "Productos": g["n"].to_numpy(int),
        "Predicción (COP)": g["pred"].to_numpy(float),
        "Real (COP)": g["real"].to_numpy(float),
    })
    out["Diferencia (real − predicción)"] = out["Real (COP)"] - out["Predicción (COP)"]
    out["Acierto %"] = gr.exactitud_grupo(out["Real (COP)"], out["Predicción (COP)"])
    out = out.sort_values("Predicción (COP)", ascending=False, kind="stable")
    return out.reset_index(drop=True)[COLUMNAS_GRUPO]


def resumen_grupos(tablas):
    """Acierto ponderado por valor (1 - sum|R-P| / sum R sobre los grupos) de
    cada mes y de todos los meses apilados ('Agrupado'). 'tablas' es
    {mes: tabla_grupos_mes}."""
    filas, todas = [], []
    for mes, t in tablas.items():
        filas.append(_fila_grupos(mes, t))
        todas.append(t)
    if todas:
        filas.append(_fila_grupos("Agrupado", pd.concat(todas, ignore_index=True)))
    return pd.DataFrame(filas)


def _fila_grupos(mes, t):
    p, r = t["Predicción (COP)"].to_numpy(float), t["Real (COP)"].to_numpy(float)
    return {
        "Mes": mes,
        "Grupos": int(len(t)),
        "Predicción (COP)": float(p.sum()),
        "Real (COP)": float(r.sum()),
        "Acierto ponderado": (float(1.0 - np.abs(r - p).sum() / r.sum())
                              if r.sum() else float("nan")),
    }
