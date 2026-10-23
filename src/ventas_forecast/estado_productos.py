"""
estado_productos.py
===================
Estado de TODOS los productos de los datos, no solo de los que reciben
predicción. Cada producto cae en exactamente uno de cinco estados:

- recomendacion_monto : el modelo da una decisión y un monto estimado.
- solo_recomendacion  : decisión del modelo sin monto (fuera de los top
                        productos del regresor).
- monto_promedio      : monto por la media de los últimos 3 meses, sin
                        decisión (no vendió el último mes).
- venta_ocasional     : menos de min_months meses con ventas.
- sin_ventas_recientes: min_months o más meses con ventas, pero sin una venta
                        reciente que lo deje en alguno de los estados
                        anteriores.

Los tres primeros salen de los resultados del modelo (lo que muestra el
dashboard); los dos últimos, del historial. Los nombres y las explicaciones
para el cliente viven solo en ESTADOS para poder cambiarlos sin tocar la vista.
"""

import numpy as np
import pandas as pd

from ventas_forecast import grupos

# Clave estable -> nombre y explicación para el cliente. {min_months} se
# reemplaza por el parámetro del modelo.
ESTADOS = {
    "recomendacion_monto": {
        "nombre": "Con recomendación y monto",
        "explicacion": "El sistema recomienda qué hacer y estima cuánto se venderá.",
    },
    "solo_recomendacion": {
        "nombre": "Solo recomendación",
        "explicacion": ("Recomendación de compra; no se estima monto porque "
                        "vende poco en pesos."),
    },
    "monto_promedio": {
        "nombre": "Monto por promedio",
        "explicacion": ("No vendió el último mes; se estima con su promedio "
                        "reciente, sin recomendación."),
    },
    "venta_ocasional": {
        "nombre": "Venta ocasional",
        "explicacion": ("Vendió en menos de {min_months} meses de los últimos dos "
                        "años: no hay historia suficiente para predecir. Conviene "
                        "comprarlo bajo pedido."),
    },
    "sin_ventas_recientes": {
        "nombre": "Sin ventas recientes",
        "explicacion": ("Se vendía antes, pero no tiene ventas recientes. Revise si "
                        "conviene seguir teniéndolo en inventario."),
    },
}

# Estados de los productos que reciben predicción este mes.
ESTADOS_CON_PREDICCION = ("recomendacion_monto", "solo_recomendacion", "monto_promedio")

# Columnas que se copian de los resultados del modelo (NaN si no aplican).
COLUMNAS_PREDICCION = ("decision", "ventas_predichas", "prob_venta", "se_vendera",
                       "stock_sugerido", "stock_sugerido_unidades")


def explicacion(clave, min_months):
    """Explicación del estado para el cliente, con min_months aplicado."""
    return ESTADOS[clave]["explicacion"].format(min_months=min_months)


def catalogo(min_months):
    """Lista ordenada de {clave, nombre, explicacion} de los cinco estados."""
    return [{"clave": k, "nombre": v["nombre"], "explicacion": explicacion(k, min_months)}
            for k, v in ESTADOS.items()]


def _es_promedio(res):
    if "origen_estimacion" in res.columns:
        return res["origen_estimacion"].eq("promedio")
    return res["decision"].astype(str).str.contains("Sin recomendación", regex=False)


def _meses_entre(desde, hasta):
    """Meses calendario de 'desde' a 'hasta' (Series de fechas)."""
    return (hasta.year - desde.dt.year) * 12 + (hasta.month - desde.dt.month)


def clasificar(historia, resultados, *, id_col, desc_col, total_col, ultimo_mes,
               min_months, mapa_grupos=None):
    """Una fila por producto de 'historia' con su estado y datos de apoyo.

    historia  : producto-mes con 'fecha' (inicio de mes) y total_col en pesos.
    resultados: salida del modelo (results) o None. Define los tres estados
                con predicción: con monto y decisión, solo decisión (monto
                NaN) o monto por promedio (origen_estimacion == "promedio").
    Columnas: id, descripción, grupo (solo con mapa_grupos), estado (clave),
    estado_nombre, meses_con_venta, ultimo_mes_venta ("AAAA-MM"),
    meses_sin_venta (meses desde la última venta; 0 = vendió el último mes),
    ventas_12m (pesos de los 12 meses que terminan en ultimo_mes),
    promedio_mensual (pesos por mes con ventas) y COLUMNAS_PREDICCION.
    No modifica sus entradas."""
    ultimo_mes = pd.Timestamp(ultimo_mes)
    h = historia[[id_col, desc_col, "fecha", total_col]]
    con_venta = h[h[total_col] > 0]

    base = h.sort_values("fecha").groupby(id_col, sort=True)[desc_col].last()
    t = pd.DataFrame({id_col: base.index, desc_col: base.to_numpy()})
    t.index = base.index

    por_prod = con_venta.groupby(id_col)
    meses = por_prod["fecha"].count().reindex(base.index).fillna(0).astype(int)
    ultima = por_prod["fecha"].max().reindex(base.index)
    suma = por_prod[total_col].sum().reindex(base.index).fillna(0.0)
    desde_12 = ultimo_mes - pd.DateOffset(months=11)
    v12 = (h[h["fecha"] >= desde_12].groupby(id_col)[total_col].sum()
           .reindex(base.index).fillna(0.0))

    t["meses_con_venta"] = meses
    t["ultimo_mes_venta"] = ultima.dt.strftime("%Y-%m")
    t["meses_sin_venta"] = _meses_entre(ultima, ultimo_mes).astype("Int64")
    t["ventas_12m"] = v12.astype(float)
    t["promedio_mensual"] = np.where(meses > 0, suma / meses.where(meses > 0), np.nan)

    # Columnas de predicción (una fila por producto aunque haya duplicados)
    res = (resultados.drop_duplicates(id_col).set_index(id_col)
           if resultados is not None and len(resultados) else None)
    for col in COLUMNAS_PREDICCION:
        if res is not None and col in res.columns:
            t[col] = res[col].reindex(t.index)
        else:
            t[col] = np.nan if col not in ("decision", "se_vendera") else None

    estado = np.where(meses < min_months, "venta_ocasional", "sin_ventas_recientes")
    estado = pd.Series(estado, index=t.index, dtype=object)
    if res is not None:
        en_res = t.index.isin(res.index)
        prom = _es_promedio(res).reindex(t.index).fillna(False).astype(bool)
        monto = (res["ventas_predichas"].reindex(t.index).notna()
                 if "ventas_predichas" in res.columns
                 else pd.Series(False, index=t.index))
        estado[en_res & prom] = "monto_promedio"
        estado[en_res & ~prom & monto] = "recomendacion_monto"
        estado[en_res & ~prom & ~monto] = "solo_recomendacion"
    t["estado"] = estado
    t["estado_nombre"] = t["estado"].map({k: v["nombre"] for k, v in ESTADOS.items()})

    if mapa_grupos is not None:
        g = pd.Series(grupos.asignar_grupo(t[id_col], mapa_grupos), index=t.index)
        t["grupo"] = g.where(g != grupos.SIN_GRUPO, grupos.ETIQUETA_SIN_GRUPO)

    orden = [id_col, desc_col, *(["grupo"] if "grupo" in t.columns else []),
             "estado", "estado_nombre", "meses_con_venta", "ultimo_mes_venta",
             "meses_sin_venta", "ventas_12m", "promedio_mensual", *COLUMNAS_PREDICCION]
    return t[orden].reset_index(drop=True)


def resumen(tabla, min_months=6):
    """Por estado (en el orden de ESTADOS): productos, % de productos, ventas
    de los últimos 12 meses y % de esas ventas, con nombre y explicación."""
    n_total = len(tabla)
    v_total = float(tabla["ventas_12m"].sum()) if n_total else 0.0
    cuenta = tabla["estado"].value_counts()
    ventas = tabla.groupby("estado")["ventas_12m"].sum()
    out = []
    for e in catalogo(min_months):
        n = int(cuenta.get(e["clave"], 0))
        v = float(ventas.get(e["clave"], 0.0))
        out.append({**e, "productos": n,
                    "pct_productos": n / n_total if n_total else 0.0,
                    "ventas_12m": v,
                    "pct_ventas_12m": v / v_total if v_total else 0.0})
    return out
