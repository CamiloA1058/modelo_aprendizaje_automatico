"""Una sola acción por producto: qué hacer, cuánto se espera vender y cuánto
stock tener.

Reglas (elegidas antes de mirar el estudio de caso y fijadas con la validación
interna del entrenamiento; ver CHANGELOG "Acción única"):

- Comprar más: el clasificador de 3 clases da Reforzar con P(Reforzar) >= t
  y la probabilidad de venta alcanza el umbral de venta (s_v). t es el menor
  valor de GRID_T con precisión de Reforzar >= 60 % en la validación interna
  (con al menos MIN_SOPORTE Reforzar predichos); si ninguno llega, el de mejor
  F0,5. Si falla t, el producto toma la mejor de las otras dos clases; si
  falla la probabilidad de venta, pasa a Mantener.
- No reponer este mes: prob_venta < s_baja, o la decisión es Reducir con
  prob_venta < s_v.
- Mantener: prob_venta >= s_v con decisión Mantener o Reducir.
- Revisar: el resto (s_baja <= prob_venta < s_v sin Reducir).
- Por promedio: productos sin venta el mes pasado (estimación por promedio).

Monto: "monto si se vende" A (media de los últimos 3 o 6 meses con venta, o la
mediana de los últimos 6, elegida con la validación interna) y venta esperada
= prob_venta * A. Lo que se muestra depende de la acción (variante V3).

Los nombres y explicaciones para el cliente viven solo en ACCIONES.
"""
import numpy as np
import pandas as pd

REDUCIR, MANTENER, REFORZAR = 0, 1, 2
DECISION_NOMBRE = {REDUCIR: "Reducir stock", MANTENER: "Mantener stock",
                   REFORZAR: "Reforzar stock"}
NOMBRE_DECISION = {v: k for k, v in DECISION_NOMBRE.items()}

MIN_SOPORTE = 30
PRECISION_META = 0.60
GRID_T = np.round(np.arange(0.34, 0.951, 0.01), 2)
CANDIDATOS_MONTO = ("A3", "A6", "M6")
MARGEN_COBERTURA = 0.03

# Clave estable -> nombre y explicación para el cliente (orden de la vista).
ACCIONES = {
    "comprar_mas": {
        "nombre": "Comprar más",
        "explicacion": "Es probable que se venda y que venda más que el mes pasado.",
    },
    "mantener": {
        "nombre": "Mantener",
        "explicacion": "Es probable que se venda; compre lo de siempre.",
    },
    "no_reponer": {
        "nombre": "No reponer este mes",
        "explicacion": "Es poco probable que se venda este mes; no compre más.",
    },
    "revisar": {
        "nombre": "Revisar",
        "explicacion": "Puede venderse o no; decida según su inventario.",
    },
    "por_promedio": {
        "nombre": "Por promedio",
        "explicacion": "No vendió el mes pasado; se estima con su promedio reciente.",
    },
}


def nombre(clave):
    """Nombre para el cliente de una acción (vacío si no hay acción)."""
    return ACCIONES[clave]["nombre"] if clave in ACCIONES else ""


def catalogo():
    """Lista ordenada de {clave, nombre, explicacion}."""
    return [{"clave": k, **v} for k, v in ACCIONES.items()]


# ── Decisión de 3 clases con confianza mínima y coherencia de venta ──────────
def decidir(P, pv=None, t=None, s=None):
    """Clase por fila (0 Reducir, 1 Mantener, 2 Reforzar) desde P (n, 3).

    Reforzar solo si argmax = Reforzar, P[:, 2] >= t (si t no es None) y
    pv >= s (si s no es None). Los que falla t pasan a la mejor de
    Reducir/Mantener; los que falla s (y pasan t) pasan a Mantener."""
    P = np.asarray(P, float)
    pred = P.argmax(axis=1)
    ref = pred == REFORZAR
    bloq_a = ref & (P[:, 2] < t) if t is not None else np.zeros(len(P), bool)
    if s is not None:
        with np.errstate(invalid="ignore"):
            bloq_b = ref & ~bloq_a & (np.asarray(pv, float) < s)
    else:
        bloq_b = np.zeros(len(P), bool)
    pred = pred.copy()
    pred[bloq_a] = P[bloq_a, :2].argmax(axis=1)
    pred[bloq_b] = MANTENER
    return pred


def _precision_recall(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    n = int((pred == REFORZAR).sum())
    tp = int(((pred == REFORZAR) & (y == REFORZAR)).sum())
    pos = int((y == REFORZAR).sum())
    return (tp / n if n else np.nan), (tp / pos if pos else np.nan), n


def _fbeta(p, r, b=0.5):
    if not (p > 0 and r > 0):
        return 0.0
    return (1 + b * b) * p * r / (b * b * p + r)


def elegir_umbral_reforzar(y, P, grid=GRID_T, meta=PRECISION_META, soporte=MIN_SOPORTE):
    """t para Reforzar con (y, P) de la validación interna: el menor t con
    precisión >= meta y >= soporte predichos; si ninguno, el que maximiza F0,5
    (empate: el menor). Devuelve (t, regla)."""
    filas = []
    for t in grid:
        p, r, n = _precision_recall(y, decidir(P, t=t))
        filas.append((float(t), p, r, n))
    ok = [f for f in filas if f[3] >= soporte and f[1] >= meta]
    if ok:
        return ok[0][0], "precision"
    f05 = [(_fbeta(p, r) if n >= soporte else -1.0) for _, p, r, n in filas]
    return filas[int(np.argmax(f05))][0], "f05"


def elegir_s_baja(prob_venta, no_vendio, s_v, soporte=MIN_SOPORTE):
    """Mayor s de 0,05..s_v - 0,01 (paso 0,01) tal que entre las filas con
    prob_venta < s haya >= soporte filas y al menos la mitad no haya vendido
    el mes siguiente. 0 si ninguno (la regla queda vacía)."""
    pv = np.asarray(prob_venta, float)
    nov = np.asarray(no_vendio, bool)
    s_baja = 0.0
    for s in np.round(np.arange(0.05, s_v - 0.0049, 0.01), 2):
        m = pv < s
        if m.sum() >= soporte and nov[m].mean() >= 0.5:
            s_baja = float(s)
    return s_baja


# ── Monto si se vende ────────────────────────────────────────────────────────
def estimadores_si_vende(df, id_col, total_col):
    """A3 / A6 (media de los últimos 3 / 6 meses CON venta) y M6 (mediana de los
    últimos 6) hasta el mes de cada fila inclusive, arrastrados a los meses sin
    venta. df: producto-mes con 'fecha'. Devuelve un DataFrame con índice
    (id, fecha)."""
    dfo = df.sort_values([id_col, "fecha"])
    pos = dfo[dfo[total_col] > 0][[id_col, total_col]]
    g = pos.groupby(id_col)[total_col]
    est = pd.DataFrame(index=pos.index)
    est["A3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
    est["A6"] = g.transform(lambda x: x.rolling(6, min_periods=1).mean())
    est["M6"] = g.transform(lambda x: x.rolling(6, min_periods=1).median())
    est = est.reindex(dfo.index)
    est[id_col] = dfo[id_col]
    est = est.groupby(id_col)[list(CANDIDATOS_MONTO)].ffill()
    est.index = pd.MultiIndex.from_arrays([dfo[id_col].to_numpy(), dfo["fecha"].to_numpy()])
    return est


def buscar(tabla, ids, fechas):
    """Filas de una tabla con índice (id, fecha) para las llaves dadas."""
    idx = pd.MultiIndex.from_arrays([np.asarray(ids), pd.to_datetime(np.asarray(fechas))])
    return tabla.reindex(idx)


def elegir_monto_si_vende(prob, real, candidatos):
    """Clave de candidatos ({clave: monto}) que minimiza |suma(prob*A) -
    suma(real)| (empate: menor MAE) sobre la validación interna. Devuelve
    (clave, {clave: (sesgo, mae, razon)})."""
    p = np.asarray(prob, float)
    r = np.asarray(real, float)
    crit = {}
    for k, a in candidatos.items():
        ev = p * np.asarray(a, float)
        crit[k] = (abs(float(np.nansum(ev)) - float(np.nansum(r))),
                   float(np.nanmean(np.abs(ev - r))),
                   float(np.nansum(ev) / np.nansum(r)) if np.nansum(r) else np.nan)
    clave = min(crit, key=lambda k: (round(crit[k][0], 6), crit[k][1]))
    return clave, crit


# ── Acción, monto mostrado y stock ───────────────────────────────────────────
def asignar_accion(es_modelo, dec, prob_venta, s_v, s_baja):
    """Clave de acción por fila (ver el docstring del módulo). dec: clase
    final (0/1/2, NaN en los productos por promedio)."""
    mod = np.asarray(es_modelo, bool)
    dec = np.asarray(dec, float)
    pv = np.asarray(prob_venta, float)
    acc = np.full(len(mod), "revisar", dtype=object)
    with np.errstate(invalid="ignore"):
        acc[mod & (pv >= s_v) & np.isin(dec, [REDUCIR, MANTENER])] = "mantener"
        acc[mod & (dec == REFORZAR)] = "comprar_mas"
        acc[mod & (dec == REDUCIR) & (pv < s_v)] = "no_reponer"
        acc[mod & (pv < s_baja)] = "no_reponer"
    acc[~mod] = "por_promedio"
    return acc


def monto_por_accion(accion, prob_venta, si_vende, tiene_monto, s_v, s_baja):
    """(venta_esperada, monto_mostrado, rango) por fila.

    venta_esperada = prob_venta * monto si se vende (para totales).
    monto_mostrado: Comprar más y Mantener -> venta esperada; No reponer -> 0;
    Revisar -> rango 0..monto si se vende (el valor es la venta esperada);
    Por promedio: prob < s_baja -> 0, s_baja <= prob < s_v -> rango, si no la
    venta esperada. Las filas sin monto (tiene_monto False) quedan en NaN."""
    acc = np.asarray(accion, object)
    pv = np.asarray(prob_venta, float)
    a = np.asarray(si_vende, float)
    ok = np.asarray(tiene_monto, bool)
    ev = np.where(ok, pv * a, np.nan)
    mostrado = ev.copy()
    rango = np.zeros(len(acc), bool)
    prom = acc == "por_promedio"
    with np.errstate(invalid="ignore"):
        mostrado[ok & (acc == "no_reponer")] = 0.0
        rango[ok & (acc == "revisar")] = True
        mostrado[ok & prom & (pv < s_baja)] = 0.0
        rango[ok & prom & (pv >= s_baja) & (pv < s_v)] = True
    return ev, mostrado, rango


def stock_por_accion(accion, stock_nivel, pred):
    """Stock por acción: No reponer -> 0, Revisar -> el monto predicho,
    resto -> el stock al nivel de servicio. Sin monto predicho: NaN."""
    acc = np.asarray(accion, object)
    stock = np.asarray(stock_nivel, float).copy()
    pred = np.asarray(pred, float)
    rev = acc == "revisar"
    stock[rev] = pred[rev]
    nr = acc == "no_reponer"
    stock[nr] = np.where(np.isnan(pred[nr]), np.nan, 0.0)
    return stock


def comparar_regla_stock(real, stock_nivel, stock_regla, margen=MARGEN_COBERTURA):
    """Compara el stock al nivel de servicio para todos con el stock por
    acción en la validación interna. La regla se adopta si su cobertura (real
    <= stock) queda a no más de 'margen' de la del nivel y baja stock / real."""
    r = np.asarray(real, float)
    s1 = np.asarray(stock_nivel, float)
    s2 = np.asarray(stock_regla, float)
    ok = ~np.isnan(r) & ~np.isnan(s1) & ~np.isnan(s2)
    r, s1, s2 = r[ok], s1[ok], s2[ok]
    nan = float("nan")

    def _m(s):
        cob = float((r <= s + 1e-6).mean()) if len(r) else nan
        razon = float(s.sum() / r.sum()) if r.sum() > 0 else nan
        return cob, razon

    cob1, raz1 = _m(s1)
    cob2, raz2 = _m(s2)
    adoptada = bool(len(r) and cob2 >= cob1 - margen - 1e-12 and raz2 < raz1)
    return {"n": int(len(r)), "cobertura_nivel": cob1, "stock_real_nivel": raz1,
            "cobertura_regla": cob2, "stock_real_regla": raz2, "adoptada": adoptada}


def contradicciones(res, s_v):
    """Combinaciones prohibidas en una tabla de resultados con accion,
    prob_venta, monto_mostrado, stock_sugerido y la base del mes anterior
    (columna 'base'). Devuelve un conteo por regla:

    comprar_sin_venta     Comprar más con prob_venta < s_v
    no_reponer_con_monto  No reponer con monto > 0
    no_reponer_con_stock  No reponer con stock > 0
    mantener_sin_stock    Mantener con stock 0 (stock conocido)
    comprar_bajo_base     Comprar más con monto < mes anterior (permitido y
                          documentado: el monto es la venta esperada, que
                          descuenta la probabilidad de venta)."""
    acc = res["accion"].to_numpy(object)
    pv = res["prob_venta"].to_numpy(float)
    monto = res["monto_mostrado"].to_numpy(float)
    stock = res["stock_sugerido"].to_numpy(float)
    base = res["base"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        return {
            "comprar_sin_venta": int(((acc == "comprar_mas") & (pv < s_v)).sum()),
            "no_reponer_con_monto": int(((acc == "no_reponer") & (monto > 0)).sum()),
            "no_reponer_con_stock": int(((acc == "no_reponer") & (stock > 0)).sum()),
            "mantener_sin_stock": int(((acc == "mantener") & (stock <= 0)).sum()),
            "comprar_bajo_base": int(((acc == "comprar_mas") & (monto < base)).sum()),
        }
