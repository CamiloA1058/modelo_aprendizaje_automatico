"""Stock sugerido por producto con nivel de servicio (cota superior).

Idea de inventario: un nivel "order-up-to" a nivel de servicio alfa es el
valor X tal que la demanda queda cubierta (real <= X) en una fraccion alfa de
los casos. Se obtiene corrigiendo la prediccion en escala log1p con el
cuantil alfa de los residuos r = log1p(real) - log1p(pred) medidos en la
validacion interna del entrenamiento (nunca con el mes a predecir ni con
ningun holdout):

    cota = max(expm1(log1p(pred) + q_alfa(r)), 0)

Los residuos se segmentan por cubetas de probabilidad de venta (prob_venta)
con los bordes EDGES; las cubetas contiguas con menos de MIN_FILAS filas se
fusionan. Las filas sin probabilidad solo cuentan en el cuantil global, que
tambien se usa como respaldo para productos sin probabilidad.
"""
import numpy as np

EDGES = (0.0, 0.4, 0.6, 0.8, 1.0)
MIN_FILAS = 50
NIVELES = (0.5, 0.7, 0.9)


def residuos_log(real, pred):
    """r = log1p(real) - log1p(pred) (ambos recortados en cero)."""
    real = np.clip(np.asarray(real, float), 0.0, None)
    pred = np.clip(np.asarray(pred, float), 0.0, None)
    return np.log1p(real) - np.log1p(pred)


def _cubeta(prob, edges):
    """Indice de cubeta [e_i, e_i+1); el ultimo borde es inclusivo."""
    prob = np.asarray(prob, float)
    return np.searchsorted(np.asarray(edges[1:-1], float), prob, side="right")


def _cuantil(r, alfa):
    return float(np.quantile(r, alfa)) if len(r) else 0.0


def _fusionar(conteos, min_filas):
    """Grupos contiguos de cubetas (lista de listas de indices) tal que cada
    grupo tenga >= min_filas filas (o quede uno solo). Se fusiona la cubeta
    mas pequena con su vecina de menor conteo."""
    grupos = [[i] for i in range(len(conteos))]
    cuenta = [int(c) for c in conteos]
    while len(grupos) > 1 and min(cuenta) < min_filas:
        i = cuenta.index(min(cuenta))
        vecinos = [j for j in (i - 1, i + 1) if 0 <= j < len(grupos)]
        j = min(vecinos, key=lambda k: (cuenta[k], k))
        a, b = sorted((i, j))
        grupos[a] = grupos[a] + grupos[b]
        cuenta[a] += cuenta[b]
        del grupos[b], cuenta[b]
    return grupos


def calibrar(prob, resid, niveles=NIVELES, edges=EDGES, min_filas=MIN_FILAS):
    """Cuantiles de los residuos por cubeta de probabilidad de venta.

    prob y resid son de la validacion interna (prob puede traer NaN: esas filas
    solo cuentan en 'global'). Devuelve un dict:
      edges    bordes finales (tras fusionar cubetas pequenas)
      n        filas por cubeta final
      q        {alfa: [cuantil por cubeta]}
      global   {alfa: cuantil de todos los residuos}
      n_total  filas totales (incluye las de prob NaN)
    Sin datos, todos los cuantiles son 0 (la cota es la prediccion)."""
    prob = np.asarray(prob, float).ravel()
    resid = np.asarray(resid, float).ravel()
    ok_r = ~np.isnan(resid)
    prob, resid = prob[ok_r], resid[ok_r]
    con_prob = ~np.isnan(prob)
    cub = _cubeta(prob[con_prob], edges)
    r_cub = resid[con_prob]
    conteos = np.bincount(cub, minlength=len(edges) - 1)
    grupos = _fusionar(conteos, min_filas)
    bordes = [float(edges[g[0]]) for g in grupos] + [float(edges[-1])]
    n = [int(sum(conteos[i] for i in g)) for g in grupos]
    q = {}
    for alfa in niveles:
        q[alfa] = [_cuantil(r_cub[np.isin(cub, g)], alfa) for g in grupos]
    return {
        "edges": bordes, "n": n, "q": q,
        "global": {alfa: _cuantil(resid, alfa) for alfa in niveles},
        "n_total": int(len(resid)),
    }


def cuantiles_para(cal, prob, alfa):
    """Cuantil de la cubeta de cada probabilidad; NaN usa el cuantil global."""
    prob = np.asarray(prob, float)
    out = np.full(prob.shape, cal["global"][alfa], dtype=float)
    ok = ~np.isnan(prob)
    idx = _cubeta(prob[ok], cal["edges"])
    out[ok] = np.asarray(cal["q"][alfa], float)[idx]
    return out


def cota_superior(pred, q):
    """max(expm1(log1p(pred) + q), 0); NaN si no hay prediccion."""
    pred = np.asarray(pred, float)
    with np.errstate(invalid="ignore"):
        cota = np.expm1(np.log1p(np.clip(pred, 0.0, None)) + np.asarray(q, float))
    return np.clip(cota, 0.0, None)


def unidades(stock, precio):
    """ceil(stock / precio); NaN sin precio positivo o sin stock."""
    stock = np.asarray(stock, float)
    precio = np.asarray(precio, float)
    ok = ~np.isnan(stock) & ~np.isnan(precio) & (precio > 0)
    out = np.full(np.broadcast(stock, precio).shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        cociente = np.where(ok, stock / np.where(ok, precio, 1.0), np.nan)
    # el redondeo evita que 3,0000000000000004 suba una unidad
    out[...] = np.where(ok, np.ceil(np.round(cociente, 9)), np.nan)
    return out
