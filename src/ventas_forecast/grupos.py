"""Prediccion mensual por grupo de producto (PRODUCTO.IDGRUPO).

Las compras del taller se deciden por linea de producto, y el monto por
producto es demasiado ruidoso para eso (ver monto_producto.py). Aqui se agregan
las ventas por grupo y se ofrecen las variantes que compara
scripts/experimento_grupos.py:

- suma de las predicciones por producto de produccion (G0);
- media de los ultimos 3 meses del grupo (G1);
- modelo directo sobre la serie mensual del grupo (G2), con la mezcla hacia la
  media de 3 meses elegida con una validacion interna (como en produccion);
- mezclas y reconciliacion de las predicciones por producto al total del grupo.

Todo se calcula con datos estrictamente anteriores al mes predicho.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

SIN_GRUPO = "SIN GRUPO"
LAGS = (1, 2, 3, 6, 12)
FEATURES = [f"lag_{k}" for k in LAGS] + [
    "mean_3", "mean_6", "activos", "facturas", "mes_sin", "mes_cos"]
GRID_PESOS = np.round(np.linspace(0.0, 1.0, 11), 2)


# ── Mapa producto -> grupo ───────────────────────────────────────────────────
def _leer(source):
    kwargs = {"sep": ";", "dtype": str, "keep_default_na": False}
    if hasattr(source, "read"):
        return pd.read_csv(source, **kwargs)
    return pd.read_csv(source, encoding="utf-8-sig", **kwargs)


def cargar_mapa(source):
    """Serie codigo (texto) -> nombre del grupo; sin grupo = SIN_GRUPO."""
    df = _leer(source)
    nombre = df["grupo"].str.strip().replace("", SIN_GRUPO)
    return pd.Series(nombre.to_numpy(), index=df["codigo"].str.strip().to_numpy())


def asignar_grupo(codigos, mapa):
    """Grupo de cada codigo; un codigo desconocido cae en SIN_GRUPO."""
    return pd.Series(codigos).astype(str).map(mapa).fillna(SIN_GRUPO).to_numpy()


# ── Ventas mensuales y series por grupo ──────────────────────────────────────
def _numero_es(s):
    return s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)


def leer_ventas(sources):
    """Ventas por (codigo, mes): total en COP y numero de facturas (FRECUENCIA)."""
    marcos = []
    for src in sources:
        d = _leer(src)
        mes = pd.to_datetime(d["FECHA"], format="%d/%m/%Y").dt.to_period("M").dt.to_timestamp()
        marcos.append(pd.DataFrame({
            "codigo": d["CODIGO"].str.strip(), "mes": mes,
            "total": _numero_es(d["TOTAL_VENDIDO"]),
            "facturas": pd.to_numeric(d["FRECUENCIA"]),
        }))
    df = pd.concat(marcos, ignore_index=True)
    return df.groupby(["codigo", "mes"], as_index=False).agg(
        total=("total", "sum"), facturas=("facturas", "sum"))


def panel_grupo(ventas, mapa, mes, productos=None):
    """Series mensuales por grupo (meses x grupos) con datos < mes ('AAAA-MM').

    Devuelve {'total', 'activos', 'facturas'}; el calendario es continuo (los
    meses sin ventas valen 0) y las columnas son los grupos con ventas antes de
    'mes'. 'productos' restringe los codigos sumados (universo de un origen)."""
    corte = pd.Timestamp(mes + "-01")
    v = ventas[ventas["mes"] < corte].copy()
    v["grupo"] = asignar_grupo(v["codigo"], mapa)
    grupos = sorted(v["grupo"].unique())
    meses = pd.date_range(v["mes"].min(), corte - pd.offsets.MonthBegin(1), freq="MS")
    if productos is not None:
        v = v[v["codigo"].isin({str(p) for p in productos})]
    v["activo"] = (v["total"] > 0).astype(float)
    v["facturas"] = v["facturas"].astype(float)
    out = {}
    for clave, col in (("total", "total"), ("activos", "activo"), ("facturas", "facturas")):
        out[clave] = (v.pivot_table(index="mes", columns="grupo", values=col,
                                    aggfunc="sum", fill_value=0.0)
                      .reindex(index=meses, columns=grupos, fill_value=0.0).astype(float))
    return out


# ── Metricas por grupo ───────────────────────────────────────────────────────
def exactitud_grupo(real, pred):
    """Acierto por grupo 1 - |R-P|/R recortado a [0, 1]; NaN si R <= 0."""
    r, p = np.asarray(real, float), np.asarray(pred, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.clip(1.0 - np.abs(r - p) / r, 0.0, 1.0)
    return np.where(r > 0, e, np.nan)


def acierto_ponderado(real, pred):
    """Acierto ponderado por valor: 1 - WAPE sobre todos los grupos."""
    r, p = np.asarray(real, float), np.asarray(pred, float)
    return float(1.0 - np.abs(r - p).sum() / r.sum()) if r.sum() else float("nan")


def resumen(real, pred, top=20):
    """Metricas de un conjunto de grupos (uno o varios meses apilados)."""
    r, p = np.asarray(real, float), np.asarray(pred, float)
    e = exactitud_grupo(r, p)
    ok = r > 0
    orden = np.argsort(-r, kind="stable")[:top]
    return {
        "ponderado": acierto_ponderado(r, p),
        "media": float(np.nanmean(e)) if ok.any() else float("nan"),
        "pct_70": float(np.mean(e[ok] >= 0.7)) if ok.any() else float("nan"),
        "top": acierto_ponderado(r[orden], p[orden]),
        "razon": float(p.sum() / r.sum()) if r.sum() else float("nan"),
        "n": int(ok.sum()),
    }


# ── Atributos del modelo directo ─────────────────────────────────────────────
def _atributos(panel):
    T = panel["total"].clip(lower=0.0)
    mean_3 = T.rolling(3, min_periods=1).mean()
    mean_6 = T.rolling(6, min_periods=1).mean()
    base = {}
    for k in LAGS:
        base[f"lag_{k}"] = np.log1p(T.shift(k - 1)).fillna(np.log1p(mean_3))
    base["mean_3"] = np.log1p(mean_3)
    base["mean_6"] = np.log1p(mean_6)
    base["activos"] = np.log1p(panel["activos"])
    base["facturas"] = np.log1p(panel["facturas"].clip(lower=0.0))
    objetivo = T.index + pd.offsets.MonthBegin(1)
    ang = 2 * np.pi * objetivo.month.to_numpy() / 12
    base["mes_sin"] = pd.DataFrame(np.repeat(np.sin(ang)[:, None], T.shape[1], 1),
                                   index=T.index, columns=T.columns)
    base["mes_cos"] = pd.DataFrame(np.repeat(np.cos(ang)[:, None], T.shape[1], 1),
                                   index=T.index, columns=T.columns)
    return base, mean_3, T


def filas_entrenamiento(panel):
    """(X, y, meta): una fila por (grupo, mes t) con objetivo log1p(total t+1)."""
    base, mean_3, T = _atributos(panel)
    n = len(T)
    filas = slice(0, n - 1)
    cols = {k: v.iloc[filas].stack() for k, v in base.items()}
    X = pd.DataFrame(cols)[FEATURES]
    y = np.log1p(T.shift(-1).iloc[filas]).stack()
    meta = pd.DataFrame({
        "grupo": X.index.get_level_values(1),
        "mes_origen": X.index.get_level_values(0),
        "media_3": mean_3.iloc[filas].stack().to_numpy(),
    })
    X = X.reset_index(drop=True)
    y = pd.Series(y.to_numpy(), index=X.index)
    return X, y, meta


def fila_origen(panel):
    """(X, grupos): atributos de cada grupo en el ultimo mes conocido."""
    base, _, T = _atributos(panel)
    X = pd.DataFrame({k: v.iloc[-1].to_numpy() for k, v in base.items()})[FEATURES]
    return X, pd.Index(T.columns)


# ── Variantes ────────────────────────────────────────────────────────────────
def media_3(panel):
    """Media de los ultimos 3 meses de cada grupo."""
    return panel["total"].clip(lower=0.0).iloc[-3:].mean()


def mezcla(w, a, b):
    return w * np.asarray(a, float) + (1.0 - w) * np.asarray(b, float)


def elegir_peso(real, a, b, grid=GRID_PESOS):
    """Peso w de w*a + (1-w)*b que minimiza el error absoluto total."""
    r = np.asarray(real, float)
    errores = [np.abs(r - mezcla(w, a, b)).sum() for w in grid]
    return float(grid[int(np.argmin(errores))])


def peso_online(reales, as_, bs, defecto=0.5, grid=GRID_PESOS):
    """Peso elegido solo con meses ya cerrados (listas de arreglos por mes);
    sin historia devuelve 'defecto'."""
    if not len(reales):
        return defecto
    return elegir_peso(np.concatenate(reales), np.concatenate(as_), np.concatenate(bs), grid)


def rf_grupo():
    return RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                                 random_state=42, n_jobs=-1)


def predecir_directo(panel, estimador=rf_grupo, val_meses=3):
    """Modelo directo del total mensual de cada grupo para el mes siguiente al
    ultimo del panel. Entrena con log1p del total y mezcla con la media de 3
    meses con el peso que minimiza el error en los ultimos 'val_meses' meses
    conocidos (el modelo de esa validacion no ve esos meses)."""
    X, y, meta = filas_entrenamiento(panel)
    n = len(panel["total"])
    origenes = panel["total"].index
    val = (meta["mes_origen"] >= origenes[n - 1 - val_meses]).to_numpy()
    entrena = ~val
    peso = 0.5
    if entrena.sum() >= 20 and val.any():
        m = estimador().fit(X[entrena], y[entrena])
        rf_val = np.expm1(m.predict(X[val]))
        peso = elegir_peso(np.expm1(y[val]), rf_val, meta.loc[val, "media_3"])
    modelo = estimador().fit(X, y)
    Xo, grupos = fila_origen(panel)
    rf = pd.Series(np.expm1(modelo.predict(Xo)).clip(min=0.0), index=grupos)
    base = media_3(panel).reindex(grupos)
    pred = pd.Series(mezcla(peso, rf, base), index=grupos)
    return {"pred": pred, "peso": peso, "rf": rf, "media_3": base}


def reconciliar(pred, grupo, total_grupo):
    """Reparte el total del grupo entre sus productos en proporcion a la
    prediccion por producto. Sin prediccion o sin total del grupo no cambia."""
    pred = pd.Series(pred).astype(float)
    grupo = pd.Series(grupo, index=pred.index)
    suma = pred.groupby(grupo).transform("sum")
    total = grupo.map(total_grupo).astype(float)
    factor = np.where((suma > 0) & total.notna(), total / suma.where(suma > 0), 1.0)
    return pd.Series(pred.to_numpy() * factor, index=pred.index)


# ── Tabla del mes predicho ───────────────────────────────────────────────────
def tabla_grupos(resultados, mapa, id_col, total_col, pred_grupo):
    """Una fila por grupo con productos que tienen monto: prediccion del grupo,
    ventas del mes anterior de esos productos y variacion. Ordenada por la
    prediccion del grupo (mayor primero)."""
    r = resultados[resultados["ventas_predichas"].notna()].copy()
    r["grupo"] = asignar_grupo(r[id_col], mapa)
    g = r.groupby("grupo").agg(n=(id_col, "size"), previo=(total_col, "sum"))
    out = pd.DataFrame({
        "Grupo": g.index,
        "Productos": g["n"].to_numpy(),
        "Predicción del grupo (COP)": pd.Series(pred_grupo).reindex(g.index)
        .fillna(0.0).to_numpy(float),
        "Ventas mes anterior (COP)": g["previo"].to_numpy(float),
    })
    prev = out["Ventas mes anterior (COP)"]
    out["Variación"] = np.where(prev > 0, out["Predicción del grupo (COP)"] / prev.where(prev > 0) - 1.0,
                                np.nan)
    return (out.sort_values("Predicción del grupo (COP)", ascending=False, kind="stable")
            .reset_index(drop=True))


# ── Vista por grupo del mes predicho (dashboard) ─────────────────────────────
ETIQUETA_SIN_GRUPO = "Sin grupo"
COLUMNAS_RESUMEN = ["grupo", "productos", "se_vendera", "prediccion",
                    "mes_anterior", "variacion", "pct_total"]


def resumen_grupos(resultados, mapa, id_col, total_col):
    """Suma de las predicciones por producto de cada grupo (variante G0).

    Una fila por grupo, ordenada por la prediccion del grupo: productos con
    monto predicho (modelo y promedio), cuantos de ellos probablemente se
    venderan (se_vendera == "Si"), prediccion, ventas del mes anterior de esos
    mismos productos, variacion (fraccion; NaN sin ventas previas) y fraccion
    del total predicho. Los codigos sin grupo van en "Sin grupo". No modifica
    'resultados'."""
    r = resultados[resultados["ventas_predichas"].notna()]
    if len(r) == 0:
        return pd.DataFrame({c: [] for c in COLUMNAS_RESUMEN})
    grupo = pd.Series(asignar_grupo(r[id_col], mapa), index=r.index)
    grupo = grupo.where(grupo != SIN_GRUPO, ETIQUETA_SIN_GRUPO)
    if "se_vendera" in r.columns:
        vende = (r["se_vendera"] == "Sí").astype(int)
    else:
        vende = pd.Series(0, index=r.index)
    g = pd.DataFrame({
        "grupo": grupo, "n": 1, "vende": vende,
        "pred": r["ventas_predichas"].astype(float),
        "previo": r[total_col].astype(float),
    }).groupby("grupo").sum()
    total = g["pred"].sum()
    previo = g["previo"]
    out = pd.DataFrame({
        "grupo": g.index,
        "productos": g["n"].to_numpy(int),
        "se_vendera": g["vende"].to_numpy(int),
        "prediccion": g["pred"].to_numpy(float),
        "mes_anterior": previo.to_numpy(float),
        "variacion": np.where(previo > 0, g["pred"] / previo.where(previo > 0) - 1.0,
                              np.nan),
        "pct_total": g["pred"].to_numpy(float) / total if total else np.nan,
    })
    return out.sort_values("prediccion", ascending=False, kind="stable").reset_index(drop=True)
