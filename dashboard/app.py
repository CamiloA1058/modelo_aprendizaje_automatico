"""
app.py — Servidor web local para el dashboard de predicción de ventas.

Ejecutar:
    python app.py
Luego abrir:  http://localhost:5000
"""

import sys
import json
import traceback
import threading
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Rutas del proyecto ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.paths import DEFAULT_DATASET, DEFAULT_PREDICTION, UPLOADS, ensure_output_dirs  # noqa: E402

ensure_output_dirs()
DEFAULT_DATA   = DEFAULT_DATASET
DEFAULT_OUTPUT = DEFAULT_PREDICTION

# ── Importar modelo ───────────────────────────────────────────────────────────
from ventas_forecast.model import SalesForecastModel, ETIQUETA_SIN_RECOMENDACION
from ventas_forecast import grupos as grupos_mod
from ventas_forecast import fuente_bd
from ventas_forecast.figuras import fmt_es, pct_es

app = Flask(__name__)
CORS(app)

_state = {
    "status":  "idle",
    "log":     [],
    "results": None,
    "model":   None,
    "error":   None,
    "mes_predicho": None,
    "fuente":  "csv",   # "csv" | "bd"
    "rango":   None,    # {"desde", "hasta"} dd/mm/aaaa de los datos leídos de la base
}

ALTERNATIVA_CSV = "Puede seguir usando la opción «Subir archivo CSV»."


def _es_promedio(df):
    """Máscara de productos con estimación por promedio (origen_estimacion).
    Sin la columna (resultados antiguos) todos son del modelo."""
    if "origen_estimacion" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["origen_estimacion"] == "promedio"


def _msg_registros(n_registros, n_productos):
    return f"  → {fmt_es(n_registros)} registros | {fmt_es(n_productos)} productos únicos"


def _msg_clasificador(m):
    """Métricas principales (macro y balanceada), las mismas de la tesis."""
    return (f"  → F1 macro {fmt_es(m['f1_macro'], 2)} | "
            f"exactitud balanceada {pct_es(m['balanced_accuracy'])} | "
            f"exactitud {pct_es(m['accuracy'])}")


def _msg_umbral(umbral, exactitud_prueba):
    """El umbral se elige con la validación interna (dentro del
    entrenamiento); la exactitud que se muestra es la de la prueba interna."""
    return (f"  → Umbral {fmt_es(umbral, 2)} (elegido con la validación interna) | "
            f"exactitud en la prueba interna {pct_es(exactitud_prueba)}")


def _msg_lectura_bd(n_registros, ventana):
    return (f"  → Leyendo la base de datos… {fmt_es(n_registros)} registros "
            f"del {ventana.desde:%d/%m/%Y} al {ventana.hasta:%d/%m/%Y}")


def _msg_ultima_venta(fecha_max, ventana):
    """La última venta y si su mes quedó fuera por estar incompleto."""
    if fecha_max is None:
        return None
    if fecha_max > ventana.hasta:
        return (f"  → Última venta: {fecha_max:%d/%m/%Y} (su mes está incompleto y no "
                f"se usa: es el mes que se predice)")
    return f"  → Última venta: {fecha_max:%d/%m/%Y}"


def _leer_base(config_bd):
    """Ventas y grupos de la base (solo lectura) para el pipeline."""
    _log("✔ Leyendo la base de datos…")
    datos = fuente_bd.leer_ventas(config_bd)
    ventana = datos.attrs["ventana"]
    _log(_msg_lectura_bd(len(datos), ventana))
    ultima = _msg_ultima_venta(datos.attrs.get("fecha_max"), ventana)
    if ultima:
        _log(ultima)
    mapa = fuente_bd.leer_grupos(config_bd)
    _state["rango"] = {"desde": f"{ventana.desde:%d/%m/%Y}",
                       "hasta": f"{ventana.hasta:%d/%m/%Y}"}
    return datos, mapa


def _log(msg):
    _state["log"].append(msg)
    print(msg)


def _run_model(filepath, col_map, params, config_bd=None):
    """config_bd (fuente_bd.ConfigBD): lee los datos de la base en lugar del CSV."""
    try:
        _state["status"] = "running"
        _state["log"]    = []
        _state["error"]  = None
        _state["fuente"] = "bd" if config_bd is not None else "csv"
        _state["rango"]  = None

        _log("▶ Iniciando pipeline de predicción…")

        origen = {}
        if config_bd is not None:
            datos, mapa = _leer_base(config_bd)
            origen = {"datos": datos, "grupos_mapa": mapa}

        model = SalesForecastModel(
            filepath=filepath,
            **origen,
            output_path=DEFAULT_OUTPUT,
            col_map=col_map or None,
            n_clusters=int(params.get("n_clusters", 3)),
            growth_threshold=float(params.get("growth_threshold", 1.15)),
            reduce_threshold=float(params.get("reduce_threshold", 0.90)),
            clf_threshold=float(params.get("clf_threshold", 0.58)),
            top_n_products=int(params.get("top_n_products", 1900)),
            train_ratio=float(params.get("train_ratio", 0.8)),
            numeric_fmt=params.get("numeric_fmt", "dot_comma"),
            min_months=int(params.get("min_months", 6)),
            estimacion_por_promedio=str(
                params.get("estimacion_por_promedio", "true")).lower() != "false",
            probabilidad_venta=str(
                params.get("probabilidad_venta", "true")).lower() != "false",
            resumen_por_grupo=str(
                params.get("resumen_por_grupo", "true")).lower() != "false",
            stock_sugerido=str(
                params.get("stock_sugerido", "true")).lower() != "false",
            nivel_servicio=float(params.get("nivel_servicio", 0.70)),
        )

        _log("✔ Cargando y limpiando datos…")
        model.load_and_clean()
        _log(_msg_registros(len(model.df), model.df[model._c('id')].nunique()))
        _log(f"  → Último mes en datos: {model._ultimo_mes.strftime('%Y-%m')}")
        _log(f"  → Mes a predecir: {model._mes_pred.strftime('%Y-%m')}")

        _log("✔ Construyendo variables predictoras…")
        model.build_features()

        _log("✔ Segmentando productos con KMeans…")
        model.cluster()

        _log("✔ Preparando conjuntos de entrenamiento y predicción…")
        model.prepare_split()
        n_pred = len(model.df_predict)
        _log(f"  → {fmt_es(n_pred)} productos elegibles para predecir {model._mes_pred.strftime('%Y-%m')}")

        _log("✔ Entrenando clasificador de crecimiento…")
        model.train_classifier()
        m = model.metrics
        _log(_msg_clasificador(m))

        _log("✔ Entrenando regresor de ventas…")
        model.train_regressor()

        _log(f"✔ Generando predicciones para {model._mes_pred.strftime('%Y-%m')}…")
        model.predict_next_month()

        _log("✔ Estimando por promedio los productos sin ventas el último mes…")
        model.estimar_por_promedio()

        if model.probabilidad_venta:
            _log("✔ Calculando la probabilidad de venta de cada producto…")
            model.agregar_probabilidad_venta()
            _log(_msg_umbral(model.sale_threshold, model.sale_test_accuracy))

        if model.stock_sugerido:
            _log("✔ Calculando el stock sugerido de cada producto…")
            model.agregar_stock_sugerido()

        if model.resumen_por_grupo:
            _log("✔ Sumando las predicciones por grupo de producto…")
            model.agregar_resumen_grupo()

        _state["model"]        = model
        _state["results"]      = model.results
        _state["mes_predicho"] = model._mes_pred.strftime("%Y-%m")
        _state["status"]       = "done"
        _log(f"✅ Pipeline completado — {fmt_es(len(model.results))} productos con predicción.")

    except fuente_bd.ErrorBaseDatos as exc:
        # Mensaje apto para el usuario (sin claves ni rutas); sin traza.
        _state["status"] = "error"
        _state["error"]  = str(exc)
        _log(f"❌ Error: {exc}. {ALTERNATIVA_CSV}")
    except Exception as exc:
        _state["status"] = "error"
        _state["error"]  = traceback.format_exc()
        _log(f"❌ Error: {exc}")


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({
        "status":       _state["status"],
        "log":          _state["log"][-60:],
        "error":        _state["error"],
        "mes_predicho": _state["mes_predicho"],
        "fuente":       _state["fuente"],
        "rango":        _state["rango"],
    })


@app.route("/api/bd/estado")
def api_bd_estado():
    """¿Está configurada la lectura de la base? No se conecta ni devuelve rutas
    ni credenciales."""
    try:
        config = fuente_bd.cargar_config()
    except fuente_bd.ErrorConfiguracionBD as exc:
        return jsonify({"configurada": False, "msg": f"{exc}. {ALTERNATIVA_CSV}"})
    return jsonify({"configurada": True, "meses": config.meses})


STOCK_COLS = ("stock_sugerido", "stock_sugerido_unidades")


def _stock_nan_a_none(registros):
    """Los productos sin monto no tienen stock sugerido: null en el JSON (no 0)."""
    for r in registros:
        for k in STOCK_COLS:
            v = r.get(k)
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
    return registros


def _fillna_sin_stock(df):
    """fillna(0) que deja en NaN las columnas de stock sugerido."""
    return df.fillna({c: 0 for c in df.columns if c not in STOCK_COLS})


@app.route("/api/run", methods=["POST"])
def api_run():
    if _state["status"] == "running":
        return jsonify({"ok": False, "msg": "El modelo ya está en ejecución."}), 409

    fuente = request.form.get("fuente", "csv")
    config_bd = None
    if fuente == "bd":
        try:
            config_bd = fuente_bd.cargar_config()
        except fuente_bd.ErrorConfiguracionBD as exc:
            return jsonify({"ok": False, "msg": f"No se puede leer la base de datos: {exc}. "
                                                f"{ALTERNATIVA_CSV}"}), 400
        filepath = None
    elif "file" in request.files:
        f = request.files["file"]
        upload_dir = UPLOADS
        upload_dir.mkdir(parents=True, exist_ok=True)
        filepath = upload_dir / f.filename
        f.save(filepath)
    else:
        filepath = DEFAULT_DATA

    params      = request.form.to_dict()
    params.pop("fuente", None)
    col_map_raw = params.pop("col_map", None)
    col_map     = json.loads(col_map_raw) if col_map_raw else None

    threading.Thread(
        target=_run_model, args=(filepath, col_map, params),
        kwargs={"config_bd": config_bd}, daemon=True
    ).start()

    return jsonify({"ok": True, "msg": "Pipeline iniciado."})


@app.route("/api/summary")
def api_summary():
    df = _state["results"]
    if df is None:
        return jsonify({"ok": False}), 404

    model = _state["model"]
    c     = model._c

    # Los productos con estimación por promedio no tienen decisión del
    # modelo: las decisiones, la dona y las tablas de probabilidad cuentan
    # solo los productos del modelo; el total en pesos suma ambos grupos.
    es_prom  = _es_promedio(df)
    df_mod   = df[~es_prom]
    n_prom   = int(es_prom.sum())

    total     = len(df_mod)
    reforzar  = int((df_mod["decision"] == "Reforzar stock").sum())
    reducir   = int((df_mod["decision"] == "Reducir stock").sum())
    mantener  = total - reforzar - reducir
    avg_prob  = float(df_mod["prob_crecimiento"].mean()) if total else 0.0

    # Ventas del último mes (base) y predichas
    base_total = float(df[c("total_sold")].sum())
    pred_col   = "ventas_predichas"
    df_con_pred = df[df[pred_col].notna()]
    mod_con_pred = df_con_pred[~_es_promedio(df_con_pred)]
    # pred_total_modelo: suma de la predicción combinada (w*modelo + (1-w)*media
    # de 3 meses) de los productos del modelo; pred_total_promedio: suma de la
    # media de 3 meses de los productos sin ventas el último mes;
    # pred_total: ambas; pred_total_rf: el modelo solo, que subestima el total.
    pred_total_modelo   = float(mod_con_pred[pred_col].sum()) if len(mod_con_pred) else 0.0
    pred_total_promedio = float(df[es_prom][pred_col].sum()) if n_prom else 0.0
    pred_total     = pred_total_modelo + pred_total_promedio
    pred_total_rf  = (float(mod_con_pred["ventas_predichas_rf"].sum())
                      if "ventas_predichas_rf" in mod_con_pred.columns else pred_total_modelo)
    blend_weight   = getattr(model, "blend_weight", None)
    base_con_pred  = float(df_con_pred[c("total_sold")].sum()) if len(df_con_pred) else 0.0
    variacion      = ((pred_total - base_con_pred) / base_con_pred * 100
                      if base_con_pred > 0 else 0)

    # Historial mensual de los productos con predicción de monto (modelo y
    # promedio), para que la línea y el punto predicho sean comparables.
    hist = (
        model.df[model.df[c("id")].isin(df_con_pred[c("id")])]
        .groupby("fecha").agg(ventas_reales=(c("total_sold"), "sum"))
        .reset_index().sort_values("fecha")
    )
    hist["fecha"] = hist["fecha"].astype(str)
    hist_list = hist.tail(18).to_dict(orient="records")

    # Probabilidad de venta (todos los productos, modelo y promedio)
    tiene_venta = "se_vendera" in df.columns
    n_se_vendera = int((df["se_vendera"] == "Sí").sum()) if tiene_venta else None
    n_con_prob_venta = int(df["prob_venta"].notna().sum()) if tiene_venta else None

    # Métricas del modelo
    metricas = model.metrics

    cols_pred = ([pred_col]
                 + (["ventas_predichas_rf"] if "ventas_predichas_rf" in df.columns else [])
                 + (["prob_venta", "se_vendera"] if tiene_venta else [])
                 + [k for k in STOCK_COLS if k in df.columns])
    df = df_mod

    # Top productos por probabilidad de crecimiento
    top = (
        df[[c("id"), c("description"), "prob_crecimiento",
            *cols_pred, c("total_sold"), "decision", "cluster"]]
        .sort_values("prob_crecimiento", ascending=False)
        .head(20)
        .pipe(_fillna_sin_stock)
        .rename(columns={
            c("id"):          "CODIGO",
            c("description"): "DESCRIPCION",
            c("total_sold"):  "ventas_base",
        })
        .to_dict(orient="records")
    )
    top = _stock_nan_a_none(top)

    # Top productos con mayor probabilidad de reducción
    top_risk = (
        df[[c("id"), c("description"), "prob_reduccion",
            *cols_pred, c("total_sold"), "decision", "cluster"]]
        .sort_values("prob_reduccion", ascending=False)
        .head(20)
        .pipe(_fillna_sin_stock)
        .rename(columns={
            c("id"):          "CODIGO",
            c("description"): "DESCRIPCION",
            c("total_sold"):  "ventas_base",
        })
        .to_dict(orient="records")
    )
    top_risk = _stock_nan_a_none(top_risk)

    return jsonify({
        "ok":          True,
        "mes_predicho": _state["mes_predicho"],
        "total":       total,
        "reforzar":    reforzar,
        "mantener":    mantener,
        "reducir":     reducir,
        "avg_prob":    round(avg_prob * 100, 1),
        "pred_total":  round(pred_total, 0),
        "pred_total_rf": round(pred_total_rf, 0),
        "pred_total_modelo":   round(pred_total_modelo, 0),
        "pred_total_promedio": round(pred_total_promedio, 0),
        "n_promedio":  n_prom,
        "n_se_vendera": n_se_vendera,
        "n_con_prob_venta": n_con_prob_venta,
        "sale_threshold": getattr(model, "sale_threshold", None),
        "sale_val_accuracy": getattr(model, "sale_val_accuracy", None),
        "sale_test_accuracy": getattr(model, "sale_test_accuracy", None),
        "blend_weight": blend_weight,
        "base_total":  round(base_total, 0),
        "variacion":   round(variacion, 1),
        "con_pred":    len(df_con_pred),
        "con_pred_modelo": len(mod_con_pred),
        "historial":   hist_list,
        "metricas":    metricas,
        "top_growth":  top,
        "top_risk":    top_risk,
    })


@app.route("/api/products")
def api_products():
    df = _state["results"]
    if df is None:
        return jsonify({"ok": False}), 404

    model = _state["model"]
    c     = model._c

    q        = request.args.get("q", "").upper()
    grupo    = request.args.get("grupo", "")
    dec      = request.args.get("decision", "")
    page     = int(request.args.get("page", 1))
    size     = int(request.args.get("size", 50))
    sort_by  = request.args.get("sort", "prob")   # prob | ventas | codigo | venta_prob
    venta    = request.args.get("venta", "")      # "Sí" | "No" (se_vendera)

    out = df.copy().rename(columns={
        c("id"):          "CODIGO",
        c("description"): "DESCRIPCION",
        c("total_sold"):  "ventas_base",
        c("frequency"):   "FRECUENCIA",
        c("avg_price"):   "PRECIO_PROMEDIO",
    })
    if "origen_estimacion" not in out.columns:
        out["origen_estimacion"] = "modelo"
    out = _fillna_sin_stock(out)

    # Grupo de producto (catálogo cargado por agregar_resumen_grupo)
    mapa = getattr(model, "mapa_grupos", None)
    if mapa is not None:
        g = pd.Series(grupos_mod.asignar_grupo(out["CODIGO"], mapa), index=out.index)
        out["grupo"] = g.where(g != grupos_mod.SIN_GRUPO, grupos_mod.ETIQUETA_SIN_GRUPO)
        if grupo:
            out = out[out["grupo"] == grupo]

    if q:
        out = out[
            out["CODIGO"].astype(str).str.upper().str.contains(q, na=False) |
            out["DESCRIPCION"].astype(str).str.upper().str.contains(q, na=False)
        ]
    if dec:
        out = out[out["decision"] == dec]
    if venta and "se_vendera" in out.columns:
        out = out[out["se_vendera"] == venta]

    sort_map = {
        "prob":   ("prob_crecimiento", False),
        "ventas": ("ventas_predichas", False),
        "codigo": ("CODIGO", True),
        "venta_prob": ("prob_venta", False),
        "stock":  ("stock_sugerido", False),
    }
    col_s, asc = sort_map.get(sort_by, ("prob_crecimiento", False))
    if col_s in out.columns:
        out = out.sort_values(col_s, ascending=asc)

    total = len(out)
    out   = out.iloc[(page - 1) * size: page * size]

    return jsonify({
        "ok":    True,
        "total": total,
        "page":  page,
        "size":  size,
        "mes_predicho": _state["mes_predicho"],
        "data":  _stock_nan_a_none(out.to_dict(orient="records")),
    })


@app.route("/api/grupos")
def api_grupos():
    """Suma de las predicciones por grupo de producto (variante G0)."""
    df = _state["results"]
    if df is None:
        return jsonify({"ok": False}), 404
    tabla = getattr(_state["model"], "resumen_grupos", None)
    if tabla is None:
        tabla = pd.DataFrame({c: [] for c in grupos_mod.COLUMNAS_RESUMEN})
    filas = tabla.astype(object).where(tabla.notna(), None).to_dict(orient="records")
    return jsonify({
        "ok":           True,
        "mes_predicho": _state["mes_predicho"],
        "total_grupos": len(filas),
        "pred_total":   float(tabla["prediccion"].sum()) if len(tabla) else 0.0,
        "data":         filas,
    })


@app.route("/api/product/<codigo>")
def api_product(codigo):
    df    = _state["results"]
    model = _state["model"]
    if df is None:
        return jsonify({"ok": False}), 404

    c = model._c

    # Fila de predicción
    pred_row = df[df[c("id")].astype(str) == str(codigo)]
    if pred_row.empty:
        return jsonify({"ok": False, "msg": "Producto no encontrado"}), 404

    # Historial completo del producto
    hist = (
        model.df[model.df[c("id")].astype(str) == str(codigo)]
        .sort_values("fecha")
        [[c("id"), c("description"), "fecha", c("total_sold"), c("frequency"), c("avg_price")]]
        .copy()
    )
    hist["fecha"] = hist["fecha"].astype(str)

    pred = pred_row.iloc[0].to_dict()
    pred = {k: ((None if k in STOCK_COLS else 0)
                if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in pred.items()}
    pred.setdefault("origen_estimacion", "modelo")
    if pred["origen_estimacion"] == "promedio":
        # Estimación por promedio: sin probabilidades ni monto del modelo.
        for k in ("prob_crecimiento", "prob_reduccion", "ventas_predichas_rf"):
            pred[k] = None
    # El panel de detalle lee las ventas del último mes como `ventas_base`,
    # igual que las tablas del resumen.
    pred["ventas_base"] = pred.get(c("total_sold"), 0)

    return jsonify({
        "ok":          True,
        "codigo":      codigo,
        "descripcion": pred.get(c("description"), ""),
        "mes_predicho": _state["mes_predicho"],
        "prediccion":  pred,
        "historial":   hist.to_dict(orient="records"),
    })


@app.route("/api/download")
def api_download():
    if not DEFAULT_OUTPUT.exists():
        return jsonify({"ok": False, "msg": "No hay archivo generado"}), 404
    return send_from_directory(
        DEFAULT_OUTPUT.parent, DEFAULT_OUTPUT.name, as_attachment=True
    )


@app.route("/")
def index():
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Dashboard de Predicción de Ventas")
    print("  Abre tu navegador en:  http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
