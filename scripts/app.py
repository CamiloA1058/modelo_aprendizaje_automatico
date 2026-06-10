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
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ventas_forecast.paths import DATA_RAW, PREDICTIONS, ensure_output_dirs
    ensure_output_dirs()
    DEFAULT_DATA   = DATA_RAW    / "Query_Result_V5.csv"
    DEFAULT_OUTPUT = PREDICTIONS / "PREDICCION_MENSUAL.csv"
except ImportError:
    DEFAULT_DATA   = Path(__file__).parent / "Query_Result_V5.csv"
    DEFAULT_OUTPUT = Path(__file__).parent / "PREDICCION_MENSUAL.csv"

# ── Importar modelo ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from train_kmeans_rf_prod import SalesForecastModel

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

_state = {
    "status":  "idle",
    "log":     [],
    "results": None,
    "model":   None,
    "error":   None,
    "mes_predicho": None,
}


def _log(msg):
    _state["log"].append(msg)
    print(msg)


def _run_model(filepath, col_map, params):
    try:
        _state["status"] = "running"
        _state["log"]    = []
        _state["error"]  = None

        _log("▶ Iniciando pipeline de predicción…")

        model = SalesForecastModel(
            filepath=filepath,
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
        )

        _log("✔ Cargando y limpiando datos…")
        model.load_and_clean()
        _log(f"  → {len(model.df):,} registros | {model.df[model._c('id')].nunique():,} productos únicos")
        _log(f"  → Último mes en datos: {model._ultimo_mes.strftime('%Y-%m')}")
        _log(f"  → Mes a predecir: {model._mes_pred.strftime('%Y-%m')}")

        _log("✔ Construyendo variables predictoras…")
        model.build_features()

        _log("✔ Segmentando productos con KMeans…")
        model.cluster()

        _log("✔ Preparando conjuntos de entrenamiento y predicción…")
        model.prepare_split()
        n_pred = len(model.df_predict)
        _log(f"  → {n_pred:,} productos elegibles para predecir {model._mes_pred.strftime('%Y-%m')}")

        _log("✔ Entrenando clasificador de crecimiento…")
        model.train_classifier()
        m = model.metrics
        _log(f"  → Accuracy {m['accuracy']:.1%} | Precision {m['precision']:.1%} | Recall {m['recall']:.1%} | F1 {m['f1']:.1%}")

        _log("✔ Entrenando regresor de ventas…")
        model.train_regressor()

        _log(f"✔ Generando predicciones para {model._mes_pred.strftime('%Y-%m')}…")
        model.predict_next_month()

        _state["model"]        = model
        _state["results"]      = model.results
        _state["mes_predicho"] = model._mes_pred.strftime("%Y-%m")
        _state["status"]       = "done"
        _log(f"✅ Pipeline completado — {len(model.results):,} productos con predicción.")

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
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    if _state["status"] == "running":
        return jsonify({"ok": False, "msg": "El modelo ya está en ejecución."}), 409

    if "file" in request.files:
        f = request.files["file"]
        upload_dir = Path(__file__).parent / "uploads"
        upload_dir.mkdir(exist_ok=True)
        filepath = upload_dir / f.filename
        f.save(filepath)
    else:
        filepath = DEFAULT_DATA

    params      = request.form.to_dict()
    col_map_raw = params.pop("col_map", None)
    col_map     = json.loads(col_map_raw) if col_map_raw else None

    threading.Thread(
        target=_run_model, args=(filepath, col_map, params), daemon=True
    ).start()

    return jsonify({"ok": True, "msg": "Pipeline iniciado."})


@app.route("/api/summary")
def api_summary():
    df = _state["results"]
    if df is None:
        return jsonify({"ok": False}), 404

    model = _state["model"]
    c     = model._c

    total     = len(df)
    reforzar  = int((df["decision"] == "Reforzar stock").sum())
    reducir   = int((df["decision"] == "Reducir stock").sum())
    mantener  = total - reforzar - reducir
    avg_prob  = float(df["prob_crecimiento"].mean())

    # Ventas del último mes (base) y predichas
    base_total = float(df[c("total_sold")].sum())
    pred_col   = "ventas_predichas"
    df_con_pred = df[df[pred_col].notna()]
    pred_total  = float(df_con_pred[pred_col].sum()) if len(df_con_pred) else 0
    variacion   = ((pred_total - df_con_pred[c("total_sold")].sum()) /
                   df_con_pred[c("total_sold")].sum() * 100) if len(df_con_pred) and df_con_pred[c("total_sold")].sum() > 0 else 0

    # Historial mensual del modelo (para el gráfico de tendencia)
    hist = model.df.groupby("fecha").agg(
        ventas_reales=(c("total_sold"), "sum")
    ).reset_index().sort_values("fecha")
    hist["fecha"] = hist["fecha"].astype(str)
    hist_list = hist.tail(18).to_dict(orient="records")

    # Métricas del modelo
    metricas = model.metrics

    # Top productos por probabilidad de crecimiento
    top = (
        df[[c("id"), c("description"), "prob_crecimiento",
            pred_col, c("total_sold"), "decision", "cluster"]]
        .sort_values("prob_crecimiento", ascending=False)
        .head(20)
        .fillna(0)
        .rename(columns={
            c("id"):          "CODIGO",
            c("description"): "DESCRIPCION",
            c("total_sold"):  "ventas_base",
        })
        .to_dict(orient="records")
    )

    # Top productos con mayor probabilidad de reducción
    top_risk = (
        df[[c("id"), c("description"), "prob_reduccion",
            pred_col, c("total_sold"), "decision", "cluster"]]
        .sort_values("prob_reduccion", ascending=False)
        .head(20)
        .fillna(0)
        .rename(columns={
            c("id"):          "CODIGO",
            c("description"): "DESCRIPCION",
            c("total_sold"):  "ventas_base",
        })
        .to_dict(orient="records")
    )

    return jsonify({
        "ok":          True,
        "mes_predicho": _state["mes_predicho"],
        "total":       total,
        "reforzar":    reforzar,
        "mantener":    mantener,
        "reducir":     reducir,
        "avg_prob":    round(avg_prob * 100, 1),
        "pred_total":  round(pred_total, 0),
        "base_total":  round(base_total, 0),
        "variacion":   round(variacion, 1),
        "con_pred":    len(df_con_pred),
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
    dec      = request.args.get("decision", "")
    page     = int(request.args.get("page", 1))
    size     = int(request.args.get("size", 50))
    sort_by  = request.args.get("sort", "prob")   # prob | ventas | codigo

    out = df.copy().rename(columns={
        c("id"):          "CODIGO",
        c("description"): "DESCRIPCION",
        c("total_sold"):  "ventas_base",
        c("frequency"):   "FRECUENCIA",
        c("avg_price"):   "PRECIO_PROMEDIO",
    })
    out = out.fillna(0)

    if q:
        out = out[
            out["CODIGO"].astype(str).str.upper().str.contains(q, na=False) |
            out["DESCRIPCION"].astype(str).str.upper().str.contains(q, na=False)
        ]
    if dec:
        out = out[out["decision"] == dec]

    sort_map = {
        "prob":   ("prob_crecimiento", False),
        "ventas": ("ventas_predichas", False),
        "codigo": ("CODIGO", True),
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
        "data":  out.to_dict(orient="records"),
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
    pred = {k: (0 if (isinstance(v, float) and np.isnan(v)) else v) for k, v in pred.items()}

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
