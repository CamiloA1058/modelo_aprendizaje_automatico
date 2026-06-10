"""
train_kmeans_rf_prod.py
=======================
Modelo de predicción de ventas reutilizable — KMeans + Random Forest.

LÓGICA PRINCIPAL
----------------
El modelo se entrena con el historial completo (donde el target es el mes
siguiente real). Luego genera predicciones REALES para el mes que aún no
existe (último mes del dataset + 1) usando como input el estado actual de
cada producto. Solo se incluyen productos con suficientes datos históricos.

Uso básico
----------
    from train_kmeans_rf_prod import SalesForecastModel

    model = SalesForecastModel(filepath="mi_dataset.csv")
    model.run()

Columnas mínimas requeridas
----------------------------
    id, description, year, month, sales, total_sold, avg_price, frequency
    (ver col_map para nombres alternativos)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

# ── Nombres de columna por defecto ────────────────────────────────────────────
DEFAULT_COL_MAP = {
    "id":          "CODIGO",
    "description": "DESCRIPCION",
    "year":        "ANIO",
    "month":       "MES",
    "sales":       "VENTAS",
    "total_sold":  "TOTAL_VENDIDO",
    "avg_price":   "PRECIO_PROMEDIO",
    "frequency":   "FRECUENCIA",
}


class SalesForecastModel:
    """
    Pipeline completo:
      1. Entrenamiento con historial (target = mes siguiente real)
      2. Predicción hacia adelante del mes que aún no existe
         — solo productos con suficientes datos en el último mes

    Parámetros
    ----------
    filepath            : ruta al CSV
    sep                 : separador (default ';')
    encoding            : codificación (default 'utf-8')
    col_map             : dict {rol: nombre_en_csv} — ver DEFAULT_COL_MAP
    output_path         : ruta del CSV de salida
    n_clusters          : clusters KMeans (default 3)
    lags                : lista de lags (default [1,2,3,6])
    growth_threshold    : umbral crecimiento clase positiva (default 1.10)
    reduce_threshold    : umbral caída para decisión "Reducir stock" (default 0.90)
    clf_threshold       : umbral probabilidad clasificador (default 0.65)
    top_n_products      : top productos para regresor (default 200)
    train_ratio         : fracción entrenamiento (default 0.8)
    numeric_fmt         : 'dot_comma' | 'plain'
    min_months          : meses mínimos de historial para predecir (default 6)
    """

    def __init__(
        self,
        filepath,
        sep=";",
        encoding="utf-8",
        col_map=None,
        output_path=None,
        n_clusters=3,
        lags=None,
        growth_threshold=1.15,
        reduce_threshold=0.90,
        clf_threshold=0.65,
        top_n_products=1900,
        train_ratio=0.8,
        numeric_fmt="dot_comma",
        min_months=6,
    ):
        self.filepath        = Path(filepath)
        self.sep             = sep
        self.encoding        = encoding
        self.col_map         = {**DEFAULT_COL_MAP, **(col_map or {})}
        self.output_path     = (
            Path(output_path) if output_path
            else self.filepath.parent / "PREDICCION_MENSUAL.csv"
        )
        self.n_clusters      = n_clusters
        self.lags            = lags or [1, 2, 3, 6]
        self.growth_threshold = growth_threshold
        self.reduce_threshold = reduce_threshold
        self.clf_threshold   = clf_threshold
        self.top_n_products  = top_n_products
        self.train_ratio     = train_ratio
        self.numeric_fmt     = numeric_fmt
        self.min_months      = min_months

        # Estado interno
        self.df           = None   # historial completo
        self.df_predict   = None   # filas del último mes (para predecir)
        self.scaler       = RobustScaler()
        self.clf          = None
        self.reg          = None
        self.results      = None   # predicciones del mes siguiente
        self.metrics      = {}
        self._features    = None
        self._top_prods   = None
        self._ultimo_mes  = None
        self._mes_pred    = None

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _c(self, role):
        return self.col_map[role]

    def _parse_numeric(self, col):
        if self.numeric_fmt == "dot_comma":
            return (
                col.astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
        return col.astype(float)

    # ── 1. CARGA Y LIMPIEZA ───────────────────────────────────────────────────
    def load_and_clean(self):
        c = self._c
        self.df = pd.read_csv(self.filepath, sep=self.sep, encoding=self.encoding)

        for role in ("sales", "total_sold", "avg_price"):
            self.df[c(role)] = self._parse_numeric(self.df[c(role)])

        self.df[c("year")] = (
            self.df[c("year")].astype(str)
            .str.replace(".", "", regex=False).astype(int)
        )
        self.df["fecha"] = pd.to_datetime(
            self.df[c("year")].astype(str) + "-" +
            self.df[c("month")].astype(str).str.zfill(2) + "-01"
        )

        # Agregar a un registro por producto-mes
        self.df = (
            self.df.groupby([c("id"), c("description"),
                             pd.Grouper(key="fecha", freq="MS")])
            .agg({
                c("sales"):      "sum",
                c("total_sold"): "sum",
                c("avg_price"):  "mean",
                c("frequency"):  "sum",
            })
            .reset_index()
        )

        self.df[c("year")]  = self.df["fecha"].dt.year
        self.df[c("month")] = self.df["fecha"].dt.month
        self.df = self.df.sort_values([c("id"), "fecha"]).reset_index(drop=True)

        self._ultimo_mes = self.df["fecha"].max()
        self._mes_pred   = self._ultimo_mes + pd.DateOffset(months=1)

        print(f"[✓] Datos cargados: {len(self.df):,} registros | "
              f"{self.df[c('id')].nunique():,} productos únicos")
        print(f"[✓] Último mes en datos: {self._ultimo_mes.strftime('%Y-%m')}")
        print(f"[✓] Mes a predecir:      {self._mes_pred.strftime('%Y-%m')}")

    # ── 2. FEATURES ───────────────────────────────────────────────────────────
    def build_features(self):
        c = self._c
        df = self.df

        # Target para entrenamiento: ventas del MES SIGUIENTE (shift -1)
        df["target"] = df.groupby(c("id"))[c("total_sold")].shift(-1)

        # Lags
        for lag in self.lags:
            df[f"lag_{lag}"] = df.groupby(c("id"))[c("total_sold")].shift(lag)

        # Medias móviles
        df["mean_3"] = df.groupby(c("id"))[c("total_sold")].transform(
            lambda x: x.rolling(3).mean().shift(1))
        df["mean_6"] = df.groupby(c("id"))[c("total_sold")].transform(
            lambda x: x.rolling(6).mean().shift(1))

        # Ratio de crecimiento
        df["growth_ratio"] = df["target"] / df[c("total_sold")]

        # Logarítmicas
        df["VENTAS_LOG"] = np.log1p(df[c("sales")])
        df["TOTAL_LOG"]  = np.log1p(df[c("total_sold")])
        df["PRECIO_LOG"] = np.log1p(df[c("avg_price")])

        # Estacionalidad cíclica
        df["mes_sin"] = np.sin(2 * np.pi * df[c("month")] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df[c("month")] / 12)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = df
        print("[✓] Features construidas")

    # ── 3. CLUSTERING ─────────────────────────────────────────────────────────
    def cluster(self):
        c = self._c
        X_c = RobustScaler().fit_transform(
            self.df[[c("total_sold"), c("frequency"), c("avg_price")]].fillna(0)
        )
        self.df["cluster"] = KMeans(
            n_clusters=self.n_clusters, random_state=42, n_init=10
        ).fit_predict(X_c)
        print(f"[✓] KMeans con {self.n_clusters} clusters")

    # ── 4. SEPARAR: histórico de entrenamiento vs último mes ─────────────────
    def prepare_split(self):
        """
        df_train : historial con target conocido (todos los meses excepto el último,
                   porque el último no tiene 'mes siguiente' real)
        df_predict: filas del último mes con suficientes datos → se predice abril
        """
        c = self._c

        lag_cols = [f"lag_{l}" for l in self.lags]
        self._features = [
            "VENTAS_LOG", "TOTAL_LOG", "PRECIO_LOG",
            c("frequency"),
            *lag_cols,
            "mean_3", "mean_6",
            "mes_sin", "mes_cos",
            "cluster",
        ]

        # ── Dataset de entrenamiento ──────────────────────────────────────────
        # Solo filas donde el target (mes siguiente) es conocido y válido
        df_train = self.df[
            self.df["target"].notna() &
            (self.df["target"] > 0) &
            (self.df[c("total_sold")] > 0)
        ].copy().reset_index(drop=True)

        # Target en 3 clases:
        #   2 = Reforzar  (growth_ratio > growth_threshold)
        #   1 = Mantener  (reduce_threshold <= growth_ratio <= growth_threshold)
        #   0 = Reducir   (growth_ratio < reduce_threshold)
        df_train["target_class"] = np.where(
            df_train["growth_ratio"] > self.growth_threshold, 2,
            np.where(df_train["growth_ratio"] < self.reduce_threshold, 0, 1)
        )

        X = df_train[self._features].fillna(0)
        y = df_train["target_class"]
        X_scaled = self.scaler.fit_transform(X)

        # Split temporal por producto
        train_idx, test_idx = [], []
        for _, g in df_train.groupby(c("id")):
            g = g.sort_values("fecha")
            cut = int(len(g) * self.train_ratio)
            train_idx += g.index[:cut].tolist()
            test_idx  += g.index[cut:].tolist()

        self._df_train   = df_train
        self._X          = X
        self._y          = y
        self._X_scaled   = X_scaled
        self._train_idx  = train_idx
        self._test_idx   = test_idx

        # ── Dataset de predicción real ────────────────────────────────────────
        # Productos que tienen dato en el último mes con suficiente historial
        conteo = self.df.groupby(c("id"))["fecha"].count()
        prods_ok = conteo[conteo >= self.min_months].index

        df_pred = self.df[
            (self.df["fecha"] == self._ultimo_mes) &
            (self.df[c("id")].isin(prods_ok)) &
            (self.df[c("total_sold")] > 0)
        ].copy()

        # Calcular features del mes siguiente (el lag_1 es el mes actual)
        # Para predecir abril: lag_1=marzo, lag_2=feb, lag_3=ene, lag_6=oct
        df_pred_feat = df_pred.copy()

        # El mes a predecir tiene mes_sin/cos del MES SIGUIENTE
        next_month = self._mes_pred.month
        df_pred_feat["mes_sin"] = np.sin(2 * np.pi * next_month / 12)
        df_pred_feat["mes_cos"] = np.cos(2 * np.pi * next_month / 12)

        df_pred_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df_predict = df_pred_feat

        excluidos = self.df[self.df["fecha"] == self._ultimo_mes][c("id")].nunique() - len(df_pred)
        print(f"[✓] Entrenamiento: {len(df_train):,} filas | "
              f"train={len(train_idx):,} test={len(test_idx):,}")
        print(f"[✓] Productos a predecir ({self._mes_pred.strftime('%Y-%m')}): "
              f"{len(df_pred):,} con >={self.min_months} meses de historial")
        if excluidos > 0:
            print(f"    (excluidos {excluidos} por historial insuficiente)")

    # ── 5. CLASIFICADOR ───────────────────────────────────────────────────────
    def train_classifier(self):
        X_tr = self._X_scaled[self._train_idx]
        y_tr = self._y.iloc[self._train_idx]
        X_te = self._X_scaled[self._test_idx]
        y_te = self._y.iloc[self._test_idx]

        self.clf = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.clf.fit(X_tr, y_tr)

        pred = self.clf.predict(X_te)

        self.metrics = {
            "accuracy":  round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred, average="weighted", zero_division=0), 4),
            "recall":    round(recall_score(y_te, pred, average="weighted", zero_division=0), 4),
            "f1":        round(f1_score(y_te, pred, average="weighted", zero_division=0), 4),
        }

        print("\n===== MÉTRICAS DE VALIDACIÓN =====")
        for k, v in self.metrics.items():
            print(f"  {k.capitalize():<10}: {v:.2%}")
        print("\n", classification_report(y_te, pred,
              target_names=["Reducir", "Mantener", "Reforzar"], zero_division=0))

    # ── 6. REGRESOR ───────────────────────────────────────────────────────────
    def train_regressor(self):
        c = self._c
        self._top_prods = (
            self._df_train.groupby(c("id"))[c("total_sold")]
            .sum().nlargest(self.top_n_products).index
        )

        mask = (
            self._df_train[c("id")].isin(self._top_prods) &
            (self._df_train["target"] > 0)
        )
        df_reg = self._df_train[mask]
        X_reg  = self._X.loc[df_reg.index]
        y_reg  = np.log1p(df_reg["target"])

        self.reg = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
        )
        self.reg.fit(self.scaler.transform(X_reg), y_reg)
        print(f"[✓] Regresor entrenado (top {self.top_n_products} productos)")

    # ── 7. PREDICCIÓN REAL DEL MES SIGUIENTE ─────────────────────────────────
    def predict_next_month(self):
        """
        Usa el estado del último mes de cada producto para predecir
        las ventas de mes_pred (el mes que todavía no ocurrió).
        """
        c = self._c
        df_pred = self.df_predict.copy()

        # Features para el mes a predecir
        X_pred = df_pred[self._features].fillna(0)
        X_pred_scaled = self.scaler.transform(X_pred)

        # Clasificación: Reducir / Mantener / Reforzar
        pred_class = self.clf.predict(X_pred_scaled)
        prob_matrix = self.clf.predict_proba(X_pred_scaled)

        # Probabilidad de la clase predicha (confianza)
        df_pred["prob_crecimiento"] = prob_matrix[:, 2]  # prob de "Reforzar"
        df_pred["prob_reduccion"]   = prob_matrix[:, 0]  # prob de "Reducir"
        df_pred["decision"] = np.where(
            pred_class == 2, "Reforzar stock",
            np.where(pred_class == 0, "Reducir stock", "Mantener stock")
        )

        # Regresión: cuánto se venderá (solo top productos)
        mask_top = df_pred[c("id")].isin(self._top_prods)
        if mask_top.sum() > 0:
            pred_sales_log = self.reg.predict(
                self.scaler.transform(X_pred.loc[mask_top[mask_top].index])
            )
            df_pred.loc[mask_top[mask_top].index, "ventas_predichas"] = (
                np.expm1(pred_sales_log)
            )

        # Mes predicho
        df_pred["mes_predicho"] = self._mes_pred.strftime("%Y-%m")

        # Limpiar columnas internas que no aportan al cliente
        cols_salida = [
            c("id"), c("description"),
            "mes_predicho",
            c("total_sold"),          # ventas del último mes (base de comparación)
            "ventas_predichas",       # predicción cuantitativa
            "prob_crecimiento",       # probabilidad de Reforzar
            "prob_reduccion",         # probabilidad de Reducir
            "decision",               # recomendación: Reforzar / Mantener / Reducir
            "cluster",                # segmento del producto
            c("frequency"),
            c("avg_price"),
        ]
        cols_salida = [col for col in cols_salida if col in df_pred.columns]
        self.results = df_pred[cols_salida].copy()

        # Exportar
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(self.output_path, index=False)
        print(f"[✓] Predicciones exportadas → {self.output_path}")
        print(f"    {len(self.results):,} productos con predicción para "
              f"{self._mes_pred.strftime('%Y-%m')}")

    # ── 8. VISUALIZACIÓN ─────────────────────────────────────────────────────
    def plot(self, product_id=None):
        c = self._c
        if self.results is None or len(self.results) == 0:
            print("[!] Sin resultados para visualizar.")
            return

        if product_id is None:
            # Producto con mayor predicción de ventas
            mask = self.results["ventas_predichas"].notna()
            if not mask.any():
                product_id = self.results[c("id")].iloc[0]
            else:
                product_id = self.results.loc[mask, "ventas_predichas"].idxmax()
                product_id = self.results.loc[product_id, c("id")]

        # Historial del producto
        hist = self.df[
            self.df[c("id")] == product_id
        ].sort_values("fecha")

        # Punto de predicción
        pred_row = self.results[self.results[c("id")] == product_id]
        if pred_row.empty:
            print(f"[!] Producto {product_id} no tiene predicción.")
            return

        pred_val  = pred_row["ventas_predichas"].values[0]
        pred_date = self._mes_pred

        plt.figure(figsize=(14, 6))
        plt.plot(hist["fecha"], hist[c("total_sold")],
                 marker="o", color="#1A1916", label="Ventas reales")

        if not np.isnan(pred_val):
            # Conectar último punto real con predicción
            plt.plot(
                [hist["fecha"].iloc[-1], pred_date],
                [hist[c("total_sold")].iloc[-1], pred_val],
                linestyle="--", color="#1D4ED8", alpha=.5,
            )
            plt.scatter([pred_date], [pred_val],
                        s=120, color="#1D4ED8", zorder=5,
                        label=f"Predicción {pred_date.strftime('%Y-%m')}")

        desc = pred_row[c("description")].values[0]
        plt.title(f"{desc}  [{product_id}]")
        plt.xlabel("Mes")
        plt.ylabel("Valor vendido (COP)")
        plt.legend()
        plt.grid(True, alpha=.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ── PIPELINE COMPLETO ─────────────────────────────────────────────────────
    def run(self):
        self.load_and_clean()
        self.build_features()
        self.cluster()
        self.prepare_split()
        self.train_classifier()
        self.train_regressor()
        self.predict_next_month()
        self.plot()
        return self.results


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from ventas_forecast.paths import DATA_RAW, PREDICTIONS, ensure_output_dirs
        ensure_output_dirs()
        data_file   = DATA_RAW    / "Query_Result_V5.csv"
        output_file = PREDICTIONS / "PREDICCION_MENSUAL.csv"
    except ImportError:
        data_file   = Path(__file__).parent / "Query_Result_V5.csv"
        output_file = Path(__file__).parent / "PREDICCION_MENSUAL.csv"

    model = SalesForecastModel(
        filepath=data_file,
        output_path=output_file,
        min_months=6,
    )
    model.run()
