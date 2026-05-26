"""
train_kmeans_rf_prod.py
=======================
Modelo de predicción de ventas reutilizable — KMeans + Random Forest.

Uso básico
----------
    from train_kmeans_rf_prod import SalesForecastModel

    model = SalesForecastModel(
        filepath="mi_dataset.csv",
        sep=";",
        col_map={
            "id":          "CODIGO",
            "description": "DESCRIPCION",
            "year":        "ANIO",
            "month":       "MES",
            "sales":       "VENTAS",
            "total_sold":  "TOTAL_VENDIDO",
            "avg_price":   "PRECIO_PROMEDIO",
            "frequency":   "FRECUENCIA",
        }
    )

    model.run()

Columnas mínimas requeridas en el dataset
------------------------------------------
    - id           : código / identificador único del producto
    - description  : descripción del producto
    - year         : año  (entero o string con puntos, p.ej. "2.023")
    - month        : mes numérico 1-12
    - sales        : valor de ventas (acepta formato 1.234,56)
    - total_sold   : total vendido en unidades/valor
    - avg_price    : precio promedio
    - frequency    : número de transacciones

Si tu CSV ya usa esos nombres exactos, no necesitas pasar col_map.
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# Nombres de columna predeterminados (los del dataset original del proyecto)
# ---------------------------------------------------------------------------
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
    Pipeline completo: limpieza → features → clustering → clasificación
    → regresión → exportación → visualización.

    Parámetros
    ----------
    filepath : str | Path
        Ruta al archivo CSV con los datos de ventas.
    sep : str
        Separador de columnas del CSV (por defecto ';').
    encoding : str
        Codificación del archivo (por defecto 'utf-8').
    col_map : dict | None
        Mapeo {rol_interno: nombre_real_en_csv}.
        Los roles internos válidos están en DEFAULT_COL_MAP.
        Si es None se usan los nombres del dataset original del proyecto.
    output_path : str | Path | None
        Ruta donde guardar el CSV de predicciones.
        Por defecto: carpeta del script / "PREDICCION_MENSUAL.csv".
    n_clusters : int
        Número de clusters para KMeans (default 3).
    lags : list[int]
        Lags a generar como features (default [1, 2, 3, 6]).
    growth_threshold : float
        Umbral de crecimiento para la clase positiva (default 1.05 = +5 %).
    clf_threshold : float
        Umbral de probabilidad para el clasificador (default 0.58).
    top_n_products : int
        Cantidad de productos top para entrenar el regresor (default 200).
    train_ratio : float
        Fracción temporal de entrenamiento por producto (default 0.8).
    numeric_fmt : str
        Formato numérico de las columnas de valor:
        - 'dot_comma'  → miles con punto, decimales con coma  "1.234,56"  (default)
        - 'plain'      → float estándar sin transformación
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
        growth_threshold=1.05,
        clf_threshold=0.58,
        top_n_products=200,
        train_ratio=0.8,
        numeric_fmt="dot_comma",
    ):
        self.filepath = Path(filepath)
        self.sep = sep
        self.encoding = encoding
        self.col_map = {**DEFAULT_COL_MAP, **(col_map or {})}
        self.output_path = (
            Path(output_path)
            if output_path
            else self.filepath.parent / "PREDICCION_MENSUAL.csv"
        )
        self.n_clusters = n_clusters
        self.lags = lags or [1, 2, 3, 6]
        self.growth_threshold = growth_threshold
        self.clf_threshold = clf_threshold
        self.top_n_products = top_n_products
        self.train_ratio = train_ratio
        self.numeric_fmt = numeric_fmt

        # Estado interno
        self.df = None
        self.scaler = RobustScaler()
        self.clf = None
        self.reg = None
        self.results = None
        self._features = None
        self._top_products = None

    # ------------------------------------------------------------------
    # Helpers de columna
    # ------------------------------------------------------------------
    def _c(self, role):
        """Devuelve el nombre real de la columna dado su rol interno."""
        return self.col_map[role]

    def _parse_numeric(self, col: pd.Series) -> pd.Series:
        """Convierte columnas numéricas según el formato declarado."""
        if self.numeric_fmt == "dot_comma":
            return (
                col.astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
        return col.astype(float)

    # ------------------------------------------------------------------
    # 1. CARGA Y LIMPIEZA
    # ------------------------------------------------------------------
    def load_and_clean(self):
        """Lee el CSV y normaliza columnas numéricas y de fecha."""
        c = self._c

        self.df = pd.read_csv(
            self.filepath,
            sep=self.sep,
            encoding=self.encoding,
        )

        # Columnas de valor
        for role in ("sales", "total_sold", "avg_price"):
            self.df[c(role)] = self._parse_numeric(self.df[c(role)])

        # Año: puede venir como "2.023"
        self.df[c("year")] = (
            self.df[c("year")]
            .astype(str)
            .str.replace(".", "", regex=False)
            .astype(int)
        )

        # Fecha
        self.df["fecha"] = pd.to_datetime(
            self.df[c("year")].astype(str)
            + "-"
            + self.df[c("month")].astype(str).str.zfill(2)
            + "-01"
        )

        # Agregar a un registro por producto-mes
        self.df = (
            self.df.groupby(
                [c("id"), c("description"), pd.Grouper(key="fecha", freq="MS")]
            )
            .agg(
                {
                    c("sales"):     "sum",
                    c("total_sold"): "sum",
                    c("avg_price"): "mean",
                    c("frequency"): "sum",
                }
            )
            .reset_index()
        )

        self.df[c("year")]  = self.df["fecha"].dt.year
        self.df[c("month")] = self.df["fecha"].dt.month
        self.df = self.df.sort_values([c("id"), "fecha"])

        print(f"[✓] Datos cargados: {len(self.df):,} registros "
              f"| {self.df[c('id')].nunique():,} productos únicos")

    # ------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # ------------------------------------------------------------------
    def build_features(self):
        """Crea lags, medias móviles, variables logarítmicas y estacionalidad."""
        c = self._c
        df = self.df

        # Target: ventas del próximo mes
        df["target"] = df.groupby(c("id"))[c("total_sold")].shift(-1)

        # Lags
        for lag in self.lags:
            df[f"lag_{lag}"] = df.groupby(c("id"))[c("total_sold")].shift(lag)

        # Medias móviles
        df["mean_3"] = df.groupby(c("id"))[c("total_sold")].transform(
            lambda x: x.rolling(3).mean().shift(1)
        )
        df["mean_6"] = df.groupby(c("id"))[c("total_sold")].transform(
            lambda x: x.rolling(6).mean().shift(1)
        )

        # Ratio de crecimiento
        df["growth_ratio"] = df["target"] / df[c("total_sold")]

        # Logarítmicas
        df["VENTAS_LOG"] = np.log1p(df[c("sales")])
        df["TOTAL_LOG"]  = np.log1p(df[c("total_sold")])
        df["PRECIO_LOG"] = np.log1p(df[c("avg_price")])

        # Estacionalidad cíclica
        df["mes_sin"] = np.sin(2 * np.pi * df[c("month")] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df[c("month")] / 12)

        # Limpieza final
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df = df[df[c("total_sold")] > 0].reset_index(drop=True)

        self.df = df
        print(f"[✓] Features construidas: {len(self.df):,} registros tras limpieza")

    # ------------------------------------------------------------------
    # 3. CLUSTERING
    # ------------------------------------------------------------------
    def cluster(self):
        """Segmenta productos con KMeans."""
        c = self._c
        X_cluster = self.df[[c("total_sold"), c("frequency"), c("avg_price")]]
        Xc = RobustScaler().fit_transform(X_cluster)
        self.df["cluster"] = KMeans(
            n_clusters=self.n_clusters, random_state=42
        ).fit_predict(Xc)
        print(f"[✓] Clustering KMeans con {self.n_clusters} clusters")

    # ------------------------------------------------------------------
    # 4. PREPARACIÓN DE FEATURES Y SPLIT
    # ------------------------------------------------------------------
    def prepare_split(self):
        """Define X / y y hace el split temporal por producto."""
        c = self._c

        # Target de clasificación
        self.df["target_class"] = (
            self.df["growth_ratio"] > self.growth_threshold
        ).astype(int)

        lag_cols = [f"lag_{l}" for l in self.lags]
        self._features = [
            "VENTAS_LOG", "TOTAL_LOG", "PRECIO_LOG",
            c("frequency"),
            *lag_cols,
            "mean_3", "mean_6",
            "mes_sin", "mes_cos",
            "cluster",
        ]

        X = self.df[self._features].fillna(0)
        y = self.df["target_class"]
        X_scaled = self.scaler.fit_transform(X)

        train_idx, test_idx = [], []
        for _, g in self.df.groupby(c("id")):
            g = g.sort_values("fecha")
            cut = int(len(g) * self.train_ratio)
            train_idx += g.index[: cut].tolist()
            test_idx  += g.index[cut:].tolist()

        self._X = X
        self._y = y
        self._X_scaled = X_scaled
        self._train_idx = train_idx
        self._test_idx = test_idx

        print(f"[✓] Split: {len(train_idx):,} train | {len(test_idx):,} test")

    # ------------------------------------------------------------------
    # 5. CLASIFICADOR
    # ------------------------------------------------------------------
    def train_classifier(self):
        """Entrena el RandomForestClassifier."""
        X_train = self._X_scaled[self._train_idx]
        y_train = self._y.iloc[self._train_idx]

        self.clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight={0: 1, 1: 3},
            random_state=42,
            n_jobs=-1,
        )
        self.clf.fit(X_train, y_train)

        X_test = self._X_scaled[self._test_idx]
        y_test = self._y.iloc[self._test_idx]

        prob = self.clf.predict_proba(X_test)[:, 1]
        pred = (prob > self.clf_threshold).astype(int)

        print("\n===== MÉTRICAS CLASIFICADOR =====")
        print("Accuracy :", accuracy_score(y_test, pred))
        print("Precision:", precision_score(y_test, pred))
        print("Recall   :", recall_score(y_test, pred))
        print("F1       :", f1_score(y_test, pred))
        print("\n", classification_report(y_test, pred))

        self._prob = prob
        self._pred = pred

    # ------------------------------------------------------------------
    # 6. REGRESOR
    # ------------------------------------------------------------------
    def train_regressor(self):
        """Entrena el RandomForestRegressor sobre los top N productos."""
        c = self._c
        self._top_products = (
            self.df.groupby(c("id"))[c("total_sold")]
            .sum()
            .nlargest(self.top_n_products)
            .index
        )

        df_reg = self.df[
            self.df[c("id")].isin(self._top_products)
            & (self.df["target"] > 0)
        ]

        X_reg = self._X.loc[df_reg.index]
        y_reg = np.log1p(df_reg["target"])
        X_reg_scaled = self.scaler.transform(X_reg)

        self.reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        self.reg.fit(X_reg_scaled, y_reg)
        print(f"[✓] Regresor entrenado sobre top {self.top_n_products} productos")

    # ------------------------------------------------------------------
    # 7. RESULTADOS Y EXPORTACIÓN
    # ------------------------------------------------------------------
    def build_results(self):
        """Construye el DataFrame de resultados y lo exporta a CSV."""
        c = self._c

        results = self.df.iloc[self._test_idx].copy()
        results["prob"]     = self._prob
        results["decision"] = np.where(
            self._pred == 1,
            "Reforzar stock próximo mes",
            "Mantener stock",
        )

        mask = results[c("id")].isin(self._top_products)
        pred_sales = self.reg.predict(
            self.scaler.transform(self._X.loc[results[mask].index])
        )
        results.loc[mask, "pred_sales"] = np.expm1(pred_sales)
        results["error_stock"] = abs(results["pred_sales"] - results["target"])

        self.results = results
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.output_path, index=False)
        print(f"[✓] Predicciones exportadas → {self.output_path}")

    # ------------------------------------------------------------------
    # 8. VISUALIZACIÓN
    # ------------------------------------------------------------------
    def plot(self, product_id=None):
        """
        Grafica ventas reales vs predicción para un producto.

        Parámetros
        ----------
        product_id : cualquier valor compatible con CODIGO/id.
            Si es None, se toma el producto con más registros en test.
        """
        c = self._c

        if product_id is None:
            product_id = self.results[c("id")].value_counts().idxmax()

        graf = (
            self.results[self.results[c("id")] == product_id]
            .sort_values("fecha")
        )

        plt.figure(figsize=(14, 6))
        plt.plot(graf["fecha"], graf[c("total_sold")],
                 marker="o", label="Ventas reales")
        plt.plot(graf["fecha"], graf["target"],
                 marker="o", linestyle="--", label="Ventas reales próximo mes")
        plt.plot(graf["fecha"], graf["pred_sales"],
                 marker="o", linewidth=3, label="Predicción próximo mes")
        plt.title(f"Predicción mensual — Producto {product_id}")
        plt.xlabel("Mes")
        plt.ylabel("Valor vendido")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # PIPELINE COMPLETO
    # ------------------------------------------------------------------
    def run(self):
        """Ejecuta el pipeline completo de principio a fin."""
        self.load_and_clean()
        self.build_features()
        self.cluster()
        self.prepare_split()
        self.train_classifier()
        self.train_regressor()
        self.build_results()
        self.plot()
        return self.results


# ---------------------------------------------------------------------------
# Punto de entrada — mantiene compatibilidad con el proyecto original
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Intenta usar las rutas centralizadas del proyecto
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from ventas_forecast.paths import DATA_RAW, PREDICTIONS, ensure_output_dirs
        ensure_output_dirs()
        data_file   = DATA_RAW   / "Query_Result_V3.csv"
        output_file = PREDICTIONS / "PREDICCION_MENSUAL.csv"
    except ImportError:
        # Fallback: misma carpeta que el script
        data_file   = Path(__file__).parent / "Query_Result_V3.csv"
        output_file = Path(__file__).parent / "PREDICCION_MENSUAL.csv"

    model = SalesForecastModel(
        filepath=data_file,
        output_path=output_file,
    )
    model.run()