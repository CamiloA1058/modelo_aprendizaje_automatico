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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, silhouette_score,
    mean_absolute_error, mean_squared_error, r2_score,
    balanced_accuracy_score,
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

# ── Hiperparámetros del clasificador (RandomForest) ──────────────────────────
# Valores seleccionados con tune_classifier() (GridSearchCV + TimeSeriesSplit
# mensual) sobre Query_Result_V5.csv; ver scripts/tune_rf.py. Re-ajustados
# con scoring="f1_macro" (ver CHANGELOG 2026-09-23 "métrica macro"): al
# seleccionar por F1 macro en vez de F1 ponderado, max_depth pasa de None a
# 14 y min_samples_leaf de 1 a 5 (un modelo algo menos profundo/más regular
# generaliza mejor en las clases minoritarias Mantener/Reforzar).
DEFAULT_CLF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 14,
    "min_samples_leaf": 5,
}

# Grilla por defecto para tune_classifier() (GridSearchCV)
DEFAULT_PARAM_GRID = {
    "rf__n_estimators": [200, 300],
    "rf__max_depth": [6, 10, 14, None],
    "rf__min_samples_leaf": [1, 3, 5],
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
    clf_params          : dict con hiperparámetros del RandomForestClassifier
                          que sobreescriben DEFAULT_CLF_PARAMS (n_estimators,
                          max_depth, min_samples_leaf). Ver también
                          tune_classifier() para ajustarlos con GridSearchCV.
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
        clf_params=None,
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
        self.clf_params      = {**DEFAULT_CLF_PARAMS, **(clf_params or {})}

        # Estado interno
        self.df           = None   # historial completo
        self.df_predict   = None   # filas del último mes (para predecir)
        self.scaler       = RobustScaler()
        self.cluster_scaler = RobustScaler()  # escalador exclusivo del clustering
        self.kmeans       = None
        self.clf          = None
        self.reg          = None
        self.results      = None   # predicciones del mes siguiente
        self.metrics      = {}
        self.reg_metrics      = {}   # métricas del regresor sobre prueba (train_regressor)
        self.baseline_metrics = {}   # métricas de los baselines ingenuo/media_3 (train_regressor)
        self.reg_test_results = None  # real vs predicho en prueba (train_regressor)
        self.tuning_results  = None   # tabla completa de GridSearchCV (tune_classifier)
        self.best_clf_params = None   # mejores hiperparámetros encontrados
        self._features    = None
        self._top_prods   = None
        self._ultimo_mes  = None
        self._mes_pred    = None
        # Poblados por prepare_split(): partición de entrenamiento/prueba
        self._df_train    = None
        self._X           = None
        self._y           = None
        self._X_scaled    = None
        self._train_idx   = None
        self._test_idx    = None

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

        # Completar el calendario mensual de cada producto (ver
        # _complete_calendar): sin esto, groupby(id).shift()/rolling() en
        # build_features() opera "por fila" (mes con ventas anterior) en vez
        # de "por mes calendario", porque un mes sin ninguna venta no tiene
        # fila en el CSV crudo.
        self.df = self._complete_calendar(self.df)

        print(f"[✓] Datos cargados: {len(self.df):,} registros | "
              f"{self.df[c('id')].nunique():,} productos únicos")
        print(f"[✓] Último mes en datos: {self._ultimo_mes.strftime('%Y-%m')}")
        print(f"[✓] Mes a predecir:      {self._mes_pred.strftime('%Y-%m')}")

    # ── 1b. COMPLETADO DE CALENDARIO ─────────────────────────────────────────
    def _complete_calendar(self, df):
        """
        Completa el calendario de cada producto con un registro por mes,
        desde su PRIMER mes con ventas hasta el último mes GLOBAL del
        dataset (self._ultimo_mes), rellenando con cero los meses en los que
        el producto no vendió (esas filas no existen en el CSV crudo).

        Sin este paso, target/lag_k/mean_3/mean_6 (build_features) se
        calculan con groupby(id).shift()/rolling(), que operan sobre la
        POSICIÓN de la fila dentro del grupo, no sobre el mes calendario: un
        producto que no vendió un mes simplemente "salta" ese mes, como si
        el mes con ventas siguiente fuera el mes inmediatamente posterior
        (HALLAZGO 2026-09-23: 40,2 % de las filas de V5 con un salto > 1 mes
        hasta la fila siguiente).

        avg_price en los meses sin ventas: se usa el ÚLTIMO precio conocido
        del producto (forward-fill), porque no hubo ninguna transacción ese
        mes de la que derivar un precio propio; el precio de lista no
        desaparece por no haber vendido.

        Devuelve el DataFrame completado (una fila por producto-mes, sin
        huecos). Imprime cuántas filas de venta 0 se agregaron.
        """
        c = self._c
        n_antes = len(df)

        grupos_completos = []
        for prod_id, g in df.groupby(c("id"), sort=False):
            g = g.sort_values("fecha").set_index("fecha")
            calendario = pd.date_range(g.index.min(), self._ultimo_mes, freq="MS")
            g = g.reindex(calendario)

            g[c("id")]          = prod_id
            g[c("description")] = g[c("description")].ffill()
            g[c("sales")]       = g[c("sales")].fillna(0.0)
            g[c("total_sold")]  = g[c("total_sold")].fillna(0.0)
            g[c("frequency")]   = g[c("frequency")].fillna(0.0)
            g[c("avg_price")]   = g[c("avg_price")].ffill()

            grupos_completos.append(g)

        df_completo = pd.concat(grupos_completos)
        df_completo.index.name = "fecha"
        df_completo = df_completo.reset_index()

        df_completo[c("year")]  = df_completo["fecha"].dt.year
        df_completo[c("month")] = df_completo["fecha"].dt.month
        df_completo = (
            df_completo.sort_values([c("id"), "fecha"]).reset_index(drop=True)
        )

        n_ceros = len(df_completo) - n_antes
        print(f"[✓] Calendario completado: {n_ceros:,} filas de mes sin venta "
              f"(total_sold=0) agregadas ({n_antes:,} → {len(df_completo):,} filas)")

        return df_completo

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

        # Media móvil de 3 meses SIN shift (incluye el mes actual): se usa
        # solo como baseline "media_3" en train_regressor() para comparar
        # contra el regresor, NO como feature del modelo (a diferencia de
        # mean_3, que sí lleva shift(1) y sirve como predictor).
        df["media_3_actual"] = df.groupby(c("id"))[c("total_sold")].transform(
            lambda x: x.rolling(3).mean())

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

    # ── Helper compartido: partición temporal de entrenamiento/prueba ────────
    def _build_train_frame(self):
        """
        Filtra las filas con target (mes siguiente calendario) conocido y con
        ventas actuales, y aplica el split temporal por producto (primeras
        train_ratio filas de cada producto, ordenadas por fecha, van a
        entrenamiento).

        target > 0 NO se exige: con el calendario completo (ver
        _complete_calendar), un mes siguiente en cero es una observación
        válida y necesaria (es justo la que entrena la clase "Reducir" en
        prepare_split). Solo se excluyen las filas cuyo MES ACTUAL no tuvo
        ventas (total_sold == 0): el growth_ratio (target / total_sold)
        quedaría indefinido (división por cero) y esas filas no aportan una
        decisión de "reforzar/mantener/reducir" con base en un mes real.

        Usado tanto por cluster() como por prepare_split() para que ambos
        entrenen con exactamente la misma partición y no haya fuga de datos
        del período de prueba ni del mes a predecir.

        Devuelve (df_train, train_idx, test_idx).
        """
        c = self._c

        df_train = self.df[
            self.df["target"].notna() &
            (self.df[c("total_sold")] > 0)
        ].copy().reset_index(drop=True)

        train_idx, test_idx = [], []
        for _, g in df_train.groupby(c("id")):
            g = g.sort_values("fecha")
            cut = int(len(g) * self.train_ratio)
            train_idx += g.index[:cut].tolist()
            test_idx  += g.index[cut:].tolist()

        return df_train, train_idx, test_idx

    # ── 3. CLUSTERING ─────────────────────────────────────────────────────────
    def _cluster_features(self, frame):
        """
        Variables de clustering (ventas, frecuencia, precio) con log1p: las
        ventas son muy asimétricas y, en escala original, KMeans separa solo
        los valores extremos en lugar de segmentos de productos.
        """
        c = self._c
        cols = [c("total_sold"), c("frequency"), c("avg_price")]
        return np.log1p(frame[cols].fillna(0).clip(lower=0))

    def cluster(self):
        """
        Ajusta el escalador y KMeans SOLO con la partición de entrenamiento
        (sin filas de prueba ni del mes a predecir) para evitar fuga de datos;
        luego asigna cluster a TODAS las filas de self.df (incluyendo prueba
        y el mes a predecir) con predict().
        """
        df_train, train_idx, _ = self._build_train_frame()
        X_train = self._cluster_features(df_train.loc[train_idx])
        X_all   = self._cluster_features(self.df)

        self.cluster_scaler = RobustScaler()
        self.cluster_scaler.fit(X_train)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=42, n_init=10
        )
        self.kmeans.fit(self.cluster_scaler.transform(X_train))

        self.df["cluster"] = self.kmeans.predict(
            self.cluster_scaler.transform(X_all)
        )
        print(f"[✓] KMeans con {self.n_clusters} clusters "
              f"(ajustado solo con {len(train_idx):,} filas de entrenamiento)")

    # ── 3b. SELECCIÓN DE k (método del codo + silueta) ───────────────────────
    def evaluate_k(self, k_range=range(2, 11), sample_size=10000):
        """
        Evalúa varios valores de k para KMeans mediante inercia (método del
        codo) y coeficiente de silueta, usando solo la partición de
        entrenamiento (mismo criterio que cluster(), sin fuga de datos).

        Requiere haber ejecutado build_features() previamente. No modifica
        self.n_clusters ni ningún otro estado del pipeline (escaladores/
        modelos locales a este método).

        Devuelve un DataFrame con columnas: k, inercia, silueta.
        """
        df_train, train_idx, _ = self._build_train_frame()
        X_train = self._cluster_features(df_train.loc[train_idx])

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train)

        filas = []
        for k in k_range:
            kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans_k.fit_predict(X_scaled)
            filas.append({
                "k": k,
                "inercia": kmeans_k.inertia_,
                "silueta": silhouette_score(
                    X_scaled, labels,
                    sample_size=min(sample_size, len(X_scaled)),
                    random_state=42,
                ),
            })

        return pd.DataFrame(filas)

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

        # ── Dataset de entrenamiento (mismo filtro y split que cluster()) ────
        df_train, train_idx, test_idx = self._build_train_frame()

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

        # El escalador se ajusta solo con entrenamiento para que las
        # métricas de prueba no incorporen estadísticos del conjunto de prueba
        self.scaler.fit(X.loc[train_idx])
        X_scaled = self.scaler.transform(X)

        self._df_train   = df_train
        self._X          = X
        self._y          = y
        self._X_scaled   = X_scaled
        self._train_idx  = train_idx
        self._test_idx   = test_idx

        # ── Dataset de predicción real ────────────────────────────────────────
        # Productos que tienen dato en el último mes con suficiente historial.
        # min_months cuenta meses CON VENTAS (total_sold > 0), no filas de
        # calendario: desde _complete_calendar, self.df tiene una fila por
        # cada mes calendario (incluyendo los de venta 0), así que contar
        # filas sobreestimaría el historial real de un producto con huecos.
        conteo = (
            self.df[self.df[c("total_sold")] > 0]
            .groupby(c("id"))["fecha"].count()
        )
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
            **self.clf_params,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.clf.fit(X_tr, y_tr)

        pred = self.clf.predict(X_te)

        self.metrics = {
            # Ponderadas por soporte de clase (mantenidas por compatibilidad:
            # scripts/app.py las lee tal cual). Bajo el desbalance de clases
            # de este problema (Reducir 79,30 % de la prueba) el promedio
            # ponderado —y sobre todo accuracy— está dominado por la clase
            # mayoritaria y puede mejorar aunque el modelo empeore en
            # Mantener/Reforzar; ver f1_macro/balanced_accuracy más abajo.
            "accuracy":  round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred, average="weighted", zero_division=0), 4),
            "recall":    round(recall_score(y_te, pred, average="weighted", zero_division=0), 4),
            "f1":        round(f1_score(y_te, pred, average="weighted", zero_division=0), 4),
            # Métricas principales bajo desbalance de clases (DECIDIDO por el
            # usuario, ver CHANGELOG 2026-09-23 "métrica macro"): f1_macro
            # promedia el F1 de cada clase con el mismo peso (no por soporte),
            # y balanced_accuracy es el promedio del recall de cada clase;
            # ambas penalizan un modelo que solo acierta en "Reducir".
            "f1_macro":          round(f1_score(y_te, pred, average="macro", zero_division=0), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_te, pred), 4),
        }

        print("\n===== MÉTRICAS DE VALIDACIÓN =====")
        for k in ("accuracy", "precision", "recall", "f1"):
            print(f"  {k.capitalize():<10}: {self.metrics[k]:.2%}")
        print("  --- Métricas principales (desbalance de clases) ---")
        print(f"  {'F1 macro':<18}: {self.metrics['f1_macro']:.2%}")
        print(f"  {'Bal. accuracy':<18}: {self.metrics['balanced_accuracy']:.2%}")
        print("\n", classification_report(y_te, pred, labels=[0, 1, 2],
              target_names=["Reducir", "Mantener", "Reforzar"], zero_division=0))

    # ── 5b. VALIDACIÓN CRUZADA TEMPORAL (splitter mensual) ───────────────────
    def _monthly_time_series_folds(self, dates, n_splits):
        """
        Genera folds de validación cruzada temporal a partir del MES de cada
        fila (no de la posición de la fila), aplicando TimeSeriesSplit sobre
        los meses únicos ordenados cronológicamente.

        Con esto se garantiza que en cada fold todos los meses de validación
        sean estrictamente posteriores a todos los meses de entrenamiento de
        ese fold, y que ningún mes quede partido entre ambos conjuntos (algo
        que un TimeSeriesSplit aplicado directamente sobre las filas no
        garantiza, porque varias filas comparten el mismo mes).

        Parámetros
        ----------
        dates    : secuencia de fechas alineada POSICIONALMENTE (mismo orden
                   y longitud) con las filas del frame que se pasará como X
                   a GridSearchCV.
        n_splits : número de folds del TimeSeriesSplit.

        Devuelve una lista de tuplas (train_positions, val_positions) con
        posiciones 0-based dentro de ese frame, tal como las espera el
        parámetro cv= de GridSearchCV (lista de folds ya calculados).
        """
        dates = pd.Series(dates).reset_index(drop=True)
        meses_unicos = np.sort(dates.unique())

        splitter = TimeSeriesSplit(n_splits=n_splits)
        folds = []
        for pos_meses_train, pos_meses_val in splitter.split(meses_unicos):
            meses_train = meses_unicos[pos_meses_train]
            meses_val = meses_unicos[pos_meses_val]
            train_positions = np.flatnonzero(dates.isin(meses_train).to_numpy())
            val_positions = np.flatnonzero(dates.isin(meses_val).to_numpy())
            folds.append((train_positions, val_positions))

        return folds

    def tune_classifier(self, param_grid=None, n_splits=5, scoring="f1_macro"):
        """
        Ajusta los hiperparámetros del RandomForestClassifier con GridSearchCV,
        usando validación cruzada temporal (TimeSeriesSplit por mes, ver
        _monthly_time_series_folds) sobre la partición de ENTRENAMIENTO
        únicamente (self._train_idx): las filas de prueba nunca participan en
        la selección de hiperparámetros.

        scoring por defecto "f1_macro" (DECIDIDO por el usuario, ver
        CHANGELOG 2026-09-23 "métrica macro"): promedia el F1 de cada clase
        con el mismo peso, sin importar su soporte. Con el desbalance de
        clases de este problema (Reducir ~79 % de la prueba tras el
        completado de calendario), "f1_weighted" —el valor anterior—
        recompensa sobre todo acertar en la clase mayoritaria y puede elegir
        hiperparámetros que ignoran Mantener/Reforzar; f1_macro exige acertar
        también en las clases minoritarias.

        El estimador es un Pipeline (RobustScaler + RandomForestClassifier)
        para que el escalador se reajuste en cada fold con sus propias filas
        de entrenamiento, sin fuga de datos entre folds; por eso se le pasa
        X sin escalar (self._X, no self._X_scaled).

        Requiere haber ejecutado prepare_split() previamente. Guarda
        self.tuning_results (tabla de cv_results_ ordenada por rank) y
        self.best_clf_params (sin el prefijo 'rf__'). NO modifica
        self.clf_params automáticamente; para aplicar el resultado hay que
        combinarlo explícitamente (ver scripts/tune_rf.py). Devuelve
        self.best_clf_params.
        """
        if self._X is None or self._train_idx is None:
            raise RuntimeError(
                "Ejecute prepare_split() antes de tune_classifier()."
            )

        param_grid = param_grid or DEFAULT_PARAM_GRID

        X_train = self._X.loc[self._train_idx]
        y_train = self._y.loc[self._train_idx]
        fechas_train = self._df_train.loc[self._train_idx, "fecha"]

        folds = self._monthly_time_series_folds(fechas_train, n_splits)

        pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("rf", RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ])

        grid = GridSearchCV(
            pipeline, param_grid=param_grid, cv=folds,
            scoring=scoring, n_jobs=1, refit=False,
        )
        grid.fit(X_train, y_train)

        columnas = [
            col for col in grid.cv_results_
            if col.startswith("param_")
            or col in ("mean_test_score", "std_test_score", "rank_test_score")
        ]
        self.tuning_results = (
            pd.DataFrame(grid.cv_results_)[columnas]
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )

        self.best_clf_params = {
            k.replace("rf__", ""): v for k, v in grid.best_params_.items()
        }

        print(f"[✓] tune_classifier: {len(self.tuning_results)} combinaciones "
              f"evaluadas ({n_splits} folds mensuales, scoring={scoring})")
        print(f"    Mejor combinación: {self.best_clf_params}")

        return self.best_clf_params

    # ── Helper compartido: métricas de regresión (modelo y baselines) ────────
    @staticmethod
    def _regression_metrics(y_true, y_pred, scale):
        """
        Calcula MAE, MSE, RMSE, R2 y MASE (con la escala de MASE ya calculada
        aparte, ver train_regressor) para un conjunto de predicciones.
        Redondea a 4 decimales. Si la escala de MASE es 0, mase queda en NaN.
        """
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
        mase = (mae / scale) if scale else float("nan")

        return {
            "mae":    round(float(mae), 4),
            "mse":    round(float(mse), 4),
            "rmse":   round(float(rmse), 4),
            "r2":     round(float(r2), 4),
            "mase":   round(float(mase), 4) if not np.isnan(mase) else float("nan"),
            "n_test": int(len(y_true)),
        }

    # ── 6. REGRESOR ───────────────────────────────────────────────────────────
    def train_regressor(self):
        """
        Entrena el RandomForestRegressor SOLO con la partición de
        entrenamiento (self._train_idx) de los top_n_products productos con
        más ventas, para no filtrar información del conjunto de prueba (el
        top de productos y el ajuste del modelo antes se calculaban sobre
        TODAS las filas de _df_train, incluyendo prueba).

        Evalúa en la partición de prueba de esos mismos productos (MAE, MSE,
        RMSE, R2, MASE) y calcula los mismos baselines para dos pronósticos
        ingenuos: "ingenuo" (ventas del mes actual) y "media_3" (media móvil
        de 3 meses incluyendo el mes actual). Guarda self.reg_metrics y
        self.baseline_metrics.
        """
        c = self._c

        # ── Top productos: SOLO con filas de entrenamiento ───────────────────
        train_rows = self._df_train.loc[self._train_idx]
        self._top_prods = (
            train_rows.groupby(c("id"))[c("total_sold")]
            .sum().nlargest(self.top_n_products).index
        )

        es_top       = self._df_train[c("id")].isin(self._top_prods)
        # target > 0 ya no se exige (self._df_train, construido por
        # _build_train_frame(), ya garantiza target.notna(); un mes
        # siguiente en cero es una observación válida de la que también se
        # quiere aprender y evaluar: log1p(0) = 0 no da problema numérico).
        tiene_target = self._df_train["target"].notna()
        es_train     = self._df_train.index.isin(self._train_idx)
        es_test      = self._df_train.index.isin(self._test_idx)

        df_reg_train = self._df_train[es_train & es_top & tiene_target]
        df_reg_test  = self._df_train[es_test & es_top & tiene_target]

        # ── Entrenamiento (solo filas de entrenamiento) ──────────────────────
        X_reg_train = self._X.loc[df_reg_train.index]
        y_reg_train = np.log1p(df_reg_train["target"])

        self.reg = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
        )
        self.reg.fit(self.scaler.transform(X_reg_train), y_reg_train)

        # ── Evaluación (solo filas de prueba) ────────────────────────────────
        X_reg_test = self._X.loc[df_reg_test.index]
        y_true     = df_reg_test["target"].to_numpy()
        y_pred     = np.expm1(self.reg.predict(self.scaler.transform(X_reg_test)))

        # Valores reales vs predichos de la prueba (top productos), para
        # graficar en scripts/evaluate_model.py (real_vs_predicho.png).
        self.reg_test_results = pd.DataFrame({
            c("id"): df_reg_test[c("id")].to_numpy(),
            "real":     y_true,
            "predicho": y_pred,
        })

        # Escala de MASE: MAE del pronóstico ingenuo (mes actual predice el
        # mes siguiente) calculada sobre la partición de ENTRENAMIENTO de los
        # top productos, para no usar información de prueba en la escala.
        naive_err_train = (
            df_reg_train["target"] - df_reg_train[c("total_sold")]
        ).abs()
        scale = naive_err_train.mean()

        self.reg_metrics = self._regression_metrics(y_true, y_pred, scale)

        # ── Baselines sobre las mismas filas de prueba ───────────────────────
        pred_ingenuo = df_reg_test[c("total_sold")].to_numpy()

        # media_3 sin shift (incluye mes actual); si un producto no tiene
        # suficiente historial (NaN, <3 meses), se usa el ingenuo como
        # respaldo para esa fila.
        pred_media_3 = df_reg_test["media_3_actual"].to_numpy()
        pred_media_3 = np.where(np.isnan(pred_media_3), pred_ingenuo, pred_media_3)

        self.baseline_metrics = {
            "ingenuo": self._regression_metrics(y_true, pred_ingenuo, scale),
            "media_3": self._regression_metrics(y_true, pred_media_3, scale),
        }

        print(f"[✓] Regresor entrenado (top {self.top_n_products} productos, "
              f"{len(df_reg_train):,} filas de entrenamiento, "
              f"{len(df_reg_test):,} filas de prueba)")

        print("\n===== MÉTRICAS DE REGRESIÓN (prueba, top productos) =====")
        print(f"{'Métrica':<10}{'Modelo':<14}{'Ingenuo':<14}{'Media_3':<14}")
        for k in ("mae", "mse", "rmse", "r2", "mase"):
            print(
                f"{k.upper():<10}"
                f"{self.reg_metrics[k]:<14}"
                f"{self.baseline_metrics['ingenuo'][k]:<14}"
                f"{self.baseline_metrics['media_3'][k]:<14}"
            )
        print(f"n_test: {self.reg_metrics['n_test']:,}")

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
