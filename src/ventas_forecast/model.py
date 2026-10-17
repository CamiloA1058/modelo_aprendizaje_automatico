"""
model.py
========
Modelo de predicción de ventas reutilizable — KMeans + Random Forest.

LÓGICA PRINCIPAL
----------------
El modelo se entrena con el historial completo (donde el target es el mes
siguiente real). Luego genera predicciones REALES para el mes que aún no
existe (último mes del dataset + 1) usando como input el estado actual de
cada producto. Solo se incluyen productos con suficientes datos históricos.

Uso básico
----------
    from ventas_forecast.model import SalesForecastModel

    model = SalesForecastModel(filepath="mi_dataset.csv")
    model.run()

Columnas mínimas requeridas
----------------------------
    id, description, year, month, sales, total_sold, avg_price, frequency
    (ver col_map para nombres alternativos)
"""

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

from ventas_forecast import eligibility as el
from ventas_forecast import grupos
from ventas_forecast import improvements as im
from ventas_forecast import sale_probability as sp
from ventas_forecast import stock_sugerido as ss
from ventas_forecast import validacion_interna as vi
from ventas_forecast.data.cleaning import remove_bulk_adjustments
from ventas_forecast.paths import DATA_RAW, DEFAULT_PREDICTION

# ── Nombres de columna por defecto ────────────────────────────────────────────
DEFAULT_COL_MAP = {
    "id":          "CODIGO",
    "description": "DESCRIPCION",
    "date":        "FECHA",
    "year":        "ANIO",
    "month":       "MES",
    "sales":       "VENTAS",
    "total_sold":  "TOTAL_VENDIDO",
    "avg_price":   "PRECIO_PROMEDIO",
    "frequency":   "FRECUENCIA",
}

# ── Hiperparámetros del clasificador (RandomForest) ──────────────────────────
# Ganador de tune_classifier() (GridSearchCV con scoring="f1_macro" sobre
# pliegues mensuales TimeSeriesSplit de la partición de ENTRENAMIENTO, ver
# scripts/tune_rf.py) con Query_Result_V6.csv hasta el 31/03/2026: F1 macro
# CV 0,4377 +/- 0,0086. Se adopta el ganador de la validación cruzada tal
# cual: la prueba interna no participa en la elección (antes se conservaba
# max_depth=14 porque sus métricas de PRUEBA eran parecidas, lo que era
# seleccionar con la prueba; ver CHANGELOG "V1 sin selección con los datos
# de prueba").
ETIQUETA_SIN_RECOMENDACION = "Sin recomendación (estimación por promedio)"

DEFAULT_CLF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 3,
}

# Hiperparámetros fijos del regresor (RandomForestRegressor sobre log1p) y del
# clasificador binario de venta; los usan los modelos de producción y los
# modelos temporales de la validación interna.
REG_PARAMS = {"n_estimators": 200, "max_depth": 12}
SALE_CLF_PARAMS = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5,
                   "class_weight": "balanced"}

# Grilla por defecto para tune_classifier() (GridSearchCV)
DEFAULT_PARAM_GRID = {
    "rf__n_estimators": [200, 300],
    "rf__max_depth": [6, 10, 14, None],
    "rf__min_samples_leaf": [1, 3, 5],
}


def _prob_vende(clf, Z):
    """Probabilidad de la clase 1 (vende); con una sola clase en el
    entrenamiento, probabilidad constante igual a esa clase."""
    p = clf.predict_proba(Z)
    if p.shape[1] == 2:
        return p[:, 1]
    return np.full(len(Z), float(clf.classes_[0]))


class SalesForecastModel:
    """
    Pipeline completo:
      1. Entrenamiento con historial (target = mes siguiente real)
      2. Predicción hacia adelante del mes que aún no existe
         — solo productos con suficientes datos en el último mes

    Parámetros
    ----------
    filepath            : ruta al CSV (None si se pasa `datos`)
    datos               : DataFrame ya leído con el mismo esquema que el CSV
                          (p. ej. ventas_forecast.fuente_bd.leer_ventas); si
                          se pasa, load_and_clean() lo usa en lugar de leer
                          filepath y no lo modifica.
    grupos_mapa         : Serie codigo -> grupo ya leída (p. ej.
                          fuente_bd.leer_grupos); si se pasa,
                          agregar_resumen_grupo() la usa en lugar de grupos_path.
    sep                 : separador (default ';')
    encoding            : codificación (default 'utf-8')
    col_map             : dict {rol: nombre_en_csv} — ver DEFAULT_COL_MAP
    output_path         : ruta del CSV de salida
    n_clusters          : clusters KMeans (default 3)
    lags                : lista de lags (default [1,2,3,6])
    growth_threshold    : umbral crecimiento para "Reforzar stock" (default 1.15)
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
    bulk_adjustment_min_products : mínimo de productos DISTINTOS que deben
                          compartir la misma combinación (fecha, cantidad,
                          precio) para tratarla como ajuste masivo de
                          inventario y excluirla en load_and_clean() (ver
                          remove_bulk_adjustments en
                          src/ventas_forecast/data/cleaning.py; CONFIRMADO
                          por el negocio el caso del 19/02/2026). Default
                          100 (conservador, ver CHANGELOG 2026-09-23
                          "Limpieza"). None desactiva la regla.
    elegibilidad_ampliada : EXPERIMENTAL, desactivado por defecto (produccion no
                          cambia). Si es True, tambien se entrenan y se
                          predicen productos SIN ventas en el ultimo mes pero
                          con media_3 > 0 (ver ventas_forecast.eligibility);
                          su razon de crecimiento y su clase usan media_3
                          como base del mes actual.
    estimacion_por_promedio : True por defecto (produccion). Tras
                          predict_next_month(), estimar_por_promedio() agrega
                          los productos sin ventas el ultimo mes con
                          media_3 > 0, >= min_months meses con ventas y dentro
                          de los top productos del regresor: su monto es la
                          media_3 (no una prediccion del modelo) y su decision
                          es "Sin recomendación (estimación por promedio)".
                          No cambia el entrenamiento. Con False la salida es
                          la de antes del paso.
    probabilidad_venta  : True por defecto (produccion). Tras
                          estimar_por_promedio(), agregar_probabilidad_venta()
                          entrena un RandomForest binario "habra ventas el mes
                          siguiente" (class_weight balanced) solo con la
                          particion de entrenamiento, elige el umbral con la
                          validacion interna y escribe prob_venta (0-1) y
                          se_vendera ("Sí"/"No") para todos los productos de
                          los resultados. No cambia el clasificador de 3
                          clases, el regresor, la mezcla ni la estimacion por
                          promedio. Con False la salida es la de antes del
                          paso.
    resumen_por_grupo   : True por defecto (produccion). Al final del
                          pipeline, agregar_resumen_grupo() suma las
                          predicciones por producto de cada grupo del
                          catalogo grupos_path (data/raw/grupos_productos.csv)
                          y lo deja en resumen_grupos. Solo lee los
                          resultados: no cambia ninguna prediccion, decision
                          ni total. Con False no se calcula.
    stock_sugerido      : True por defecto (produccion). Tras
                          agregar_probabilidad_venta(), agregar_stock_sugerido()
                          escribe stock_sugerido (COP) y, con precio conocido,
                          stock_sugerido_unidades: un nivel "order-up-to" tal
                          que la demanda del mes queda cubierta en una fraccion
                          nivel_servicio de los casos. Corrige la prediccion
                          con el cuantil de los residuos log1p(real) -
                          log1p(pred) de la validacion interna, por cubeta de
                          prob_venta (ver ventas_forecast.stock_sugerido).
                          Solo agrega columnas: no cambia decisiones, montos,
                          totales ni prob_venta. Con False la salida es la de
                          antes del paso.
    nivel_servicio      : nivel de servicio del stock sugerido (default 0,70).
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
        bulk_adjustment_min_products=100,
        elegibilidad_ampliada=False,
        estimacion_por_promedio=True,
        probabilidad_venta=True,
        resumen_por_grupo=True,
        grupos_path=None,
        stock_sugerido=True,
        nivel_servicio=0.70,
        datos=None,
        grupos_mapa=None,
    ):
        if filepath is None and datos is None:
            raise ValueError("Indique filepath (CSV) o datos (DataFrame)")
        self.filepath        = Path(filepath) if filepath is not None else None
        self.datos           = datos
        self.grupos_mapa     = grupos_mapa
        self.sep             = sep
        self.encoding        = encoding
        self.col_map         = {**DEFAULT_COL_MAP, **(col_map or {})}
        self.output_path     = (
            Path(output_path) if output_path
            else DEFAULT_PREDICTION
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
        self.bulk_adjustment_min_products = bulk_adjustment_min_products
        self.elegibilidad_ampliada = elegibilidad_ampliada
        self.estimacion_por_promedio = estimacion_por_promedio
        self.probabilidad_venta = probabilidad_venta
        self.resumen_por_grupo = resumen_por_grupo
        self.grupos_path = Path(grupos_path) if grupos_path else DATA_RAW / "grupos_productos.csv"
        self.stock_sugerido = stock_sugerido
        self.nivel_servicio = nivel_servicio
        # Poblados por agregar_stock_sugerido()
        self.calibracion_stock = None
        self.stock_validacion = None
        self._sale_val = None
        # Poblados por agregar_resumen_grupo()
        self.resumen_grupos = None
        self.mapa_grupos = None
        # Opcional (solo experimentos): funcion que recibe los meses actuales
        # (fecha) de las filas de entrenamiento y devuelve sus pesos de
        # muestra. None = sin pesos (comportamiento de produccion).
        self.sample_weight_fn = None
        # Poblados por agregar_probabilidad_venta()
        self.sale_clf = None
        self.sale_threshold = None
        self.sale_val_accuracy = None   # exactitud en la validación interna (selección)
        self.sale_test_accuracy = None  # exactitud en la prueba interna (reporte)
        self._sale_features = None
        self._sale_n_train = None
        self._sale_n_val = None

        # Estado interno
        self.df           = None   # historial completo
        self.cleaning_summary = None  # resumen de remove_bulk_adjustments (load_and_clean)
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
        self.reg_test_media_3 = None  # media_3 de las filas de prueba (train_regressor)
        self.blend_weight  = None     # w de la mezcla w*RF + (1-w)*media_3 (train_regressor)
        self.blend_metrics = {}       # métricas de la mezcla sobre prueba (train_regressor)
        self.blend_validacion = None  # validación interna con la que se elige w (train_regressor)
        self._top_interno = None      # top productos del entrenamiento interno
        self._interna     = None      # partición interna del entrenamiento (_particion_interna)
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
        if self.datos is not None:
            self.df = self.datos.copy()
        else:
            self.df = pd.read_csv(self.filepath, sep=self.sep, encoding=self.encoding)

        for role in ("sales", "total_sold", "avg_price"):
            self.df[c(role)] = self._parse_numeric(self.df[c(role)])

        # Ajustes masivos de inventario (HALLAZGO/CONFIRMADO 2026-09-23, ver
        # remove_bulk_adjustments): se aplican a nivel de FILA cruda, antes
        # de agregar a producto-mes, porque el ajuste se detecta por
        # combinación (fecha, cantidad, precio) DIARIA — a nivel mensual
        # varios productos legítimos podrían coincidir por casualidad.
        if self.bulk_adjustment_min_products is not None:
            self.df, self.cleaning_summary = remove_bulk_adjustments(
                self.df,
                date_col=c("date"),
                id_col=c("id"),
                qty_col=c("sales"),
                price_col=c("avg_price"),
                min_products=self.bulk_adjustment_min_products,
            )
            print(
                f"[✓] Ajustes masivos de inventario excluidos: "
                f"{self.cleaning_summary['rows_removed']:,} filas "
                f"({self.cleaning_summary['distinct_products']:,} productos, "
                f"{len(self.cleaning_summary['affected_dates'])} fecha(s), "
                f"${self.cleaning_summary['cop_amount']:,.0f} COP)"
            )

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

        if self.elegibilidad_ampliada:
            return self._build_train_frame_ampliada()

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

    def _build_train_frame_ampliada(self):
        """Variante experimental de _build_train_frame (elegibilidad_ampliada):
        agrega las filas con mes actual en cero y media_3 > 0. Las filas de
        produccion se parten exactamente igual (ver
        eligibility.temporal_train_mask)."""
        c = self._c
        mask = el.training_mask(
            self.df[c("total_sold")], self.df["media_3_actual"],
            self.df["target"], True)
        df_train = self.df[mask].copy().reset_index(drop=True)
        is_base = (df_train[c("total_sold")] > 0).to_numpy()
        train_idx, test_idx = [], []
        for _, g in df_train.groupby(c("id")):
            g = g.sort_values("fecha")
            en_train = el.temporal_train_mask(is_base[g.index.to_numpy()],
                                              self.train_ratio)
            train_idx += g.index[en_train].tolist()
            test_idx  += g.index[~en_train].tolist()
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
        if self.elegibilidad_ampliada:
            # Base de la decision: mes actual si hubo ventas; si no, media_3.
            df_train["growth_ratio"] = el.growth_ratio_base(
                df_train["target"], df_train[c("total_sold")],
                df_train["media_3_actual"])

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
        self._interna    = None
        self._top_interno = None

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

        if self.elegibilidad_ampliada:
            df_pred = self.df[
                (self.df["fecha"] == self._ultimo_mes) &
                el.prediction_mask(
                    self.df[c("total_sold")], self.df["media_3_actual"],
                    self.df[c("id")].isin(prods_ok), True)
            ].copy()
        else:
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

    def _fit_kwargs(self, rows):
        """sample_weight para fit() si hay sample_weight_fn; vacio si no."""
        if self.sample_weight_fn is None:
            return {}
        return {"sample_weight": np.asarray(
            self.sample_weight_fn(rows["fecha"].to_numpy()), float)}

    # ── 4b. VALIDACIÓN INTERNA (dentro de la partición de entrenamiento) ─────
    def _particion_interna(self):
        """
        Divide la partición de ENTRENAMIENTO (self._train_idx) en
        entrenamiento interno y validación interna con la misma regla
        temporal por producto de _build_train_frame (primeras train_ratio
        filas por fecha -> entrenamiento interno; el resto -> validación
        interna). Ajusta un escalador y un KMeans de clustering SOLO con el
        entrenamiento interno, para que la variable cluster de la validación
        interna no dependa de sus propios meses.

        Toda selección (peso de la mezcla, umbral de venta, cuantiles del
        stock) se hace con esta validación interna; la partición de prueba
        interna (self._test_idx) solo se usa para reportar.

        Devuelve (y guarda en caché hasta el próximo prepare_split) un dict
        con "tr" y "va" (etiquetas de self._df_train) y "cluster"
        ((escalador, kmeans) o None si hay menos filas que clusters).
        """
        if self._interna is not None:
            return self._interna
        c = self._c
        filas = self._df_train.loc[self._train_idx].sort_values([c("id"), "fecha"])
        if self.elegibilidad_ampliada:
            es_base = (filas[c("total_sold")] > 0).to_numpy()
            mask = np.zeros(len(filas), bool)
            pos = np.arange(len(filas))
            for _, g in filas.assign(_pos=pos).groupby(c("id"), sort=False):
                p = g["_pos"].to_numpy()
                mask[p] = el.temporal_train_mask(es_base[p], self.train_ratio)
        else:
            mask = vi.particion_temporal(filas[c("id")], filas["fecha"], self.train_ratio)
        tr = filas.index[mask].tolist()
        va = filas.index[~mask].tolist()
        cluster = None
        if len(tr) >= self.n_clusters:
            Xc = self._cluster_features(self._df_train.loc[tr])
            esc = RobustScaler().fit(Xc)
            km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            km.fit(esc.transform(Xc))
            cluster = (esc, km)
        self._interna = {"tr": tr, "va": va, "cluster": cluster}
        return self._interna

    def _cluster_interno(self, frame):
        """Cluster de las filas de frame según el KMeans del entrenamiento
        interno (o la columna cluster de producción si no hay KMeans
        interno)."""
        modelo = self._particion_interna()["cluster"]
        if modelo is None:
            return frame["cluster"].to_numpy()
        esc, km = modelo
        return km.predict(esc.transform(self._cluster_features(frame)))

    def _validacion_interna_regresor(self):
        """
        Repite el procedimiento del regresor con el entrenamiento interno:
        top productos, escalador, variable cluster y RandomForestRegressor
        (log1p, mismos hiperparámetros REG_PARAMS) ajustados SOLO con el
        entrenamiento interno, y predice la validación interna de esos top
        productos. Ningún objetivo de la validación interna, de la prueba
        interna ni del mes a predecir entra al ajuste.

        Devuelve un DataFrame id, fecha, real, rf, media_3 (media_3 con
        respaldo a las ventas del mes actual) y n_train, o None si la
        partición interna no alcanza (sin filas de entrenamiento o de
        validación interna).
        """
        c = self._c
        p = self._particion_interna()
        if not p["tr"] or not p["va"] or p["cluster"] is None:
            return None
        d = self._df_train
        filas = p["tr"] + p["va"]
        X = self._X.loc[filas].copy()
        X["cluster"] = self._cluster_interno(d.loc[filas])
        top = (d.loc[p["tr"]].groupby(c("id"))[c("total_sold")]
               .sum().nlargest(self.top_n_products).index)
        self._top_interno = top
        tr = d.loc[p["tr"]]
        tr = tr[tr[c("id")].isin(top)]
        va = d.loc[p["va"]]
        va = va[va[c("id")].isin(top)]
        if len(tr) == 0 or len(va) == 0:
            return None
        esc = RobustScaler().fit(X.loc[p["tr"]])
        reg = RandomForestRegressor(**REG_PARAMS, random_state=42, n_jobs=-1)
        reg.fit(esc.transform(X.loc[tr.index]), np.log1p(tr["target"]),
                **self._fit_kwargs(tr))
        rf = np.expm1(reg.predict(esc.transform(X.loc[va.index])))
        m3 = va["media_3_actual"].to_numpy(float)
        m3 = np.where(np.isnan(m3), va[c("total_sold")].to_numpy(float), m3)
        return pd.DataFrame({
            "id": va[c("id")].to_numpy(), "fecha": va["fecha"].to_numpy(),
            "real": va["target"].to_numpy(float), "rf": rf, "media_3": m3,
            "n_train": len(tr),
        })

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
        self.clf.fit(X_tr, y_tr, **self._fit_kwargs(self._df_train.iloc[self._train_idx]))

        pred = self.clf.predict(X_te)

        self.metrics = {
            # Ponderadas por soporte de clase (mantenidas por compatibilidad:
            # dashboard/app.py las lee tal cual). Bajo el desbalance de clases
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

        self.reg = RandomForestRegressor(**REG_PARAMS, random_state=42, n_jobs=-1)
        self.reg.fit(self.scaler.transform(X_reg_train), y_reg_train,
                     **self._fit_kwargs(df_reg_train))

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

        # ── Mezcla w*RF + (1-w)*media_3 ──────────────────────────────────────
        # El RF entrenado en log1p subestima el total en pesos (ratio de la
        # suma predicha / real < 1). w se elige SOLO con la validación
        # interna DENTRO de la partición de entrenamiento (un regresor
        # temporal ajustado con el entrenamiento interno predice los últimos
        # meses de entrenamiento de cada producto), minimizando
        # |suma(pred) - suma(real)|. La partición de prueba interna solo
        # reporta el resultado con ese w; tampoco se usa el mes objetivo ni
        # ningún holdout. Sin validación interna posible, w = 1 (RF puro).
        self.reg_test_media_3 = pred_media_3
        self.blend_validacion = self._validacion_interna_regresor()
        v = self.blend_validacion
        if v is not None and len(v):
            self.blend_weight = im.choose_blend_weight(
                v["real"].to_numpy(float), v["rf"].to_numpy(float),
                v["media_3"].to_numpy(float), criterion="sesgo")
        else:
            self.blend_weight = 1.0
            print("[!] Sin validación interna para la mezcla: w = 1 (RF puro)")
        pred_blend = im.blend(self.blend_weight, y_pred, pred_media_3)
        total_real = float(np.sum(y_true))

        def _ratio(pred):
            return (round(float(np.sum(pred)) / total_real, 4)
                    if total_real else float("nan"))

        self.reg_metrics["sum_ratio"] = _ratio(y_pred)
        self.blend_metrics = self._regression_metrics(y_true, pred_blend, scale)
        self.blend_metrics["sum_ratio"] = _ratio(pred_blend)
        self.blend_metrics["w"] = self.blend_weight

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
        if v is not None and len(v):
            print(f"Validación interna de la mezcla: {len(v):,} filas "
                  f"(regresor interno con {int(v['n_train'].iloc[0]):,} filas)")
        print(f"Mezcla w*RF + (1-w)*media_3: w={self.blend_weight}, "
              f"MAE={self.blend_metrics['mae']}, "
              f"razón de sumas={self.blend_metrics['sum_ratio']} "
              f"(RF {self.reg_metrics['sum_ratio']})")

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
            idx_top = mask_top[mask_top].index
            rf_pred = np.expm1(pred_sales_log)
            df_pred.loc[idx_top, "ventas_predichas_rf"] = rf_pred

            # Mezcla del sistema: w*RF + (1-w)*media_3; media_3 faltante →
            # ventas del mes actual; sin w (modelo no entrenado) → RF puro.
            m3 = df_pred.loc[idx_top, "media_3_actual"].to_numpy(dtype=float)
            m3 = np.where(np.isnan(m3),
                          df_pred.loc[idx_top, c("total_sold")].to_numpy(dtype=float), m3)
            m3 = np.where(np.isnan(m3), rf_pred, m3)
            w = 1.0 if self.blend_weight is None else self.blend_weight
            df_pred.loc[idx_top, "ventas_predichas"] = im.blend(w, rf_pred, m3)

        # Mes predicho
        df_pred["mes_predicho"] = self._mes_pred.strftime("%Y-%m")

        # Limpiar columnas internas que no aportan al cliente
        cols_salida = [
            c("id"), c("description"),
            "mes_predicho",
            c("total_sold"),          # ventas del último mes (base de comparación)
            "ventas_predichas",       # predicción cuantitativa (mezcla modelo + media_3)
            "ventas_predichas_rf",    # predicción cuantitativa solo del RF
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

    # ── 7b. ESTIMACIÓN POR PROMEDIO (productos sin venta el último mes) ──────
    def estimar_por_promedio(self):
        """
        Agrega a self.results los productos que NO vendieron en el último mes
        pero tienen media_3 > 0 (media de los últimos 3 meses, la misma de la
        elegibilidad ampliada), al menos min_months meses con ventas y están
        entre los top productos del regresor. Su monto (ventas_predichas) es
        media_3; ventas_predichas_rf, prob_crecimiento y prob_reduccion quedan
        en NaN y su decisión es ETIQUETA_SIN_RECOMENDACION: el clasificador no
        supera a la clase mayoritaria en estos productos (ver CHANGELOG
        "[Experimento] Elegibilidad ampliada"), por eso no se usa. El
        entrenamiento no se toca. Agrega la columna origen_estimacion
        ("modelo" / "promedio") y vuelve a exportar el CSV. Con
        estimacion_por_promedio=False no hace nada.
        """
        if not self.estimacion_por_promedio:
            return
        c = self._c
        ultimo = self.df[self.df["fecha"] == self._ultimo_mes]
        conteo = (self.df[self.df[c("total_sold")] > 0]
                  .groupby(c("id"))["fecha"].count())
        prods_ok = conteo[conteo >= self.min_months].index
        sel = ultimo[
            (ultimo[c("total_sold")] <= 0) &
            (ultimo["media_3_actual"] > 0) &
            ultimo[c("id")].isin(prods_ok) &
            ultimo[c("id")].isin(self._top_prods)
        ]
        nuevos = pd.DataFrame(index=sel.index)
        nuevos[c("id")] = sel[c("id")]
        nuevos[c("description")] = sel[c("description")]
        nuevos["mes_predicho"] = self._mes_pred.strftime("%Y-%m")
        nuevos[c("total_sold")] = sel[c("total_sold")]
        nuevos["ventas_predichas"] = sel["media_3_actual"]
        nuevos["ventas_predichas_rf"] = np.nan
        nuevos["prob_crecimiento"] = np.nan
        nuevos["prob_reduccion"] = np.nan
        nuevos["decision"] = ETIQUETA_SIN_RECOMENDACION
        nuevos["cluster"] = sel["cluster"]
        nuevos[c("frequency")] = sel[c("frequency")]
        nuevos[c("avg_price")] = sel[c("avg_price")]

        modelo = self.results.copy()
        modelo["origen_estimacion"] = "modelo"
        nuevos["origen_estimacion"] = "promedio"
        nuevos = nuevos[[col for col in modelo.columns if col in nuevos.columns]]
        self.results = pd.concat([modelo, nuevos])

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(self.output_path, index=False)
        print(f"[✓] Estimación por promedio: {len(nuevos):,} productos sin "
              f"ventas el último mes (monto = media de 3 meses, sin decisión)")

    # ── 7c. PROBABILIDAD DE VENTA (habra ventas el mes siguiente: si/no) ─────
    def agregar_probabilidad_venta(self):
        """
        Entrena un RandomForest binario (vende = ventas del mes siguiente > 0,
        SALE_CLF_PARAMS, misma configuracion del experimento "[Experimento]
        Probabilidad de venta") con las variables de SalesForecastModel mas las
        de ventas_forecast.sale_probability, SOLO con la particion temporal de
        entrenamiento. El umbral se elige con la validacion interna DENTRO de
        ese entrenamiento: un clasificador temporal con los mismos
        hiperparametros (y la variable cluster del KMeans interno) se ajusta
        con el entrenamiento interno y el umbral maximiza su exactitud en la
        validacion interna. La particion de prueba interna solo reporta la
        exactitud con ese umbral (sale_test_accuracy); nunca se usa el mes a
        predecir. Escribe en self.results prob_venta (0-1) y se_vendera
        ("Sí"/"No") para todos los productos (modelo y promedio) y vuelve a
        exportar el CSV. Guarda sale_threshold, sale_val_accuracy (exactitud
        en la validacion interna, la de la seleccion) y sale_test_accuracy.
        No toca el clasificador de 3 clases, el regresor ni la mezcla. Con
        probabilidad_venta=False no hace nada.
        """
        if not self.probabilidad_venta:
            return
        c = self._c
        d = sp.add_probability_features(
            self.df, c("id"), c("total_sold"), c("frequency"))
        frame, tr, te = sp.sale_train_frame(
            d, c("id"), c("total_sold"), self.train_ratio)
        feats = [
            "VENTAS_LOG", "TOTAL_LOG", "PRECIO_LOG", c("frequency"),
            *[f"lag_{l}" for l in self.lags], "mean_3", "mean_6",
            "mes_sig_sin", "mes_sig_cos", "cluster", *sp.NEW_FEATURES,
        ]
        self._sale_features = feats
        X = frame[feats].fillna(0)
        y = frame["vende"].to_numpy()
        tr, te = list(tr), list(te)

        # ── Umbral: validacion interna dentro del entrenamiento ─────────────
        sub = frame.iloc[tr]
        en_tr = vi.particion_temporal(sub[c("id")], sub["fecha"], self.train_ratio)
        tr_in = [i for i, m in zip(tr, en_tr) if m]
        va_in = [i for i, m in zip(tr, en_tr) if not m]
        if tr_in and va_in:
            X_in = X.copy()
            X_in["cluster"] = self._cluster_interno(frame)
            clf_in = RandomForestClassifier(**SALE_CLF_PARAMS, random_state=42, n_jobs=-1)
            clf_in.fit(X_in.iloc[tr_in].to_numpy(), y[tr_in])
            p_val = _prob_vende(clf_in, X_in.iloc[va_in].to_numpy())
            umbral = sp.choose_threshold(y[va_in], p_val)
            acc_val = float(((p_val >= umbral).astype(int) == y[va_in]).mean())
        else:
            p_val = np.array([])
            umbral, acc_val = 0.5, float("nan")

        # ── Modelo de produccion (todo el entrenamiento) y reporte en prueba ─
        clf = RandomForestClassifier(**SALE_CLF_PARAMS, random_state=42, n_jobs=-1)
        clf.fit(X.iloc[tr].to_numpy(), y[tr])
        if te:
            p_te = _prob_vende(clf, X.iloc[te].to_numpy())
            acc_test = float(((p_te >= umbral).astype(int) == y[te]).mean())
        else:
            acc_test = float("nan")

        ultimo = d[d["fecha"] == self._ultimo_mes].drop_duplicates(c("id"))
        p_ult = pd.Series(_prob_vende(clf, ultimo[feats].fillna(0).to_numpy()),
                          index=ultimo[c("id")].to_numpy())
        res = self.results.copy()
        prob = res[c("id")].map(p_ult).astype(float)
        res["prob_venta"] = prob.to_numpy()
        res["se_vendera"] = np.where(res["prob_venta"] >= umbral, "Sí", "No")
        self.results = res

        self.sale_clf = clf
        self.sale_threshold = float(umbral)
        self.sale_val_accuracy = acc_val
        self.sale_test_accuracy = acc_test
        self._sale_n_train = len(tr)
        self._sale_n_val = len(va_in)
        self._sale_n_test = len(te)
        # Validacion interna con probabilidad fuera de muestra del
        # clasificador temporal (la usa agregar_stock_sugerido para calibrar
        # por cubeta de prob_venta). Nunca filas de la prueba interna.
        fv = frame.iloc[va_in]
        self._sale_val = pd.DataFrame({
            "id": fv[c("id")].to_numpy(),
            "fecha": fv["fecha"].to_numpy(),
            "actual": fv[c("total_sold")].to_numpy(float),
            "media_3": fv["media_3_actual"].to_numpy(float),
            "target": fv["target"].to_numpy(float),
            "meses_con_venta": np.rint(
                fv["frac_hist"].to_numpy(float) * (fv["edad_meses"].to_numpy(float) + 1)),
            "prob": p_val,
        })

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(self.output_path, index=False)
        n_si = int((res["se_vendera"] == "Sí").sum())
        print(f"[✓] Probabilidad de venta: umbral {umbral:.2f} (validación "
              f"interna, exactitud {acc_val:.1%}); exactitud en la prueba "
              f"interna {acc_test:.1%}; {n_si:,} de {len(res):,} productos se "
              f"venderán")

    # ── 7e. STOCK SUGERIDO (nivel de servicio por producto) ──────────────────
    def _validacion_stock(self):
        """Residuos de la validacion interna DENTRO del entrenamiento (nunca la
        prueba interna, el mes a predecir ni un holdout). Productos del
        modelo: validacion interna del regresor (blend_validacion: regresor
        temporal ajustado con el entrenamiento interno) con la prediccion
        w*RF + (1-w)*media_3 y el w elegido en esa misma validacion.
        Productos por promedio: filas de la validacion interna del
        clasificador de venta con el mes actual en cero, media_3 > 0,
        >= min_months meses con ventas y top producto del entrenamiento
        interno, con prediccion media_3. La probabilidad de venta es la
        fuera de muestra del clasificador temporal en esa validacion (NaN si
        la fila no esta en ella o no hay modelo de venta). Devuelve un
        DataFrame id, fecha, origen, prob, pred, real, resid."""
        c = self._c
        cols = ["id", "fecha", "origen", "prob", "pred", "real"]
        partes = []
        v = self.blend_validacion
        if v is not None and len(v):
            w = 1.0 if self.blend_weight is None else self.blend_weight
            partes.append(pd.DataFrame({
                "id": v["id"].to_numpy(), "fecha": v["fecha"].to_numpy(),
                "origen": "modelo", "prob": np.nan,
                "pred": im.blend(w, v["rf"].to_numpy(float), v["media_3"].to_numpy(float)),
                "real": v["real"].to_numpy(float),
            }))
        sv = self._sale_val
        if sv is not None and len(sv):
            if partes:
                llave = sv.set_index(["id", "fecha"])["prob"]
                llave = llave[~llave.index.duplicated()]
                idx = pd.MultiIndex.from_arrays(
                    [partes[0]["id"], partes[0]["fecha"]])
                partes[0]["prob"] = llave.reindex(idx).to_numpy(float)
            if self.estimacion_por_promedio:
                top = (self._top_interno if self._top_interno is not None
                       else self._top_prods)
                p = sv[(sv["actual"] <= 0) & (sv["media_3"] > 0)
                       & (sv["meses_con_venta"] >= self.min_months)
                       & sv["id"].isin(top) & sv["target"].notna()]
                if len(p):
                    partes.append(pd.DataFrame({
                        "id": p["id"].to_numpy(), "fecha": p["fecha"].to_numpy(),
                        "origen": "promedio", "prob": p["prob"].to_numpy(float),
                        "pred": p["media_3"].to_numpy(float),
                        "real": p["target"].to_numpy(float),
                    }))
        val = (pd.concat(partes, ignore_index=True) if partes
               else pd.DataFrame({k: [] for k in cols}))
        val["resid"] = ss.residuos_log(val["real"], val["pred"])
        return val

    def agregar_stock_sugerido(self):
        """
        Escribe en self.results stock_sugerido (COP) y stock_sugerido_unidades
        para los productos con monto predicho: la cota superior a nivel de
        servicio nivel_servicio, expm1(log1p(pred) + q), donde q es el cuantil
        de los residuos log1p(real) - log1p(pred) de la validacion interna
        dentro del entrenamiento (ver _validacion_stock; nunca la prueba
        interna) por cubeta de prob_venta. Sin probabilidad de
        venta se usa el cuantil global. Las unidades son el techo del stock
        entre el ultimo precio promedio conocido. Guarda calibracion_stock
        (cuantiles de 0,5 / 0,7 / 0,9 y el nivel elegido) y stock_validacion.
        No cambia ninguna otra columna. Con stock_sugerido=False no hace nada.
        """
        if not self.stock_sugerido:
            return
        c = self._c
        val = self._validacion_stock()
        niveles = sorted(set(ss.NIVELES) | {float(self.nivel_servicio)})
        cal = ss.calibrar(val["prob"], val["resid"], niveles=niveles)
        res = self.results.copy()
        pred = res["ventas_predichas"].to_numpy(float)
        if "prob_venta" in res.columns:
            prob = res["prob_venta"].to_numpy(float)
        else:
            prob = np.full(len(res), np.nan)
        q = ss.cuantiles_para(cal, prob, float(self.nivel_servicio))
        stock = ss.cota_superior(pred, q)
        res["stock_sugerido"] = stock
        precio = (res[c("avg_price")].to_numpy(float)
                  if c("avg_price") in res.columns else np.full(len(res), np.nan))
        res["stock_sugerido_unidades"] = ss.unidades(stock, precio)
        self.results = res
        self.calibracion_stock = cal
        self.stock_validacion = val
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(self.output_path, index=False)
        print(f"[✓] Stock sugerido (nivel {self.nivel_servicio:.0%}): "
              f"{int(np.isfinite(stock).sum()):,} productos; calibrado con "
              f"{cal['n_total']:,} residuos de validación interna "
              f"({len(cal['n'])} cubetas de probabilidad de venta)")
        for i, n in enumerate(cal["n"]):
            q = cal["q"][float(self.nivel_servicio)][i]
            print(f"    prob_venta [{cal['edges'][i]:.1f}-{cal['edges'][i + 1]:.1f}): "
                  f"n={n:,}, cuantil {self.nivel_servicio:.0%} de log1p(real) - "
                  f"log1p(pred) = {q:+.3f}")

    # ── 7d. RESUMEN POR GRUPO DE PRODUCTO (suma de predicciones) ─────────────
    def agregar_resumen_grupo(self):
        """
        Construye self.resumen_grupos: por grupo de producto (catalogo
        grupos_path), cuantos productos tienen monto predicho, cuantos
        probablemente se venderan, la suma de sus predicciones, sus ventas del
        mes anterior, la variacion y el porcentaje del total. Es la variante G0
        del experimento "[Experimento] Predicción mensual por grupo". No
        modifica self.results ni el CSV. Si el catalogo no existe, todos los
        productos quedan en "Sin grupo". Con resumen_por_grupo=False no hace
        nada.
        """
        if not self.resumen_por_grupo:
            return
        c = self._c
        if self.grupos_mapa is not None:
            mapa = self.grupos_mapa
        elif self.grupos_path.exists():
            mapa = grupos.cargar_mapa(self.grupos_path)
        else:
            print(f"[!] Sin catálogo de grupos ({self.grupos_path}): "
                  f"todos los productos quedan en 'Sin grupo'")
            mapa = pd.Series(dtype=object)
        self.mapa_grupos = mapa
        self.resumen_grupos = grupos.resumen_grupos(
            self.results, mapa, c("id"), c("total_sold"))
        print(f"[✓] Resumen por grupo: {len(self.resumen_grupos):,} grupos")

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
        self.estimar_por_promedio()
        self.agregar_probabilidad_venta()
        self.agregar_stock_sugerido()
        self.agregar_resumen_grupo()
        self.plot()
        return self.results

