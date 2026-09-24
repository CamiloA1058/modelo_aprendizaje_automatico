"""
benchmarks.py
=============
Utilidades genéricas para comparar modelos de pronóstico sobre las MISMAS
filas/partición que scripts/train_kmeans_rf_prod.SalesForecastModel, usadas
por scripts/compare_models.py:

  - rolling_one_step_forecast(): pronóstico a un paso re-entrenado mes a
    mes (sin fuga hacia el futuro: cada ajuste solo ve el historial hasta
    el mes actual).
  - arima_fit_predict() / prophet_fit_predict(): ajustan un modelo de serie
    de tiempo sobre log1p(historial) y devuelven la predicción del mes
    siguiente en escala original (expm1), con respaldo a un pronóstico
    ingenuo (contado) cuando el historial es insuficiente o el ajuste falla.
  - build_eval_frame() / sample_products_with_test_rows(): construyen el
    mismo conjunto de filas de evaluación para todos los modelos comparados,
    para que la comparación sea justa.
"""

import contextlib
import io
import logging
import warnings

import numpy as np
import pandas as pd


def _pronostico_ingenuo(historia):
    """Último valor del historial, o 0.0 si el historial está vacío. Se usa
    como respaldo cuando un modelo de serie de tiempo no puede ajustarse."""
    if len(historia) == 0:
        return 0.0
    return max(float(historia[-1]), 0.0)


def rolling_one_step_forecast(series, test_positions, fit_predict):
    """
    Pronóstico a un paso re-entrenado mes a mes, SIN fuga de datos hacia el
    futuro: para cada posición de prueba t, ajusta con el historial
    series[:t+1] (hasta e incluyendo el mes t) y pronostica el mes t+1.

    Esto reproduce exactamente la convención del modelo de producción
    (build_features(): target = total_sold.shift(-1)): la fila cuyo "mes
    actual" está en la posición t tiene como target el valor de la
    posición t+1. Por eso las predicciones se devuelven indexadas por t (no
    por t+1): el resultado predicciones[t] debe compararse contra
    series[t+1], nunca contra series[t].

    Parámetros
    ----------
    series         : secuencia de valores mensuales (ordenados por fecha,
                     una posición por mes calendario, sin huecos).
    test_positions : posiciones t (0-based) a pronosticar. Si t es la
                     última posición de la serie (no hay mes t+1), esa
                     posición se omite del resultado (nada que pronosticar).
    fit_predict     : función que recibe el historial series[:t+1] (array
                      de numpy) y devuelve la predicción escalar del mes
                      t+1. No recibe información posterior a t bajo ninguna
                      circunstancia.

    Devuelve un dict {t: predicción} con una entrada por cada posición de
    prueba que sí tiene mes siguiente.
    """
    serie = np.asarray(series, dtype=float)
    predicciones = {}
    for t in test_positions:
        if t + 1 >= len(serie):
            continue  # no existe un mes t+1 que pronosticar
        historia = serie[: t + 1]
        predicciones[t] = float(fit_predict(historia))
    return predicciones


def arima_fit_predict(history, order=(1, 1, 1), min_history=6, fallback_counter=None):
    """
    Ajusta un ARIMA(1,1,1) (statsmodels) sobre log1p(history) y devuelve la
    predicción de un paso en escala original (expm1), recortada a >= 0.

    Cae a un pronóstico ingenuo (último valor del historial, recortado a
    >= 0) cuando el historial tiene menos de min_history puntos, o cuando el
    ajuste falla (no converge, produce un valor no finito, o lanza
    cualquier excepción). Si fallback_counter es una lista, se le agrega un
    elemento cada vez que ocurre un respaldo (permite contar respaldos a
    través de muchas llamadas, ver scripts/compare_models.py).

    Las advertencias de convergencia de statsmodels se silencian: un ARIMA
    que no converge bien en una serie corta es un resultado esperable (cae
    al respaldo ingenuo), no algo que deba imprimirse por cada producto.
    """
    historia = np.asarray(history, dtype=float)

    if len(historia) < min_history:
        if fallback_counter is not None:
            fallback_counter.append(1)
        return _pronostico_ingenuo(historia)

    try:
        historia_log = np.log1p(np.clip(historia, 0, None))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from statsmodels.tsa.arima.model import ARIMA

            modelo = ARIMA(historia_log, order=order)
            resultado = modelo.fit()
            pred_log = resultado.forecast(steps=1)[0]

        pred = float(np.expm1(pred_log))
        if not np.isfinite(pred):
            raise ValueError("predicción de ARIMA no finita")
        return max(pred, 0.0)
    except Exception:
        if fallback_counter is not None:
            fallback_counter.append(1)
        return _pronostico_ingenuo(historia)


def prophet_fit_predict(history, min_history=6, fallback_counter=None):
    """
    Ajusta Prophet sobre log1p(history) (una fecha mensual arbitraria por
    punto, ya que solo importa la posición relativa para un pronóstico de
    un paso) y devuelve la predicción del mes siguiente en escala original
    (expm1), recortada a >= 0.

    La estacionalidad anual se desactiva explícitamente: con ~2 años de
    historial no hay ciclos suficientes para estimarla y, en series
    intermitentes, su ajuste dispara el pronóstico. El registro de
    cmdstanpy/prophet y la salida de la compilación de Stan se silencian
    (son muy verbosos y se llama esta función una vez por producto y mes
    de prueba).

    Cae a un pronóstico ingenuo (contado en fallback_counter, igual que
    arima_fit_predict) cuando el historial es muy corto o el ajuste falla.
    """
    historia = np.asarray(history, dtype=float)

    if len(historia) < min_history:
        if fallback_counter is not None:
            fallback_counter.append(1)
        return _pronostico_ingenuo(historia)

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)

    try:
        import cmdstanpy
        from prophet import Prophet

        fechas = pd.date_range("2000-01-01", periods=len(historia), freq="MS")
        df_prophet = pd.DataFrame({
            "ds": fechas,
            "y": np.log1p(np.clip(historia, 0, None)),
        })

        # Estacionalidad anual desactivada: Prophet la activa sola con ~2 años
        # de datos, pero estimar 20 términos de Fourier con ~25 puntos
        # mensuales intermitentes sobreajusta y dispara el pronóstico (hasta
        # 4,2e10 frente a un máximo histórico de 1,3e6). Semanal y diaria no
        # aplican a datos mensuales.
        modelo = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        buffer_salida = io.StringIO()
        # cmdstanpy fija su propio logger a DEBUG con un handler a INFO la
        # primera vez que se usa (independiente del nivel que se le haya
        # puesto antes de esa primera llamada), así que además de subir el
        # nivel hay que deshabilitarlo explícitamente para silenciar el
        # "Chain [1] start/done processing" en cada ajuste.
        with cmdstanpy.disable_logging(), \
                contextlib.redirect_stdout(buffer_salida), \
                contextlib.redirect_stderr(buffer_salida):
            modelo.fit(df_prophet)
            futuro = modelo.make_future_dataframe(periods=1, freq="MS")
            pronostico = modelo.predict(futuro)

        pred_log = float(pronostico["yhat"].iloc[-1])
        pred = float(np.expm1(pred_log))
        if not np.isfinite(pred):
            raise ValueError("predicción de Prophet no finita")
        return max(pred, 0.0)
    except Exception:
        if fallback_counter is not None:
            fallback_counter.append(1)
        return _pronostico_ingenuo(historia)


def build_eval_frame(df_train, test_idx, id_col, product_ids):
    """
    Filas de PRUEBA de df_train restringidas a product_ids: el mismo
    conjunto de filas se usa para evaluar TODOS los modelos comparados
    (RF, XGBoost, ingenuo, media_3, Prophet, ARIMA), para que la
    comparación sea justa.

    Determinista: dos llamadas con los mismos argumentos devuelven
    exactamente las mismas filas, en el mismo orden.
    """
    mascara = df_train.index.isin(test_idx) & df_train[id_col].isin(product_ids)
    return df_train.loc[mascara].copy()


def sample_products_with_test_rows(df_train, test_idx, id_col, product_ids, n=150, random_state=42):
    """
    Muestra semillada (random_state) de hasta n productos, tomados de
    product_ids, que tengan AL MENOS una fila de prueba en df_train (según
    test_idx). Un producto sin ninguna fila de prueba no aportaría ninguna
    observación evaluable, así que nunca se incluye.

    Determinista: la misma semilla siempre devuelve la misma muestra. Si
    hay n o menos productos elegibles, se devuelven todos (sin muestrear).

    Devuelve una lista ordenada de identificadores de producto.
    """
    en_prueba = df_train.index.isin(test_idx)
    elegibles = (
        df_train.loc[en_prueba & df_train[id_col].isin(product_ids), id_col]
        .unique()
    )
    elegibles = np.sort(elegibles)  # orden determinista antes de muestrear

    if len(elegibles) <= n:
        return sorted(elegibles.tolist())

    rng = np.random.default_rng(random_state)
    muestra = rng.choice(elegibles, size=n, replace=False)
    return sorted(muestra.tolist())
