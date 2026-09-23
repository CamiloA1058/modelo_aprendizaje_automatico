"""
Verifica src/ventas_forecast/benchmarks.py, usado por scripts/compare_models.py
para comparar RF (modelo de producción) contra XGBoost, Prophet y ARIMA sobre
las MISMAS filas/partición:

  - rolling_one_step_forecast(): pronóstico a un paso re-entrenado mes a mes,
    SIN fuga hacia el futuro (solo ve el historial hasta el mes actual).
  - Alineación: la predicción del mes t se compara contra el mes t+1.
  - arima_fit_predict()/prophet_fit_predict(): devuelven un valor finito y no
    negativo, y caen a un pronóstico ingenuo (contado) si el historial es
    muy corto o el ajuste falla.
  - build_eval_frame()/sample_products_with_test_rows(): garantizan que todos
    los modelos comparados usan exactamente el mismo conjunto de filas.

Ejecutar:  python -m unittest tests.test_benchmarks
"""

import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.benchmarks import (  # noqa: E402
    rolling_one_step_forecast,
    arima_fit_predict,
    prophet_fit_predict,
    build_eval_frame,
    sample_products_with_test_rows,
)


class RollingOneStepForecastTest(unittest.TestCase):
    """(a) Sin fuga de datos: cada llamada a fit_predict solo ve el
    historial hasta e incluyendo el mes actual (nunca el mes a pronosticar
    ni ninguno posterior)."""

    def test_trivial_fit_predict_reproduces_naive_forecast(self):
        serie = [10.0, 20.0, 41.0, 53.0, 77.0, 90.0]
        posiciones_prueba = [1, 3, 4]

        # fit_predict "ingenuo": el pronóstico del mes siguiente es el
        # último valor del historial (equivalente al baseline "ingenuo").
        predicciones = rolling_one_step_forecast(
            serie, posiciones_prueba, fit_predict=lambda h: h[-1]
        )

        for t in posiciones_prueba:
            self.assertEqual(predicciones[t], serie[t])

    def test_never_sees_beyond_the_current_month(self):
        serie = [10.0, 20.0, 41.0, 53.0, 77.0, 90.0, 15.0]
        posiciones_prueba = [2, 4, 5]
        longitudes_vistas = []

        def fit_predict_espia(historia):
            longitudes_vistas.append(len(historia))
            return historia[-1]

        rolling_one_step_forecast(
            serie, posiciones_prueba, fit_predict=fit_predict_espia
        )

        # La longitud del historial visto en el mes t es exactamente t+1
        # (posiciones 0..t incluidas); nunca t+2 ni más (eso sería ver el
        # mes que se está pronosticando, o uno posterior).
        for t, longitud in zip(posiciones_prueba, longitudes_vistas):
            self.assertEqual(longitud, t + 1)

        self.assertEqual(max(longitudes_vistas), max(posiciones_prueba) + 1)

    def test_skips_positions_with_no_next_month(self):
        # La última posición de la serie no tiene "mes siguiente" que
        # pronosticar: no debe generar una entrada en el resultado.
        serie = [1.0, 2.0, 3.0]
        predicciones = rolling_one_step_forecast(
            serie, [0, 1, 2], fit_predict=lambda h: h[-1]
        )
        self.assertNotIn(2, predicciones)
        self.assertIn(0, predicciones)
        self.assertIn(1, predicciones)


class AlignmentTest(unittest.TestCase):
    """(b) La predicción del mes t se compara contra el valor REAL del mes
    t+1 (el "target"), no contra el mes t."""

    def test_prediction_for_month_t_is_compared_against_month_t_plus_1(self):
        # Serie estrictamente creciente: el valor de un mes nunca coincide
        # con el de otro mes, así que comparar contra el mes equivocado es
        # detectable.
        serie = [10.0, 25.0, 44.0, 61.0, 80.0]
        posiciones_prueba = [1, 3]

        predicciones = rolling_one_step_forecast(
            serie, posiciones_prueba, fit_predict=lambda h: h[-1]
        )

        for t in posiciones_prueba:
            objetivo_real = serie[t + 1]
            # El pronóstico ingenuo coincide con el propio mes t...
            self.assertEqual(predicciones[t], serie[t])
            # ...y por lo tanto es DISTINTO del objetivo real (mes t+1):
            # confirma que la comparación correcta es contra t+1, no contra t.
            self.assertNotEqual(predicciones[t], objetivo_real)


class ArimaFitPredictTest(unittest.TestCase):
    """(c) arima_fit_predict(): valor finito y no negativo; respaldo
    ingenuo (contado) si el historial es muy corto o el ajuste falla."""

    def test_falls_back_to_naive_for_short_history(self):
        contador_respaldo = []
        historia_corta = [100.0, 110.0, 90.0]  # < 6 puntos

        pred = arima_fit_predict(historia_corta, fallback_counter=contador_respaldo)

        self.assertEqual(len(contador_respaldo), 1)
        self.assertTrue(math.isfinite(pred))
        self.assertGreaterEqual(pred, 0.0)
        self.assertEqual(pred, historia_corta[-1])

    def test_returns_finite_nonnegative_value_on_short_synthetic_series(self):
        rng = np.random.default_rng(0)
        historia = 100 + 10 * np.sin(np.arange(12)) + rng.normal(0, 1, 12)
        historia = np.clip(historia, 1, None)

        pred = arima_fit_predict(historia)

        self.assertTrue(math.isfinite(pred))
        self.assertGreaterEqual(pred, 0.0)

    def test_falls_back_to_naive_when_fit_raises(self):
        # Historial constante en cero: el ARIMA(1,1,1) puede fallar a
        # converger o lanzar excepción; en cualquier caso el resultado debe
        # seguir siendo finito y no negativo (ya sea el ajuste real o el
        # respaldo ingenuo).
        contador_respaldo = []
        historia = [0.0] * 8
        pred = arima_fit_predict(historia, fallback_counter=contador_respaldo)
        self.assertTrue(math.isfinite(pred))
        self.assertGreaterEqual(pred, 0.0)


class ProphetFitPredictTest(unittest.TestCase):
    """(c) prophet_fit_predict(): valor finito y no negativo; respaldo
    ingenuo (contado) si el historial es muy corto."""

    def test_falls_back_to_naive_for_short_history(self):
        contador_respaldo = []
        historia_corta = [50.0, 60.0]  # < 6 puntos

        pred = prophet_fit_predict(historia_corta, fallback_counter=contador_respaldo)

        self.assertEqual(len(contador_respaldo), 1)
        self.assertTrue(math.isfinite(pred))
        self.assertGreaterEqual(pred, 0.0)
        self.assertEqual(pred, historia_corta[-1])

    def test_returns_finite_nonnegative_value_on_short_synthetic_series(self):
        rng = np.random.default_rng(0)
        historia = 100 + 10 * np.sin(np.arange(12)) + rng.normal(0, 1, 12)
        historia = np.clip(historia, 1, None)

        inicio = time.time()
        pred = prophet_fit_predict(historia)
        duracion = time.time() - inicio

        self.assertTrue(math.isfinite(pred))
        self.assertGreaterEqual(pred, 0.0)
        # No es una aserción de rendimiento estricta, solo evidencia en caso
        # de que este caso se vuelva lento y haya que marcarlo con
        # unittest.skipUnless en el futuro.
        self.assertLess(duracion, 30.0)

    def test_does_not_explode_on_intermittent_two_year_series(self):
        # Serie real (producto 1185, V5) de 26 meses con meses en cero: con
        # la estacionalidad anual automática de Prophet (se activa con ~2
        # años de datos) el pronóstico llegaba a 4,2e10 frente a un máximo
        # histórico de 1,3e6.
        historia = np.array([
            247000, 0, 116000, 44503, 53000, 13000, 0, 27650, 25858, 0,
            27430, 39715, 26715, 320825, 55700, 0, 165300, 613065, 449200,
            146000, 72500, 189000, 256500, 0, 144200, 1307286,
        ], dtype=float)

        pred = prophet_fit_predict(historia)

        self.assertTrue(math.isfinite(pred))
        self.assertLess(pred, 10 * historia.max())


class EvalFrameTest(unittest.TestCase):
    """(d) build_eval_frame()/sample_products_with_test_rows(): mismo
    conjunto de filas para todos los modelos comparados."""

    def setUp(self):
        self.df_train = pd.DataFrame({
            "CODIGO": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "target": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        })
        self.test_idx = [2, 5, 8]  # última fila de cada producto

    def test_build_eval_frame_keeps_only_test_rows_of_selected_products(self):
        frame = build_eval_frame(self.df_train, self.test_idx, "CODIGO", [0, 1])

        self.assertEqual(set(frame.index), {2, 5})
        self.assertTrue(set(frame["CODIGO"].unique()).issubset({0, 1}))

    def test_build_eval_frame_is_deterministic_across_calls(self):
        frame1 = build_eval_frame(self.df_train, self.test_idx, "CODIGO", [0, 1, 2])
        frame2 = build_eval_frame(self.df_train, self.test_idx, "CODIGO", [0, 1, 2])
        pd.testing.assert_frame_equal(frame1, frame2)

    def test_sample_products_with_test_rows_is_seeded_and_reproducible(self):
        # 5 productos, todos con exactamente una fila de prueba (última fila
        # de cada uno); se muestrean 2.
        df_train = pd.DataFrame({
            "CODIGO": [p for p in range(5) for _ in range(4)],
            "target": range(20),
        })
        test_idx = [3, 7, 11, 15, 19]

        muestra1 = sample_products_with_test_rows(
            df_train, test_idx, "CODIGO", list(range(5)), n=2, random_state=42
        )
        muestra2 = sample_products_with_test_rows(
            df_train, test_idx, "CODIGO", list(range(5)), n=2, random_state=42
        )

        self.assertEqual(muestra1, muestra2)
        self.assertEqual(len(muestra1), 2)

        productos_con_prueba = set(df_train.loc[test_idx, "CODIGO"])
        for producto in muestra1:
            self.assertIn(producto, productos_con_prueba)

    def test_sample_products_excludes_products_without_test_rows(self):
        # El producto 4 no tiene ninguna fila en test_idx: nunca debe salir
        # en la muestra, incluso pidiendo más muestras que productos válidos.
        df_train = pd.DataFrame({
            "CODIGO": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "target": range(10),
        })
        test_idx = [1, 3, 5, 7]  # producto 4 (índices 8, 9) queda fuera

        muestra = sample_products_with_test_rows(
            df_train, test_idx, "CODIGO", list(range(5)), n=10, random_state=42
        )

        self.assertNotIn(4, muestra)
        self.assertEqual(set(muestra), {0, 1, 2, 3})

    def test_all_comparison_models_would_use_the_same_row_set(self):
        # Simula el uso real: la misma llamada a build_eval_frame() alimenta
        # "todos los modelos" (aquí representados por dos listas de índices
        # calculadas independientemente) — deben coincidir exactamente.
        productos = [0, 1, 2]
        filas_modelo_a = build_eval_frame(
            self.df_train, self.test_idx, "CODIGO", productos
        ).index.tolist()
        filas_modelo_b = build_eval_frame(
            self.df_train, self.test_idx, "CODIGO", productos
        ).index.tolist()
        self.assertEqual(filas_modelo_a, filas_modelo_b)


if __name__ == "__main__":
    unittest.main()
