"""Reglas de la acción única por producto (ventas_forecast.accion).

Ejecutar:  python -m unittest tests.test_accion
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import accion as ac  # noqa: E402


class DecidirTest(unittest.TestCase):
    P = np.array([
        [0.1, 0.2, 0.7],   # Reforzar con confianza
        [0.3, 0.2, 0.5],   # Reforzar con poca confianza -> mejor de las otras: Reducir
        [0.1, 0.35, 0.55],  # poca confianza -> Mantener
        [0.6, 0.3, 0.1],   # Reducir
    ])

    def test_sin_umbrales_es_argmax(self):
        self.assertEqual(ac.decidir(self.P).tolist(), [2, 2, 2, 0])

    def test_confianza_minima_manda_a_la_mejor_de_las_otras(self):
        self.assertEqual(ac.decidir(self.P, t=0.6).tolist(), [2, 0, 1, 0])

    def test_sin_probabilidad_de_venta_pasa_a_mantener(self):
        pv = np.array([0.5, 0.9, 0.9, 0.1])
        self.assertEqual(ac.decidir(self.P, pv, t=0.6, s=0.6).tolist(), [1, 0, 1, 0])


class UmbralReforzarTest(unittest.TestCase):

    def test_menor_t_con_precision_meta(self):
        rng = np.random.default_rng(1)
        n = 400
        p2 = rng.uniform(0.34, 0.95, n)
        y = np.where(rng.uniform(size=n) < p2 * 0.9, 2, 0)  # mas confianza, mas acierto
        P = np.column_stack([1 - p2, np.zeros(n), p2])
        t, regla = ac.elegir_umbral_reforzar(y, P)
        self.assertEqual(regla, "precision")
        pred = ac.decidir(P, t=t)
        self.assertGreaterEqual(((y == 2) & (pred == 2)).sum() / (pred == 2).sum(), 0.6)
        anterior = round(t - 0.01, 2)
        if anterior >= 0.34:
            pa = ac.decidir(P, t=anterior)
            prec = ((y == 2) & (pa == 2)).sum() / (pa == 2).sum()
            self.assertLess(prec, 0.6)

    def test_sin_precision_meta_usa_f05(self):
        n = 200
        P = np.column_stack([np.full(n, 0.2), np.zeros(n), np.full(n, 0.8)])
        y = np.array([2] * 40 + [0] * 160)  # precision 20 % para cualquier t <= 0,8
        t, regla = ac.elegir_umbral_reforzar(y, P)
        self.assertEqual(regla, "f05")
        self.assertEqual(t, 0.34)


class SBajaTest(unittest.TestCase):

    def test_mayor_s_con_mayoria_sin_venta(self):
        pv = np.r_[np.full(40, 0.1), np.full(40, 0.3), np.full(40, 0.5)]
        nov = np.r_[np.ones(40), np.ones(40) * 0, np.zeros(40)].astype(bool)
        nov[40:50] = True  # entre 0,1 y 0,3 vende la mayoria
        # p < 0,11..0,30: 40 filas, todas sin venta; p < 0,31..: 80 filas, 50 sin venta
        self.assertEqual(ac.elegir_s_baja(pv, nov, 0.6), 0.5)

    def test_sin_soporte_es_cero(self):
        self.assertEqual(ac.elegir_s_baja(np.full(10, 0.1), np.ones(10, bool), 0.6), 0.0)


class AccionTest(unittest.TestCase):

    def test_reglas(self):
        es_modelo = np.array([True] * 6 + [False])
        dec = np.array([2, 1, 0, 0, 1, 1, np.nan])
        pv = np.array([0.9, 0.9, 0.9, 0.55, 0.55, 0.3, 0.2])
        acc = ac.asignar_accion(es_modelo, dec, pv, s_v=0.6, s_baja=0.5)
        self.assertEqual(acc.tolist(), ["comprar_mas", "mantener", "mantener", "no_reponer",
                                        "revisar", "no_reponer", "por_promedio"])
        self.assertEqual(ac.nombre("no_reponer"), "No reponer este mes")

    def test_monto_por_accion(self):
        acc = np.array(["comprar_mas", "mantener", "no_reponer", "revisar",
                        "por_promedio", "por_promedio", "por_promedio", "mantener"], object)
        pv = np.array([0.9, 0.8, 0.3, 0.55, 0.2, 0.55, 0.9, 0.9])
        a = np.full(8, 100.0)
        tiene = np.array([True] * 7 + [False])
        ev, mostrado, rango = ac.monto_por_accion(acc, pv, a, tiene, s_v=0.6, s_baja=0.5)
        np.testing.assert_allclose(ev[:7], pv[:7] * 100)
        np.testing.assert_allclose(mostrado[:7], [90, 80, 0, 55, 0, 55, 90])
        self.assertEqual(rango.tolist(), [False, False, False, True, False, True, False, False])
        self.assertTrue(np.isnan(ev[7]) and np.isnan(mostrado[7]))

    def test_stock_por_accion(self):
        acc = np.array(["comprar_mas", "mantener", "no_reponer", "revisar", "no_reponer",
                        "por_promedio"], object)
        stock = np.array([10.0, 20, 30, 40, np.nan, 50])
        pred = np.array([5.0, 6, 7, 8, np.nan, 9])
        np.testing.assert_allclose(ac.stock_por_accion(acc, stock, pred),
                                   [10, 20, 0, 8, np.nan, 50])


class MontoSiSeVendeTest(unittest.TestCase):

    def test_estimadores_solo_con_meses_con_venta(self):
        df = pd.DataFrame({
            "id": ["a"] * 5,
            "fecha": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "total": [10.0, 0.0, 20.0, 30.0, 0.0],
        })
        est = ac.estimadores_si_vende(df, "id", "total")
        np.testing.assert_allclose(est["A3"].to_numpy(), [10, 10, 15, 20, 20])
        np.testing.assert_allclose(est["M6"].to_numpy(), [10, 10, 15, 20, 20])

    def test_elige_el_menor_sesgo(self):
        clave, crit = ac.elegir_monto_si_vende(
            [0.5, 0.5], [10.0, 10.0], {"A3": [40.0, 40.0], "A6": [20.0, 20.0]})
        self.assertEqual(clave, "A6")
        self.assertAlmostEqual(crit["A6"][0], 0.0)


class ReglaStockTest(unittest.TestCase):

    def test_adopta_si_cobertura_cercana_y_menos_stock(self):
        real = np.array([0.0, 0, 10, 10])
        nivel = np.array([5.0, 5, 12, 12])
        regla = np.array([0.0, 0, 12, 12])
        r = ac.comparar_regla_stock(real, nivel, regla)
        self.assertTrue(r["adoptada"])
        self.assertEqual(r["cobertura_regla"], r["cobertura_nivel"])
        self.assertLess(r["stock_real_regla"], r["stock_real_nivel"])

    def test_rechaza_si_pierde_mas_de_tres_puntos(self):
        real = np.array([1.0, 0, 10, 10])
        nivel = np.array([5.0, 5, 12, 12])
        regla = np.array([0.0, 0, 12, 12])
        self.assertFalse(ac.comparar_regla_stock(real, nivel, regla)["adoptada"])


class ContradiccionesTest(unittest.TestCase):

    def test_cuenta_cada_combinacion(self):
        res = pd.DataFrame({
            "accion": ["comprar_mas", "no_reponer", "no_reponer", "mantener", "comprar_mas"],
            "prob_venta": [0.5, 0.2, 0.2, 0.9, 0.9],
            "monto_mostrado": [100.0, 5.0, 0.0, 10.0, 50.0],
            "stock_sugerido": [100.0, 0.0, 3.0, 0.0, 80.0],
            "base": [10.0, 1, 1, 1, 60.0],
        })
        self.assertEqual(ac.contradicciones(res, 0.6), {
            "comprar_sin_venta": 1, "no_reponer_con_monto": 1, "no_reponer_con_stock": 1,
            "mantener_sin_stock": 1, "comprar_bajo_base": 1})


if __name__ == "__main__":
    unittest.main()
