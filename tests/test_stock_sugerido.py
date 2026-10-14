"""Ayudantes puros del stock sugerido (nivel de servicio): residuos en log1p
por cubeta de probabilidad de venta, cuantiles, cota superior y unidades."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import stock_sugerido as ss  # noqa: E402


class ResiduosTest(unittest.TestCase):
    def test_residuo_es_diferencia_de_log1p(self):
        r = ss.residuos_log([100.0, 0.0], [50.0, 10.0])
        np.testing.assert_allclose(
            r, [np.log1p(100) - np.log1p(50), 0.0 - np.log1p(10)])


class CalibrarTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        # dos regimenes bien separados: prob baja -> residuos negativos
        self.p = np.r_[np.full(200, 0.2), np.full(200, 0.9)]
        self.r = np.r_[rng.normal(-1.0, 0.3, 200), rng.normal(0.5, 0.3, 200)]

    def test_cuantil_por_cubeta(self):
        cal = ss.calibrar(self.p, self.r)
        self.assertEqual(cal["edges"], [0.0, 0.8, 1.0])
        q = cal["q"][0.7]
        # una cubeta para prob < 0,4 y otra para prob >= 0,8 (las intermedias
        # estan vacias y se fusionan)
        self.assertEqual(len(q), len(cal["n"]))
        self.assertAlmostEqual(q[0], float(np.quantile(self.r[:200], 0.7)), places=9)
        self.assertAlmostEqual(q[-1], float(np.quantile(self.r[200:], 0.7)), places=9)

    def test_edges_por_defecto(self):
        self.assertEqual(list(ss.EDGES), [0.0, 0.4, 0.6, 0.8, 1.0])

    def test_cuantil_global_con_todos_los_residuos(self):
        cal = ss.calibrar(self.p, self.r)
        self.assertAlmostEqual(cal["global"][0.7], float(np.quantile(self.r, 0.7)))
        self.assertEqual(cal["n_total"], 400)

    def test_niveles_por_defecto(self):
        cal = ss.calibrar(self.p, self.r)
        self.assertEqual(set(cal["q"]), {0.5, 0.7, 0.9})
        self.assertLess(cal["global"][0.5], cal["global"][0.7])
        self.assertLess(cal["global"][0.7], cal["global"][0.9])

    def test_cubetas_con_menos_de_50_filas_se_fusionan(self):
        p = np.r_[np.full(10, 0.1), np.full(100, 0.5), np.full(100, 0.9)]
        r = np.arange(len(p), dtype=float)
        cal = ss.calibrar(p, r)
        self.assertTrue(all(n >= 50 for n in cal["n"]))
        self.assertEqual(sum(cal["n"]), 210)
        self.assertLess(len(cal["n"]), 4)

    def test_sin_filas_suficientes_queda_una_cubeta(self):
        cal = ss.calibrar(np.full(30, 0.5), np.linspace(-1, 1, 30))
        self.assertEqual(cal["n"], [30])
        self.assertEqual(cal["edges"], [0.0, 1.0])
        self.assertAlmostEqual(cal["q"][0.7][0], cal["global"][0.7])

    def test_prob_nan_solo_cuenta_en_el_global(self):
        p = np.r_[np.full(100, 0.5), np.full(60, np.nan)]
        r = np.r_[np.zeros(100), np.full(60, 10.0)]
        cal = ss.calibrar(p, r)
        self.assertEqual(sum(cal["n"]), 100)
        self.assertEqual(cal["n_total"], 160)
        self.assertAlmostEqual(cal["q"][0.7][0], 0.0)
        self.assertGreater(cal["global"][0.7], 0.0)

    def test_sin_datos_devuelve_cero(self):
        cal = ss.calibrar(np.array([]), np.array([]))
        self.assertEqual(cal["global"][0.7], 0.0)
        self.assertEqual(cal["q"][0.7], [0.0])

    def test_cuantiles_para_asigna_por_cubeta_y_nan_usa_global(self):
        cal = ss.calibrar(self.p, self.r)
        q = ss.cuantiles_para(cal, [0.1, 0.95, np.nan], 0.7)
        self.assertAlmostEqual(q[0], cal["q"][0.7][0])
        self.assertAlmostEqual(q[1], cal["q"][0.7][-1])
        self.assertAlmostEqual(q[2], cal["global"][0.7])

    def test_prob_igual_a_uno_cae_en_la_ultima_cubeta(self):
        cal = ss.calibrar(self.p, self.r)
        q = ss.cuantiles_para(cal, [1.0], 0.7)
        self.assertAlmostEqual(q[0], cal["q"][0.7][-1])

    def test_cobertura_en_muestra_cercana_al_nivel(self):
        rng = np.random.default_rng(7)
        p = rng.uniform(0, 1, 5000)
        pred = rng.uniform(1e3, 1e5, 5000)
        real = np.expm1(np.log1p(pred) + rng.normal(0, 1, 5000) - 0.3 * p)
        r = ss.residuos_log(real, pred)
        cal = ss.calibrar(p, r)
        cota = ss.cota_superior(pred, ss.cuantiles_para(cal, p, 0.7))
        self.assertAlmostEqual(float((real <= cota).mean()), 0.70, delta=0.02)


class CotaTest(unittest.TestCase):
    def test_formula(self):
        # expm1(log1p(100) + ln 2) = 101 * 2 - 1
        self.assertAlmostEqual(float(ss.cota_superior(100.0, np.log(2))), 201.0)

    def test_q_cero_devuelve_la_prediccion(self):
        np.testing.assert_allclose(ss.cota_superior([0.0, 5.0, 1e6], 0.0),
                                   [0.0, 5.0, 1e6])

    def test_recorta_en_cero(self):
        self.assertEqual(float(ss.cota_superior(0.0, -5.0)), 0.0)
        self.assertEqual(float(ss.cota_superior(10.0, -50.0)), 0.0)

    def test_pred_nan_da_nan(self):
        self.assertTrue(np.isnan(ss.cota_superior(np.nan, 0.5)))

    def test_vectorizada(self):
        out = ss.cota_superior([100.0, 100.0], [np.log(2), 0.0])
        np.testing.assert_allclose(out, [201.0, 100.0])


class UnidadesTest(unittest.TestCase):
    def test_techo(self):
        np.testing.assert_array_equal(
            ss.unidades([1000.0, 900.0, 0.0], [300.0, 300.0, 300.0]), [4, 3, 0])

    def test_sin_precio_es_nan(self):
        out = ss.unidades([1000.0, 1000.0, 1000.0], [np.nan, 0.0, -1.0])
        self.assertTrue(np.isnan(out).all())

    def test_stock_nan_es_nan(self):
        self.assertTrue(np.isnan(ss.unidades([np.nan], [100.0])).all())

    def test_error_de_punto_flotante_no_sube_una_unidad(self):
        # 3 * 1,1 / 1,1 = 3,0000000000000004 en coma flotante
        self.assertEqual(float(ss.unidades([3 * 1.1], [1.1])[0]), 3.0)


class CoberturaTest(unittest.TestCase):
    def setUp(self):
        # real, stock: cubre, cubre (real 0), no cubre, agotado (stock 0, vendio)
        self.real = np.array([10.0, 0.0, 50.0, 5.0])
        self.stock = np.array([10.0, 0.0, 40.0, 0.0])

    def test_cobertura_incluye_los_que_no_vendieron(self):
        m = ss.cobertura(self.real, self.stock)
        self.assertEqual(m["n"], 4)
        self.assertAlmostEqual(m["cobertura"], 0.5)

    def test_cobertura_entre_los_que_vendieron(self):
        m = ss.cobertura(self.real, self.stock)
        self.assertEqual(m["n_vendieron"], 3)
        self.assertAlmostEqual(m["cobertura_vendieron"], 1 / 3)

    def test_razones_stock_real(self):
        m = ss.cobertura(self.real, self.stock)
        # solo los que vendieron: 10/10, 40/50, 0/5
        self.assertAlmostEqual(m["mediana_stock_real"], 0.8)
        self.assertAlmostEqual(m["media_stock_real"], (1.0 + 0.8 + 0.0) / 3)
        self.assertAlmostEqual(m["total_stock_real"], 50.0 / 65.0)

    def test_agotados(self):
        m = ss.cobertura(self.real, self.stock)
        self.assertAlmostEqual(m["agotados"], 0.25)

    def test_unidades(self):
        m = ss.cobertura(self.real, self.stock,
                         real_unid=[2.0, 0.0, np.nan, 1.0],
                         stock_unid=[2.0, 0.0, 3.0, 0.0])
        self.assertEqual(m["n_unid"], 3)
        self.assertAlmostEqual(m["cobertura_unid"], 2 / 3)

    def test_sin_filas(self):
        m = ss.cobertura([], [])
        self.assertEqual(m["n"], 0)
        self.assertTrue(np.isnan(m["cobertura"]))

    def test_filas_sin_stock_se_excluyen(self):
        m = ss.cobertura([1.0, 2.0], [np.nan, 5.0])
        self.assertEqual(m["n"], 1)
        self.assertAlmostEqual(m["cobertura"], 1.0)


class UnidadesRealesTest(unittest.TestCase):
    def test_suma_de_unidades_por_producto_y_mes(self):
        import io
        from ventas_forecast import case_study as cs
        csv = (
            "CODIGO;DESCRIPCION;FECHA;ANIO;MES;DIA;DIA_SEMANA;VENTAS;TOTAL_VENDIDO;PRECIO_PROMEDIO;FRECUENCIA;HUBO_PROMOCION;TEMPORADA\n"
            "1;A;01/04/2026;2026;4;1;3;30,00;54.000,00;1.800,00;2;0;Baja\n"
            "1;A;02/04/2026;2026;4;2;4;2,50;5.000,00;2.000,00;1;0;Baja\n"
            "1;A;03/05/2026;2026;5;3;1;9,00;9.000,00;1.000,00;1;0;Baja\n"
            "2;B;05/04/2026;2026;4;5;6;4,00;8.000,00;2.000,00;1;0;Baja\n")
        u = cs.monthly_units(io.StringIO(csv), "2026-04")
        self.assertAlmostEqual(u["1"], 32.5)
        self.assertAlmostEqual(u["2"], 4.0)
        self.assertEqual(len(u), 2)


if __name__ == "__main__":
    unittest.main()
