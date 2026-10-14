"""Ayudantes puros del experimento de mejoras de la prediccion en pesos."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ventas_forecast import improvements as im  # noqa: E402


class TestWape(unittest.TestCase):
    def test_valor(self):
        self.assertAlmostEqual(im.wape([10, 20, 30], [12, 18, 30]), 4 / 60)

    def test_real_cero_es_nan(self):
        self.assertTrue(np.isnan(im.wape([0, 0], [1, 2])))


class TestTopDown(unittest.TestCase):
    def test_suma_igual_al_total_y_proporcional(self):
        out = im.topdown_reconcile(1000.0, [1.0, 3.0])
        self.assertAlmostEqual(out.sum(), 1000.0)
        self.assertAlmostEqual(out[0], 250.0)
        self.assertAlmostEqual(out[1], 750.0)

    def test_suma_rf_cero_reparte_parejo(self):
        out = im.topdown_reconcile(90.0, [0.0, 0.0, 0.0])
        self.assertEqual(out.tolist(), [30.0, 30.0, 30.0])


class TestBlend(unittest.TestCase):
    def test_mezcla(self):
        out = im.blend(0.25, [100.0, 0.0], [0.0, 40.0])
        self.assertEqual(out.tolist(), [25.0, 30.0])

    def test_elige_w_que_minimiza_mae(self):
        real = np.array([10.0, 20.0, 30.0])
        rf = np.array([0.0, 0.0, 0.0])
        base = real.copy()
        self.assertEqual(im.choose_blend_weight(real, rf, base), 0.0)
        self.assertEqual(im.choose_blend_weight(real, base, rf), 1.0)

    def test_empate_toma_el_menor_w(self):
        real = np.array([10.0])
        self.assertEqual(im.choose_blend_weight(real, [10.0], [10.0]), 0.0)

    def test_criterio_sesgo_elige_w_que_iguala_la_suma(self):
        real = np.array([100.0, 100.0])
        rf = np.array([50.0, 50.0])      # suma 100
        base = np.array([150.0, 150.0])  # suma 300; real suma 200
        w = im.choose_blend_weight(real, rf, base, criterion="sesgo")
        self.assertEqual(w, 0.5)

    def test_criterio_desconocido_falla(self):
        with self.assertRaises(ValueError):
            im.choose_blend_weight([1.0], [1.0], [1.0], criterion="otro")

    def test_rejilla_por_defecto(self):
        self.assertEqual(len(im.BLEND_WEIGHTS), 11)
        self.assertEqual(im.BLEND_WEIGHTS[0], 0.0)
        self.assertEqual(im.BLEND_WEIGHTS[-1], 1.0)


class TestIntervals(unittest.TestCase):
    def test_cobertura_y_ancho(self):
        real = [5.0, 15.0, 25.0, 100.0]
        lo = [0.0, 10.0, 30.0, 90.0]
        hi = [10.0, 20.0, 40.0, 110.0]
        self.assertEqual(im.interval_coverage(real, lo, hi), 0.75)
        self.assertEqual(im.interval_width(lo, hi), (10 + 10 + 10 + 20) / 4)

    def test_extremos_incluidos(self):
        self.assertEqual(im.interval_coverage([10.0], [10.0], [10.0]), 1.0)

    def test_cuantil_conformal(self):
        res = np.arange(1, 10, dtype=float)  # 1..9, n=9
        # ceil(10*0,8)=8 -> octavo menor = 8
        self.assertEqual(im.conformal_quantile(res, 0.20), 8.0)

    def test_cuantil_conformal_con_pocos_datos_usa_el_maximo(self):
        self.assertEqual(im.conformal_quantile([1.0, 2.0], 0.05), 2.0)

    def test_cuantiles_por_arbol(self):
        arboles = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        lo, hi = im.tree_quantiles(arboles, 0.0, 1.0)
        self.assertEqual(lo.tolist(), [1.0, 10.0])
        self.assertEqual(hi.tolist(), [3.0, 30.0])


if __name__ == "__main__":
    unittest.main()
