import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import experimento_grupos as eg  # noqa: E402


def _mes(real, previo, pred):
    return pd.DataFrame({"real": real, "previo": previo, "G0": pred},
                        index=[f"g{i}" for i in range(len(real))])


class TopSinMirarElMesEvaluado(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.previo = rng.uniform(1, 100, 30)
        self.real = rng.uniform(1, 100, 30)
        self.pred = rng.uniform(1, 100, 30)

    def test_seleccion_ignora_las_ventas_reales_del_mes(self):
        a = eg.seleccion_top(_mes(self.real, self.previo, self.pred))
        b = eg.seleccion_top(_mes(self.real[::-1].copy() * 7, self.previo, self.pred))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(int(a.sum()), 20)
        esperado = set(np.argsort(-self.previo, kind="stable")[:20])
        self.assertEqual(set(np.flatnonzero(a)), esperado)

    def test_tabla_metricas_top_usa_el_mes_previo(self):
        df = _mes(self.real, self.previo, self.pred)
        t = eg.tabla_metricas({"2026-04": df}, ["G0"], "universo")
        sel = np.argsort(-self.previo, kind="stable")[:20]
        esp = eg.gr.acierto_ponderado(self.real[sel], self.pred[sel])
        fila = t[t["mes"] == "2026-04"].iloc[0]
        self.assertAlmostEqual(fila["top"], esp)
        pool = t[t["mes"] == "agrupado"].iloc[0]
        self.assertAlmostEqual(pool["top"], esp)


if __name__ == "__main__":
    unittest.main()
