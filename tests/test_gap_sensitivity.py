"""Ayudantes puros del experimento de sensibilidad al tramo de baja facturacion
y el peso de muestras opcional del modelo (apagado por defecto)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ventas_forecast import gap_sensitivity as gs  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402
from tests._helpers import synthetic_history  # noqa: E402

GAP = ("2024-02", "2024-04")      # 3 meses
REF = ("2024-05", "2024-06")      # 2 meses


def _long(rows):
    return pd.DataFrame(rows, columns=["id", "month", "amount"])


class TestFactors(unittest.TestCase):
    def setUp(self):
        rows = []
        # producto A: 10/mes en el tramo, 40/mes fuera -> factor 4
        for m in ("2024-02", "2024-03", "2024-04"):
            rows.append(("A", m, 10.0))
        for m in ("2024-05", "2024-06"):
            rows.append(("A", m, 40.0))
        # producto B: 1/mes dentro, 100/mes fuera -> ratio 100, tope 5
        for m in ("2024-02", "2024-03", "2024-04"):
            rows.append(("B", m, 1.0))
        for m in ("2024-05", "2024-06"):
            rows.append(("B", m, 100.0))
        # producto C: casi sin ventas fuera -> debe usar el global
        for m in ("2024-02", "2024-03", "2024-04"):
            rows.append(("C", m, 10.0))
        rows.append(("C", "2024-06", 5.0))
        self.long = _long(rows)

    def test_factor_por_producto_y_tope(self):
        f, g, _ = gs.gap_scale_factors(self.long, GAP, REF, cap=5.0, min_active=2)
        self.assertAlmostEqual(f.loc["A", "factor"], 4.0)
        self.assertEqual(f.loc["A", "source"], "producto")
        self.assertAlmostEqual(f.loc["B", "factor"], 5.0)

    def test_pocos_meses_activos_usa_global(self):
        f, g, _ = gs.gap_scale_factors(self.long, GAP, REF, cap=5.0, min_active=2)
        self.assertEqual(f.loc["C", "source"], "global")
        self.assertAlmostEqual(f.loc["C", "factor"], g)

    def test_global_es_razon_de_totales_mensuales(self):
        _, g, _ = gs.gap_scale_factors(self.long, GAP, REF, cap=50.0, min_active=2)
        dentro = (10 + 1 + 10) * 3 / 3          # suma del tramo / 3 meses
        fuera = (40 * 2 + 100 * 2 + 5) / 2      # suma fuera / 2 meses
        self.assertAlmostEqual(g, fuera / dentro)

    def test_nunca_menor_que_uno(self):
        rows = [("A", "2024-02", 50.0), ("A", "2024-03", 50.0),
                ("A", "2024-05", 10.0), ("A", "2024-06", 10.0)]
        f, g, _ = gs.gap_scale_factors(_long(rows), GAP, REF, min_active=2)
        self.assertEqual(f.loc["A", "factor"], 1.0)
        self.assertGreaterEqual(g, 1.0)

    def test_resumen(self):
        _, _, s = gs.gap_scale_factors(self.long, GAP, REF, cap=5.0, min_active=2)
        self.assertEqual(s["n_gap_months"], 3)
        self.assertEqual(s["n_ref_months"], 2)
        self.assertEqual(s["n_products"], 3)


class TestScaleRows(unittest.TestCase):
    def test_escala_solo_filas_del_tramo_y_columnas_indicadas(self):
        df = pd.DataFrame({
            "CODIGO": ["A", "A", "Z"],
            "FECHA": pd.to_datetime(["2024-03-05", "2024-05-05", "2024-03-07"]),
            "VENTAS": [2.0, 2.0, 1.0],
            "TOTAL_VENDIDO": [100.0, 100.0, 10.0],
            "PRECIO_PROMEDIO": [50.0, 50.0, 10.0],
        })
        f = pd.DataFrame({"factor": [3.0]}, index=["A"])
        out, mask = gs.scale_gap_rows(
            df, f, global_factor=2.0, id_col="CODIGO", date_col="FECHA",
            gap=GAP, scale_cols=["VENTAS", "TOTAL_VENDIDO"])
        self.assertEqual(mask.tolist(), [True, False, True])
        self.assertEqual(out["TOTAL_VENDIDO"].tolist(), [300.0, 100.0, 20.0])
        self.assertEqual(out["VENTAS"].tolist(), [6.0, 2.0, 2.0])
        self.assertEqual(out["PRECIO_PROMEDIO"].tolist(), [50.0, 50.0, 10.0])
        # el original no se modifica
        self.assertEqual(df["TOTAL_VENDIDO"].tolist(), [100.0, 100.0, 10.0])

    def test_no_crea_filas(self):
        df = pd.DataFrame({
            "CODIGO": ["A"], "FECHA": pd.to_datetime(["2024-03-05"]),
            "TOTAL_VENDIDO": [1.0],
        })
        out, _ = gs.scale_gap_rows(
            df, pd.DataFrame({"factor": [2.0]}, index=["A"]), 2.0, "CODIGO", "FECHA",
            GAP, ["TOTAL_VENDIDO"])
        self.assertEqual(len(out), 1)


class TestSampleWeights(unittest.TestCase):
    def test_peso_para_filas_cuyo_mes_actual_o_siguiente_cae_en_el_tramo(self):
        fechas = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01",
                                 "2024-05-01", "2024-06-01"])
        w = gs.gap_sample_weights(fechas, GAP, 0.3)
        # 2024-01 (su objetivo es feb), feb y abr (dentro) pesan 0,3; mayo no
        # (su objetivo, junio, esta fuera); junio tampoco
        self.assertEqual(w.tolist(), [0.3, 0.3, 0.3, 1.0, 1.0])

    def test_incluye_la_fila_cuyo_objetivo_es_el_primer_mes_fuera(self):
        # fila de abril: mes actual dentro del tramo, objetivo mayo (recuperacion)
        w = gs.gap_sample_weights(pd.to_datetime(["2024-04-01"]), GAP, 0.0)
        self.assertEqual(w.tolist(), [0.0])


class TestHelpersExperimento(unittest.TestCase):
    def test_format_es_roundtrip_con_el_formato_del_csv(self):
        out = gs.format_es(pd.Series([1234.5, 0.0, 1234567.891, 2.0]))
        self.assertEqual(out.tolist(), ["1.234,50", "0,00", "1.234.567,89", "2,00"])
        back = (out.str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False).astype(float))
        self.assertAlmostEqual(back.iloc[2], 1234567.89)

    def test_monthly_long_suma_por_producto_y_mes(self):
        df = pd.DataFrame({
            "CODIGO": ["A", "A", "B"],
            "FECHA": pd.to_datetime(["2024-03-05", "2024-03-20", "2024-03-07"]),
            "TOTAL_VENDIDO": [1.0, 2.0, 5.0],
        })
        long = gs.monthly_long(df, "CODIGO", "FECHA", "TOTAL_VENDIDO")
        d = {(r.id, r.month): r.amount for r in long.itertuples()}
        self.assertEqual(d, {("A", "2024-03"): 3.0, ("B", "2024-03"): 5.0})

    def test_class_share(self):
        self.assertAlmostEqual(gs.class_share([2, 2, 0, 1], 2), 0.5)
        self.assertTrue(np.isnan(gs.class_share([], 2)))

    def test_label_counts_por_mes_de_origen(self):
        df = pd.DataFrame({
            "fecha": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01",
                                     "2025-04-01"]),
            "target_class": [0, 0, 1, 2],
        })
        c = gs.label_counts(df, ["2024-01", "2025-04"])
        self.assertEqual(c["2024-01"], [2, 1, 0])
        self.assertEqual(c["2025-04"], [0, 0, 1])


class TestModelSampleWeightHook(unittest.TestCase):
    def _model(self, tmp, fn=None):
        m = SalesForecastModel(filepath="unused.csv",
                               output_path=Path(tmp) / "p.csv")
        if fn is not None:
            m.sample_weight_fn = fn
        m.df = synthetic_history(n_products=6, n_months=24)
        m._ultimo_mes = m.df["fecha"].max()
        m._mes_pred = m._ultimo_mes + pd.DateOffset(months=1)
        m.build_features()
        m.cluster()
        m.prepare_split()
        return m

    def test_por_defecto_apagado(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(self._model(tmp).sample_weight_fn)

    def test_pesos_llegan_al_clasificador_y_al_regresor(self):
        seen = []

        def fn(fechas):
            seen.append(len(fechas))
            return np.full(len(fechas), 0.5)

        with tempfile.TemporaryDirectory() as tmp:
            m = self._model(tmp, fn)
            m.train_classifier()
            m.train_regressor()
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], len(m._train_idx))

    def test_sin_pesos_no_se_pasa_sample_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._model(tmp)
            self.assertEqual(m._fit_kwargs(m._df_train.iloc[m._train_idx]), {})
            from unittest import mock
            with mock.patch("ventas_forecast.model.RandomForestClassifier.fit",
                            autospec=True, return_value=None) as f:
                try:
                    m.train_classifier()
                except Exception:
                    pass  # predict falla sin ajuste real; solo importa la llamada
            self.assertNotIn("sample_weight", f.call_args.kwargs)

    def test_sin_pesos_es_determinista_y_reproduce_el_ajuste_previo(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._model(tmp)
            a.train_classifier()
            a.train_regressor()
            b = self._model(tmp)
            b.train_classifier()
            b.train_regressor()
        self.assertEqual(a.metrics, b.metrics)
        self.assertEqual(a.reg_metrics["mae"], b.reg_metrics["mae"])


if __name__ == "__main__":
    unittest.main()
