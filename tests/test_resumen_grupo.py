"""Vista por grupo de producto (produccion, activada por defecto): suma de las
predicciones por producto de cada grupo para el mes predicho. No cambia ninguna
prediccion, decision ni total; con la opcion en False no hay resumen."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import grupos as gr  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402
from tests.test_probabilidad_venta import DROP, N_MONTHS  # noqa: E402

MAPA = gr.cargar_mapa(io.StringIO(
    "codigo;id_grupo;grupo\nA;1;FRENOS\nB;1;FRENOS\nC;2;ACEITES\nD;2;ACEITES\n"
    "E;3;\n"))


def _resultados():
    return pd.DataFrame({
        "CODIGO": ["A", "B", "C", "D", "E", "Z"],
        "TOTAL_VENDIDO": [100.0, 200.0, 50.0, 80.0, 10.0, 40.0],
        "ventas_predichas": [120.0, 180.0, 60.0, np.nan, 30.0, 20.0],
        "se_vendera": ["Sí", "No", "Sí", "Sí", "Sí", "No"],
        "origen_estimacion": ["modelo", "modelo", "modelo", "modelo",
                              "promedio", "promedio"],
    })


class ResumenGruposTest(unittest.TestCase):
    def setUp(self):
        self.t = gr.resumen_grupos(_resultados(), MAPA, "CODIGO",
                                   "TOTAL_VENDIDO").set_index("grupo")

    def test_columnas_y_orden_por_prediccion(self):
        t = gr.resumen_grupos(_resultados(), MAPA, "CODIGO", "TOTAL_VENDIDO")
        self.assertEqual(list(t.columns), [
            "grupo", "productos", "se_vendera", "prediccion", "mes_anterior",
            "variacion", "pct_total"])
        self.assertEqual(list(t["grupo"]), ["FRENOS", "ACEITES", "Sin grupo"])

    def test_suma_de_predicciones_y_productos_con_monto(self):
        self.assertEqual(self.t.loc["FRENOS", "prediccion"], 300.0)
        self.assertEqual(self.t.loc["FRENOS", "productos"], 2)
        # D no tiene monto: no cuenta ni en productos ni en ventas previas
        self.assertEqual(self.t.loc["ACEITES", "productos"], 1)
        self.assertEqual(self.t.loc["ACEITES", "prediccion"], 60.0)
        self.assertEqual(self.t.loc["ACEITES", "mes_anterior"], 50.0)

    def test_cuenta_los_que_se_venderan_entre_los_del_grupo(self):
        self.assertEqual(self.t.loc["FRENOS", "se_vendera"], 1)
        self.assertEqual(self.t.loc["ACEITES", "se_vendera"], 1)

    def test_variacion_y_porcentaje_del_total(self):
        self.assertAlmostEqual(self.t.loc["FRENOS", "variacion"], 0.0)
        self.assertAlmostEqual(self.t.loc["ACEITES", "variacion"], 60 / 50 - 1)
        self.assertAlmostEqual(self.t["pct_total"].sum(), 1.0)
        self.assertAlmostEqual(self.t.loc["FRENOS", "pct_total"], 300 / 410)

    def test_sin_grupo_y_codigo_desconocido(self):
        self.assertEqual(self.t.loc["Sin grupo", "productos"], 2)  # E y Z
        self.assertEqual(self.t.loc["Sin grupo", "prediccion"], 50.0)

    def test_total_igual_a_la_suma_de_los_productos(self):
        self.assertAlmostEqual(self.t["prediccion"].sum(),
                               _resultados()["ventas_predichas"].sum())

    def test_variacion_vacia_si_no_hubo_ventas_previas(self):
        r = _resultados()
        r.loc[r["CODIGO"].isin(["C"]), "TOTAL_VENDIDO"] = 0.0
        t = gr.resumen_grupos(r, MAPA, "CODIGO", "TOTAL_VENDIDO").set_index("grupo")
        self.assertTrue(np.isnan(t.loc["ACEITES", "variacion"]))

    def test_sin_se_vendera_la_columna_queda_en_cero(self):
        r = _resultados().drop(columns="se_vendera")
        t = gr.resumen_grupos(r, MAPA, "CODIGO", "TOTAL_VENDIDO")
        self.assertTrue((t["se_vendera"] == 0).all())

    def test_resultados_vacios(self):
        r = _resultados().iloc[0:0]
        t = gr.resumen_grupos(r, MAPA, "CODIGO", "TOTAL_VENDIDO")
        self.assertEqual(len(t), 0)
        self.assertIn("grupo", t.columns)


def _entrenado(tmp, resumen=None, grupos_path=None):
    kw = {}
    if resumen is not None:
        kw["resumen_por_grupo"] = resumen
    # sin catalogo (archivo inexistente) todo queda en "Sin grupo"
    kw["grupos_path"] = grupos_path or Path(tmp) / "no_existe.csv"
    m = SalesForecastModel(filepath="unused.csv", min_months=6,
                           output_path=Path(tmp) / "pred.csv", **kw)
    m.df = synthetic_history(n_products=6, n_months=N_MONTHS, drop_months=DROP)
    m._ultimo_mes = m.df["fecha"].max()
    m._mes_pred = m._ultimo_mes + pd.DateOffset(months=1)
    m.df = m._complete_calendar(m.df)
    m.build_features()
    m.cluster()
    m.prepare_split()
    m.train_classifier()
    m.train_regressor()
    m.predict_next_month()
    m.estimar_por_promedio()
    m.agregar_probabilidad_venta()
    m.agregar_resumen_grupo()
    return m


class ModeloTest(unittest.TestCase):
    def test_activado_por_defecto(self):
        self.assertTrue(SalesForecastModel(filepath="x.csv").resumen_por_grupo)

    def test_genera_el_resumen_y_suma_el_total_de_las_predicciones(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = _entrenado(tmp)
            self.assertIsNotNone(m.resumen_grupos)
            self.assertAlmostEqual(m.resumen_grupos["prediccion"].sum(),
                                   m.results["ventas_predichas"].sum())
            self.assertEqual(m.resumen_grupos["productos"].sum(),
                             int(m.results["ventas_predichas"].notna().sum()))
            # los codigos sinteticos no estan en el catalogo: todo "Sin grupo"
            self.assertEqual(list(m.resumen_grupos["grupo"]), ["Sin grupo"])

    def test_usa_el_catalogo_indicado(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _entrenado(tmp)
            ids = base.results[base._c("id")].astype(str).tolist()
            csv = Path(tmp) / "g.csv"
            filas = ["codigo;id_grupo;grupo"] + [
                f"{i};{k % 2};G{k % 2}" for k, i in enumerate(ids)]
            csv.write_text("\n".join(filas), encoding="utf-8")
            m = _entrenado(tmp, grupos_path=csv)
            self.assertEqual(set(m.resumen_grupos["grupo"]) - {"G0", "G1"}, set())
            self.assertGreater(len(m.resumen_grupos), 1)

    def test_apagado_no_hay_resumen_y_la_salida_es_identica(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            on = _entrenado(t1)
            off = _entrenado(t2, resumen=False)
            self.assertIsNone(off.resumen_grupos)
            pd.testing.assert_frame_equal(on.results, off.results)
            pd.testing.assert_frame_equal(pd.read_csv(on.output_path),
                                          pd.read_csv(off.output_path))
            self.assertEqual(list(pd.read_csv(on.output_path).columns),
                             list(on.results.columns))


if __name__ == "__main__":
    unittest.main()
