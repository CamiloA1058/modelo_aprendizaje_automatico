"""Estado de TODOS los productos de los datos (no solo los que reciben
predicción): cinco estados disjuntos cuya suma es el total de productos.

- Con recomendación y monto: decisión del modelo y monto estimado.
- Solo recomendación: decisión del modelo sin monto (fuera de los top).
- Monto por promedio: monto por la media de 3 meses, sin decisión.
- Venta ocasional: menos de min_months meses con ventas.
- Sin ventas recientes: min_months o más meses con ventas, pero sin una venta
  reciente que lo deje en los estados anteriores.
"""
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from ventas_forecast import estado_productos as ep  # noqa: E402
from ventas_forecast import grupos as gr  # noqa: E402
from ventas_forecast.model import (  # noqa: E402
    ETIQUETA_SIN_RECOMENDACION, SalesForecastModel,
)
from ventas_forecast.paths import DEFAULT_DATASET  # noqa: E402

from tests._helpers import synthetic_history  # noqa: E402

ULTIMO = pd.Timestamp("2025-12-01")
MESES = pd.date_range("2024-01-01", ULTIMO, freq="MS")


def _historia(ventas):
    """ventas: {codigo: {indice de mes (0-23): monto}}. Calendario completo
    desde la primera venta de cada producto hasta ULTIMO, como el modelo."""
    filas = []
    for cod, por_mes in ventas.items():
        primero = min(por_mes)
        for i in range(primero, len(MESES)):
            filas.append({"CODIGO": cod, "DESCRIPCION": f"DESC {cod}",
                          "fecha": MESES[i],
                          "TOTAL_VENDIDO": float(por_mes.get(i, 0.0))})
    return pd.DataFrame(filas)


def _todos(monto, desde=0, hasta=23):
    return {i: monto for i in range(desde, hasta + 1)}


VENTAS = {
    "A": _todos(1000.0),                                 # recomendación y monto
    "B": _todos(200.0),                                  # solo recomendación
    "C": {**_todos(300.0, 0, 21)},                       # promedio (0 en 22 y 23)
    "D": {21: 100.0, 22: 100.0, 23: 100.0},              # ocasional, vendió el último mes
    "E": _todos(500.0, 8, 17),                           # sin ventas recientes (6 meses sin)
    "F": _todos(50.0, 0, 5),                             # exactamente 6 meses: sin ventas recientes
    "G": _todos(70.0, 0, 4),                             # 5 meses: ocasional
}

RESULTADOS = pd.DataFrame({
    "CODIGO": ["A", "B", "C"],
    "DESCRIPCION": ["DESC A", "DESC B", "DESC C"],
    "TOTAL_VENDIDO": [1000.0, 200.0, 0.0],
    "ventas_predichas": [1100.0, np.nan, 300.0],
    "decision": ["Reforzar stock", "Reducir stock", ETIQUETA_SIN_RECOMENDACION],
    "origen_estimacion": ["modelo", "modelo", "promedio"],
    "prob_venta": [0.9, 0.4, 0.6],
    "se_vendera": ["Sí", "No", "Sí"],
    "stock_sugerido": [1500.0, np.nan, 450.0],
    "stock_sugerido_unidades": [3.0, np.nan, 2.0],
})


def _clasificar(min_months=6, resultados=RESULTADOS, mapa=None, ventas=VENTAS):
    return ep.clasificar(
        _historia(ventas), resultados, id_col="CODIGO", desc_col="DESCRIPCION",
        total_col="TOTAL_VENDIDO", ultimo_mes=ULTIMO, min_months=min_months,
        mapa_grupos=mapa)


class CatalogoEstadosTest(unittest.TestCase):
    def test_cinco_estados_en_orden_con_nombre_y_explicacion(self):
        cat = ep.catalogo(6)
        self.assertEqual([e["clave"] for e in cat], list(ep.ESTADOS))
        self.assertEqual(
            [e["nombre"] for e in cat],
            ["Con recomendación y monto", "Solo recomendación", "Monto por promedio",
             "Venta ocasional", "Sin ventas recientes"])
        for e in cat:
            self.assertTrue(e["explicacion"].strip())

    def test_explicacion_de_venta_ocasional_usa_min_months(self):
        self.assertIn("menos de 6 meses", ep.explicacion("venta_ocasional", 6))
        self.assertIn("menos de 4 meses", ep.explicacion("venta_ocasional", 4))


class ClasificarTest(unittest.TestCase):
    def setUp(self):
        self.t = _clasificar()
        self.por = self.t.set_index("CODIGO")

    def test_un_estado_por_producto(self):
        self.assertEqual(self.por["estado"].to_dict(), {
            "A": "recomendacion_monto", "B": "solo_recomendacion",
            "C": "monto_promedio", "D": "venta_ocasional",
            "E": "sin_ventas_recientes", "F": "sin_ventas_recientes",
            "G": "venta_ocasional",
        })

    def test_particion_disjunta_que_suma_todos_los_productos(self):
        self.assertEqual(len(self.t), 7)
        self.assertFalse(self.t["CODIGO"].duplicated().any())
        self.assertTrue(set(self.t["estado"]) <= set(ep.ESTADOS))
        self.assertEqual(int(self.t["estado"].value_counts().sum()), 7)

    def test_nombre_del_estado_sale_del_catalogo(self):
        for _, f in self.t.iterrows():
            self.assertEqual(f["estado_nombre"], ep.ESTADOS[f["estado"]]["nombre"])

    def test_vendio_el_ultimo_mes_con_pocos_meses_es_ocasional(self):
        self.assertEqual(self.por.loc["D", "estado"], "venta_ocasional")
        self.assertEqual(self.por.loc["D", "meses_sin_venta"], 0)

    def test_exactamente_min_months_no_es_ocasional(self):
        self.assertEqual(self.por.loc["F", "meses_con_venta"], 6)
        self.assertEqual(self.por.loc["F", "estado"], "sin_ventas_recientes")
        self.assertEqual(self.por.loc["G", "meses_con_venta"], 5)
        self.assertEqual(self.por.loc["G", "estado"], "venta_ocasional")

    def test_cambiar_min_months_cambia_la_clasificacion(self):
        t = _clasificar(min_months=5).set_index("CODIGO")
        self.assertEqual(t.loc["G", "estado"], "sin_ventas_recientes")
        t = _clasificar(min_months=7).set_index("CODIGO")
        self.assertEqual(t.loc["F", "estado"], "venta_ocasional")

    def test_datos_de_apoyo(self):
        a, e = self.por.loc["A"], self.por.loc["E"]
        self.assertEqual(a["DESCRIPCION"], "DESC A")
        self.assertEqual(a["meses_con_venta"], 24)
        self.assertEqual(a["ultimo_mes_venta"], "2025-12")
        self.assertEqual(a["meses_sin_venta"], 0)
        self.assertAlmostEqual(a["ventas_12m"], 12000.0)
        self.assertAlmostEqual(a["promedio_mensual"], 1000.0)
        # E vendió de 2024-09 a 2025-06: 6 de esos meses caen en los últimos 12.
        self.assertEqual(e["ultimo_mes_venta"], "2025-06")
        self.assertEqual(e["meses_sin_venta"], 6)
        self.assertAlmostEqual(e["ventas_12m"], 3000.0)
        self.assertAlmostEqual(e["promedio_mensual"], 500.0)

    def test_columnas_de_prediccion_solo_para_productos_con_prediccion(self):
        a, b, d = self.por.loc["A"], self.por.loc["B"], self.por.loc["D"]
        self.assertEqual(a["decision"], "Reforzar stock")
        self.assertAlmostEqual(a["ventas_predichas"], 1100.0)
        self.assertAlmostEqual(a["prob_venta"], 0.9)
        self.assertEqual(a["se_vendera"], "Sí")
        self.assertAlmostEqual(a["stock_sugerido"], 1500.0)
        self.assertAlmostEqual(a["stock_sugerido_unidades"], 3.0)
        self.assertTrue(np.isnan(b["ventas_predichas"]))
        self.assertTrue(pd.isna(d["decision"]))
        self.assertTrue(np.isnan(d["ventas_predichas"]))
        self.assertTrue(np.isnan(d["prob_venta"]))

    def test_sin_columnas_opcionales_en_resultados(self):
        r = RESULTADOS.drop(columns=["prob_venta", "se_vendera", "stock_sugerido",
                                     "stock_sugerido_unidades"])
        t = _clasificar(resultados=r).set_index("CODIGO")
        self.assertEqual(t.loc["A", "estado"], "recomendacion_monto")
        self.assertTrue(np.isnan(t.loc["A", "prob_venta"]))

    def test_resultados_duplicados_no_duplican_productos(self):
        r = pd.concat([RESULTADOS, RESULTADOS.iloc[[0]]])
        t = _clasificar(resultados=r)
        self.assertEqual(len(t), 7)

    def test_sin_resultados_todos_quedan_fuera_de_prediccion(self):
        t = _clasificar(resultados=None).set_index("CODIGO")
        self.assertEqual(t.loc["A", "estado"], "sin_ventas_recientes")
        self.assertEqual(t.loc["D", "estado"], "venta_ocasional")

    def test_grupo_con_catalogo(self):
        mapa = gr.cargar_mapa(io.StringIO("codigo;id_grupo;grupo\nA;1;FRENOS\nB;2;\n"))
        t = _clasificar(mapa=mapa).set_index("CODIGO")
        self.assertEqual(t.loc["A", "grupo"], "FRENOS")
        self.assertEqual(t.loc["B", "grupo"], gr.ETIQUETA_SIN_GRUPO)
        self.assertEqual(t.loc["E", "grupo"], gr.ETIQUETA_SIN_GRUPO)

    def test_sin_catalogo_no_hay_columna_grupo(self):
        self.assertNotIn("grupo", self.t.columns)

    def test_resumen_por_estado(self):
        r = ep.resumen(self.t)
        self.assertEqual([f["clave"] for f in r], list(ep.ESTADOS))
        n = {f["clave"]: f["productos"] for f in r}
        self.assertEqual(n, {"recomendacion_monto": 1, "solo_recomendacion": 1,
                             "monto_promedio": 1, "venta_ocasional": 2,
                             "sin_ventas_recientes": 2})
        self.assertAlmostEqual(sum(f["pct_productos"] for f in r), 1.0)
        self.assertAlmostEqual(sum(f["pct_ventas_12m"] for f in r), 1.0)
        rec = r[0]
        self.assertAlmostEqual(rec["ventas_12m"], 12000.0)
        self.assertAlmostEqual(rec["pct_productos"], 1 / 7)


# ── Con el modelo entrenado (datos sintéticos) ──────────────────────────────
N_MONTHS = 12
# 0: vendió todo el año, el de menos ventas -> fuera de los top: solo recomendación
# 1: 5 meses con ventas (7-11), vendió el último mes -> venta ocasional
# 2: exactamente 6 meses con ventas (6-11) -> con recomendación y monto
# 3: sin ventas el último mes ni en el 8 -> monto por promedio
# 4: sin ventas en los 3 últimos meses -> sin ventas recientes
# 5: solo 2 meses con ventas (9 y 10) -> venta ocasional
# 6, 7: vendieron todo el año -> con recomendación y monto
DROP = {1: range(0, 7), 2: range(0, 6), 3: [8, 11], 4: [9, 10, 11],
        5: [*range(0, 9), 11]}
ESPERADO = {0: "solo_recomendacion", 1: "venta_ocasional",
            2: "recomendacion_monto", 3: "monto_promedio",
            4: "sin_ventas_recientes", 5: "venta_ocasional",
            6: "recomendacion_monto", 7: "recomendacion_monto"}


def _entrenado(tmp, min_months=6):
    m = SalesForecastModel(filepath="unused.csv", min_months=min_months,
                           output_path=Path(tmp) / "pred.csv", top_n_products=5)
    m.df = synthetic_history(n_products=8, n_months=N_MONTHS, drop_months=DROP)
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
    m.clasificar_estado_productos()
    return m


class ClasificarConModeloTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.m = _entrenado(cls._tmp.name)
        cls.t = cls.m.estado_productos

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_fuera_de_los_top_solo_el_producto_0(self):
        # Premisa del escenario: el producto 0 queda fuera de los top.
        self.assertNotIn(0, set(self.m._top_prods))
        self.assertTrue({2, 3, 6, 7} <= set(self.m._top_prods))

    def test_estado_de_cada_producto(self):
        self.assertEqual(self.t.set_index(self.m._c("id"))["estado"].to_dict(), ESPERADO)

    def test_coincide_con_los_resultados_del_modelo(self):
        c = self.m._c
        con_pred = self.t[self.t["estado"].isin(ep.ESTADOS_CON_PREDICCION)]
        self.assertEqual(set(con_pred[c("id")]), set(self.m.results[c("id")]))

    def test_particion(self):
        c = self.m._c
        self.assertEqual(len(self.t), self.m.df[c("id")].nunique())
        self.assertFalse(self.t[c("id")].duplicated().any())

    def test_no_cambia_los_resultados(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = _entrenado(tmp)
            antes = m.results.copy()
            m.clasificar_estado_productos()
            pd.testing.assert_frame_equal(m.results, antes)

    def test_min_months_menor_saca_al_producto_1_de_ocasional(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = _entrenado(tmp, min_months=5)
            t = m.estado_productos.set_index(m._c("id"))
            self.assertNotEqual(t.loc[1, "estado"], "venta_ocasional")
            self.assertEqual(t.loc[5, "estado"], "venta_ocasional")


class RunLlamaClasificacionTest(unittest.TestCase):
    def test_run_clasifica_despues_de_la_estimacion_por_promedio(self):
        orden = []
        pasos = ["load_and_clean", "build_features", "cluster", "prepare_split",
                 "train_classifier", "train_regressor", "predict_next_month",
                 "estimar_por_promedio", "agregar_probabilidad_venta",
                 "agregar_stock_sugerido", "agregar_resumen_grupo",
                 "clasificar_estado_productos", "plot"]
        m = SalesForecastModel(filepath="unused.csv")
        with mock.patch.multiple(
                SalesForecastModel,
                **{p: (lambda p: (lambda self, *a, **k: orden.append(p)))(p)
                   for p in pasos}):
            m.run()
        self.assertIn("clasificar_estado_productos", orden)
        self.assertGreater(orden.index("clasificar_estado_productos"),
                           orden.index("estimar_por_promedio"))
        self.assertGreater(orden.index("clasificar_estado_productos"),
                           orden.index("agregar_stock_sugerido"))


# ── Datos reales (opcional: VENTAS_PRUEBA_DATOS_REALES=1, ~1-2 min) ──────────
@unittest.skipUnless(os.environ.get("VENTAS_PRUEBA_DATOS_REALES") == "1"
                     and DEFAULT_DATASET.exists(),
                     "Prueba con el CSV real: defina VENTAS_PRUEBA_DATOS_REALES=1")
class DatosRealesTest(unittest.TestCase):
    def test_conteos_de_abril_2026(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = SalesForecastModel(filepath=DEFAULT_DATASET,
                                   output_path=Path(tmp) / "pred.csv")
            with mock.patch.object(SalesForecastModel, "plot"):
                m.run()
        t = m.estado_productos
        conteo = t["estado"].value_counts().to_dict()
        self.assertEqual(conteo, {"recomendacion_monto": 645, "solo_recomendacion": 52,
                                  "monto_promedio": 348, "venta_ocasional": 2260,
                                  "sin_ventas_recientes": 279})
        self.assertEqual(len(t), 3584)
        self.assertFalse(t[m._c("id")].duplicated().any())


if __name__ == "__main__":
    unittest.main()
