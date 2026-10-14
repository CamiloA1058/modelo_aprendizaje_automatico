"""Prediccion mensual por grupo de producto: mapa producto -> grupo, series
mensuales por grupo, variantes (media 3, modelo directo, mezclas,
reconciliacion) y metricas de acierto por grupo."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast import grupos as gr  # noqa: E402


def ventas_csv(filas):
    cab = ("CODIGO;DESCRIPCION;FECHA;ANIO;MES;DIA;DIA_SEMANA;VENTAS;TOTAL_VENDIDO;"
           "PRECIO_PROMEDIO;FRECUENCIA;HUBO_PROMOCION;TEMPORADA")
    lineas = [cab] + [
        f"{c};X;{f};2024;1;1;1;1,00;{t};1,00;{fr};0;Baja " for c, f, t, fr in filas]
    return io.StringIO("\n".join(lineas))


def panel_sintetico(meses=18, grupos=("A", "B"), base=(1000.0, 50.0)):
    idx = pd.date_range("2024-01-01", periods=meses, freq="MS")
    total = pd.DataFrame({g: [b * (1 + 0.1 * np.sin(i)) for i in range(meses)]
                          for g, b in zip(grupos, base)}, index=idx)
    activos = (total > 0).astype(float) * 5
    facturas = total / 100
    return {"total": total, "activos": activos, "facturas": facturas}


class MapaGruposTest(unittest.TestCase):
    def test_carga_y_rellena_sin_grupo(self):
        src = io.StringIO("codigo;id_grupo;grupo\n1;4;RETENES\n2;;\n")
        mapa = gr.cargar_mapa(src)
        self.assertEqual(mapa["1"], "RETENES")
        self.assertEqual(mapa["2"], gr.SIN_GRUPO)

    def test_asignar_codigo_desconocido_es_sin_grupo(self):
        mapa = pd.Series({"1": "RETENES"})
        out = gr.asignar_grupo(pd.Series(["1", "9"]), mapa)
        self.assertEqual(list(out), ["RETENES", gr.SIN_GRUPO])


class VentasMensualesTest(unittest.TestCase):
    def test_agrega_por_producto_y_mes(self):
        src = ventas_csv([("1", "05/01/2024", "1.000,00", 2),
                          ("1", "20/01/2024", "500,00", 1),
                          ("1", "03/02/2024", "200,00", 1)])
        df = gr.leer_ventas([src])
        ene = df[(df["codigo"] == "1") & (df["mes"] == pd.Timestamp("2024-01-01"))].iloc[0]
        self.assertEqual(ene["total"], 1500.0)
        self.assertEqual(ene["facturas"], 3)
        self.assertEqual(len(df), 2)


class PanelGrupoTest(unittest.TestCase):
    def setUp(self):
        self.mapa = pd.Series({"1": "A", "2": "A", "3": "B"})
        m = pd.Timestamp
        self.ventas = pd.DataFrame({
            "codigo": ["1", "2", "3", "1", "3"],
            "mes": [m("2024-01-01"), m("2024-01-01"), m("2024-01-01"),
                    m("2024-03-01"), m("2024-04-01")],
            "total": [100.0, 50.0, 10.0, 70.0, 99.0],
            "facturas": [1, 2, 1, 1, 1],
        })

    def test_estrictamente_antes_del_mes_y_calendario_completo(self):
        p = gr.panel_grupo(self.ventas, self.mapa, "2024-04")
        self.assertEqual(p["total"].index.max(), pd.Timestamp("2024-03-01"))
        self.assertEqual(len(p["total"]), 3)  # ene, feb (ceros), mar
        self.assertEqual(p["total"].loc["2024-02-01", "A"], 0.0)
        self.assertEqual(p["total"].loc["2024-01-01", "A"], 150.0)
        self.assertEqual(p["activos"].loc["2024-01-01", "A"], 2)
        self.assertEqual(p["facturas"].loc["2024-01-01", "A"], 3)

    def test_universo_restringe_los_productos(self):
        p = gr.panel_grupo(self.ventas, self.mapa, "2024-04", productos=["1"])
        self.assertEqual(p["total"].loc["2024-01-01", "A"], 100.0)
        self.assertEqual(p["total"].loc["2024-01-01", "B"], 0.0)


class MetricasTest(unittest.TestCase):
    def test_exactitud_por_grupo_recorta_y_omite_real_cero(self):
        e = gr.exactitud_grupo([100.0, 100.0, 0.0, 100.0], [90.0, 300.0, 5.0, 100.0])
        np.testing.assert_allclose(e, [0.9, 0.0, np.nan, 1.0])

    def test_acierto_ponderado_es_uno_menos_wape(self):
        self.assertAlmostEqual(gr.acierto_ponderado([100, 300], [50, 300]), 1 - 50 / 400)

    def test_resumen_metricas(self):
        real = np.array([1000.0, 500.0, 100.0, 0.0])
        pred = np.array([900.0, 400.0, 100.0, 50.0])
        m = gr.resumen(real, pred, top=2)
        self.assertAlmostEqual(m["ponderado"], 1 - (100 + 100 + 0 + 50) / 1600)
        self.assertAlmostEqual(m["media"], np.mean([0.9, 0.8, 1.0]))
        self.assertAlmostEqual(m["pct_70"], 1.0)  # los tres con real > 0
        self.assertAlmostEqual(m["top"], 1 - (100 + 100) / 1500)
        self.assertAlmostEqual(m["razon"], 1450 / 1600)
        self.assertEqual(m["n"], 3)


class FeaturesTest(unittest.TestCase):
    def test_fila_usa_solo_datos_hasta_t_y_objetivo_es_t_mas_uno(self):
        p = panel_sintetico(meses=14)
        X, y, meta = gr.filas_entrenamiento(p)
        # el ultimo mes conocido no tiene objetivo: la ultima fila es t = penultimo mes
        self.assertEqual(meta["mes_origen"].max(), p["total"].index[-2])
        fila = meta.index[(meta["grupo"] == "A") & (meta["mes_origen"] == p["total"].index[5])][0]
        self.assertAlmostEqual(y.loc[fila], np.log1p(p["total"]["A"].iloc[6]))
        self.assertAlmostEqual(X.loc[fila, "lag_1"], np.log1p(p["total"]["A"].iloc[5]))
        self.assertAlmostEqual(X.loc[fila, "lag_2"], np.log1p(p["total"]["A"].iloc[4]))
        self.assertAlmostEqual(
            X.loc[fila, "mean_3"], np.log1p(p["total"]["A"].iloc[3:6].mean()))

    def test_los_rezagos_inexistentes_se_rellenan(self):
        p = panel_sintetico(meses=14)
        X, _, _ = gr.filas_entrenamiento(p)
        self.assertFalse(X.isna().any().any())

    def test_la_fila_de_origen_describe_el_ultimo_mes(self):
        p = panel_sintetico(meses=14)
        X, grupos = gr.fila_origen(p)
        self.assertEqual(list(grupos), ["A", "B"])
        self.assertAlmostEqual(X.loc[0, "lag_1"], np.log1p(p["total"]["A"].iloc[-1]))
        mes_objetivo = p["total"].index[-1] + pd.offsets.MonthBegin(1)
        self.assertAlmostEqual(X.loc[0, "mes_sin"], np.sin(2 * np.pi * mes_objetivo.month / 12))

    def test_mismas_columnas_en_entrenamiento_y_origen(self):
        p = panel_sintetico(meses=14)
        X, _, _ = gr.filas_entrenamiento(p)
        Xo, _ = gr.fila_origen(p)
        self.assertEqual(list(X.columns), list(Xo.columns))


class VariantesTest(unittest.TestCase):
    def test_media_3(self):
        p = panel_sintetico(meses=6)
        m = gr.media_3(p)
        self.assertAlmostEqual(m["A"], p["total"]["A"].iloc[-3:].mean())

    def test_elegir_peso_minimiza_el_error(self):
        real = np.array([100.0, 200.0, 300.0])
        self.assertEqual(gr.elegir_peso(real, real, real * 0.5), 1.0)
        self.assertEqual(gr.elegir_peso(real, real * 0.5, real), 0.0)

    def test_mezcla(self):
        np.testing.assert_allclose(gr.mezcla(0.25, [100.0], [200.0]), [175.0])

    def test_peso_online_sin_historia_usa_el_defecto(self):
        self.assertEqual(gr.peso_online([], [], [], defecto=0.5), 0.5)

    def test_peso_online_con_historia(self):
        real = np.array([100.0, 200.0])
        self.assertEqual(gr.peso_online([real], [real], [real * 2]), 1.0)

    def test_predice_serie_estable(self):
        p = panel_sintetico(meses=30, base=(1000.0, 50.0))
        p["total"] = pd.DataFrame({"A": 1000.0, "B": 50.0}, index=p["total"].index)
        pred = gr.predecir_directo(p, estimador=gr.rf_grupo)
        self.assertAlmostEqual(pred["pred"]["A"], 1000.0, delta=60)
        self.assertAlmostEqual(pred["pred"]["B"], 50.0, delta=15)
        self.assertGreaterEqual(pred["peso"], 0.0)
        self.assertLessEqual(pred["peso"], 1.0)

    def test_reconciliar_conserva_el_total_del_grupo(self):
        pred = pd.Series([10.0, 30.0, 5.0])
        grupo = pd.Series(["A", "A", "B"])
        out = gr.reconciliar(pred, grupo, pd.Series({"A": 80.0, "B": 5.0}))
        np.testing.assert_allclose(out.to_numpy(), [20.0, 60.0, 5.0])

    def test_reconciliar_sin_prediccion_en_el_grupo_no_toca(self):
        pred = pd.Series([0.0, 0.0])
        grupo = pd.Series(["A", "A"])
        out = gr.reconciliar(pred, grupo, pd.Series({"A": 80.0}))
        np.testing.assert_allclose(out.to_numpy(), [0.0, 0.0])

    def test_reconciliar_grupo_sin_total_conserva_la_prediccion(self):
        out = gr.reconciliar(pd.Series([10.0]), pd.Series(["Z"]), pd.Series({"A": 80.0}))
        np.testing.assert_allclose(out.to_numpy(), [10.0])


class TablaGruposTest(unittest.TestCase):
    def test_tabla_para_el_mes_predicho(self):
        resultados = pd.DataFrame({
            "CODIGO": ["1", "2", "3"], "TOTAL_VENDIDO": [100.0, 50.0, 10.0],
            "ventas_predichas": [120.0, np.nan, 12.0],
            "origen_estimacion": ["modelo", "modelo", "promedio"],
        })
        mapa = pd.Series({"1": "A", "2": "A", "3": "B"})
        t = gr.tabla_grupos(resultados, mapa, "CODIGO", "TOTAL_VENDIDO",
                            pred_grupo=pd.Series({"A": 130.0, "B": 11.0}))
        a = t[t["Grupo"] == "A"].iloc[0]
        self.assertEqual(a["Productos"], 1)  # el producto sin monto no cuenta
        self.assertEqual(a["Predicción del grupo (COP)"], 130.0)
        self.assertEqual(a["Ventas mes anterior (COP)"], 100.0)
        self.assertAlmostEqual(a["Variación"], 0.3)
        self.assertEqual(list(t["Grupo"]), ["A", "B"])  # ordenada por prediccion


if __name__ == "__main__":
    unittest.main()
