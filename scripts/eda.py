"""
eda.py
======
Análisis exploratorio de datos (EDA) sobre Query_Result_V5.csv — fase 2
(comprensión de datos) y fase 3 (preparación de datos) de CRISP-DM.

Usa las funciones puras de src/ventas_forecast/eda.py (probadas en
tests/test_eda.py) sobre:
  - el CSV crudo, ANTES de cualquier agregación (calidad de datos), y
  - el resultado de SalesForecastModel.load_and_clean() / build_features() /
    cluster() / prepare_split() (todo lo demás: tamaños, ABC, intermitencia,
    estacionalidad, perfil de clusters).

Genera:
  outputs/reports/eda_resumen.txt
  outputs/figures/eda_ventas_mensuales.png
  outputs/figures/eda_estacionalidad.png
  outputs/figures/eda_pareto_abc.png
  outputs/figures/eda_distribucion_ventas.png

Uso
---
    PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/eda.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.paths import DATA_RAW, FIGURES, REPORTS, ensure_output_dirs  # noqa: E402
from ventas_forecast.eda import (  # noqa: E402
    abc_classification,
    data_quality_summary,
    intermittency_summary,
    monthly_totals,
    seasonal_profile,
)

from train_kmeans_rf_prod import SalesForecastModel, DEFAULT_COL_MAP  # noqa: E402

# ── Paleta (misma referencia usada en el resto del proyecto) ────────────────
COLOR_LINEA   = "#2a78d6"
COLOR_GRID    = "#e1e0d9"
COLOR_INK     = "#52514e"
COLOR_MUTED   = "#898781"
COLOR_SURFACE = "#fcfcfb"
COLOR_FILL_A  = "#f0efec"

MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
            "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def _estilo_ejes(ax, fig):
    """Aplica el estilo compartido de las 4 figuras (fondo, grid, ejes)."""
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    ax.tick_params(colors=COLOR_INK, labelsize=9)
    ax.xaxis.label.set_color(COLOR_INK)
    ax.yaxis.label.set_color(COLOR_INK)
    for spine in ax.spines.values():
        spine.set_color(COLOR_GRID)


def main():
    inicio = time.time()
    ensure_output_dirs()

    # ── 1. CALIDAD DE DATOS: sobre el CSV crudo, antes de agregar ───────────
    raw_path = DATA_RAW / "Query_Result_V5.csv"
    raw_df = pd.read_csv(raw_path, sep=";", encoding="utf-8")

    col_map_raw = {**DEFAULT_COL_MAP, "date": "FECHA"}
    calidad = data_quality_summary(raw_df, col_map_raw)

    # Cómo maneja el pipeline los registros con id de producto nulo: HECHO
    # verificado (no supuesto) — load_and_clean() agrupa con
    # groupby([id, description, fecha]); pandas.groupby() usa dropna=True
    # por defecto, así que las filas con id NULO se EXCLUYEN silenciosamente
    # del agregado (nunca llegan a self.df, ni a clustering, entrenamiento o
    # predicción). No es un error del código: son ventas registradas sin
    # código de producto asociado (DESCRIPCION="VARIOS" en los 8 casos de
    # V5), pero SÍ representa una pérdida silenciosa de información que
    # conviene declarar.
    id_col = col_map_raw["id"]
    total_col = col_map_raw["total_sold"]

    def _parse_monto(col):
        return (
            col.astype(str).str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False).astype(float)
        )

    monto_total = _parse_monto(raw_df[total_col]).sum()
    monto_excluido = _parse_monto(
        raw_df.loc[raw_df[id_col].isna(), total_col]
    ).sum()

    # Primer/último día del histórico (crudo, granularidad diaria): para
    # saber si el primer/último mes calendario del dataset están completos.
    fechas_diarias = pd.to_datetime(raw_df["FECHA"], format="%d/%m/%Y")
    primer_dia, ultimo_dia = fechas_diarias.min(), fechas_diarias.max()
    ultimo_dia_del_mes = (
        pd.Timestamp(ultimo_dia.year, ultimo_dia.month, 1) + pd.offsets.MonthEnd(0)
    )
    primer_mes_parcial = primer_dia.day != 1
    ultimo_mes_parcial = ultimo_dia.normalize() != ultimo_dia_del_mes

    # ── 2. PIPELINE COMPLETO: agregación, calendario, features, clusters ───
    model = SalesForecastModel(filepath=raw_path)
    c = model._c

    # Se instrumenta _complete_calendar (sin tocar su lógica) solo para
    # capturar el tamaño ANTES/DESPUÉS del completado de calendario, dato
    # que load_and_clean() ya imprime pero no expone en ningún atributo.
    calendario_info = {}
    _complete_calendar_original = model._complete_calendar

    def _complete_calendar_instrumentado(df):
        calendario_info["filas_antes"] = len(df)
        calendario_info["productos_antes"] = df[c("id")].nunique()
        resultado = _complete_calendar_original(df)
        calendario_info["filas_despues"] = len(resultado)
        calendario_info["productos_despues"] = resultado[c("id")].nunique()
        calendario_info["ceros_agregados"] = len(resultado) - len(df)
        return resultado

    model._complete_calendar = _complete_calendar_instrumentado

    model.load_and_clean()
    model.build_features()
    model.cluster()
    model.prepare_split()

    n_meses_calendario = model.df["fecha"].nunique()

    # ── 3. ABC, intermitencia y estacionalidad sobre el histórico completo ──
    tabla_abc, resumen_abc = abc_classification(model.df, c("id"), c("total_sold"))
    intermit = intermittency_summary(model.df, c("id"), c("total_sold"))
    mensual = monthly_totals(model.df, "fecha", c("total_sold"))
    perfil_estacional = seasonal_profile(mensual)

    # ── 4. Perfil de clusters (SOLO partición de entrenamiento, sin fuga) ───
    entrenamiento = model._df_train.loc[model._train_idx]
    perfil_clusters = entrenamiento.groupby("cluster").agg(
        n_producto_mes=("cluster", "size"),
        mediana_total_vendido=(c("total_sold"), "median"),
        mediana_frecuencia=(c("frequency"), "median"),
        mediana_precio=(c("avg_price"), "median"),
    ).sort_index()

    # ── Reporte de texto ─────────────────────────────────────────────────────
    lineas = []
    lineas.append("ANÁLISIS EXPLORATORIO DE DATOS (EDA) — Query_Result_V5.csv")
    lineas.append("=" * 70)

    lineas.append("")
    lineas.append("1. CALIDAD DE DATOS (CSV crudo, antes de agregar)")
    lineas.append("-" * 70)
    lineas.append(f"Filas: {calidad['n_rows']:,} | Columnas: {calidad['n_cols']}")
    nulos_no_cero = {k: v for k, v in calidad["nulls"].items() if v > 0}
    if nulos_no_cero:
        lineas.append("Nulos por columna (solo columnas con al menos 1 nulo):")
        for col, n in nulos_no_cero.items():
            lineas.append(f"  {col:<20}: {n:,}")
    else:
        lineas.append("Nulos por columna: ninguno.")
    lineas.append(f"Filas exactamente duplicadas: {calidad['n_duplicates']:,}")
    lineas.append(
        f"Filas con id de producto nulo ({id_col} nulo): {calidad['n_null_id']:,} "
        f"({monto_excluido:,.0f} COP, {monto_excluido / monto_total:.4%} del total)"
    )
    lineas.append(
        "  -> Manejo real del pipeline (verificado en train_kmeans_rf_prod."
        "load_and_clean): groupby([id, description, fecha]) usa dropna=True "
        "por defecto, así que estas filas se EXCLUYEN silenciosamente del "
        "agregado — nunca llegan a self.df ni a clustering/entrenamiento/"
        "predicción. Monto excluido económicamente despreciable "
        f"({monto_excluido / monto_total:.4%} del total), pero es una "
        "pérdida silenciosa de datos que conviene declarar."
    )
    lineas.append(f"Productos distintos (id no nulo): {calidad['n_products']:,}")
    lineas.append(
        f"Rango de fechas (crudo, diario): {calidad['date_min']:%Y-%m-%d} a "
        f"{calidad['date_max']:%Y-%m-%d} ({calidad['n_distinct_months']} "
        "combinaciones año-mes distintas)"
    )
    lineas.append(
        f"Primer mes ({primer_dia:%Y-%m}): "
        + ("PARCIAL, arranca el día " + str(primer_dia.day) if primer_mes_parcial
           else "completo")
    )
    lineas.append(
        f"Último mes ({ultimo_dia:%Y-%m}): "
        + ("PARCIAL, termina el día " + str(ultimo_dia.day) if ultimo_mes_parcial
           else "completo")
    )

    lineas.append("")
    lineas.append("2. TAMAÑO DEL DATASET (agregación y completado de calendario)")
    lineas.append("-" * 70)
    lineas.append(f"Filas crudas (transacciones diarias): {calidad['n_rows']:,}")
    lineas.append(
        f"Tras agregar a producto-mes (antes de completar calendario): "
        f"{calendario_info['filas_antes']:,} filas | "
        f"{calendario_info['productos_antes']:,} productos"
    )
    lineas.append(
        f"Tras completar el calendario mensual (ver HALLAZGO 2026-09-23, "
        f"CHANGELOG): {calendario_info['filas_despues']:,} filas | "
        f"{calendario_info['productos_despues']:,} productos | "
        f"{calendario_info['ceros_agregados']:,} filas de mes sin venta "
        "agregadas"
    )
    lineas.append(f"Meses calendario distintos en el histórico completo: {n_meses_calendario}")

    lineas.append("")
    lineas.append("3. CLASIFICACIÓN ABC (por producto, participación de TOTAL_VENDIDO)")
    lineas.append("-" * 70)
    lineas.append(
        "Regla de frontera: clase A si el acumulado (incluyendo el producto) "
        "es <= 80 %, B si es <= 95 %, C en otro caso. Un producto que por sí "
        "solo hace que el acumulado SUPERE un corte queda en la clase "
        "siguiente (ver src/ventas_forecast/eda.py)."
    )
    for clase in ("A", "B", "C"):
        r = resumen_abc[clase]
        lineas.append(
            f"  Clase {clase}: {r['n_products']:,} productos "
            f"({r['n_products'] / len(tabla_abc):.1%} del total) -> "
            f"{r['pct_sales']:.2f} % de las ventas"
        )

    lineas.append("")
    lineas.append("4. INTERMITENCIA (meses en cero, sobre el calendario completo)")
    lineas.append("-" * 70)
    lineas.append(
        f"Filas producto-mes con venta cero: {intermit['share_zero_overall']:.2%} "
        "del total"
    )
    lineas.append(
        "Distribución por producto de la fracción de sus propios meses en "
        f"cero — mediana: {intermit['median']:.2%} | Q1: {intermit['q1']:.2%} "
        f"| Q3: {intermit['q3']:.2%}"
    )

    lineas.append("")
    lineas.append("5. PERFIL ESTACIONAL (promedio de ventas por mes calendario)")
    lineas.append("-" * 70)
    lineas.append(
        "ADVERTENCIA: con ~2-3 años de historial, el promedio por mes se "
        "apoya en muy pocas observaciones (ver columna n_años) — estimación "
        "estacional débil, no concluyente."
    )
    lineas.append(f"{'Mes':<6}{'Promedio (COP)':>20}{'Años observados':>18}")
    for mes_num, nombre in enumerate(MESES_ES, start=1):
        fila = perfil_estacional.loc[mes_num]
        promedio = fila["promedio"]
        promedio_txt = f"{promedio:,.0f}" if pd.notna(promedio) else "sin datos"
        lineas.append(f"{nombre:<6}{promedio_txt:>20}{int(fila['n_anios']):>18}")

    lineas.append("")
    lineas.append("6. PERFIL DE CLUSTERS (partición de ENTRENAMIENTO, escala original)")
    lineas.append("-" * 70)
    lineas.append(
        f"{'Cluster':<10}{'n (prod-mes)':>14}{'Mediana TOTAL_VENDIDO':>24}"
        f"{'Mediana FRECUENCIA':>20}{'Mediana PRECIO_PROM.':>22}"
    )
    for cluster_id, fila in perfil_clusters.iterrows():
        lineas.append(
            f"{cluster_id:<10}{int(fila['n_producto_mes']):>14,}"
            f"{fila['mediana_total_vendido']:>24,.0f}"
            f"{fila['mediana_frecuencia']:>20,.1f}"
            f"{fila['mediana_precio']:>22,.0f}"
        )
    lineas.append(
        "(Interpretación en palabras: describir cada cluster con base en su "
        "mediana de TOTAL_VENDIDO/FRECUENCIA/PRECIO_PROMEDIO relativa a los "
        "demás — p.ej. alta rotación y ventas altas vs. baja rotación y "
        "ventas bajas — se deja para el documento de tesis, con estos "
        "números como evidencia.)"
    )

    elapsed = time.time() - inicio
    lineas.append("")
    lineas.append(f"Tiempo total: {elapsed:.1f} s")

    resumen_txt = "\n".join(lineas)
    print(resumen_txt)

    salida_txt = REPORTS / "eda_resumen.txt"
    salida_txt.write_text(resumen_txt, encoding="utf-8")
    print(f"\n[✓] Reporte exportado -> {salida_txt}")

    # ── Figura a: ventas mensuales totales ──────────────────────────────────
    x = mensual.index
    y = mensual.to_numpy() / 1e6

    fig, ax = plt.subplots(figsize=(10, 5))
    _estilo_ejes(ax, fig)
    ax.plot(x, y, color=COLOR_LINEA, linewidth=2, marker="o", markersize=4.5,
            markerfacecolor=COLOR_LINEA, markeredgecolor=COLOR_LINEA)

    # Primer y último mes: marcador hueco si son meses parciales (día crudo
    # distinto de 1 o del último día del mes, ver sección 1 del reporte).
    if primer_mes_parcial:
        ax.plot(x[0], y[0], marker="o", markersize=7, markerfacecolor=COLOR_SURFACE,
                markeredgecolor=COLOR_LINEA, markeredgewidth=1.6, zorder=5)
    if ultimo_mes_parcial:
        ax.plot(x[-1], y[-1], marker="o", markersize=7, markerfacecolor=COLOR_SURFACE,
                markeredgecolor=COLOR_LINEA, markeredgewidth=1.6, zorder=5)

    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Ventas (millones de COP)")
    fig.autofmt_xdate(rotation=45)
    if primer_mes_parcial or ultimo_mes_parcial:
        fig.text(0.5, 0.005, "○ Mes incompleto (primer y/o último mes del histórico)",
                  ha="center", fontsize=8, color=COLOR_MUTED)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()

    salida_a = FIGURES / "eda_ventas_mensuales.png"
    fig.savefig(salida_a, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Figura exportada -> {salida_a}")

    # ── Figura b: estacionalidad (promedio por mes calendario) ──────────────
    valores = (perfil_estacional["promedio"].to_numpy() / 1e6)
    valores_plot = np.nan_to_num(valores, nan=0.0)
    n_anios_plot = perfil_estacional["n_anios"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    _estilo_ejes(ax, fig)
    barras = ax.bar(MESES_ES, valores_plot, color=COLOR_LINEA, width=0.7)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Ventas promedio (millones de COP)")

    y_max = valores_plot.max() if valores_plot.max() > 0 else 1.0
    ax.set_ylim(top=y_max * 1.18)
    for barra, n in zip(barras, n_anios_plot):
        ax.annotate(f"n={int(n)}", (barra.get_x() + barra.get_width() / 2,
                    barra.get_height()), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.5, color=COLOR_MUTED)

    fig.tight_layout()
    salida_b = FIGURES / "eda_estacionalidad.png"
    fig.savefig(salida_b, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Figura exportada -> {salida_b}")

    # ── Figura c: Pareto / curva ABC ─────────────────────────────────────────
    n_prod = len(tabla_abc)
    x_pct = np.arange(1, n_prod + 1) / n_prod * 100
    y_pct = tabla_abc["cum_share"].to_numpy() * 100

    n_a = resumen_abc["A"]["n_products"]
    n_b = resumen_abc["B"]["n_products"]
    corte_ab = n_a / n_prod * 100
    corte_bc = (n_a + n_b) / n_prod * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    _estilo_ejes(ax, fig)
    ax.axvspan(0, corte_ab, color=COLOR_FILL_A, zorder=0)
    ax.axvspan(corte_ab, corte_bc, color=COLOR_SURFACE, zorder=0)
    ax.axvspan(corte_bc, 100, color=COLOR_FILL_A, zorder=0)
    ax.plot(x_pct, y_pct, color=COLOR_LINEA, linewidth=2, zorder=3)
    ax.axvline(corte_ab, color=COLOR_MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.axvline(corte_bc, color=COLOR_MUTED, linestyle="--", linewidth=1, zorder=2)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% de productos (ordenados por ventas, de mayor a menor)")
    ax.set_ylabel("% acumulado de ventas")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # Letras de región centradas arriba (caben incluso en la región A, que es
    # angosta) y el detalle numérico en un bloque dentro de la región C, que
    # es amplia y está vacía; así ningún texto cruza las líneas de corte.
    for letra, x0, x1 in (("A", 0, corte_ab), ("B", corte_ab, corte_bc), ("C", corte_bc, 100)):
        ax.text((x0 + x1) / 2, 103, letra, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLOR_INK)
    pct_prod = {k: resumen_abc[k]["n_products"] / n_prod * 100 for k in ("A", "B", "C")}
    detalle = "\n".join(
        f"{k}: {pct_prod[k]:.1f} % de productos → {resumen_abc[k]['pct_sales']:.1f} % de ventas"
        for k in ("A", "B", "C")
    )
    ax.text(corte_bc + 4, 40, detalle, ha="left", va="center", fontsize=8.5,
            color=COLOR_INK, linespacing=1.6)
    ax.set_ylim(0, 108)

    fig.tight_layout()
    salida_c = FIGURES / "eda_pareto_abc.png"
    fig.savefig(salida_c, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Figura exportada -> {salida_c}")

    # ── Figura d: distribución de ventas producto-mes (log10, solo > 0) ────
    positivos = model.df.loc[model.df[c("total_sold")] > 0, c("total_sold")].to_numpy()
    bins = np.logspace(np.log10(positivos.min()), np.log10(positivos.max()), 40)

    fig, ax = plt.subplots(figsize=(9, 5))
    _estilo_ejes(ax, fig)
    ax.hist(positivos, bins=bins, color=COLOR_LINEA)
    ax.set_xscale("log")
    ax.set_xlabel("Ventas del mes por producto (COP, escala log)")
    ax.set_ylabel("Número de observaciones producto-mes")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # Margen superior para que la nota no tape ninguna barra
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.text(
        0.02, 0.96,
        f"{intermit['share_zero_overall']:.1%} de los meses producto-mes tienen "
        "venta = 0 y no se muestran aquí\n(no representables en escala "
        "logarítmica; ver sección 4 del reporte)",
        transform=ax.transAxes, ha="left", va="top", fontsize=8, color=COLOR_INK,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=COLOR_FILL_A, edgecolor=COLOR_GRID),
    )

    fig.tight_layout()
    salida_d = FIGURES / "eda_distribucion_ventas.png"
    fig.savefig(salida_d, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Figura exportada -> {salida_d}")

    print(f"\n[✓] Tiempo total: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
