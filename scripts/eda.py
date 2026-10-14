"""
eda.py
======
Análisis exploratorio de datos (EDA) sobre Query_Result_V6.csv — fase 2
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
from ventas_forecast.paths import DEFAULT_DATASET, FIGURES, REPORTS, ensure_output_dirs  # noqa: E402
from ventas_forecast.eda import (  # noqa: E402
    abc_classification,
    data_quality_summary,
    intermittency_summary,
    monthly_totals,
    seasonal_profile,
)

from ventas_forecast.model import SalesForecastModel, DEFAULT_COL_MAP  # noqa: E402
from ventas_forecast.figuras import (  # noqa: E402
    ANCHO, COLOR_GRID, COLOR_INK, COLOR_MUTED, COLOR_SURFACE, FUENTE, MESES_ES,
    aplicar_estilo, estilo_ejes, fmt_es, formateador_es,
)

# ── Paleta (misma referencia usada en el resto del proyecto) ────────────────
COLOR_LINEA   = "#008300"   # Figura de ventas mensuales
COLOR_ESTACION = "#eb6834"  # Figura de estacionalidad
COLOR_PARETO   = "#199e70"  # Figura de Pareto ABC
COLOR_HIST     = "#4a3aa7"  # Figura de distribución de ventas
COLOR_FILL_A  = "#f0efec"


def main():
    inicio = time.time()
    ensure_output_dirs()

    # ── 1. CALIDAD DE DATOS: sobre el CSV crudo, antes de agregar ───────────
    raw_path = DEFAULT_DATASET
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
    lineas.append("ANÁLISIS EXPLORATORIO DE DATOS (EDA) — Query_Result_V6.csv")
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
        "  -> Manejo real del pipeline (verificado en ventas_forecast.model."
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
    if model.cleaning_summary is not None:
        resumen_ajuste = model.cleaning_summary
        productos_perdidos = calidad["n_products"] - calendario_info["productos_antes"]
        lineas.append(
            "Ajuste masivo de inventario excluido (CONFIRMADO por el negocio, "
            "ver CHANGELOG 2026-09-23 'Limpieza', remove_bulk_adjustments): "
            f"{resumen_ajuste['rows_removed']:,} filas "
            f"({resumen_ajuste['distinct_products']:,} productos distintos, "
            f"{len(resumen_ajuste['affected_dates'])} fecha(s): "
            f"{', '.join(resumen_ajuste['affected_dates'])}), "
            f"${resumen_ajuste['cop_amount']:,.0f} COP — no es venta real."
        )
        lineas.append(
            f"De esos {resumen_ajuste['distinct_products']:,} productos, "
            f"{productos_perdidos:,} tenían ESA fila de ajuste como su ÚNICO "
            "registro y por lo tanto desaparecen por completo del histórico "
            f"limpio ({calidad['n_products']:,} productos crudos -> "
            f"{calendario_info['productos_antes']:,} tras la limpieza)."
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

    # ── Figuras ──────────────────────────────────────────────────────────────
    aplicar_estilo()
    figura_ventas_mensuales(mensual, primer_mes_parcial, ultimo_mes_parcial)
    figura_estacionalidad(perfil_estacional)
    figura_pareto(tabla_abc, resumen_abc)
    positivos = model.df.loc[model.df[c("total_sold")] > 0, c("total_sold")].to_numpy()
    figura_distribucion(positivos, intermit["share_zero_overall"])

    print(f"\n[✓] Tiempo total: {elapsed:.1f} s")


def _guardar(fig, nombre):
    salida = FIGURES / nombre
    fig.savefig(salida, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Figura exportada -> {salida}")


def figura_ventas_mensuales(mensual, primer_mes_parcial, ultimo_mes_parcial):
    """Ventas totales por mes (millones de COP); marcador hueco en meses incompletos."""
    x = mensual.index
    y = mensual.to_numpy() / 1e6

    fig, ax = plt.subplots(figsize=(ANCHO, 4.6))
    estilo_ejes(ax)
    ax.plot(x, y, color=COLOR_LINEA, linewidth=2, marker="o", markersize=5)
    parciales = []
    if primer_mes_parcial:
        parciales.append(0)
    if ultimo_mes_parcial:
        parciales.append(len(x) - 1)
    if parciales:
        ax.plot([x[i] for i in parciales], [y[i] for i in parciales], linestyle="none",
                marker="o", markersize=8, markerfacecolor=COLOR_SURFACE,
                markeredgecolor=COLOR_LINEA, markeredgewidth=1.8, zorder=5,
                label="Mes incompleto")
        ax.legend(frameon=False, loc="upper left")

    pos = list(range(0, len(x), 3))
    ax.set_xticks([x[i] for i in pos])
    ax.set_xticklabels([f"{MESES_ES[x[i].month - 1]} {x[i].year % 100:02d}" for i in pos],
                       rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(formateador_es(0))
    ax.set_xlabel("Mes")
    ax.set_ylabel("Ventas (millones de COP)")
    fig.tight_layout()
    _guardar(fig, "eda_ventas_mensuales.png")


def figura_estacionalidad(perfil_estacional):
    """Promedio de ventas por mes calendario; n = años observados de cada mes."""
    valores = np.nan_to_num(perfil_estacional["promedio"].to_numpy() / 1e6, nan=0.0)
    n_anios = perfil_estacional["n_anios"].to_numpy()

    fig, ax = plt.subplots(figsize=(ANCHO, 4.6))
    estilo_ejes(ax)
    barras = ax.bar(MESES_ES, valores, color=COLOR_ESTACION, width=0.7)
    ax.set_ylim(0, max(valores.max(), 1.0) * 1.15)
    ax.yaxis.set_major_formatter(formateador_es(0))
    ax.set_xlabel("Mes (n = años observados)")
    ax.set_ylabel("Ventas promedio (millones de COP)")
    for barra, n in zip(barras, n_anios):
        ax.annotate(f"n={int(n)}", (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=FUENTE - 1, color=COLOR_INK)
    plt.setp(ax.get_xticklabels(), fontsize=FUENTE - 1)
    fig.tight_layout()
    _guardar(fig, "eda_estacionalidad.png")


def figura_pareto(tabla_abc, resumen_abc):
    """Curva de Pareto: % acumulado de ventas según % de productos."""
    n_prod = len(tabla_abc)
    x_pct = np.arange(1, n_prod + 1) / n_prod * 100
    y_pct = tabla_abc["cum_share"].to_numpy() * 100
    n_a = resumen_abc["A"]["n_products"]
    n_b = resumen_abc["B"]["n_products"]
    corte_ab = n_a / n_prod * 100
    corte_bc = (n_a + n_b) / n_prod * 100

    fig, ax = plt.subplots(figsize=(ANCHO, 5.2))
    estilo_ejes(ax)
    ax.axvspan(0, corte_ab, color=COLOR_FILL_A, zorder=0)
    ax.axvspan(corte_bc, 100, color=COLOR_FILL_A, zorder=0)
    ax.plot(x_pct, y_pct, color=COLOR_PARETO, linewidth=2.2, zorder=3)
    for corte in (corte_ab, corte_bc):
        ax.axvline(corte, color=COLOR_MUTED, linestyle="--", linewidth=1, zorder=2)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("% de productos (de mayor a menor venta)")
    ax.set_ylabel("% acumulado de ventas")
    for letra, x0, x1 in (("A", 0, corte_ab), ("B", corte_ab, corte_bc), ("C", corte_bc, 100)):
        ax.text((x0 + x1) / 2, 102, letra, ha="center", va="bottom",
                fontweight="bold", color=COLOR_INK)
    detalle = "\n".join(
        f"{k}: {fmt_es(resumen_abc[k]['n_products'] / n_prod * 100, 1)} % de productos"
        f" → {fmt_es(resumen_abc[k]['pct_sales'], 1)} % de ventas"
        for k in ("A", "B", "C")
    )
    ax.text(99, 30, detalle, ha="right", va="center", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_SURFACE, edgecolor=COLOR_GRID))
    fig.tight_layout()
    _guardar(fig, "eda_pareto_abc.png")


def figura_distribucion(positivos, proporcion_ceros):
    """Histograma de ventas producto-mes > 0 en escala logarítmica."""
    bins = np.logspace(np.log10(positivos.min()), np.log10(positivos.max()), 40)

    fig, ax = plt.subplots(figsize=(ANCHO, 4.6))
    estilo_ejes(ax)
    ax.hist(positivos, bins=bins, color=COLOR_HIST)
    ax.set_xscale("log")
    ax.set_xticks([1e2, 1e4, 1e6])
    ax.xaxis.set_major_formatter(formateador_es(0))
    ax.yaxis.set_major_formatter(formateador_es(0))
    ax.set_xlabel("Ventas del mes por producto (COP, escala logarítmica)")
    ax.set_ylabel("Casos (producto-mes)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    ax.text(0.02, 0.96,
            f"Sin los meses sin ventas\n({fmt_es(proporcion_ceros * 100, 1)} % de los casos)",
            transform=ax.transAxes, ha="left", va="top", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=COLOR_FILL_A, edgecolor=COLOR_GRID))
    fig.tight_layout()
    _guardar(fig, "eda_distribucion_ventas.png")


if __name__ == "__main__":
    main()
