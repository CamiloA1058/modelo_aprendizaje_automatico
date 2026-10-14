"""
figura_estudio_caso_totales.py
==============================
Figura del estudio de caso para la tesis: total vendido REAL frente al total
PREDICHO por el sistema (modelo combinado + estimación por promedio) en cada
mes del estudio de caso, y porcentaje de aciertos de "¿se venderá?".

Lee la hoja "Resumen" de outputs/reports/prediccion_vs_real.xlsx (la genera
scripts/reporte_prediccion_vs_real.py); no recalcula ni modifica ninguna cifra.

Genera: outputs/figures/estudio_caso_totales.png

Uso:  MPLBACKEND=Agg PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/figura_estudio_caso_totales.py
"""
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.figuras import (  # noqa: E402
    ANCHO, FUENTE, MESES_ES, aplicar_estilo, estilo_ejes, fmt_es, formateador_es,
)
from ventas_forecast.paths import FIGURES, REPORTS  # noqa: E402

COLOR_REAL = "#898781"
COLOR_PREDICHO = "#d99a00"
COLOR_ACIERTO = "#9b4dca"
MES_INCOMPLETO = "2026-09"


def cargar_resumen(xlsx=None):
    """Filas mensuales (hasta la primera fila vacía) de la hoja Resumen."""
    df = pd.read_excel(xlsx or REPORTS / "prediccion_vs_real.xlsx", sheet_name="Resumen")
    df = df[df["Mes"].astype(str).str.match(r"^\d{4}-\d{2}$")]
    return df.reset_index(drop=True)


def etiquetas_mes(meses):
    """'2026-04' -> 'Abr'; el mes con ventas reales incompletas lleva asterisco."""
    return [MESES_ES[int(m[5:7]) - 1] + ("*" if m == MES_INCOMPLETO else "") for m in meses]


def figura(resumen, salida):
    aplicar_estilo()
    meses = list(resumen["Mes"].astype(str))
    x = np.arange(len(meses))
    real = resumen["Real total (COP)"].to_numpy() / 1e6
    pred = resumen["Predicción total (COP)"].to_numpy() / 1e6
    acierto = resumen["% acierto de venta"].to_numpy() * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ANCHO, 8.2),
                                   gridspec_kw={"height_ratios": [1.25, 1]})
    estilo_ejes(ax1)
    estilo_ejes(ax2)

    ancho = 0.38
    b1 = ax1.bar(x - ancho / 2, real, ancho, color=COLOR_REAL, label="Vendido (real)")
    b2 = ax1.bar(x + ancho / 2, pred, ancho, color=COLOR_PREDICHO, label="Predicho por el sistema")
    for barras in (b1, b2):
        for b in barras:
            ax1.annotate(fmt_es(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                         textcoords="offset points", xytext=(0, 3), ha="center",
                         fontsize=FUENTE - 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas_mes(meses))
    ax1.set_ylabel("Total del mes (millones de COP)")
    ax1.set_ylim(0, max(real.max(), pred.max()) * 1.32)
    ax1.yaxis.set_major_formatter(formateador_es(0))
    ax1.legend(frameon=False, loc="upper center", ncol=2, columnspacing=1.2, handlelength=1.2)

    barras = ax2.bar(x, acierto, 0.55, color=COLOR_ACIERTO)
    for b, v in zip(barras, acierto):
        ax2.annotate(f"{fmt_es(v, 1)} %", (b.get_x() + b.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 3), ha="center")
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiquetas_mes(meses))
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Aciertos de «¿se venderá?» (%)")
    ax2.set_xlabel("Mes predicho")

    fig.text(0.02, 0.005, "* Mes incompleto: las ventas reales están registradas solo en parte.",
             fontsize=FUENTE - 1, color="#52514e", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(salida, dpi=200)
    plt.close(fig)


def main():
    salida = FIGURES / "estudio_caso_totales.png"
    figura(cargar_resumen(), salida)
    print(f"[OK] {salida}")


if __name__ == "__main__":
    main()
