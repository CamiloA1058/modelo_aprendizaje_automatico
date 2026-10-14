"""
select_k.py
============
Selección del número de clústeres (k) para el KMeans de segmentación de
productos, mediante el método del codo (inercia) y el coeficiente de
silueta, evaluados solo sobre la partición de entrenamiento (sin fuga de
datos, ver SalesForecastModel.evaluate_k).

Uso
---
    PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/select_k.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.paths import DEFAULT_DATASET, FIGURES, REPORTS, ensure_output_dirs  # noqa: E402

from ventas_forecast.model import SalesForecastModel  # noqa: E402
from ventas_forecast.figuras import ANCHO, aplicar_estilo, estilo_ejes, formateador_es  # noqa: E402


def main():
    ensure_output_dirs()

    model = SalesForecastModel(filepath=DEFAULT_DATASET)
    model.load_and_clean()
    model.build_features()

    tabla = model.evaluate_k(k_range=range(2, 11))

    # ── Exportar tabla ────────────────────────────────────────────────────────
    salida_csv = REPORTS / "seleccion_k_kmeans.csv"
    tabla.to_csv(salida_csv, index=False)
    print(tabla.to_string(index=False))

    k_optimo = int(tabla.loc[tabla["silueta"].idxmax(), "k"])
    print(f"\n[✓] k con mayor silueta: {k_optimo}")
    print(f"[✓] Tabla exportada → {salida_csv}")

    salida_fig = FIGURES / "seleccion_k_kmeans.png"
    figura_seleccion_k(tabla, salida_fig)
    print(f"[✓] Figura exportada → {salida_fig}")


def figura_seleccion_k(tabla, salida):
    """Codo (inercia) y silueta por k, en dos paneles apilados."""
    aplicar_estilo()
    color = "#e34948"
    fig, (ax_codo, ax_silueta) = plt.subplots(2, 1, figsize=(ANCHO, 7.2), sharex=True)
    for ax in (ax_codo, ax_silueta):
        estilo_ejes(ax, grid="both")

    ax_codo.plot(tabla["k"], tabla["inercia"], marker="o", color=color, linewidth=2)
    ax_codo.set_ylabel("Inercia")
    ax_codo.yaxis.set_major_formatter(formateador_es(0))

    ax_silueta.plot(tabla["k"], tabla["silueta"], marker="o", color=color, linewidth=2)
    fila = tabla.loc[tabla["silueta"].idxmax()]
    ax_silueta.scatter([fila["k"]], [fila["silueta"]], s=220, marker="*", color="#eda100",
                       zorder=5, label=f"k óptimo = {int(fila['k'])}")
    ax_silueta.set_xlabel("Número de clústeres (k)")
    ax_silueta.set_ylabel("Coeficiente de silueta")
    ax_silueta.yaxis.set_major_formatter(formateador_es(2))
    ax_silueta.set_xticks(list(tabla["k"]))
    ax_silueta.set_ylim(top=float(tabla["silueta"].max()) + 0.012)
    ax_silueta.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
