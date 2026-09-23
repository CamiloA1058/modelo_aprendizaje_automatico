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
from ventas_forecast.paths import DATA_RAW, FIGURES, REPORTS, ensure_output_dirs  # noqa: E402

from train_kmeans_rf_prod import SalesForecastModel  # noqa: E402


def main():
    ensure_output_dirs()

    model = SalesForecastModel(filepath=DATA_RAW / "Query_Result_V5.csv")
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

    # ── Gráfico: codo + silueta ───────────────────────────────────────────────
    fig, (ax_codo, ax_silueta) = plt.subplots(1, 2, figsize=(14, 5))
    color = "#1D4ED8"

    ax_codo.plot(tabla["k"], tabla["inercia"], marker="o", color=color)
    ax_codo.set_title("Método del codo")
    ax_codo.set_xlabel("Número de clústeres (k)")
    ax_codo.set_ylabel("Inercia")
    ax_codo.grid(True, alpha=.3)

    ax_silueta.plot(tabla["k"], tabla["silueta"], marker="o", color=color)
    fila_optima = tabla.loc[tabla["silueta"].idxmax()]
    ax_silueta.scatter(
        [fila_optima["k"]], [fila_optima["silueta"]],
        s=140, marker="*", color="#DC2626", zorder=5,
        label=f"k óptimo = {int(fila_optima['k'])}",
    )
    ax_silueta.annotate(
        f"k={int(fila_optima['k'])}",
        (fila_optima["k"], fila_optima["silueta"]),
        textcoords="offset points", xytext=(8, 8),
    )
    ax_silueta.set_title("Coeficiente de silueta")
    ax_silueta.set_xlabel("Número de clústeres (k)")
    ax_silueta.set_ylabel("Silueta")
    ax_silueta.legend()
    ax_silueta.grid(True, alpha=.3)

    fig.tight_layout()
    salida_fig = FIGURES / "seleccion_k_kmeans.png"
    fig.savefig(salida_fig, dpi=200)
    print(f"[✓] Figura exportada → {salida_fig}")


if __name__ == "__main__":
    main()
