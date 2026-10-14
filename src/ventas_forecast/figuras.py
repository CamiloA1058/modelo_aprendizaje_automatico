"""
Utilidades de estilo y formato para las figuras de la tesis.

Todas las figuras comparten: ancho de 7 pulgadas (a 15 cm de ancho en el
documento, una fuente de 12 pt se ve de unos 10 pt), etiquetas en español,
coma decimal y punto de miles.
"""
from matplotlib.ticker import FuncFormatter

ANCHO = 7.0
FUENTE = 12

COLOR_GRID = "#e1e0d9"
COLOR_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_SURFACE = "#fcfcfb"

MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
            "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def fmt_es(valor, decimales=0):
    """Número con punto de miles y coma decimal: 1234567.5 -> '1.234.567,5'."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "\0").replace(".", ",").replace("\0", ".")


def formateador_es(decimales=0, sufijo=""):
    """FuncFormatter de matplotlib con coma decimal y punto de miles."""
    return FuncFormatter(lambda v, _pos: fmt_es(v, decimales) + sufijo)


def aplicar_estilo():
    """Parámetros globales de matplotlib para todas las figuras."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": FUENTE,
        "axes.labelsize": FUENTE,
        "xtick.labelsize": FUENTE,
        "ytick.labelsize": FUENTE,
        "legend.fontsize": FUENTE,
        "axes.edgecolor": COLOR_GRID,
        "axes.labelcolor": COLOR_INK,
        "xtick.color": COLOR_INK,
        "ytick.color": COLOR_INK,
        "text.color": COLOR_INK,
        "figure.facecolor": COLOR_SURFACE,
        "axes.facecolor": COLOR_SURFACE,
        "savefig.facecolor": COLOR_SURFACE,
    })


def estilo_ejes(ax, grid="y"):
    """Grilla recesiva y ejes suaves."""
    ax.set_facecolor(COLOR_SURFACE)
    for spine in ax.spines.values():
        spine.set_color(COLOR_GRID)
    if grid:
        ax.grid(axis=grid, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
