"""Rutas centralizadas del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_HISTORICO = PROJECT_ROOT / "data" / "historico"
DEFAULT_DATASET = DATA_RAW / "Query_Result_V6.csv"

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
PREDICTIONS = OUTPUTS / "predictions"
REPORTS = OUTPUTS / "reports"
UPLOADS = OUTPUTS / "uploads"
DEFAULT_PREDICTION = PREDICTIONS / "PREDICCION_MENSUAL.csv"


def ensure_output_dirs():
    """Crea las carpetas de salida si no existen."""
    for path in (FIGURES, PREDICTIONS, REPORTS, UPLOADS):
        path.mkdir(parents=True, exist_ok=True)
