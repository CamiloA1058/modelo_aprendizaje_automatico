"""Rutas centralizadas del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
PREDICTIONS = OUTPUTS / "predictions"
REPORTS = OUTPUTS / "reports"
SCRIPTS = PROJECT_ROOT / "scripts"
LEGACY = PROJECT_ROOT / "models" / "legacy"
DATABASE = PROJECT_ROOT / "database"
CLEANING_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "ventas_forecast"
    / "data"
    / "cleaning.py"
)


def ensure_output_dirs():
    """Crea las carpetas de salida si no existen."""
    for path in (FIGURES, PREDICTIONS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
