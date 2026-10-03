"""
Entrena el modelo de produccion (KMeans + Random Forest) sobre el dataset por
defecto (data/raw/Query_Result_V5.csv) y escribe la prediccion del mes
siguiente en outputs/predictions/PREDICCION_MENSUAL.csv.

Ejecutar desde la raiz del repo:  python scripts/entrenar.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.model import SalesForecastModel  # noqa: E402
from ventas_forecast.paths import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_PREDICTION,
    ensure_output_dirs,
)

if __name__ == "__main__":
    ensure_output_dirs()

    model = SalesForecastModel(
        filepath=DEFAULT_DATASET,
        output_path=DEFAULT_PREDICTION,
        min_months=6,
    )
    model.run()
