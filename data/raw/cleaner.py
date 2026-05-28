from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "src"

sys.path.append(str(SRC_PATH))
from ventas_forecast.data.cleaning import DatasetCleaner

cleaner = DatasetCleaner("Query_Result_V4.csv", separator=';')
cleaner.load_data()

# Limpieza original (mano de obra y gasolina)
cleaner.remove_by_keywords(
    column_name="DESCRIPCION",
    keywords=["MANTENIMIENTO", "SOLUCION"]
)

# ✅ NUEVO: eliminar registros sin precio real
cleaner.remove_price_anomalies(
    price_column="PRECIO_PROMEDIO",
    qty_column="VENTAS",
    total_column="TOTAL_VENDIDO",
    min_price=10
)

cleaner.save_dataset("Query_Result_V4_clean.csv")