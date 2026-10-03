"""Rutas por defecto del proyecto: dataset de produccion y salida de predicciones."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ventas_forecast import paths  # noqa: E402
from ventas_forecast.model import SalesForecastModel  # noqa: E402


class TestDefaultPaths(unittest.TestCase):
    def test_default_dataset_is_v5_in_data_raw(self):
        self.assertEqual(paths.DEFAULT_DATASET, ROOT / "data" / "raw" / "Query_Result_V5.csv")

    def test_default_prediction_under_outputs_predictions(self):
        self.assertEqual(
            paths.DEFAULT_PREDICTION,
            ROOT / "outputs" / "predictions" / "PREDICCION_MENSUAL.csv",
        )

    def test_uploads_under_outputs(self):
        self.assertEqual(paths.UPLOADS, ROOT / "outputs" / "uploads")

    def test_model_default_output_not_next_to_input(self):
        model = SalesForecastModel(filepath=paths.DEFAULT_DATASET)
        self.assertEqual(model.output_path, paths.DEFAULT_PREDICTION)
        self.assertNotEqual(model.output_path.parent, paths.DEFAULT_DATASET.parent)


if __name__ == "__main__":
    unittest.main()
