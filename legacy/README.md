# Código heredado

Experimentos y versiones anteriores conservados como referencia histórica.
**No forman parte del sistema de producción** y no están cubiertos por las
pruebas del proyecto (`tests/`).

- `scripts/`: primeros pipelines (KMeans + RF inicial, XGBoost, Prophet, reportes).
- `models/`: versiones 1.0 y 2.0 del modelo y la demo ARIMA + RF.
- `tests/`: scripts de humo de XGBoost sin aserciones.

Leen los datos antiguos de `data/historico/` y se ejecutan desde la raíz del
repositorio, por ejemplo `python legacy/models/model_k-means_random_forest_v2.0.py`.
Pueden requerir dependencias adicionales (`xgboost`, `prophet`, `statsmodels`).
El modelo vigente está en `src/ventas_forecast/model.py`.
