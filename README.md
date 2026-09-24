# Modelo de Predicción de Ventas K-Means + Random Forest

Sistema de predicción de ventas y decisiones de stock (Reforzar / Mantener / Reducir) utilizando K-Means + Random Forest.

**Modelo final (producción)**: `scripts/train_kmeans_rf_prod.py` (clase `SalesForecastModel`)  
**`scripts/train_kmeans_rf.py`** ("predicción por ítem") es **legacy**: no forma parte del modelo final, no lo importa ningún script de producción, evaluación o comparación, y el dashboard (`scripts/app.py`) solo usa `train_kmeans_rf_prod`.  
**Modelos de comparación** (benchmark): `scripts/compare_models.py` (RF vs XGBoost vs Prophet vs ARIMA, sobre la misma partición del modelo final); `scripts/run_xgboost.py` y `scripts/run_prophet.py` quedan sustituidos por `compare_models.py` para efectos del trabajo de grado.

Dataset vigente: `data/raw/Query_Result_V5.csv` (extracción de ventas reales del taller de los últimos dos años; 78.921 registros diarios, 5.654 productos, 5.074 tras la limpieza de ajustes masivos de inventario).

### Métricas — clasificador (`outputs/reports/evaluacion_modelo.txt`, prueba n=8.942)

Métricas principales (robustas al desbalance de clases; Reducir = 78,42 % de la prueba):

| Métrica | Modelo | Línea base (siempre "Reducir") |
|---|---|---|
| F1 macro | 45.25% | 29.30% |
| Accuracy balanceada | 50.16% | 33.33% |

Métricas ponderadas por soporte de clase (referencia):

| Métrica | Modelo | Línea base (siempre "Reducir") |
|---|---|---|
| Accuracy | 67.51% | 78.42% |
| F1 ponderado | 70.83% | 68.93% |

### Métricas — regresor (`outputs/reports/evaluacion_modelo.txt`, prueba top productos, n=5.062)

| Métrica | Modelo | Ingenuo | Media_3 |
|---|---|---|---|
| MAE (COP) | 385,016 | 644,901 | 493,789 |
| RMSE (COP) | 1,244,584 | 1,651,228 | 1,232,872 |
| R² | 0.6005 | 0.2969 | 0.6080 |
| MASE | 0.8887 | 1.4885 | 1.1397 |

El regresor supera a ambas líneas base en MAE y MASE (MASE < 1); Media_3 obtiene mejor RMSE y R² en esta partición (ver detalle y discusión en `docs/CHANGELOG.md`, entrada "[Limpieza] - 2026-09-23").

Ver [docs/CHANGELOG.md](docs/CHANGELOG.md) para el historial completo de decisiones y resultados, y [docs/BITACORA.md](docs/BITACORA.md) para reportes detallados.

---

## Estructura del proyecto

```
modelo_aprendizaje_automatico/
│
├── README.md
├── requirements.txt
│
├── src/ventas_forecast/          # Código compartido
│   ├── paths.py                  # Rutas centralizadas
│   ├── eda.py                    # Funciones de análisis exploratorio
│   ├── benchmarks.py             # Utilidades de comparación RF/XGBoost/Prophet/ARIMA
│   └── data/cleaning.py          # DatasetCleaner + remove_bulk_adjustments
│
├── scripts/                      # Scripts ejecutables
│   ├── train_kmeans_rf_prod.py   # Modelo FINAL (producción) — SalesForecastModel
│   ├── train_kmeans_rf.py        # LEGACY — predicción por ítem, no usado por el modelo final
│   ├── select_k.py               # Selección de k (codo + silueta) para K-means
│   ├── tune_rf.py                # Ajuste de hiperparámetros del clasificador (GridSearchCV)
│   ├── evaluate_model.py         # Evaluación del clasificador y el regresor
│   ├── compare_models.py         # Comparación RF vs XGBoost vs Prophet vs ARIMA
│   ├── eda.py                    # Reporte y figuras de análisis exploratorio
│   ├── run_xgboost.py, run_prophet.py  # Benchmarks antiguos (sustituidos por compare_models.py)
│   ├── app.py                    # Servidor Flask del dashboard
│   ├── dashboard.html            # Interfaz web del dashboard
│   ├── Iniciar_Dashboard.bat     # Lanzador Windows — doble clic
│   ├── generar_reportes.py       # Reportes para cliente
│   ├── monitor_prophet.py        # Monitor de progreso Prophet
│   └── debug_capture.py          # Depuración de ejecución
│
├── models/legacy/                # Versiones históricas (v1.0, v2.0, ARIMA demo)
│
├── data/raw/                     # Datos fuente
│   ├── Query_Result.csv          # Exportación v1
│   ├── Query_Result_V2.csv       # Con columnas diarias
│   ├── Query_Result_V3.csv       # Legacy
│   ├── Query_Result_V4.csv       # Legacy
│   ├── Query_Result_V5.csv       # Producción — usado por train_kmeans_rf_prod y el dashboard
│   ├── ventas.csv
│   └── README.md                 # Documentación de datasets
│
├── outputs/                      # Generado al ejecutar (gitignored)
│   ├── figures/                  # Gráficos PNG
│   ├── predictions/              # CSV de predicciones
│   └── reports/                  # Resúmenes TXT y CSV (evaluación, tuning, comparación, EDA)
│
├── docs/                         # Documentación del proyecto
│   ├── BITACORA.md
│   ├── CHANGELOG.md              # Historial detallado de decisiones y resultados
│   ├── INDICE_DOCUMENTACION.md
│   └── VERSION_SUMMARY.md
│
├── tests/                        # Pruebas (unittest)
└── database/                     # Backups Firebird (gitignored)
```

---

## Cómo usar

### 1. Preparar entorno

```bash
cd modelo_aprendizaje_automatico
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ejecutar modelos

Desde la **raíz del proyecto** (usar `PYTHONIOENCODING=utf-8` para evitar errores de consola con los caracteres ✓/✅/❌):

```bash
# Modelo FINAL — producción
PYTHONIOENCODING=utf-8 python scripts/train_kmeans_rf_prod.py

# Selección de k, ajuste de hiperparámetros, evaluación
PYTHONIOENCODING=utf-8 python scripts/select_k.py
PYTHONIOENCODING=utf-8 python scripts/tune_rf.py
PYTHONIOENCODING=utf-8 python scripts/evaluate_model.py

# Comparación de modelos (RF vs XGBoost vs Prophet vs ARIMA)
PYTHONIOENCODING=utf-8 python scripts/compare_models.py

# Análisis exploratorio de datos (EDA)
PYTHONIOENCODING=utf-8 python scripts/eda.py

# Dashboard (Flask + interfaz web)
PYTHONIOENCODING=utf-8 python scripts/app.py    # o doble clic en scripts/Iniciar_Dashboard.bat
```

`scripts/train_kmeans_rf.py` (predicción por ítem) es legacy y no forma parte del modelo final; `scripts/run_xgboost.py` y `scripts/run_prophet.py` quedan sustituidos por `scripts/compare_models.py`.

### 3. Pruebas

```bash
PYTHONIOENCODING=utf-8 python -m unittest tests.test_scaler_no_leakage tests.test_kmeans_selection tests.test_rf_tuning tests.test_regression_metrics tests.test_monthly_calendar tests.test_benchmarks tests.test_eda tests.test_bulk_adjustments
```

### 4. Salidas

Los resultados se guardan en `outputs/` (carpeta gitignored, se genera al ejecutar los scripts):

- `outputs/predictions/` — CSV de predicciones (decisiones, probabilidades, ventas predichas)
- `outputs/figures/` — gráficos PNG (selección de k, matriz de confusión, real vs. predicho, comparación de modelos, EDA)
- `outputs/reports/` — resúmenes TXT/CSV (evaluación del modelo, tuning, comparación de modelos, EDA)

---

## Datos

| Archivo | Uso |
|---------|-----|
| `Query_Result_V5.csv` | **Producción** — `train_kmeans_rf_prod`, todos los scripts de evaluación/comparación y el dashboard |
| `Query_Result_V3.csv`, `Query_Result_V4.csv` | Legacy |
| `Query_Result.csv`, `Query_Result_V2.csv` | Legacy, fuente antes de limpieza |

Ver `data/raw/README.md` para el historial de versiones del dataset y `docs/CHANGELOG.md` para el detalle de la limpieza aplicada sobre `Query_Result_V5.csv` (completado de calendario mensual y exclusión de ajustes masivos de inventario).

---

## Documentación

- [docs/BITACORA.md](docs/BITACORA.md) — Mejoras y cambios
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — Historial de versiones
- [docs/INDICE_DOCUMENTACION.md](docs/INDICE_DOCUMENTACION.md) — Índice general

---

## Dependencias

Ver [requirements.txt](requirements.txt).
