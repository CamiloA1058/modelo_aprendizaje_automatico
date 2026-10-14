# Modelo de Predicción de Ventas K-Means + Random Forest

Sistema de predicción de ventas y decisiones de stock (Reforzar / Mantener / Reducir) utilizando K-Means + Random Forest.

**Modelo final (producción)**: `src/ventas_forecast/model.py` (clase `SalesForecastModel`), que se entrena con `scripts/entrenar.py` o desde el dashboard (`dashboard/app.py`).  
**Código heredado** (no es producción): `legacy/` (primeros pipelines, versiones 1.0 y 2.0, benchmarks sueltos de XGBoost/Prophet).  
**Modelos de comparación** (benchmark): `scripts/compare_models.py` (RF vs XGBoost vs Prophet vs ARIMA, sobre la misma partición del modelo final).

Dataset vigente: `data/raw/Query_Result_V6.csv` (facturas de venta reales del taller de los últimos dos años; 48.517 registros diarios, 3.576 productos).

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

## Inicio rápido

**Windows (cliente final)**: doble clic en `Iniciar_Dashboard.bat`. Usa `.venv` si existe, instala las dependencias la primera vez y abre `http://localhost:5000`.

**Desarrollador**:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Solo si se va a ejecutar scripts/compare_models.py:
pip install -r requirements-comparacion.txt
python dashboard/app.py          # dashboard en http://localhost:5000
python scripts/entrenar.py       # entrena y escribe outputs/predictions/PREDICCION_MENSUAL.csv
```

Usar `PYTHONIOENCODING=utf-8` para evitar errores de consola con los caracteres ✓/✅/❌. `scripts/entrenar.py` termina mostrando una figura con matplotlib; en entornos sin pantalla definir `MPLBACKEND=Agg`.

---

## Estructura del proyecto

```
modelo_aprendizaje_automatico/
├── README.md
├── requirements.txt             # Dependencias del modelo y el dashboard
├── requirements-comparacion.txt # Extra para compare_models.py (XGBoost, Prophet, ARIMA)
├── Iniciar_Dashboard.bat        # Lanzador Windows, doble clic
│
├── src/ventas_forecast/         # Código compartido
│   ├── model.py                 # Modelo FINAL: SalesForecastModel
│   ├── paths.py                 # Rutas centralizadas (DEFAULT_DATASET, UPLOADS, ...)
│   ├── eda.py                   # Funciones de análisis exploratorio
│   ├── benchmarks.py            # Utilidades de comparación RF/XGBoost/Prophet/ARIMA
│   └── data/cleaning.py         # DatasetCleaner + remove_bulk_adjustments
│
├── dashboard/                   # Interfaz web
│   ├── app.py                   # Servidor Flask (API JSON)
│   └── dashboard.html           # Interfaz
│
├── scripts/                     # Scripts ejecutables
│   ├── entrenar.py              # Entrena el modelo final y exporta la predicción
│   ├── select_k.py              # Selección de k (codo + silueta) para K-means
│   ├── tune_rf.py               # Ajuste de hiperparámetros del clasificador
│   ├── evaluate_model.py        # Evaluación del clasificador y el regresor
│   ├── compare_models.py        # Comparación RF vs XGBoost vs Prophet vs ARIMA
│   └── eda.py                   # Reporte y figuras de análisis exploratorio
│
├── sql/                         # Extracción desde Firebird
│   ├── extraccion_ventas_v6.sql # Consulta vigente (genera Query_Result_V6.csv)
│   ├── extraer_ventas.py        # Extractor offline a CSV (formato V5)
│   ├── consultas_versiones_anteriores.sql
│   └── exploracion/             # Scripts de exploración (credenciales por variables de entorno)
│
├── data/
│   ├── raw/                     # Query_Result_V6.csv (producción)
│   ├── holdout/                 # Ventas posteriores al entrenamiento (estudio de caso)
│   └── historico/               # V1 a V5 y utilidades antiguas
│
├── tests/                       # Pruebas (unittest)
├── docs/                        # BITACORA.md (histórico) y CHANGELOG.md
├── legacy/                      # Código heredado, fuera de producción
└── outputs/                     # Generado al ejecutar (gitignored): figures/, predictions/, reports/, uploads/
```

---

## Cómo reentrenar con datos nuevos

1. Ejecutar `sql/extraer_ventas.py` con `sql/extraccion_ventas_v6.sql` sobre una copia de la base Firebird (instrucciones en [sql/README.md](sql/README.md)).
2. Guardar el resultado como `data/raw/Query_Result_V6.csv`, o subirlo desde la pestaña "Ejecutar modelo" del dashboard (se guarda en `outputs/uploads/`).
3. Entrenar con `python scripts/entrenar.py` o con el botón de ejecución del dashboard.
4. La predicción queda en `outputs/predictions/PREDICCION_MENSUAL.csv`.

---

## Cómo ejecutar los scripts

Desde la **raíz del proyecto**:

```bash
PYTHONIOENCODING=utf-8 python scripts/entrenar.py        # Modelo final
PYTHONIOENCODING=utf-8 python scripts/select_k.py        # Selección de k
PYTHONIOENCODING=utf-8 python scripts/tune_rf.py         # Hiperparámetros
PYTHONIOENCODING=utf-8 python scripts/evaluate_model.py  # Evaluación
PYTHONIOENCODING=utf-8 python scripts/compare_models.py  # Comparación de modelos
PYTHONIOENCODING=utf-8 python scripts/eda.py             # EDA
```

### Pruebas

```bash
PYTHONIOENCODING=utf-8 python -m unittest discover -s tests
```

### Salidas

Los resultados se guardan en `outputs/` (gitignored):

- `outputs/predictions/` — CSV de predicciones (decisiones, probabilidades, ventas predichas)
- `outputs/figures/` — gráficos PNG (selección de k, matriz de confusión, real vs. predicho, comparación de modelos, EDA)
- `outputs/reports/` — resúmenes TXT/CSV (evaluación, tuning, comparación, EDA)
- `outputs/uploads/` — CSV subidos desde el dashboard

---

## Dashboard

Interfaz web local para usuarios sin conocimientos técnicos. Pestañas: resumen general (KPIs, historial, distribución de decisiones, top de productos), catálogo de predicciones con búsqueda, filtros y descarga del CSV, detalle por producto y ejecución del modelo con subida de CSV y parámetros.

```
Navegador (dashboard/dashboard.html)
        │  HTTP / JSON
        ▼
  dashboard/app.py (Flask, puerto 5000)
        ├── GET  /api/status        Estado del pipeline + mes predicho
        ├── POST /api/run           Lanza el modelo en un hilo secundario
        ├── GET  /api/summary       KPIs, historial, métricas, top productos
        ├── GET  /api/products      Lista paginada con filtros y orden
        ├── GET  /api/product/<id>  Historial + predicción de un producto
        └── GET  /api/download      Descarga PREDICCION_MENSUAL.csv
              ▼
  SalesForecastModel (src/ventas_forecast/model.py)
```

## Uso programático del modelo

```python
import sys
sys.path.insert(0, "src")
from ventas_forecast.model import SalesForecastModel
from ventas_forecast.paths import DEFAULT_DATASET

model = SalesForecastModel(filepath=DEFAULT_DATASET, min_months=6)
model.run()                 # o paso a paso: load_and_clean, build_features, cluster, ...
print(model.results.head())
```

Si no se indica `output_path`, la predicción se escribe en `outputs/predictions/PREDICCION_MENSUAL.csv`. Los parámetros del constructor (`n_clusters`, `lags`, `growth_threshold`, `clf_threshold`, `top_n_products`, `col_map`, ...) están documentados en el docstring de la clase.

---

## Datos

| Ubicación | Uso |
|-----------|-----|
| `data/raw/Query_Result_V6.csv` | **Producción**: el modelo, los scripts y el dashboard |
| `data/historico/` | V1 a V4, solo referencia |

Ver [data/historico/README.md](data/historico/README.md) para el linaje V1 a V5 y [sql/README.md](sql/README.md) para la extracción. El detalle de la limpieza aplicada sobre V5 (completado de calendario mensual y exclusión de ajustes masivos de inventario) está en [docs/CHANGELOG.md](docs/CHANGELOG.md).

---

## Documentación

- [docs/CHANGELOG.md](docs/CHANGELOG.md) — Historial de decisiones y resultados (incluye la tabla de rutas antiguas y nuevas de la reestructuración)
- [docs/BITACORA.md](docs/BITACORA.md) — Bitácora histórica de desarrollo
- [sql/README.md](sql/README.md) — Extracción de datos

## Dependencias

Ver [requirements.txt](requirements.txt) y, para la comparación de modelos, [requirements-comparacion.txt](requirements-comparacion.txt).
