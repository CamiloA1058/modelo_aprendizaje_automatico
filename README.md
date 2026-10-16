# Modelo de Predicción de Ventas K-Means + Random Forest

Sistema de predicción de ventas y decisiones de stock (Reforzar / Mantener / Reducir) utilizando K-Means + Random Forest.

**Modelo final (producción)**: `src/ventas_forecast/model.py` (clase `SalesForecastModel`), que se entrena con `scripts/entrenar.py` o desde el dashboard (`dashboard/app.py`).  
**Código heredado** (no es producción): `legacy/` (primeros pipelines, versiones 1.0 y 2.0, benchmarks sueltos de XGBoost/Prophet).  
**Modelos de comparación** (benchmark): `scripts/compare_models.py` (RF vs XGBoost vs Prophet vs ARIMA, sobre la misma partición del modelo final).

Dataset vigente: `data/raw/Query_Result_V6.csv` (facturas de venta reales del taller de los últimos dos años, sin servicios; 48.028 registros diarios, 3.584 productos).

Regla de validación: nada de lo reportado se elige con los datos sobre los que se reporta. Hiperparámetros, peso de la mezcla, umbral de venta y cuantiles del stock se eligen solo con el entrenamiento (validación cruzada mensual o validación interna temporal); la prueba interna (último 20 % de los meses de cada producto) solo reporta y el estudio de caso es la evaluación final (entrada "[Validación] V1" de `docs/CHANGELOG.md`).

### Métricas — clasificador (`outputs/reports/evaluacion_modelo.txt`, prueba interna n=6.242)

Métricas principales (robustas al desbalance de clases; Reducir = 79,80 % de la prueba):

| Métrica | Modelo | Línea base (siempre "Reducir") |
|---|---|---|
| F1 macro | 0,4502 | 0,2959 |
| Exactitud balanceada | 0,4974 | 0,3333 |

Métricas ponderadas por soporte de clase (referencia):

| Métrica | Modelo | Línea base (siempre "Reducir") |
|---|---|---|
| Exactitud | 0,6831 | 0,7980 |
| F1 ponderado | 0,7202 | 0,7083 |

### Métricas — regresor (`outputs/reports/evaluacion_modelo.txt`, prueba interna, top productos, n=4.472)

| Métrica | Modelo (RF) | Mezcla (w = 0,0) | Ingenuo | Media_3 |
|---|---|---|---|---|
| MAE (COP) | 232.349 | 314.974 | 427.404 | 314.974 |
| RMSE (COP) | 907.724 | 930.639 | 1.254.946 | 930.639 |
| R² | 0,4213 | 0,3917 | −0,1061 | 0,3917 |
| MASE | 1,0382 | 1,4074 | 1,9097 | 1,4074 |
| Razón de sumas | 0,5937 | 1,2919 | — | 1,2919 |

Con w elegido en la validación interna (w = 0,0) la mezcla de abril es la media de 3 meses.

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
PYTHONIOENCODING=utf-8 python scripts/reporte_prediccion_vs_real.py  # Excel predicción vs real por producto (requiere openpyxl)
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

Interfaz web local para usuarios sin conocimientos técnicos. Pestañas: resumen general (KPIs, historial, distribución de decisiones, top de productos), vista por grupo de producto, catálogo de predicciones con búsqueda, filtros y descarga del CSV, detalle por producto y ejecución del modelo con subida de CSV y parámetros.

**Predicción combinada.** El monto predicho por producto (`ventas_predichas`) es `w x RF + (1 - w) x media_3`, donde `media_3` es el promedio de los últimos 3 meses del producto y `w` (0, 0,1 ... 1) se elige con la validación interna dentro del entrenamiento (un regresor temporal ajustado con los primeros meses de entrenamiento de cada producto predice sus últimos meses de entrenamiento), minimizando |suma predicha - suma real|; la prueba interna solo reporta. El RF puro se conserva en `ventas_predichas_rf`. El KPI "Ventas predichas" muestra la suma de la mezcla (con `w` en porcentaje), el detalle del producto agrega "Solo modelo: $..." y el CSV descargable trae ambas columnas. En la última corrida real `w` = 0,0 (en la validación interna el RF predice el 23,6 % de la suma real y la media de 3 meses el 94,8 %), así que el monto combinado de abril es la media de 3 meses; en la prueba interna la razón suma predicha/real es 0,59 (RF) y 1,29 (mezcla) (ver `outputs/reports/evaluacion_modelo.txt` y la entrada "[Validación] V1" de `docs/CHANGELOG.md`).

**Estimación por promedio.** Los productos que no vendieron en el último mes pero sí en alguno de los 2 meses anteriores (con al menos `min_months` meses con ventas y dentro de los 1.900 con más ventas) aparecen aparte: su monto es el promedio de los últimos 3 meses (`media_3`), no una predicción del modelo, y su decisión es "Sin recomendación (estimación por promedio)", porque el clasificador es peor que la clase mayoritaria en ellos. Se calculan en `SalesForecastModel.estimar_por_promedio()` (activo por defecto, `estimacion_por_promedio=False` lo desactiva) y el CSV trae la columna `origen_estimacion` ("modelo" / "promedio"). El KPI "Ventas predichas" suma ambos grupos y detalla cada uno; Reforzar/Mantener/Reducir y la dona cuentan solo los productos del modelo, y un KPI aparte muestra cuántos productos quedan sin recomendación. El catálogo tiene el filtro "Sin recomendación (estimación por promedio)". En abril de 2026 (V6): 697 productos con decisión, 348 por promedio, 247,0 M + 60,6 M = 307,6 M (entrada "[Sistema] Estimación por promedio..." de `docs/CHANGELOG.md`).

**Probabilidad de venta.** Para todos los productos (los del modelo y los de estimación por promedio) el sistema calcula la probabilidad de que tengan ventas el mes predicho y la traduce a "Se venderá: Sí/No". `SalesForecastModel.agregar_probabilidad_venta()` (activo por defecto, `probabilidad_venta=False` lo desactiva) entrena un Random Forest binario solo con la partición de entrenamiento y elige el umbral con la validación interna dentro de ese entrenamiento (la exactitud que muestra el dashboard es la de la prueba interna, 79,2 % en abril); el CSV trae las columnas `prob_venta` (0 a 1) y `se_vendera`. El resumen muestra el KPI "Probablemente se venderán: N de M productos" (con el umbral y la exactitud en la prueba interna), las tablas del resumen y el catálogo tienen las columnas "Prob. de venta" y "Se venderá", el catálogo permite filtrar por "Se venderá / No se venderá" y ordenar por probabilidad de venta, y el detalle del producto la muestra también para los de promedio. En el estudio de caso (abril a septiembre de 2026) acierta si habrá venta en 70,1 % de los producto-mes agrupados (69,0 % a 71,6 % por mes); los montos no cambian (entrada "[Sistema] Probabilidad de venta en el dashboard" de `docs/CHANGELOG.md`).

**Stock sugerido (70 %).** Para cada producto con monto predicho el sistema calcula un nivel de abastecimiento: si se tiene en inventario hasta ese valor, en los meses de prueba (abril a septiembre de 2026) se cubrió la demanda de cerca de 7 de cada 10 productos (71,8 % agrupado, incluye los que no vendieron; 53,0 % entre los que vendieron; stock total 1,45 veces lo vendido). `SalesForecastModel.agregar_stock_sugerido()` (activo por defecto, `stock_sugerido=False` lo desactiva, `nivel_servicio=0.70`) corrige la predicción con el cuantil 70 % de los errores de la validación interna (dentro del entrenamiento) por cubeta de probabilidad de venta; el CSV trae `stock_sugerido` (COP) y `stock_sugerido_unidades`, y el dashboard la columna "Stock sugerido (70 %)" en Productos, detalle y Grupos. No cambia decisiones, montos ni totales (entrada "[Sistema] Stock sugerido por producto" de `docs/CHANGELOG.md`; la hoja de producto de `prediccion_vs_real.xlsx` trae "¿Cubrió la demanda?").

**Vista por grupo de producto.** La vista "Grupos" suma las predicciones por producto de cada grupo del catálogo `data/raw/grupos_productos.csv` (la línea o proveedor con la que compra el taller): productos con monto, cuántos se venderán, predicción del grupo, ventas del mes anterior, variación y porcentaje del total, con buscador; un clic en un grupo abre "Productos" filtrado por él. `SalesForecastModel.agregar_resumen_grupo()` (activo por defecto, `resumen_por_grupo=False` lo desactiva) solo lee los resultados y no cambia ninguna predicción, decisión ni total. En abril a septiembre de 2026 esta suma acertó el 68,9 % del valor vendido por grupo (75,6 % en los 20 grupos con más ventas); úsela como guía para planear pedidos por línea (entrada "[Sistema] Vista por grupo de producto" de `docs/CHANGELOG.md`). La hoja "Grupos" de `prediccion_vs_real.xlsx` trae la comparación contra lo real mes a mes.

```
Navegador (dashboard/dashboard.html)
        │  HTTP / JSON
        ▼
  dashboard/app.py (Flask, puerto 5000)
        ├── GET  /api/status        Estado del pipeline + mes predicho
        ├── POST /api/run           Lanza el modelo en un hilo secundario
        ├── GET  /api/summary       KPIs, pred_total (modelo + promedio), pred_total_modelo, pred_total_promedio, n_promedio, n_se_vendera, sale_threshold, pred_total_rf, blend_weight, historial, métricas, top productos
        ├── GET  /api/products      Lista paginada con filtros (incluye grupo) y orden
        ├── GET  /api/grupos        Suma de predicciones por grupo de producto
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
