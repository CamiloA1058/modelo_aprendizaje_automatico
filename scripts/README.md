# Modelo de Predicción de Ventas — KMeans + Random Forest

Sistema de predicción de ventas y decisiones de stock.  
**Modelo final**: `scripts/train_kmeans_rf_prod.py` — clase `SalesForecastModel`

---

## Métricas del modelo

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 68.60% |
| Precision | 61.56% |
| Recall    | 90.01% |
| F1-Score  | 73.11% |

Ver [docs/BITACORA.md](docs/BITACORA.md) para el historial completo de métricas.

---

## Lógica de predicción

El modelo **no predice el historial**: predice el mes que todavía no ocurrió.

| Paso | Qué hace |
|------|----------|
| Entrenamiento | Usa todo el historial donde el "mes siguiente" es conocido y real |
| Predicción | Toma el estado actual de cada producto (último mes del dataset) y estima sus ventas del mes siguiente |
| Filtro de calidad | Solo incluye productos con ≥6 meses de historial (configurable con `min_months`) |

**Ejemplo con el dataset actual:**
- Datos disponibles hasta: `2026-03`
- Mes predicho: `2026-04`
- Productos con predicción: `1,288` (de 1,788 con dato en marzo; 500 excluidos por historial insuficiente)
- Predicción cuantitativa (regresión): top 200 productos por volumen de ventas
- Predicción cualitativa (clasificación): todos los 1,288 productos elegibles

---

## Dashboard para el cliente

Interfaz web interactiva pensada para usuarios sin conocimientos técnicos.  
Corre localmente en el navegador — sin instalaciones complejas.

### Archivos del dashboard

| Archivo | Descripción |
|---------|-------------|
| `app.py` | Servidor Flask — API JSON + sirve el dashboard |
| `dashboard.html` | Interfaz web completa |
| `Iniciar_Dashboard.bat` | Lanzador Windows — doble clic |
| `requirements_dashboard.txt` | Dependencias del servidor |

### Cómo usarlo — cliente final (Windows)

1. Poner los 4 archivos del dashboard en la misma carpeta que `train_kmeans_rf_prod.py` y el CSV.
2. Doble clic en `Iniciar_Dashboard.bat`.
3. El navegador se abre en `http://localhost:5000` automáticamente.
4. La primera vez instala dependencias solo (1–2 min).

### Cómo usarlo — desarrollador

```bash
pip install flask flask-cors pandas numpy scikit-learn matplotlib
python app.py
# Abrir http://localhost:5000
```

### Qué ve el cliente en el dashboard

**Resumen general**
- Banner con el mes que se está prediciendo (ej. "Predicción para 2026-04")
- KPIs: productos predichos, cuántos reforzar, cuántos mantener, total ventas predichas con variación vs mes anterior
- Gráfico de línea: historial real de los últimos 18 meses + punto de predicción del mes siguiente
- Donut: distribución Reforzar / Mantener
- Métricas de validación del modelo (Accuracy, Precision, Recall, F1)
- Tabla top 20 productos con mayor probabilidad de crecimiento

**Catálogo de predicciones**
- Tabla completa con búsqueda por código o nombre, filtro por decisión, ordenamiento
- Columnas: ventas base (mes anterior), predicción, variación ↑↓, probabilidad de crecimiento, decisión, segmento
- Paginación de 50 en 50
- Descarga directa del CSV de predicciones

**Detalle por producto** (clic en cualquier fila)
- KPIs individuales: ventas base, predicción, variación, probabilidad
- Gráfico de historial completo con el punto de predicción marcado en azul
- Tabla mes a mes con la fila de predicción destacada

**Ejecutar modelo**
- Subida de CSV nuevo (arrastrar y soltar)
- Parámetros configurables: meses mínimos de historial, umbral de crecimiento, segmentos KMeans
- Barra de progreso y log en tiempo real

### Arquitectura del dashboard

```
Navegador (dashboard.html)
        │  HTTP / JSON
        ▼
  app.py (Flask, puerto 5000)
        │
        ├── GET  /api/status          Estado del pipeline + mes predicho
        ├── POST /api/run             Lanza el modelo en hilo secundario
        ├── GET  /api/summary         KPIs, historial, métricas, top productos
        ├── GET  /api/products        Lista paginada con filtros y ordenamiento
        ├── GET  /api/product/<id>    Historial + predicción de un producto
        └── GET  /api/download        Descarga PREDICCION_MENSUAL.csv
              │
              ▼
  SalesForecastModel (train_kmeans_rf_prod.py)
```

El modelo corre en hilo secundario. El frontend hace polling cada 1.5 s y actualiza log y progreso en tiempo real.

---

## Clase `SalesForecastModel` — uso programático

### Parámetros del constructor

```python
SalesForecastModel(
    filepath,                    # Ruta al CSV
    sep=";",                     # Separador
    encoding="utf-8",
    col_map=None,                # Mapeo de columnas (ver tabla abajo)
    output_path=None,            # Ruta CSV de salida
    n_clusters=3,                # Clusters KMeans
    lags=[1, 2, 3, 6],          # Lags de ventas
    growth_threshold=1.05,       # Umbral crecimiento (+5 %)
    clf_threshold=0.58,          # Umbral probabilidad clasificador
    top_n_products=200,          # Productos para el regresor
    train_ratio=0.8,             # Proporción entrenamiento
    numeric_fmt="dot_comma",     # "dot_comma" | "plain"
    min_months=6,                # Meses mínimos para predecir ← NUEVO
)
```

### Mapeo de columnas (`col_map`)

| Rol interno   | Nombre por defecto | Descripción                      |
|---------------|--------------------|----------------------------------|
| `id`          | `CODIGO`           | Identificador único del producto |
| `description` | `DESCRIPCION`      | Nombre del producto              |
| `year`        | `ANIO`             | Año                              |
| `month`       | `MES`              | Mes (1–12)                       |
| `sales`       | `VENTAS`           | Valor de ventas                  |
| `total_sold`  | `TOTAL_VENDIDO`    | Total vendido                    |
| `avg_price`   | `PRECIO_PROMEDIO`  | Precio promedio                  |
| `frequency`   | `FRECUENCIA`       | Número de transacciones          |

### Pasos del pipeline

```python
model.load_and_clean()      # Carga, parseo, agregación mensual
model.build_features()      # Lags, medias móviles, log, estacionalidad
model.cluster()             # Segmentación KMeans
model.prepare_split()       # Separa historial de entrenamiento vs último mes
model.train_classifier()    # RandomForestClassifier + métricas de validación
model.train_regressor()     # RandomForestRegressor (top N productos)
model.predict_next_month()  # Predicción real del mes siguiente ← NUEVO
model.plot()                # Visualización: historial + punto de predicción
```

O todo de una vez: `model.run()`

### Atributos útiles tras ejecutar

| Atributo | Contenido |
|----------|-----------|
| `model.results` | DataFrame con predicciones del mes siguiente |
| `model.metrics` | Dict con accuracy, precision, recall, f1 |
| `model._ultimo_mes` | Último mes en los datos (Timestamp) |
| `model._mes_pred` | Mes predicho (Timestamp) |
| `model.df` | Historial completo con features |

### Columnas del CSV de salida (`PREDICCION_MENSUAL.csv`)

| Columna | Descripción |
|---------|-------------|
| `CODIGO` | Identificador del producto |
| `DESCRIPCION` | Nombre del producto |
| `mes_predicho` | Mes de la predicción (YYYY-MM) |
| `TOTAL_VENDIDO` | Ventas del último mes (base de comparación) |
| `ventas_predichas` | Predicción cuantitativa (top productos) |
| `prob_crecimiento` | Probabilidad de crecimiento [0–1] |
| `decision` | `"Reforzar stock"` o `"Mantener stock"` |
| `cluster` | Segmento KMeans del producto |
| `FRECUENCIA` | Transacciones en el último mes |
| `PRECIO_PROMEDIO` | Precio promedio en el último mes |

### Ejemplos de uso

**Dataset original:**
```python
from train_kmeans_rf_prod import SalesForecastModel

model = SalesForecastModel(filepath="data/raw/Query_Result_V3.csv")
model.run()
```

**Dataset con columnas en inglés:**
```python
model = SalesForecastModel(
    filepath="sales.csv",
    sep=",",
    numeric_fmt="plain",
    col_map={
        "id": "product_id", "description": "product_name",
        "year": "year", "month": "month",
        "sales": "revenue", "total_sold": "units_sold",
        "avg_price": "avg_price", "frequency": "transactions",
    },
)
model.run()
```

**Ajustar criterio de inclusión y umbral:**
```python
model = SalesForecastModel(
    filepath="data/raw/Query_Result_V3.csv",
    min_months=3,              # incluir productos con ≥3 meses
    growth_threshold=1.10,     # reforzar solo si crece >10%
    top_n_products=100,
)
model.run()
```

**Acceder a resultados sin gráfico:**
```python
model = SalesForecastModel(filepath="data/raw/Query_Result_V3.csv")
model.load_and_clean()
model.build_features()
model.cluster()
model.prepare_split()
model.train_classifier()
model.train_regressor()
model.predict_next_month()

print(f"Prediciendo: {model._mes_pred.strftime('%Y-%m')}")
print(model.results[["CODIGO", "decision", "ventas_predichas", "prob_crecimiento"]])
```

---

## Estructura del proyecto

```
modelo_aprendizaje_automatico/
│
├── README.md
├── requirements.txt
│
├── Iniciar_Dashboard.bat         ← lanzador cliente Windows
├── app.py                        ← servidor Flask del dashboard
├── dashboard.html                ← interfaz web
├── requirements_dashboard.txt    ← dependencias web
│
├── src/ventas_forecast/
│   ├── paths.py
│   └── data/cleaning.py
│
├── scripts/
│   ├── train_kmeans_rf_prod.py   ← modelo FINAL — clase SalesForecastModel
│   ├── train_kmeans_rf.py
│   ├── run_xgboost.py
│   └── run_prophet.py
│
├── data/raw/
│   ├── Query_Result_V3.csv       ← producción
│   ├── Query_Result_V2.csv
│   └── Query_Result.csv
│
├── outputs/
│   ├── predictions/PREDICCION_MENSUAL.csv
│   ├── figures/
│   └── reports/
│
└── docs/
    ├── BITACORA.md
    ├── CHANGELOG.md
    └── INDICE_DOCUMENTACION.md
```

---

## Cómo ejecutar desde terminal

```bash
cd modelo_aprendizaje_automatico
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Modelo directo
python scripts/train_kmeans_rf_prod.py

# Dashboard
python app.py
```

---

## Dependencias

Ver `requirements.txt` y `requirements_dashboard.txt`.
