# Modelo de Predicción de Ventas K-Means + Random Forest

Sistema de predicción de ventas y decisiones de stock utilizando K-Means + Random Forest.

**Modelo final (producción)**: `scripts/train_kmeans_rf_prod.py` — pendiente de revisión  
**Predicción por ítem**: `scripts/train_kmeans_rf.py`  
**Modelos de comparación** (benchmark): `scripts/run_xgboost.py`, `scripts/run_prophet.py`

### Métricas — modelo final (`train_kmeans_rf_prod`)

| Métrica | Valor |
|---------|-------|
| Accuracy | 67.32% |
| Precision | 60.25% |
| Recall | 86.28% |
| F1-Score | 70.95% |

### Métricas — predicción por ítem (`train_kmeans_rf`)

| Métrica | Valor |
|---------|-------|
| Accuracy | 78.03% |
| Precision | 46.24% |
| Recall | 40.82% |
| F1-Score | 43.36% |

Ver [docs/BITACORA.md](docs/BITACORA.md) para reportes detallados.

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
│   └── data/cleaning.py          # DatasetCleaner
│
├── scripts/                      # Scripts ejecutables
│   ├── train_kmeans_rf_prod.py   # Modelo FINAL (producción)
│   ├── train_kmeans_rf.py        # Predicción por ítem
│   ├── run_xgboost.py            # Comparación (benchmark)
│   ├── run_prophet.py            # Comparación (benchmark)
│   ├── generar_reportes.py       # Reportes para cliente
│   ├── monitor_prophet.py        # Monitor de progreso Prophet
│   └── debug_capture.py          # Depuración de ejecución
│
├── models/legacy/                # Versiones históricas (v1.0, v2.0, ARIMA demo)
│
├── data/raw/                     # Datos fuente
│   ├── Query_Result.csv          # Exportación v1
│   ├── Query_Result_V2.csv       # Con columnas diarias
│   ├── Query_Result_V3.csv       # Limpio (producción)
│   ├── ventas.csv
│   └── README.md                 # Documentación de datasets
│
├── outputs/                      # Generado al ejecutar (gitignored)
│   ├── figures/                  # Gráficos PNG
│   ├── predictions/              # CSV de predicciones
│   └── reports/                  # Resúmenes TXT
│
├── docs/                         # Documentación del proyecto
│   ├── BITACORA.md
│   ├── CHANGELOG.md
│   ├── INDICE_DOCUMENTACION.md
│   └── VERSION_SUMMARY.md
│
├── tests/                        # Pruebas
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

Desde la **raíz del proyecto**:

```bash
# Modelo FINAL — producción
python scripts/train_kmeans_rf_prod.py

# Predicción por ítem (producto a producto)
python scripts/train_kmeans_rf.py

# Modelos de comparación (no son el modelo final)
python scripts/run_xgboost.py
python scripts/run_prophet.py
```

### 3. Salidas

Los resultados se guardan en `outputs/`:

- `outputs/predictions/` — CSV (predicciones, decisiones, métricas)
- `outputs/figures/` — gráficos PNG
- `outputs/reports/` — resúmenes ejecutivos TXT

---

## Datos

| Archivo | Uso |
|---------|-----|
| `Query_Result_V3.csv` | **Producción** — `train_kmeans_rf_prod` y `train_kmeans_rf` |
| `Query_Result.csv` | Legacy, comparaciones (XGBoost), tests |
| `Query_Result_V2.csv` | Fuente antes de limpieza |

Ver `data/raw/README.md` para el historial de versiones del dataset.

---

## Documentación

- [docs/BITACORA.md](docs/BITACORA.md) — Mejoras y cambios
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — Historial de versiones
- [docs/INDICE_DOCUMENTACION.md](docs/INDICE_DOCUMENTACION.md) — Índice general

---

## Dependencias

Ver [requirements.txt](requirements.txt).
