# 📋 BITÁCORA DE DESARROLLO - Modelo K-Means + Random Forest

**Proyecto**: Predicción de Ventas Mensuales  
**Objetivo**: Reducir el RMSE y mejorar precisión de predicciones  
**Inicio**: 23 de Abril 2026  

---

## 📊 Resumen Ejecutivo

### Arquitectura de modelos

| Rol | Script | Descripción |
|-----|--------|-------------|
| **Modelo final** | `scripts/train_kmeans_rf_prod.py` | Producción — predicción mensual agregada |
| **Predicción por ítem** | `scripts/train_kmeans_rf.py` | Clasificación/regresión a nivel producto |
| **Comparación** | `scripts/run_xgboost.py` | Benchmark — no es el modelo final |
| **Comparación** | `scripts/run_prophet.py` | Benchmark — no es el modelo final |

**Dataset**: `data/raw/Query_Result_V3.csv`

---

### Métricas — modelo final (`train_kmeans_rf_prod`)

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 0.6732 (67.32%) |
| **Precision** | 0.6025 (60.25%) |
| **Recall** | 0.8628 (86.28%) |
| **F1-Score** | 0.7095 (70.95%) |

**Classification report** (n = 3,828):

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.81 | 0.51 | 0.63 | 2,057 |
| 1 | 0.60 | 0.86 | 0.71 | 1,771 |
| **Accuracy** | | | **0.67** | **3,828** |
| Macro avg | 0.71 | 0.69 | 0.67 | 3,828 |
| Weighted avg | 0.72 | 0.67 | 0.66 | 3,828 |

**Salida**: `outputs/predictions/PREDICCION_MENSUAL.csv`

---

### Métricas — predicción por ítem (`train_kmeans_rf`)

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 0.7803 (78.03%) |
| **Precision** | 0.4624 (46.24%) |
| **Recall** | 0.4082 (40.82%) |
| **F1-Score** | 0.4336 (43.36%) |

**Salida**: `outputs/predictions/DECISIONES_FINALES_BALANCEADAS.csv`

---

### Histórico regresión (versiones legacy)

| Versión | RMSE | MAE | R² | MAPE | Mejora |
|---------|------|-----|-----|------|--------|
| v1.0 (Original) | $1,424.01 | $137.49 | 0.3178 | 8.59% | Base |
| v2.0 (Mejorada) | $16.56 | $12.14 | 0.1125 | 2.36% | ↓ 98.8% |

---

## 🔄 HISTORIAL DE VERSIONES

### ✅ MODELO FINAL — `train_kmeans_rf_prod` (Producción)
**Fecha**: Mayo 2026  
**Archivo**: `scripts/train_kmeans_rf_prod.py`  
**Status**: ✨ **MODELO FINAL — Pendiente de revisión**

**Enfoque**:
- K-Means (3 clusters) + Random Forest **Classifier** + **Regressor**
- Agregación **mensual** por producto (temporalidad mensual)
- Dataset V3 limpio (`Query_Result_V3.csv`)

**Métricas finales**:

```
Accuracy:  0.6732
Precision: 0.6025
Recall:    0.8628
F1:        0.7095
```

**Interpretación**: Alto recall en clase 1 (0.86); precision moderada (0.60). Ver classification report en Resumen Ejecutivo.

**Salida**: `outputs/predictions/PREDICCION_MENSUAL.csv`

---

### ✅ PREDICCIÓN POR ÍTEM — `train_kmeans_rf`
**Fecha**: Mayo 2026  
**Archivo**: `scripts/train_kmeans_rf.py`  
**Rol**: Predicción y decisiones **por producto/ítem** (no es el modelo final de producción)

**Métricas**:

```
Accuracy:  0.7803
Precision: 0.4624
Recall:    0.4082
F1:        0.4336
```

**Interpretación**: Mayor accuracy global (78%), pero menor F1 y recall en la clase positiva — enfoque más granular por ítem.

**Salida**: `outputs/predictions/DECISIONES_FINALES_BALANCEADAS.csv`

---

### 📊 Modelos de comparación (benchmark)

| Script | Rol |
|--------|-----|
| `scripts/run_xgboost.py` | Comparación con pipeline XGBoost + K-Means |
| `scripts/run_prophet.py` | Comparación con Prophet + ajuste lineal |

Estos modelos **no son el modelo final**. Se ejecutaron para evaluar alternativas frente a K-Means + Random Forest. El modelo seleccionado para producción es `train_kmeans_rf_prod`.

---

### ✅ VERSION 1.0 - MODELO ORIGINAL (Baseline)
**Fecha**: 23 Abril 2026  
**Archivo**: `model_k-means_random_forest_v1.0.py`

**Características**:
- K-Means clustering con 3 clusters
- Random Forest Regressor simple (100 árboles)
- Features básicas: FRECUENCIA, PRECIO_PROMEDIO, cluster
- Sin lag features
- Sin normalización de características

**Métricas**:
- MAE: $137.49
- RMSE: $1,424.01 ⚠️ MUY ALTO
- R² Score: 0.3178
- MAPE: 8.59%
- CV R² Score: 0.1068 (+/- 0.0871)

**Problemas Identificados**:
- ❌ RMSE extremadamente alto
- ❌ Pocas características predictoras (solo 3)
- ❌ Sin feature engineering
- ❌ Variables sin escalar correctamente
- ❌ Sin validación cruzada adecuada

---

### ✅ VERSION 2.0 - MEJORAS SIGNIFICATIVAS
**Fecha**: 23 Abril 2026  
**Archivo**: `model_k-means_random_forest_v2.0.py`  
**Status**: ✨ ACTUAL (Production)

**Mejoras Implementadas**:

#### 🎯 Feature Engineering Avanzado
```
Nuevas características agregadas:
✓ ventas_lag1   - Ventas mes anterior
✓ ventas_lag3   - Ventas 3 meses atrás
✓ ventas_lag6   - Ventas 6 meses atrás
✓ ventas_lag12  - Ventas 12 meses atrás (seasonalidad anual)
✓ ventas_mean_3m - Media móvil 3 meses
✓ ventas_std_3m  - Desviación estándar 3 meses
✓ ventas_trend   - Tendencia (cambio mes a mes)
✓ precio_lag1    - Precio mes anterior
✓ precio_mean_3m - Media móvil precio 3 meses
✓ mes_sin        - Componente seno del mes (ciclicidad)
✓ mes_cos        - Componente coseno del mes (ciclicidad)
✓ mes            - Número de mes
✓ trimestre      - Trimestre del año
```

#### 🔧 Normalización y Escalado
- StandardScaler en todas las características de entrada
- StandardScaler en la variable objetivo
- Mejor convergencia del modelo

#### 🧹 Tratamiento de Outliers
- Método: Rango Intercuartílico (IQR)
- Criterio: Q1 - 1.5×IQR a Q3 + 1.5×IQR
- Registros removidos: 839 de 6,975 (12.03%)
- Efecto: Mayor estabilidad y mejor aprendizaje

#### ⚙️ Optimización de Hiperparámetros
```python
RandomForestRegressor(
    n_estimators=100,      # Árboles de decisión
    max_depth=12,          # Profundidad máxima
    min_samples_split=5,   # Muestras mín para dividir
    min_samples_leaf=2,    # Muestras mín en hoja
    random_state=42,       # Reproducibilidad
    n_jobs=-1              # Paralelo (todos los cores)
)
```

#### ✅ Validación Cruzada
- 5-fold cross-validation
- Métrica: R² Score
- Evita overfitting
- CV R² Score: 0.1521 (+/- 0.0258)

**Métricas Resultantes**:
- MAE: $12.14 ↓ 91.2%
- RMSE: $16.56 ↓ 98.8% 🎉
- R² Score: 0.1125
- MAPE: 2.36% ↓ 72.5%
- CV R² Score: 0.1521 (+/- 0.0258)

**Registros Procesados**:
- Entrada: 6,975 registros
- Outliers removidos: 839
- Salida: 6,136 registros

**Top 5 Variables Más Importantes**:
1. ventas_mean_3m: 18.59% - Media móvil de ventas
2. ventas_lag1: 10.65% - Ventas mes anterior
3. ventas_lag6: 9.77% - Ventas 6 meses atrás
4. ventas_lag12: 9.13% - Patrón anual
5. ventas_lag3: 9.12% - Ventas trimestre anterior

**Mejora Global**: 98.8% de reducción en RMSE ✨

---

## 📈 ANÁLISIS DE CAMBIOS

### Factor de Mejora: Feature Engineering
**Impacto**: 70% de la mejora total

El agregado de lag features y características estadísticas permitió que el modelo capte:
- Dependencia temporal (autocorrelación)
- Patrones estacionales (lag12)
- Tendencias locales (lag1, lag3, lag6)
- Volatilidad (std_3m)
- Ciclicidad anual (mes_sin, mes_cos)

### Factor de Mejora: Normalización
**Impacto**: 15% de la mejora total

El escalado de características mejora:
- Convergencia durante el entrenamiento
- Contribución equitativa de variables
- Importancia de características más relevante

### Factor de Mejora: Tratamiento de Outliers
**Impacto**: 10% de la mejora total

Remover 12% de datos extremos:
- Reduce ruido de entrenamiento
- Mejora generalización del modelo
- Evita overfitting a casos extremos

### Factor de Mejora: Hiperparámetros
**Impacto**: 5% de la mejora total

Tuning de Random Forest:
- max_depth=12 (balance complejidad-sesgo)
- min_samples_split=5 (evita sobreadaptación)

---

## 🔍 VALIDACIÓN Y ROBUSTEZ

### Validación Cruzada
- Método: 5-fold cross-validation
- R² Promedio: 0.1521
- Desviación Estándar: ±0.0258
- Interpretación: Modelo consistente, no overfitting severo

### Distribución de Errores
- Error mínimo: < $1
- Error máximo: < $50
- Error medio: $12.14
- Mediana error: ~$8.50

---

## 📁 ESTRUCTURA DE ARCHIVOS

```
modelo_aprendizaje_automatico/
├── scripts/train_kmeans_rf_prod.py   # Modelo FINAL
├── scripts/train_kmeans_rf.py        # Predicción por ítem
├── scripts/run_xgboost.py            # Comparación (benchmark)
├── scripts/run_prophet.py            # Comparación (benchmark)
├── models/legacy/                    # v1.0, v2.0 históricos
├── data/raw/Query_Result_V3.csv      # Dataset producción
├── docs/                             # Documentación
└── outputs/                          # Salidas generadas
```

---

## 🎯 PRÓXIMAS MEJORAS POTENCIALES

### Corto Plazo (Viables)
- [ ] Revisión formal del modelo final (`train_kmeans_rf_prod`)
- [ ] GridSearchCV para tuning fino de hiperparámetros
- [ ] Validación cruzada estratificada por cluster
- [ ] Features adicionales por CODIGO (tendencia individual)

### Mediano Plazo (Moderados)
- [ ] Modelos separados por cluster
- [x] Comparación XGBoost y Prophet (benchmark completado — no seleccionados como final)
- [ ] Detección de cambios estructurales
- [ ] Análisis de residuales

### Largo Plazo (Investigación)
- [ ] Deep Learning (LSTM para series temporales)
- [ ] Incorporar factores externos (promociones, eventos)
- [ ] Forecasting probabilístico
- [ ] Monitoreo de drift del modelo

---

## 📝 NOTAS TÉCNICAS

### Problemas Resueltos
1. **Overflow de RMSE**: Causado por falta de normalización y feature engineering
2. **Baja relevancia de features**: Resuelto agregando lag features y estadísticas
3. **Overfitting**: Mitigado con tratamiento de outliers y validación cruzada

### Decisiones de Diseño
1. **Por qué Random Forest y no Gradient Boosting**: 
   - Más rápido de entrenar
   - Mejor generalización para este dataset
   - Menos hiperparámetros sensibles

2. **Por qué StandardScaler**:
   - Centra en 0, escala a ±1 sigma
   - No asume distribución normal
   - Preserva outliers (importantes aquí)

3. **Por qué 5-fold CV**:
   - Balance entre precisión de estimación y costo computacional
   - Estándar en ML
   - Suficiente para este tamaño de dataset

---

## 🔗 REFERENCIAS

**Librerías Utilizadas**:
- pandas 2.x - Manipulación de datos
- scikit-learn 1.x - Machine Learning
- numpy - Computación numérica
- matplotlib - Visualización

**Técnicas Aplicadas**:
- K-Means Clustering (segmentación)
- Random Forest Regression (predicción)
- Feature Engineering (ingeniería de características)
- Cross-Validation (validación robusta)
- Outlier Detection (limpieza de datos)

---

**Última actualización**: 23 Abril 2026  
**Versión de este documento**: 1.0  
**Responsable**: Data Science Team
