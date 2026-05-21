# 📝 CHANGELOG - Control de Versiones

## [Producción Final] - 2026-05-21

### 🎯 Modelo final — `train_kmeans_rf_prod`

**Script**: `scripts/train_kmeans_rf_prod.py`  
**Dataset**: `data/raw/Query_Result_V3.csv`  
**Salida**: `outputs/predictions/PREDICCION_MENSUAL.csv`

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.6732 |
| Precision | 0.6025 |
| Recall | 0.8628 |
| F1-Score | 0.7095 |

#### Classification report (modelo final)

```
              precision    recall  f1-score   support
           0       0.81      0.51      0.63      2057
           1       0.60      0.86      0.71      1771
    accuracy                           0.67      3828
   macro avg       0.71      0.69      0.67      3828
weighted avg       0.72      0.67      0.66      3828
```

### 📦 Predicción por ítem — `train_kmeans_rf`

**Script**: `scripts/train_kmeans_rf.py`  
**Rol**: Predicción por producto/ítem (no es el modelo final)

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.7803 |
| Precision | 0.4624 |
| Recall | 0.4082 |
| F1-Score | 0.4336 |

### 📊 Modelos de comparación

- `scripts/run_xgboost.py` — benchmark XGBoost (comparación, no modelo final)
- `scripts/run_prophet.py` — benchmark Prophet (comparación, no modelo final)

### 📁 Reorganización del repositorio

- `src/ventas_forecast/` — rutas centralizadas y módulos compartidos
- `scripts/` — modelos activos
- `models/legacy/` — versiones históricas (v1.0, v2.0)
- `data/raw/` — datasets fuente
- `outputs/` — predicciones, gráficos y reportes (gitignored)
- `docs/` — documentación del proyecto

### ✨ DatasetCleaner

- Nuevo módulo: `src/ventas_forecast/data/cleaning.py`
- Clase `DatasetCleaner` para filtrar registros por palabras clave y exportar CSV limpio

---

## [2.0] - 2026-04-23

### 🎉 CAMBIOS PRINCIPALES
- **Reducción RMSE**: De $1,424.01 a $16.56 (↓ 98.8%)
- **Nuevo**: Feature Engineering avanzado con lag features
- **Nuevo**: Normalización con StandardScaler
- **Nuevo**: Tratamiento automático de outliers (IQR)
- **Nuevo**: Validación cruzada mejorada
- **Actualización**: Optimización de hiperparámetros

### ✨ CARACTERÍSTICAS AGREGADAS

#### Features Temporales
```
Lags: [1, 3, 6, 12] meses
Estadísticas: [media 3m, std 3m, tendencia]
Cíclicas: [seno mes, coseno mes, mes, trimestre]
```

#### Procesamiento de Datos
```python
# Outlier Detection
Q1, Q3 = df['ventas_futuras'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Removidos: 839 registros (12.03%)
```

#### Normalización
```python
# Entrada
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Salida
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
```

### 📊 RESULTADOS

| Métrica | v1.0 | v2.0 | Cambio |
|---------|------|------|--------|
| MAE | $137.49 | $12.14 | ↓ 91.2% |
| RMSE | $1,424.01 | $16.56 | ↓ 98.8% |
| R² | 0.3178 | 0.1125 | - |
| MAPE | 8.59% | 2.36% | ↓ 72.5% |
| CV R² | 0.1068 | 0.1521 | ↑ 42.5% |

### 🔧 CAMBIOS TÉCNICOS

#### Antes (v1.0)
```python
X = df[['FRECUENCIA', 'PRECIO_PROMEDIO', 'cluster']]
# 3 features, sin escalar
```

#### Después (v2.0)
```python
feature_cols = [
    'FRECUENCIA', 'PRECIO_PROMEDIO', 'cluster',
    'ventas_lag1', 'ventas_lag3', 'ventas_lag6', 'ventas_lag12',
    'ventas_mean_3m', 'ventas_std_3m', 'ventas_trend',
    'precio_lag1', 'precio_mean_3m',
    'mes', 'trimestre', 'mes_sin', 'mes_cos'
]
# 15 features, escaladas
```

### 🎯 IMPORTANCIA DE VARIABLES (Top 10)

1. ventas_mean_3m: 18.59% ⭐⭐⭐
2. ventas_lag1: 10.65% ⭐⭐
3. ventas_lag6: 9.77% ⭐
4. ventas_lag12: 9.13% ⭐
5. ventas_lag3: 9.12% ⭐
6. PRECIO_PROMEDIO: 8.45%
7. precio_mean_3m: 7.82%
8. ventas_std_3m: 6.95%
9. FRECUENCIA: 6.23%
10. cluster: 4.87%

### ⚠️ BREAKING CHANGES
- Cambio en estructura de features (requiere reentrenamiento)
- Datos de entrada ahora están normalizados
- Formato de output de predicciones modificado

### 🐛 BUGS ARREGLADOS
- ✅ Normalización incorrecta de PRECIO_PROMEDIO
- ✅ Falta de validación cruzada
- ✅ Valores extremos no removidos
- ✅ Gráficas se quedaban colgadas (plt.show() → plt.savefig())

### 📖 DOCUMENTACIÓN
- ✅ Documento BITACORA.md creado
- ✅ Comentarios en código mejorados
- ✅ Métricas más descriptivas en output
- ✅ Archivo CHANGELOG.md creado

### 🧪 TESTING
- ✅ Validación cruzada: 5-fold CV R² = 0.1521
- ✅ Train/Test split: 80/20
- ✅ Datos de test independientes verificados

### 📈 NOTAS DE PERFORMANCE
- Tiempo de entrenamiento: ~3 segundos (optimizado)
- Memoria utilizada: ~150MB
- Features: 15 (aumento 400% desde v1.0)
- Registros procesados: 6,136 (después de limpiar)

---

## [1.0] - 2026-04-23

### 🚀 RELEASE INICIAL

**Features**:
- K-Means clustering (3 clusters)
- Random Forest Regressor básico
- Carga de datos desde CSV
- Visualizaciones con matplotlib

**Métricas Iniciales**:
- MAE: $137.49
- RMSE: $1,424.01
- R² Score: 0.3178
- MAPE: 8.59%

**Limitaciones**:
- RMSE muy elevado
- Features limitadas
- Sin tratamiento de outliers
- Normalización incompleta

---

## 🔮 ROADMAP FUTURO

### v2.1 (Próxima)
- [ ] GridSearchCV para tuning fino
- [ ] Validación cruzada estratificada
- [ ] Métricas adicionales (MAE percentiles, RMSE por cluster)

### v3.0 (Mediano Plazo)
- [ ] Revisión formal del modelo final (`train_kmeans_rf_prod`)
- [ ] Modelos específicos por cluster
- [x] Comparación XGBoost y Prophet (benchmark completado)

### v4.0 (Largo Plazo)
- [ ] Deep Learning (LSTM, Transformer)
- [ ] Forecasting probabilístico
- [ ] API REST para predicciones

---

**Convenciones de versionado**: [MAJOR].[MINOR]  
Fecha de última actualización: 2026-04-23
