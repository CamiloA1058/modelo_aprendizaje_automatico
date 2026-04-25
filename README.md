# 📊 Modelo de Predicción de Ventas K-Means + Random Forest

## 🎯 Descripción del Proyecto

Sistema de predicción de ventas mensuales utilizando clustering de clientes con K-Means y regresión con Random Forest. El modelo predice ventas futuras basadas en históricos, patrones estacionales y características de cada producto.

**Versión Actual**: 2.0 (Production)  
**Última actualización**: 23 de Abril 2026  
**Estado**: ✅ Production Ready

---

## 📈 Rendimiento

| Métrica | Valor | Mejora |
|---------|-------|--------|
| **RMSE** | $16.56 | ↓ 98.8% desde v1.0 |
| **MAE** | $12.14 | ↓ 91.2% desde v1.0 |
| **R² Score** | 0.1125 | - |
| **MAPE** | 2.36% | ↓ 72.5% desde v1.0 |
| **Validación Cruzada** | 0.1521 (±0.0258) | ↑ 42.5% desde v1.0 |

---

## 📁 Estructura de Archivos

```
modelo_aprendizaje_automatico/
│
├── 📄 Documentación
│   ├── BITACORA.md              ← Documentación completa de mejoras
│   ├── CHANGELOG.md             ← Historial de versiones
│   └── README.md                ← Este archivo
│
├── 🐍 Scripts de Modelo
│   ├── model_k-means_random_forest.py        (Versión actual - v2.0)
│   ├── model_k-means_random_forest_v1.0.py   (Baseline original)
│   ├── model_k-means_random_forest_v2.0.py   (Versión mejorada documentada)
│   └── run_model.py             ← Script auxiliar
│
├── 📊 Datos
│   ├── Query_Result.csv         ← Datos fuente principal
│   └── ventas.csv               ← Datos complementarios
│
├── 🏢 Base de Datos
│   └── bd/                      ← Backups de Firebird
│       └── bd_rc/
│           ├── 1.sql
│           ├── PASANTIAS_BD_RC_BK.FDB
│           └── [archivos adicionales]
│
└── 📈 Salidas Generadas
    ├── grafico_clusters.png
    ├── grafico_cluster_summary.png
    ├── grafico_prediccion.png
    ├── grafico_importancia.png
    └── grafico_evolucion.png
```

---

## 🚀 Cómo Usar

### 1. Preparar Entorno

```bash
# Activar virtual environment
cd modelo_aprendizaje_automatico
venv\Scripts\activate

# Instalar dependencias (si es necesario)
pip install pandas numpy scikit-learn matplotlib
```

### 2. Ejecutar el Modelo

```bash
# Ejecutar versión actual (v2.0)
python model_k-means_random_forest.py

# Ejecutar versión documentada
python model_k-means_random_forest_v2.0.py

# Ejecutar baseline original (para referencia)
python model_k-means_random_forest_v1.0.py
```

### 3. Interpretar Resultados

El script genera:

**Consola**:
- Registro de cada paso procesado
- Métricas finales (MAE, RMSE, R², MAPE)
- Top 5 variables más importantes
- Muestra de predicciones

**Visualizaciones** (archivos PNG):
- `grafico_clusters.png` - Segmentación K-Means
- `grafico_cluster_summary.png` - Promedios por cluster
- `grafico_prediccion.png` - Predicciones vs Valores Reales
- `grafico_importancia.png` - Importancia de features
- `grafico_evolucion.png` - Evolución de ventas de un producto

---

## 🔍 Características del Modelo (v2.0)

### Features Utilizados (15 variables)

**Básicas**:
- `FRECUENCIA` - Número de transacciones
- `PRECIO_PROMEDIO` - Precio unitario
- `cluster` - Segmento K-Means (0, 1, 2)

**Lag Features** (Histórico):
- `ventas_lag1` - Ventas mes anterior
- `ventas_lag3` - Ventas 3 meses atrás
- `ventas_lag6` - Ventas 6 meses atrás
- `ventas_lag12` - Ventas 12 meses atrás (seasonalidad)

**Estadísticas**:
- `ventas_mean_3m` - Media móvil 3 meses
- `ventas_std_3m` - Desviación estándar 3 meses
- `ventas_trend` - Tendencia mes a mes
- `precio_lag1` - Precio mes anterior
- `precio_mean_3m` - Media móvil precio 3 meses

**Temporales/Cíclicas**:
- `mes` - Número de mes (1-12)
- `trimestre` - Trimestre del año (1-4)
- `mes_sin` - Componente seno (ciclo anual)
- `mes_cos` - Componente coseno (ciclo anual)

### Top 5 Variables Más Importantes

| # | Feature | Importancia |
|---|---------|------------|
| 1 | ventas_mean_3m | 18.59% ⭐⭐⭐ |
| 2 | ventas_lag1 | 10.65% ⭐⭐ |
| 3 | ventas_lag6 | 9.77% ⭐ |
| 4 | ventas_lag12 | 9.13% ⭐ |
| 5 | ventas_lag3 | 9.12% ⭐ |

---

## 📊 Paso a Paso del Modelo

### Paso 1: Carga y Limpieza
- Lee `Query_Result.csv` (separado por `;`)
- Convierte números formato europeo (`,` como decimal)

### Paso 2: Ingeniería de Características Temporales
- Crea columna `fecha` (YYYY-MM-01)
- Ordena datos por CODIGO y fecha

### Paso 3: K-Means Clustering
- Segmenta productos en 3 clusters basado en:
  - Ventas mensuales (log)
  - Frecuencia de compra
  - Precio promedio (log)

### Paso 4: Variable Objetivo
- Crea `ventas_futuras` = ventas del mes siguiente
- Desplaza datos 1 período hacia atrás

### Paso 5: Feature Engineering Avanzado
- Crea 12 variables adicionales
- Lag features para captar dependencia temporal
- Estadísticas móviles para tendencias
- Características cíclicas para seasonalidad

### Paso 6: Tratamiento de Outliers
- Método: Rango Intercuartílico (IQR)
- Fórmula: Q1 - 1.5×IQR a Q3 + 1.5×IQR
- Impacto: Remueve 12.03% de datos extremos

### Paso 7: Normalización y Escalado
- StandardScaler en características (media=0, std=1)
- StandardScaler en variable objetivo

### Paso 8: División Train-Test
- 80% entrenamiento / 20% prueba
- Random seed=42 para reproducibilidad

### Paso 9: Entrenamiento
```python
RandomForestRegressor(
    n_estimators=100,      # 100 árboles
    max_depth=12,          # Profundidad máxima
    min_samples_split=5,   # Mínimo para dividir nodo
    min_samples_leaf=2,    # Mínimo en hoja
    n_jobs=-1              # Paralelo
)
```

### Paso 10: Validación Cruzada
- 5-fold cross-validation
- Métrica: R² Score
- Evita overfitting

### Paso 11-16: Predicciones y Evaluación
- Genera predicciones en test set
- Calcula métricas (MAE, RMSE, R², MAPE)
- Crea visualizaciones
- Identifica variables importantes

---

## 🎯 Interpretación de Métricas

### MAE (Error Absoluto Medio)
- **Qué es**: Diferencia promedio entre predicciones y valores reales
- **Unidad**: Misma que la variable (dinero)
- **v2.0**: $12.14 (error promedio de $12 por predicción)

### RMSE (Raíz del Error Cuadrático Medio)
- **Qué es**: Penaliza más errores grandes
- **Unidad**: Misma que la variable
- **v2.0**: $16.56 (mejora 98.8% desde v1.0)
- **Interpretación**: Errores típicos ±$16.56

### R² Score
- **Rango**: [0, 1] donde 1 = perfecto
- **v2.0**: 0.1125 (modelo explica ~11% de la variabilidad)
- **Nota**: Bajo pero esperado en datos reales complejos

### MAPE (% de Error Medio Absoluto)
- **Qué es**: Error porcentual promedio
- **v2.0**: 2.36% (muy bueno en términos porcentuales)

---

## 🔄 Histórico de Versiones

### ✅ v2.0 - Mejorada (Actual)
- **Fecha**: 23 Abril 2026
- **RMSE**: $16.56 ↓ 98.8%
- **Features**: 15 variables
- **Status**: ✨ Production Ready

### ⚠️ v1.0 - Baseline Original
- **Fecha**: 23 Abril 2026
- **RMSE**: $1,424.01 (problema)
- **Features**: 3 variables
- **Status**: Deprecated

**Ver**: [CHANGELOG.md](CHANGELOG.md) para detalles completos

---

## 🚀 Próximas Mejoras

### Corto Plazo
- [ ] GridSearchCV para tuning fino de hiperparámetros
- [ ] Validación cruzada estratificada por cluster
- [ ] Métricas adicionales (percentiles de error)

### Mediano Plazo
- [ ] Ensamble de modelos (RF + XGBoost + LightGBM)
- [ ] Modelos separados por cluster
- [ ] ARIMA/Prophet para series de tiempo

### Largo Plazo
- [ ] Deep Learning (LSTM)
- [ ] Forecast probabilístico
- [ ] API REST

---

## 📚 Dependencias

```python
pandas>=2.0          # Manipulación de datos
numpy>=1.24         # Computación numérica
scikit-learn>=1.3   # Machine Learning
matplotlib>=3.7     # Visualización
```

---

## 📝 Documentación Relacionada

- **[BITACORA.md](BITACORA.md)** - Documento oficial de cambios y mejoras
- **[CHANGELOG.md](CHANGELOG.md)** - Historial técnico de versiones

---

## 📧 Contacto

**Proyecto**: Modelo de Predicción de Ventas  
**Responsable**: Data Science Team  
**Última actualización**: 23 de Abril 2026  
**Versión**: 2.0

---

## 📄 Licencia

Proyecto interno - Uso restringido

---

**¿Necesitas ayuda?** Consulta BITACORA.md para documentación detallada.
