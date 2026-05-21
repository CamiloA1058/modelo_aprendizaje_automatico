# 📚 ÍNDICE DE DOCUMENTACIÓN

**Proyecto**: Modelo K-Means + Random Forest para Predicción de Ventas  
**Versión**: Producción final (V3)  
**Fecha**: 21 de Mayo 2026  
**Modelo final**: `scripts/train_kmeans_rf_prod.py` (Accuracy 67.32%, F1 70.95%)  
**Por ítem**: `scripts/train_kmeans_rf.py` (Accuracy 78.03%, F1 43.36%)  
**Comparaciones**: XGBoost y Prophet (benchmark, no modelo final)  

---

## 🎯 Empieza Por Aquí

### 1️⃣ **[README.md](README.md)** - Para Usuarios
**Tiempo de lectura**: 5-10 minutos  
**Para**: Entender qué es el modelo y cómo usarlo

Contiene:
- ✅ Descripción general del proyecto
- ✅ Cómo ejecutar el modelo
- ✅ Interpretación de resultados
- ✅ Estructura de archivos

**Acción**: Lee esto primero si es tu primer contacto

---

### 2️⃣ **[BITACORA.md](BITACORA.md)** - Para Entender la Mejora
**Tiempo de lectura**: 15-20 minutos  
**Para**: Comprender qué cambió y por qué

Contiene:
- ✅ Historial completo de versiones (v1.0 vs v2.0)
- ✅ Análisis detallado de cada mejora
- ✅ Resultados antes y después
- ✅ Justificación técnica de decisiones
- ✅ Variables más importantes
- ✅ Próximos pasos sugeridos

**Acción**: Lee esto para entender la solución al problema de RMSE alto

---

### 3️⃣ **[CHANGELOG.md](CHANGELOG.md)** - Para Control de Versiones
**Tiempo de lectura**: 10 minutos  
**Para**: Ver cambios específicos entre versiones

Contiene:
- ✅ [2.0] - Cambios principales (actual)
- ✅ [1.0] - Versión original
- ✅ Roadmap futuro
- ✅ Comparativa de métricas

**Acción**: Usa esto como referencia técnica rápida

---

### 4️⃣ **[VERSION_SUMMARY.md](VERSION_SUMMARY.md)** - Resumen Ejecutivo
**Tiempo de lectura**: 5 minutos  
**Para**: Ver benchmark y comparación visual

Contiene:
- ✅ Comparación v1.0 vs v2.0
- ✅ Cambios principales explicados
- ✅ Análisis factorial de mejoras
- ✅ Timeline de desarrollo
- ✅ Lecciones aprendidas

**Acción**: Léelo para la visión general rápida

---

## 🐍 SCRIPTS DISPONIBLES

### Modelo final (producción)
```bash
python scripts/train_kmeans_rf_prod.py
```
**Archivo**: `scripts/train_kmeans_rf_prod.py` — modelo seleccionado para producción

### Predicción por ítem
```bash
python scripts/train_kmeans_rf.py
```
**Archivo**: `scripts/train_kmeans_rf.py` — predicción/decisiones por producto

### Modelos de comparación (benchmark)
```bash
python scripts/run_xgboost.py
python scripts/run_prophet.py
```
**Nota**: Solo para comparar con el modelo final K-Means + RF. No usar en producción.

### Legacy (referencia histórica)
```bash
python models/legacy/model_k-means_random_forest_v1.0.py
python models/legacy/model_k-means_random_forest_v2.0.py
```

---

## 📊 FLUJO DE APRENDIZAJE RECOMENDADO

```
Soy Nuevo en el Proyecto
    ↓
Lee README.md (5 min)
    ↓
Ejecuta: python scripts/train_kmeans_rf_prod.py
    ↓
Lee BITACORA.md (15 min)
    ↓
Examina grafico_*.png generadas
    ↓
Lee VERSION_SUMMARY.md para detalles
    ↓
Consultá CHANGELOG.md para referencia
```

---

```
Soy Data Scientist
    ↓
Lee BITACORA.md primero (visión técnica)
    ↓
Abre scripts/train_kmeans_rf_prod.py (modelo final)
    ↓
Compara con scripts/run_xgboost.py y scripts/run_prophet.py (benchmark)
    ↓
Lee CHANGELOG.md para detalles específicos
    ↓
Examina resultados detallados
```

---

```
Soy Ejecutivo/Manager
    ↓
Lee VERSION_SUMMARY.md (5 min)
    ↓
Mira los números de mejora:
    RMSE: $1,424 → $16.56 (↓ 98.8%)
    MAE: $137 → $12 (↓ 91%)
    ↓
¡Listo! Proyecto exitoso ✅
```

---

## 📁 ARCHIVOS DE REFERENCIA RÁPIDA

| Archivo | Líneas | Tiempo | Tipo | Uso |
|---------|--------|--------|------|-----|
| README.md | ~400 | 5 min | Doc | Guía general |
| BITACORA.md | ~350 | 15 min | Doc | Análisis detallado |
| CHANGELOG.md | ~250 | 10 min | Doc | Control versiones |
| VERSION_SUMMARY.md | ~400 | 5 min | Doc | Resumen ejecutivo |
| train_kmeans_rf_prod.py | ~380 | - | Código | **Modelo final** |
| train_kmeans_rf.py | ~250 | - | Código | Predicción por ítem |
| run_xgboost.py | ~430 | - | Código | Comparación |
| run_prophet.py | ~230 | - | Código | Comparación |

---

## 🎯 BUSCA ESPECÍFICAMENTE

### ¿Cómo ejecuto el modelo?
→ [README.md](README.md) sección "Cómo Usar"

### ¿Cuáles fueron las mejoras?
→ [BITACORA.md](BITACORA.md) sección "HISTORIAL DE VERSIONES"

### ¿Qué variables son importantes?
→ [BITACORA.md](BITACORA.md) sección "Top 5 Variables Más Importantes"

### ¿Por qué RMSE mejoró tanto?
→ [VERSION_SUMMARY.md](VERSION_SUMMARY.md) sección "ANÁLISIS FACTORIAL DE MEJORAS"

### ¿Cuál es el plan futuro?
→ [BITACORA.md](BITACORA.md) sección "Próximas Mejoras Potenciales"

### ¿Cómo interpreto las métricas?
→ [README.md](README.md) sección "Interpretación de Métricas"

### ¿Cuál es la diferencia v1.0 vs v2.0?
→ [VERSION_SUMMARY.md](VERSION_SUMMARY.md) sección "CAMBIOS PRINCIPALES"

### ¿Qué features se usan?
→ [README.md](README.md) sección "Características del Modelo (v2.0)"

### ¿Cómo funcionan los 16 pasos?
→ [models/legacy/model_k-means_random_forest_v2.0.py](../models/legacy/model_k-means_random_forest_v2.0.py) + comentarios

### ¿Qué pasó el 23 de Abril?
→ [VERSION_SUMMARY.md](VERSION_SUMMARY.md) sección "TIMELINE"

---

## 🔑 PUNTOS CLAVE A RECORDAR

### ✅ Problema Resuelto
- RMSE bajó de **$1,424.01** a **$16.56**
- Eso es una **reducción de 98.8%** ✨

### 🎯 Causa del Problema
1. Solo 3 variables predictoras
2. Sin lag features (histórico)
3. Sin normalización
4. Valores extremos sin remover

### 💡 Solución Aplicada
1. Agregamos 12 variables nuevas (lag, media móvil, trend, cíclicas)
2. Normalizamos entrada y salida
3. Removimos outliers (12% de datos)
4. Agregamos validación cruzada

### 📊 Mejora por Factor
- Feature Engineering: **70%**
- Normalización: **15%**
- Outlier Removal: **10%**
- Hyper-parameters: **5%**

### 📈 Variable Más Importante
**ventas_mean_3m** (Media móvil 3 meses) = **18.59%** de importancia

---

## 🚀 PRÓXIMOS PASOS SUGERIDOS

1. **Ahora**: Ejecuta el modelo final → `python scripts/train_kmeans_rf_prod.py`
2. **Por ítem**: `python scripts/train_kmeans_rf.py`
3. **Luego**: Lee BITACORA.md — XGBoost y Prophet son solo comparaciones
4. **Futuro**: Revisión formal antes de despliegue

---

## 📞 REFERENCIA RÁPIDA

```
Datos de Entrada:
  └─ Query_Result.csv (6,975 registros)

Proceso:
  ├─ K-Means Clustering → 3 clusters
  ├─ Feature Engineering → 15 variables
  ├─ Normalización → StandardScaler
  ├─ Outlier Removal → IQR method (-12%)
  ├─ Train/Test Split → 80/20
  └─ Random Forest → 100 árboles

Resultados:
  ├─ RMSE: $16.56 ✅
  ├─ MAE: $12.14 ✅
  ├─ Validación CV: 0.1521 ✅
  └─ Visualizaciones: 5 gráficos PNG

Documentación:
  ├─ README.md (guía)
  ├─ BITACORA.md (detalle)
  ├─ CHANGELOG.md (versiones)
  └─ VERSION_SUMMARY.md (resumen)
```

---

## ✨ CONCLUSIÓN

El **modelo final** (`train_kmeans_rf_prod`) fue seleccionado tras comparar con XGBoost y Prophet. Toda la documentación está disponible para:
- ✅ Reproducir resultados
- ✅ Entender cambios
- ✅ Continuar mejorando
- ✅ Escalar a producción

**Comienza por el README.md → Luego BITACORA.md → Consulta según necesites**

---

**Creado**: 23 Abril 2026  
**Versión**: 1.0  
**Estado**: ✅ Completo
