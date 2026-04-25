"""
MODELO K-MEANS + RANDOM FOREST - VERSIÓN 1.0 (BASELINE)
=========================================================

FECHA: 23 de Abril 2026
VERSIÓN: 1.0 - Original
RMSE: $1,424.01 ⚠️ MUY ALTO

NOTA: Esta es la versión original con problemas de RMSE alto.
      Se incluye como referencia histórica.
      VER VERSIÓN 2.0 para la versión mejorada.

RESULTADOS ORIGINALES (ANTES DE MEJORAS):
- MAE: $137.49
- RMSE: $1,424.01 ← Problema principal
- R² Score: 0.3178
- MAPE: 8.59%

PROBLEMAS IDENTIFICADOS:
✗ RMSE extremadamente alto
✗ Solo 3 features predictoras
✗ Sin feature engineering
✗ Variables sin escalar correctamente
✗ Sin validación cruzada robusta

HISTÓRICO: Véase BITACORA.md y CHANGELOG.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("="*70)
print("VERSIÓN 1.0 - MODELO ORIGINAL (BASELINE)")
print("="*70)

# ============================================================================
# CARGA DE DATOS
# ============================================================================
df = pd.read_csv("Query_Result.csv", sep=';')

def limpiar_numero(col):
    return (
        col.astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

df['VENTAS_MENSUALES'] = limpiar_numero(df['VENTAS_MENSUALES'])
df['PRECIO_PROMEDIO'] = limpiar_numero(df['PRECIO_PROMEDIO'])
df['TOTAL_VENDIDO'] = limpiar_numero(df['TOTAL_VENDIDO'])

print(f"Datos cargados: {len(df)} registros\n")

# ============================================================================
# PROCESAMIENTO BÁSICO
# ============================================================================
df['ANIO'] = df['ANIO'].astype(int)

if df['ANIO'].max() < 100:  
    df['ANIO'] = df['ANIO'] + 2020

df['fecha'] = pd.to_datetime(
    df['ANIO'].astype(str) + '-' +
    df['MES'].astype(int).astype(str).str.zfill(2) + '-01',
    format='%Y-%m-%d'
)

df = df.sort_values(['CODIGO', 'fecha'])

# ============================================================================
# TRANSFORMACIONES LOGARÍTMICAS
# ============================================================================
df['VENTAS_MENSUALES_LOG'] = np.log1p(df['VENTAS_MENSUALES'])
df['PRECIO_PROMEDIO_LOG'] = np.log1p(df['PRECIO_PROMEDIO'])

# ============================================================================
# K-MEANS CLUSTERING (Simple, 3 clusters)
# ============================================================================
X_cluster = df[['VENTAS_MENSUALES_LOG', 'FRECUENCIA', 'PRECIO_PROMEDIO_LOG']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(f"K-Means: {len(df)} productos en 3 clusters")

# ============================================================================
# VARIABLE OBJETIVO Y TRAIN-TEST SPLIT
# ============================================================================
df['ventas_futuras'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(-1)
df = df.dropna()

# ⚠️ PROBLEMA: Solo 3 features, sin escalar entrada
X = df[['FRECUENCIA', 'PRECIO_PROMEDIO', 'cluster']]
y = df['ventas_futuras']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Train: {len(X_train)} registros | Test: {len(X_test)} registros\n")

# ============================================================================
# RANDOM FOREST BÁSICO
# ============================================================================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ============================================================================
# PREDICCIONES Y MÉTRICAS
# ============================================================================
df['prediccion'] = rf.predict(X)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50)
print("MÉTRICAS - VERSIÓN 1.0 (ORIGINAL)")
print("="*50)
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f} ⚠️ ← PROBLEMA DETECTADO")
print("="*50)

# ============================================================================
# VISUALIZACIONES
# ============================================================================
plt.figure(figsize=(8,6))
plt.scatter(df['VENTAS_MENSUALES'], df['FRECUENCIA'], c=df['cluster'])
plt.title("Segmentación de productos (K-Means)")
plt.xlabel("Ventas mensuales")
plt.ylabel("Frecuencia")
plt.savefig('grafico_clusters_v1.0.png')
plt.close()

cluster_summary = df.groupby('cluster')[['VENTAS_MENSUALES','FRECUENCIA']].mean()
cluster_summary.plot(kind='bar')
plt.title("Promedio por cluster")
plt.savefig('grafico_cluster_summary_v1.0.png')
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Predicción vs Real (V1.0)")
plt.savefig('grafico_prediccion_v1.0.png')
plt.close()

importances = rf.feature_importances_
plt.figure(figsize=(6,4))
plt.bar(X.columns, importances)
plt.title("Importancia de variables (V1.0)")
plt.savefig('grafico_importancia_v1.0.png')
plt.close()

# ============================================================================
# RESULTADOS
# ============================================================================
print("\nRESULTADOS FINALES (PRIMEROS 10 REGISTROS):")
print(df[['CODIGO','VENTAS_MENSUALES','ventas_futuras','prediccion','cluster']].head(10))

print("\n⚠️ NOTA: RMSE muy alto - véase versión 2.0 para mejoras")
print("📄 Documentación: BITACORA.md y CHANGELOG.md")
