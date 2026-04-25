import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

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


df['ANIO'] = df['ANIO'].astype(int)

if df['ANIO'].max() < 100:  
    df['ANIO'] = df['ANIO'] + 2020


df['fecha'] = pd.to_datetime(
    df['ANIO'].astype(str) + '-' +
    df['MES'].astype(int).astype(str).str.zfill(2) + '-01',
    format='%Y-%m-%d'
)

df = df.sort_values(['CODIGO', 'fecha'])


df['VENTAS_MENSUALES_LOG'] = np.log1p(df['VENTAS_MENSUALES'])
df['PRECIO_PROMEDIO_LOG'] = np.log1p(df['PRECIO_PROMEDIO'])

X_cluster = df[['VENTAS_MENSUALES_LOG', 'FRECUENCIA', 'PRECIO_PROMEDIO_LOG']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

df['ventas_futuras'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(-1)

# Feature Engineering mejorado: Más lag features y características estadísticas
df['ventas_lag1'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(1)
df['ventas_lag2'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(2)
df['ventas_lag3'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(3)
df['ventas_lag4'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(4)
df['ventas_lag6'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(6)
df['ventas_lag9'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(9)
df['ventas_lag12'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(12)

df['ventas_growth_1'] = df['VENTAS_MENSUALES'] / df['ventas_lag1']
df['ventas_growth_3'] = df['VENTAS_MENSUALES'] / df['ventas_lag3']
df['ventas_seasonal_ratio'] = df['ventas_lag1'] / df['ventas_lag12']
df['ventas_diff_1'] = df['VENTAS_MENSUALES'] - df['ventas_lag1']
df['ventas_diff_3'] = df['VENTAS_MENSUALES'] - df['ventas_lag3']
df['ventas_mean_6m'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].transform(lambda x: x.rolling(6, min_periods=1).mean().shift(1))
df['ventas_std_6m'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].transform(lambda x: x.rolling(6, min_periods=1).std().shift(1))
df['ventas_min_6m'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].transform(lambda x: x.rolling(6, min_periods=1).min().shift(1))
df['ventas_max_6m'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].transform(lambda x: x.rolling(6, min_periods=1).max().shift(1))
df['ventas_mean_12m'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].transform(lambda x: x.rolling(12, min_periods=1).mean().shift(1))
df['ventas_trend'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].transform(lambda x: (x - x.shift(1)).shift(1))

df['precio_lag1'] = df.groupby('CODIGO')['PRECIO_PROMEDIO'].shift(1)
df['precio_lag3'] = df.groupby('CODIGO')['PRECIO_PROMEDIO'].shift(3)
df['precio_lag6'] = df.groupby('CODIGO')['PRECIO_PROMEDIO'].shift(6)
df['precio_growth_1'] = df['PRECIO_PROMEDIO'] / df['precio_lag1']
df['precio_diff_1'] = df['PRECIO_PROMEDIO'] - df['precio_lag1']
df['precio_x_ventas'] = df['PRECIO_PROMEDIO'] * df['ventas_lag1']
df['precio_mean_3m'] = df.groupby('CODIGO')['PRECIO_PROMEDIO'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
df['precio_mean_6m'] = df.groupby('CODIGO')['PRECIO_PROMEDIO'].transform(lambda x: x.rolling(6, min_periods=1).mean().shift(1))

# Evitar divisiones infinitas
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df['mes'] = df['fecha'].dt.month
df['trimestre'] = df['fecha'].dt.quarter
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

df = df.dropna()

# Detección y tratamiento de outliers usando IQR
Q1 = df['ventas_futuras'].quantile(0.25)
Q3 = df['ventas_futuras'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df['ventas_futuras'] >= lower_bound) & (df['ventas_futuras'] <= upper_bound)].copy()

print(f"\nRegistros originales: {len(df)}")
print(f"Registros después de remover outliers: {len(df_clean)}")
print(f"Outliers removidos: {len(df) - len(df_clean)} ({(len(df) - len(df_clean))/len(df)*100:.2f}%)")

df = df_clean

# Features escaladas
feature_cols = ['FRECUENCIA', 'PRECIO_PROMEDIO', 'VENTAS_MENSUALES_LOG', 'PRECIO_PROMEDIO_LOG', 'cluster',
                'ventas_lag1', 'ventas_lag2', 'ventas_lag3', 'ventas_lag4', 'ventas_lag6', 'ventas_lag9', 'ventas_lag12',
                'ventas_growth_1', 'ventas_growth_3', 'ventas_seasonal_ratio', 'ventas_diff_1', 'ventas_diff_3',
                'ventas_mean_6m', 'ventas_std_6m', 'ventas_min_6m', 'ventas_max_6m', 'ventas_mean_12m', 'ventas_trend',
                'precio_lag1', 'precio_lag3', 'precio_lag6', 'precio_growth_1', 'precio_diff_1',
                'precio_mean_3m', 'precio_mean_6m', 'precio_x_ventas', 'mes', 'trimestre', 'mes_sin', 'mes_cos']
X = df[feature_cols].copy()

# Rellenar NaN que puedan quedar
X = X.fillna(X.mean())

# Escalar todas las características
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Escalar la variable objetivo
y = df['ventas_futuras']
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y_scaled, y, test_size=0.2, random_state=42
)

# Random Forest mejorado con búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 12],
    'min_samples_split': [3, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
model_name = "Random Forest Tuned"
print(f"\nMejores hiperparámetros: {grid_search.best_params_}")

cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=1)
print(f"\nValidación Cruzada (5-fold) - R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Comparar con Gradient Boosting para ver si mejora el R²
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gbr.fit(X_train, y_train)

gbr_cv_scores = cross_val_score(gbr, X_train, y_train, cv=5, scoring='r2', n_jobs=1)
print(f"\nGradient Boosting (5-fold) - R² Score: {gbr_cv_scores.mean():.4f} (+/- {gbr_cv_scores.std():.4f})")

if gbr_cv_scores.mean() > cv_scores.mean():
    best_model = gbr
    model_name = "Gradient Boosting Tuned"
    print("\nSeleccionando Gradient Boosting por mejor R² en CV.")

df['prediccion_scaled'] = best_model.predict(X_scaled)
df['prediccion'] = scaler_y.inverse_transform(df['prediccion_scaled'].values.reshape(-1, 1)).flatten()

y_pred_scaled = best_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Calcular métricas en escala original
mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)
mape = mean_absolute_percentage_error(y_test_orig, y_pred)

print("\n" + "="*60)
print(f"MÉTRICAS DEL MODELO ({model_name})")
print("="*60)
print(f"MAE (Error Absoluto Medio):              ${mae:,.2f}")
print(f"RMSE (Raíz del Error Cuadrático):       ${rmse:,.2f}")
print(f"R² Score:                                {r2:.4f}")
print(f"MAPE (% de Error Medio Absoluto):       {mape:.2f}%")
print("="*60)

plt.figure(figsize=(8,6))
plt.scatter(df['VENTAS_MENSUALES'], df['FRECUENCIA'], c=df['cluster'])
plt.title("Segmentación de productos (K-Means)")
plt.xlabel("Ventas mensuales")
plt.ylabel("Frecuencia")
plt.savefig('grafico_clusters.png')
plt.close()

cluster_summary = df.groupby('cluster')[['VENTAS_MENSUALES','FRECUENCIA']].mean()

cluster_summary.plot(kind='bar')
plt.title("Promedio por cluster")
plt.savefig('grafico_cluster_summary.png')
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(y_test_orig, y_pred, alpha=0.6)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title(f"Predicción vs Real (R² = {r2:.4f}, RMSE = ${rmse:,.0f})")
plt.grid(True, alpha=0.3)
plt.savefig('grafico_prediccion.png')
plt.close()

importances = best_model.feature_importances_
plt.figure(figsize=(6,4))
plt.bar(X.columns, importances)
plt.title(f"Importancia de variables ({model_name})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('grafico_importancia.png')
plt.close()

# Mostrar top 5 features más importantes
top_features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(5)

print("\nTOP 5 VARIABLES MÁS IMPORTANTES:")
print(top_features.to_string(index=False))


producto = df[df['CODIGO'] == df['CODIGO'].iloc[0]]

plt.figure(figsize=(10,5))
plt.plot(producto['fecha'], producto['VENTAS_MENSUALES'], label="Real")
plt.plot(producto['fecha'], producto['prediccion'], label="Predicción")
plt.legend()
plt.title("Evolución de ventas de un producto")
plt.savefig('grafico_evolucion.png')
plt.close()

print("\nRESULTADOS FINALES")
print(df[['CODIGO','VENTAS_MENSUALES','ventas_futuras','prediccion','cluster']].head(10))

# ==================== REPORTES PARA EL CLIENTE ====================

# Crear reporte detallado de predicciones
df_reporte = df[['CODIGO', 'fecha', 'VENTAS_MENSUALES', 'ventas_futuras', 
                 'prediccion', 'PRECIO_PROMEDIO', 'FRECUENCIA', 'cluster']].copy()

# Calcular error por predicción
df_reporte['error_absoluto'] = abs(df_reporte['prediccion'] - df_reporte['ventas_futuras'])
df_reporte['error_porcentaje'] = (df_reporte['error_absoluto'] / df_reporte['ventas_futuras']) * 100

# Ordenar por fecha más reciente
df_reporte = df_reporte.sort_values('fecha', ascending=False)

# ===== 1. REPORTE CSV DETALLADO =====
df_reporte_csv = df_reporte.copy()
df_reporte_csv['VENTAS_MENSUALES'] = df_reporte_csv['VENTAS_MENSUALES'].round(2)
df_reporte_csv['ventas_futuras'] = df_reporte_csv['ventas_futuras'].round(2)
df_reporte_csv['prediccion'] = df_reporte_csv['prediccion'].round(2)
df_reporte_csv['PRECIO_PROMEDIO'] = df_reporte_csv['PRECIO_PROMEDIO'].round(2)
df_reporte_csv['error_absoluto'] = df_reporte_csv['error_absoluto'].round(2)
df_reporte_csv['error_porcentaje'] = df_reporte_csv['error_porcentaje'].round(2)
df_reporte_csv['fecha'] = df_reporte_csv['fecha'].dt.strftime('%Y-%m-%d')

# Renombrar columnas para mejor presentación
df_reporte_csv.columns = [
    'Código Producto', 'Fecha', 'Ventas Actuales', 'Ventas Reales Futuras', 
    'Ventas Predichas', 'Precio Promedio', 'Frecuencia', 'Cluster',
    'Error Absoluto ($)', 'Error (%)'
]

df_reporte_csv.to_csv('PREDICCIONES_DETALLADAS.csv', index=False, encoding='utf-8-sig')
print("\n✓ Reporte detallado guardado: PREDICCIONES_DETALLADAS.csv")

# ===== 2. REPORTE RESUMEN POR CLUSTER =====
cluster_analysis = df_reporte.groupby('cluster').agg({
    'VENTAS_MENSUALES': ['count', 'mean', 'min', 'max'],
    'ventas_futuras': 'mean',
    'prediccion': 'mean',
    'error_porcentaje': ['mean', 'max'],
    'PRECIO_PROMEDIO': 'mean',
    'FRECUENCIA': 'mean'
}).round(2)

print("\n" + "="*70)
print("ANÁLISIS POR CLUSTER")
print("="*70)
for cluster_id in sorted(df_reporte['cluster'].unique()):
    cluster_data = df_reporte[df_reporte['cluster'] == cluster_id]
    
    print(f"\nCLUSTER {int(cluster_id)}:")
    print(f"  Productos: {len(cluster_data)}")
    print(f"  Ventas promedio actual: ${cluster_data['VENTAS_MENSUALES'].mean():,.2f}")
    print(f"  Ventas predichas promedio: ${cluster_data['prediccion'].mean():,.2f}")
    print(f"  Error promedio: {cluster_data['error_porcentaje'].mean():.2f}%")
    print(f"  Precio promedio: ${cluster_data['PRECIO_PROMEDIO'].mean():,.2f}")
    print(f"  Frecuencia promedio: {cluster_data['FRECUENCIA'].mean():.2f}")

# ===== 3. TOP 10 PREDICCIONES (PRODUCTOS MAS IMPORTANTES) =====
print("\n" + "="*70)
print("TOP 10 PRODUCTOS CON MAYORES VENTAS PREDICHAS")
print("="*70)

top_10_predictions = df_reporte.nlargest(10, 'prediccion')[
    ['CODIGO', 'fecha', 'VENTAS_MENSUALES', 'prediccion', 'error_porcentaje', 'FRECUENCIA']
].copy()

top_10_predictions['fecha'] = top_10_predictions['fecha'].dt.strftime('%Y-%m-%d')
top_10_predictions.columns = ['Código', 'Fecha', 'Ventas Actuales', 'Ventas Predichas', 'Error (%)', 'Frecuencia']

print("\n" + top_10_predictions.to_string(index=False))

top_10_predictions.to_csv('TOP_10_PREDICCIONES.csv', index=False, encoding='utf-8-sig')

# ===== 4. RESUMEN EJECUTIVO EN TXT =====
resumen_ejecutivo = f"""
{'='*70}
RESUMEN EJECUTIVO - MODELO DE PREDICCIÓN DE VENTAS
{'='*70}

MODELO UTILIZADO: {model_name}

1. MÉTRICAS GENERALES DE DESEMPEÑO:
   • Precisión (R²):                     {r2:.4f}
   • Error Absoluto Promedio (MAE):     ${mae:,.2f}
   • Raíz del Error Cuadrático (RMSE):  ${rmse:,.2f}
   • Error Porcentual Promedio (MAPE):  {mape:.2f}%

2. DATOS PROCESADOS:
   • Total de registros analizados:      {len(df):,}
   • Períodos de tiempo:                 {df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}
   • Número de productos únicos:         {df['CODIGO'].nunique():,}
   • Segmentación (clusters):            {int(df['cluster'].max()) + 1}

3. ANÁLISIS POR CLUSTER:
"""

for cluster_id in sorted(df_reporte['cluster'].unique()):
    cluster_data = df_reporte[df_reporte['cluster'] == cluster_id]
    resumen_ejecutivo += f"""
   CLUSTER {int(cluster_id)}:
   • Cantidad de productos:              {len(cluster_data):,}
   • Ventas promedio actual:             ${cluster_data['VENTAS_MENSUALES'].mean():,.2f}
   • Ventas predichas promedio:          ${cluster_data['prediccion'].mean():,.2f}
   • Variación esperada:                 {((cluster_data['prediccion'].mean() / cluster_data['VENTAS_MENSUALES'].mean() - 1) * 100):+.2f}%
   • Error promedio de predicción:       {cluster_data['error_porcentaje'].mean():.2f}%
"""

resumen_ejecutivo += f"""

4. PRINCIPALES VARIABLES PREDICTORAS:
"""

for idx, row in top_features.iterrows():
    resumen_ejecutivo += f"\n   • {row['Feature']:<30} {row['Importance']:.4f}"

resumen_ejecutivo += f"""

5. RECOMENDACIONES:
   • El modelo tiene un R² de {r2:.4f}, indicando {"excelente" if r2 > 0.8 else "buena" if r2 > 0.6 else "aceptable"} precisión.
   • El MAPE de {mape:.2f}% indica un {"excelente" if mape < 5 else "muy bueno" if mape < 10 else "bueno" if mape < 15 else "aceptable"} margen de error.
   • Se recomienda usar estas predicciones para planificación de inventario y recursos.
   • Monitorear continuamente el error de predicción para mejoras futuras.

6. ARCHIVOS GENERADOS:
   • PREDICCIONES_DETALLADAS.csv    → Predicciones completas con errores
   • TOP_10_PREDICCIONES.csv         → Los 10 productos con mayores ventas predichas
   • grafico_prediccion.png          → Gráfico Real vs Predicción
   • grafico_clusters.png            → Visualización de clusters
   • grafico_importancia.png         → Importancia de variables

{'='*70}
Reporte generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

with open('RESUMEN_EJECUTIVO.txt', 'w', encoding='utf-8') as f:
    f.write(resumen_ejecutivo)

print(resumen_ejecutivo)
print("✓ Resumen ejecutivo guardado: RESUMEN_EJECUTIVO.txt")

# ===== 5. TABLA RESUMEN SIMPLE =====
summary_table = pd.DataFrame({
    'Métrica': ['Precisión (R²)', 'Error Absoluto (MAE)', 'Error Cuadrático (RMSE)', 
                'Error Porcentual (MAPE)', 'Productos Analizados', 'Registros Procesados'],
    'Valor': [f"{r2:.4f}", f"${mae:,.2f}", f"${rmse:,.2f}", 
              f"{mape:.2f}%", f"{df['CODIGO'].nunique():,}", f"{len(df):,}"]
})

summary_table.to_csv('RESUMEN_METRICAS.csv', index=False, encoding='utf-8-sig')
print("✓ Tabla de métricas guardada: RESUMEN_METRICAS.csv")