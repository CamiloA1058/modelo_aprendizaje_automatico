import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Cargamos el dataset y limpiamos las columnas numéricas

df = pd.read_csv("data\Query_Result_V3.csv", sep=';')

def limpiar(col):
    return (
        col.astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

df['VENTAS'] = limpiar(df['VENTAS'])
df['TOTAL_VENDIDO'] = limpiar(df['TOTAL_VENDIDO'])
df['PRECIO_PROMEDIO'] = limpiar(df['PRECIO_PROMEDIO'])

# Cambiamos el formato de la fecha y ordenamos por producto y fecha

df['ANIO'] = (
    df['ANIO']
    .astype(str)
    .str.replace('.', '', regex=False)
    .astype(int)
)

df['fecha'] = pd.to_datetime(
    df['ANIO'].astype(str) + '-' +
    df['MES'].astype(str).str.zfill(2) + '-01'
)

#Regularizamos a un producto por mes
# Ahora cada fila sera un producto en un mes
# un producto en un mes

df = (
    df.groupby([
        'CODIGO',
        'DESCRIPCION',
        pd.Grouper(key='fecha', freq='MS')
    ])
    .agg({
        'VENTAS': 'sum',
        'TOTAL_VENDIDO': 'sum',
        'PRECIO_PROMEDIO': 'mean',
        'FRECUENCIA': 'sum'
    })
    .reset_index()
)

# Extraemos año y mes de la fecha para futuras referencias

df['ANIO'] = df['fecha'].dt.year
df['MES'] = df['fecha'].dt.month

df = df.sort_values(['CODIGO', 'fecha'])

# Creamos el target de regresión: ventas del próximo mes

df['target'] = (
    df.groupby('CODIGO')['TOTAL_VENDIDO']
    .shift(-1)
)

# Features temporales con lags y medias móviles
for lag in [1,2,3,6]:
    df[f'lag_{lag}'] = (
        df.groupby('CODIGO')['TOTAL_VENDIDO']
        .shift(lag)
    )

# MEDIAS MÓVILES

df['mean_3'] = (
    df.groupby('CODIGO')['TOTAL_VENDIDO']
    .transform(
        lambda x: x.rolling(3).mean().shift(1)
    )
)

df['mean_6'] = (
    df.groupby('CODIGO')['TOTAL_VENDIDO']
    .transform(
        lambda x: x.rolling(6).mean().shift(1)
    )
)


# 7. CRECIMIENTO


df['growth_ratio'] = (
    df['target'] / df['TOTAL_VENDIDO']
)

# 8. LOGARÍTMICAS

df['VENTAS_LOG'] = np.log1p(df['VENTAS'])
df['TOTAL_LOG'] = np.log1p(df['TOTAL_VENDIDO'])
df['PRECIO_LOG'] = np.log1p(df['PRECIO_PROMEDIO'])

# 9. Estacionalidad

df['mes_sin'] = np.sin(
    2*np.pi*df['MES']/12
)

df['mes_cos'] = np.cos(
    2*np.pi*df['MES']/12
)

# 10. LIMPIEZA
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df = df.dropna().reset_index(drop=True)

df = df[
    (df['TOTAL_VENDIDO'] > 0)
].reset_index(drop=True)

# 11. CLUSTERING

X_cluster = df[
    ['TOTAL_VENDIDO', 'FRECUENCIA', 'PRECIO_PROMEDIO']
]

Xc = RobustScaler().fit_transform(X_cluster)

df['cluster'] = KMeans(
    n_clusters=3,
    random_state=42
).fit_predict(Xc)


# 12. TARGET CLASIFICACIÓN

# crecerá más de 5% el próximo mes

df['target_class'] = (
    df['growth_ratio'] > 1.05
).astype(int)


# 13. FEATURES


features = [
    'VENTAS_LOG',
    'TOTAL_LOG',
    'PRECIO_LOG',
    'FRECUENCIA',
    'lag_1',
    'lag_2',
    'lag_3',
    'lag_6',
    'mean_3',
    'mean_6',
    'mes_sin',
    'mes_cos',
    'cluster'
]

X = df[features].fillna(0)

y = df['target_class']


# 14. ESCALAR


scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)


# 15. SPLIT TEMPORAL


train_idx = []
test_idx = []

for _, g in df.groupby('CODIGO'):

    g = g.sort_values('fecha')

    cut = int(len(g) * 0.8)

    train_idx += g.index[:cut].tolist()
    test_idx += g.index[cut:].tolist()

X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# 16. CLASIFICADOR


clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=3,
    class_weight={0:1, 1:3},
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)


# 17. PREDICCIÓN CLASIFICACIÓN


prob = clf.predict_proba(X_test)[:,1]

threshold = 0.58

pred = (prob > threshold).astype(int)


# 18. MÉTRICAS


print("\n===== MÉTRICAS =====")

print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

print(
    "Precision:",
    precision_score(y_test, pred)
)

print(
    "Recall:",
    recall_score(y_test, pred)
)

print(
    "F1:",
    f1_score(y_test, pred)
)

print("\n===== REPORT =====")

print(
    classification_report(y_test, pred)
)


# 19. REGRESIÓN


top = (
    df.groupby('CODIGO')['TOTAL_VENDIDO']
    .sum()
    .nlargest(200)
    .index
)

df_reg = df[
    (df['CODIGO'].isin(top)) &
    (df['target'] > 0)
]

X_reg = X.loc[df_reg.index]

y_reg = np.log1p(df_reg['target'])

X_reg_scaled = scaler.transform(X_reg)

reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

reg.fit(X_reg_scaled, y_reg)


# 20. RESULTADOS


results = df.iloc[test_idx].copy()

results['prob'] = prob

results['decision'] = np.where(
    pred == 1,
    'Reforzar stock próximo mes',
    'Mantener stock'
)

# SOLO TOP 

mask = results['CODIGO'].isin(top)

pred_sales = reg.predict(
    scaler.transform(
        X.loc[results[mask].index]
    )
)

results.loc[mask, 'pred_sales'] = np.expm1(pred_sales)

# ERROR

results['error_stock'] = abs(
    results['pred_sales'] -
    results['target']
)


# 21. EXPORTAR


results.to_csv(
    "PREDICCION_MENSUAL.csv",
    index=False
)

print("\nArchivo generado: PREDICCION_MENSUAL.csv")


# 22. VISUALIZACIÓN


producto_visual = (
    results['CODIGO']
    .value_counts()
    .idxmax()
)

graf = results[
    results['CODIGO'] == producto_visual
].copy()

graf = graf.sort_values('fecha')

plt.figure(figsize=(14,6))

# REAL ACTUAL
plt.plot(
    graf['fecha'],
    graf['TOTAL_VENDIDO'],
    marker='o',
    label='Ventas reales'
)

# REAL FUTURO
plt.plot(
    graf['fecha'],
    graf['target'],
    marker='o',
    linestyle='--',
    label='Ventas reales próximo mes'
)

# PREDICCIÓN
plt.plot(
    graf['fecha'],
    graf['pred_sales'],
    marker='o',
    linewidth=3,
    label='Predicción próximo mes'
)

plt.title(
    f'Predicción mensual - Producto {producto_visual}'
)

plt.xlabel('Mes')

plt.ylabel('Valor vendido (COP)')

plt.legend()

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()