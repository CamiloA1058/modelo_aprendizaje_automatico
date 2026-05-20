import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# 1. CARGA

df = pd.read_csv("data\Query_Result_V3.csv", sep=';')

def limpiar(col):
    return col.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

df['VENTAS'] = limpiar(df['VENTAS'])
df['TOTAL_VENDIDO'] = limpiar(df['TOTAL_VENDIDO'])
df['PRECIO_PROMEDIO'] = limpiar(df['PRECIO_PROMEDIO'])


# 2. FECHA

df['ANIO'] = df['ANIO'].astype(str).str.replace('.', '', regex=False).astype(int)

df['fecha'] = pd.to_datetime(
    df['ANIO'].astype(str) + '-' +
    df['MES'].astype(str).str.zfill(2) + '-' +
    df['DIA'].astype(str).str.zfill(2)
)

df = df.sort_values(['CODIGO', 'fecha'])


# 3. TARGET

df['target'] = df.groupby('CODIGO')['TOTAL_VENDIDO'].shift(-1)


# 4. FEATURES

for lag in [1,2,3,6]:
    df[f'lag_{lag}'] = df.groupby('CODIGO')['TOTAL_VENDIDO'].shift(lag)

df['mean_3'] = df.groupby('CODIGO')['TOTAL_VENDIDO'].transform(lambda x: x.rolling(3).mean().shift(1))
df['mean_6'] = df.groupby('CODIGO')['TOTAL_VENDIDO'].transform(lambda x: x.rolling(6).mean().shift(1))

df['growth_ratio'] = df['target'] / df['TOTAL_VENDIDO']

# LOG
df['VENTAS_LOG'] = np.log1p(df['VENTAS'])
df['TOTAL_LOG'] = np.log1p(df['TOTAL_VENDIDO'])
df['PRECIO_LOG'] = np.log1p(df['PRECIO_PROMEDIO'])

# TEMPORAL
df['mes_sin'] = np.sin(2*np.pi*df['MES']/12)
df['mes_cos'] = np.cos(2*np.pi*df['MES']/12)

# LIMPIEZA
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna().reset_index(drop=True)

# FILTRO
df = df[(df['TOTAL_VENDIDO'] > 0) & (df['FRECUENCIA'] > 3)].reset_index(drop=True)


# 5. CLUSTERING

X_cluster = df[['TOTAL_VENDIDO', 'FRECUENCIA', 'PRECIO_PROMEDIO']]
Xc = RobustScaler().fit_transform(X_cluster)
df['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(Xc)


# 6. TARGET CLASIFICACIÓN

df['target_class'] = (df['growth_ratio'] > 1.05).astype(int)


# 7. FEATURES

features = [
    'VENTAS_LOG','TOTAL_LOG','PRECIO_LOG','FRECUENCIA',
    'lag_1','lag_2','lag_3','lag_6',
    'mean_3','mean_6',
    'mes_sin','mes_cos',
    'cluster'
]

X = df[features].fillna(0)
y = df['target_class']

# ESCALAR
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)


# 8. SPLIT POR PRODUCTO

train_idx, test_idx = [], []

for _, g in df.groupby('CODIGO'):
    cut = int(len(g)*0.8)
    train_idx += g.index[:cut].tolist()
    test_idx += g.index[cut:].tolist()

X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# 9. MODELO CLASIFICACIÓN

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=3,
    class_weight={0:1, 1:3},  # más peso a crecimiento
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

#PROBS
prob = clf.predict_proba(X_test)[:,1]

#UMBRAL BALANCEADO
threshold = 0.58
pred = (prob > threshold).astype(int)

#MÉTRICAS

print("\n===== MÉTRICAS CLASIFICACIÓN =====")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))

print("\n===== REPORT =====")
print(classification_report(y_test, pred))


# 11. REGRESIÓN SOLO TOP

top = df.groupby('CODIGO')['TOTAL_VENDIDO'].sum().nlargest(200).index
df_reg = df[df['CODIGO'].isin(top) & (df['target'] > 0)]

X_reg = X.loc[df_reg.index]
y_reg = np.log1p(df_reg['target'])

X_reg_scaled = scaler.transform(X_reg)

reg = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

reg.fit(X_reg_scaled, y_reg)


# 12. RESULTADOS

results = df.iloc[test_idx].copy()
results['prob'] = prob

results['decision'] = np.where(
    pred==1,
    'Reforzar stock',
    'Mantener'
)

# Predicción solo top
mask = results['CODIGO'].isin(top)
pred_sales = reg.predict(scaler.transform(X.loc[results[mask].index]))
results.loc[mask, 'pred_sales'] = np.expm1(pred_sales)

results['error_stock'] = abs(results['pred_sales'] - results['target'])

print("\n DECISIONES: ")
print(results['decision'].value_counts())

print("\nError stock promedio:", results['error_stock'].mean())

# ================================ # 14. EXPORTAR # ================================ 
results.to_csv("DECISIONES_FINALES_BALANCEADAS(1).csv", index=False) 
#print("\nArchivo generado: DECISIONES_FINALES_BALANCEADAS.csv")


producto_visual = results['CODIGO'].value_counts().idxmax()

print("Producto visual:", producto_visual)


# SOLO DATOS TEST

graf = results[
    results['CODIGO'] == producto_visual
].copy()
# ordenar
graf = graf.sort_values('fecha')

# VERIFICAR DATOS


print(graf[['fecha','TOTAL_VENDIDO','target','pred_sales']].head())


# GRAFICAR


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
    label='Valor real futuro'
)

# PREDICCION
plt.plot(
    graf['fecha'],
    graf['pred_sales'],
    marker='o',
    linewidth=3,
    label='Predicción modelo'
)

plt.title(f'Predicción vs Realidad - Producto {producto_visual}')

plt.xlabel('Fecha')
plt.ylabel('Ventas')

plt.legend()
plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()