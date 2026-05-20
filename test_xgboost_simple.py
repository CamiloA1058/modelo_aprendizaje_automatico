import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

print("Iniciando prueba XGBoost...")

df = pd.read_csv("data\Query_Result.csv", sep=';')

def limpiar_numero(col):
    return col.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

df['VENTAS_MENSUALES'] = limpiar_numero(df['VENTAS_MENSUALES'])
df['PRECIO_PROMEDIO'] = limpiar_numero(df['PRECIO_PROMEDIO'])

df['ANIO'] = df['ANIO'].astype(int)
if df['ANIO'].max() < 100:
    df['ANIO'] = df['ANIO'] + 2020

df['fecha'] = pd.to_datetime(df['ANIO'].astype(str) + '-' + df['MES'].astype(int).astype(str).str.zfill(2) + '-01', format='%Y-%m-%d')
df = df.sort_values(['CODIGO', 'fecha'])

df['ventas_futuras'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(-1)
df['ventas_lag1'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(1)
df['ventas_mean_6m'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].transform(lambda x: x.rolling(6, min_periods=1).mean().shift(1))

df = df.dropna()

# Outliers
Q1 = df['ventas_futuras'].quantile(0.25)
Q3 = df['ventas_futuras'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['ventas_futuras'] >= Q1 - 1.5 * IQR) & (df['ventas_futuras'] <= Q3 + 1.5 * IQR)]

feature_cols = ['FRECUENCIA', 'PRECIO_PROMEDIO', 'ventas_lag1', 'ventas_mean_6m']
X = df[feature_cols]
y = np.log1p(df['ventas_futuras'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_orig = df.loc[y_test.index, 'ventas_futuras']

print(f"Datos preparados: {len(X_train)} train, {len(X_test)} test")

xgb = XGBRegressor(n_estimators=100, random_state=42)
print("Entrenando XGBoost...")
xgb.fit(X_train, y_train)

y_pred_scaled = xgb.predict(X_test)
y_pred = np.expm1(y_pred_scaled)

r2 = r2_score(y_test_orig, y_pred)

print(f"XGBoost R² Score: {r2:.4f}")
print(f"Random Forest anterior: 0.1549")
print(f"Diferencia: {r2 - 0.1549:.4f}")
print("Prueba completada.")