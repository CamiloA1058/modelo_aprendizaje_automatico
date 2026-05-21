import os
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.paths import DATA_RAW, PREDICTIONS, ensure_output_dirs

ensure_output_dirs()
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# versión moderna de Prophet
try:
    from prophet import Prophet
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'prophet'])
    from prophet import Prophet

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def limpiar_numero(col):
    return (
        col.astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )


df = pd.read_csv(DATA_RAW / 'Query_Result.csv', sep=';')

df['VENTAS_MENSUALES'] = limpiar_numero(df['VENTAS_MENSUALES'])
df['PRECIO_PROMEDIO'] = limpiar_numero(df['PRECIO_PROMEDIO'])
df['TOTAL_VENDIDO'] = limpiar_numero(df['TOTAL_VENDIDO'])

def parse_year(value):
    text = str(value).strip().replace(',', '.')
    if '.' in text:
        parts = text.split('.')
        if len(parts) == 2 and len(parts[0]) == 1 and len(parts[1]) == 3:
            return int(parts[0] + parts[1])
        if parts[-1] == '0':
            text = parts[0]
    return int(text)

df['ANIO'] = df['ANIO'].apply(parse_year)

# Construir fecha mensual
_df_mes = df['MES'].astype(str).str.zfill(2)
df['fecha'] = pd.to_datetime(df['ANIO'].astype(str) + '-' + _df_mes + '-01', format='%Y-%m-%d')

df = df.sort_values(['CODIGO', 'fecha']).reset_index(drop=True)

df['ventas_futuras'] = df.groupby('CODIGO')['VENTAS_MENSUALES'].shift(-1)

df = df.dropna(subset=['VENTAS_MENSUALES', 'fecha'])

# usamos el último mes de cada producto como prueba
print('=== Prophet - Modelo Global ===')

predicciones = []
processed = 0

def fit_prophet(series):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,  
        seasonality_prior_scale=10.0  
    )
    model.fit(series)
    return model

print('=== Prophet - Modelo Individual Optimizado (PRUEBA RÁPIDA - 50 productos) ===')

# limitar a 50 productos
max_productos = 50
processed = 0

for codigo, group in df.groupby('CODIGO'):
    if processed >= max_productos:
        break
    group = group.sort_values('fecha')
    if len(group) < 8:
        continue

    # Usar solo los últimos 12 meses para entrenamiento
    train = group.tail(12).iloc[:-1].copy()  # Últimos 12 meses, menos el último
    test = group.iloc[-1:].copy()

    train_df = train[['fecha', 'VENTAS_MENSUALES']].rename(columns={'fecha': 'ds', 'VENTAS_MENSUALES': 'y'})
    train_df['y'] = np.log1p(train_df['y'])

    try:
        model = fit_prophet(train_df)
    except Exception:
        continue

    future = test[['fecha']].rename(columns={'fecha': 'ds'})
    forecast = model.predict(future)
    yhat = np.expm1(forecast['yhat'].values[0])
    yhat = float(max(0.0, yhat))

    predicciones.append({
        'CODIGO': codigo,
        'fecha': test['fecha'].iloc[0],
        'ventas_reales': test['VENTAS_MENSUALES'].iloc[0],
        'prediccion_prophet': yhat
    })

    processed += 1
    if processed % 100 == 0:
        print(f'Procesados: {processed} códigos')
        temp_results = pd.DataFrame(predicciones)
        temp_path = PREDICTIONS / 'PREDICCIONES_PROPHET_temp.csv'
        temp_results.to_csv(temp_path, index=False, encoding='utf-8-sig')
        print(f'[TEMP] Resultados parciales guardados en {temp_path}')

print(f'[INFO] Prophet completado: {processed} productos procesados')

# modelo híbrido 
print('=== Creando modelo híbrido ===')

from sklearn.linear_model import LinearRegression

print(f'[DEBUG] Iniciando preparación de datos. Predicciones disponibles: {len(predicciones)}')

# Preparar datos
ajuste_data = []
for pred in predicciones:
    group = df[df['CODIGO'] == pred['CODIGO']].sort_values('fecha')
    if len(group) >= 3:
        # Usar las últimas 3 ventas para predecir la siguiente
        ultimas_ventas = group.tail(3)['VENTAS_MENSUALES'].values
        ajuste_data.append({
            'codigo': pred['CODIGO'],
            'ventas_t1': ultimas_ventas[-1],
            'ventas_t2': ultimas_ventas[-2] if len(ultimas_ventas) > 1 else 0,
            'ventas_t3': ultimas_ventas[-3] if len(ultimas_ventas) > 2 else 0,
            'pred_prophet': pred['prediccion_prophet'],
            'real': pred['ventas_reales']
        })

ajuste_df = pd.DataFrame(ajuste_data)
print(f'[DEBUG] Datos de ajuste preparados: {len(ajuste_df)} registros')

# Entrenar modelo lineal
X = ajuste_df[['ventas_t1', 'ventas_t2', 'ventas_t3', 'pred_prophet']]
y = ajuste_df['real']

lr_model = LinearRegression()
lr_model.fit(X, y)

print(f'[OK] Modelo de ajuste entrenado. Score: {lr_model.score(X, y):.4f}')

# Aplicar ajuste híbrido
print('[DEBUG] Aplicando ajuste híbrido...')
predicciones_hibridas = []
for i, pred in enumerate(predicciones):
    group = df[df['CODIGO'] == pred['CODIGO']].sort_values('fecha')
    ultimas_ventas = group.tail(3)['VENTAS_MENSUALES'].values

    X_pred = [[
        ultimas_ventas[-1],
        ultimas_ventas[-2] if len(ultimas_ventas) > 1 else 0,
        ultimas_ventas[-3] if len(ultimas_ventas) > 2 else 0,
        pred['prediccion_prophet']
    ]]

    pred_ajustada = lr_model.predict(X_pred)[0]
    pred_ajustada = max(0.0, pred_ajustada)  # No negativas

    predicciones_hibridas.append({
        'CODIGO': pred['CODIGO'],
        'fecha': pred['fecha'],
        'ventas_reales': pred['ventas_reales'],
        'prediccion_prophet': pred['prediccion_prophet'],
        'prediccion_hibrida': pred_ajustada
    })

    if (i + 1) % 100 == 0:
        print(f'[DEBUG] Procesados híbridos: {i + 1}/{len(predicciones)}')

print(f'[DEBUG] Ajuste híbrido completado. Total predicciones híbridas: {len(predicciones_hibridas)}')

results = pd.DataFrame(predicciones_hibridas)
print(f'[DEBUG] DataFrame creado con {len(results)} filas')

# Guardar resultados intermedios
results.to_csv(PREDICTIONS / 'PREDICCIONES_PROPHET_intermedio.csv', index=False, encoding='utf-8-sig')
print('[DEBUG] Resultados intermedios guardados')

results['error_prophet'] = (results['prediccion_prophet'] - results['ventas_reales']).abs()
results['error_hibrida'] = (results['prediccion_hibrida'] - results['ventas_reales']).abs()

results['error_pct_prophet'] = results['error_prophet'] / results['ventas_reales'].replace(0, np.nan) * 100
results['error_pct_hibrida'] = results['error_hibrida'] / results['ventas_reales'].replace(0, np.nan) * 100

print('[DEBUG] Métricas calculadas')

# Calcular métricas para cada modelo
for model_name in ['prophet', 'hibrida']:
    mae = results[f'error_{model_name}'].mean()
    rmse = np.sqrt((results[f'error_{model_name}'] ** 2).mean())
    r2 = r2_score(results['ventas_reales'], results[f'prediccion_{model_name}']) if len(results) > 1 else None

    print(f'=== Modelo {model_name.upper()} ===')
    print(f'Registros evaluados: {len(results):,}')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    if r2 is not None:
        print(f'R²: {r2:.4f}')
    print(f'Outliers de predicción (>50% error): {len(results[results[f"error_pct_{model_name}"] > 50]):,}')
    print()

final_path = PREDICTIONS / 'PREDICCIONES_PROPHET.csv'
results.to_csv(final_path, index=False, encoding='utf-8-sig')
print(f'[OK] Predicciones guardadas en {final_path}')

temp_path = PREDICTIONS / 'PREDICCIONES_PROPHET_temp.csv'
if temp_path.exists():
    os.remove(temp_path)
    print('[CLEAN] Archivo temporal eliminado')
