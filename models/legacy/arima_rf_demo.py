import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ventas_forecast.paths import DATA_RAW

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


#  CARGAR DATOS

data = pd.read_csv(DATA_RAW / "ventas.csv")

data['fecha'] = pd.to_datetime(data['fecha'])
data.set_index('fecha', inplace=True)

# Asegurar frecuencia (mensual en este caso)
data = data.asfreq('MS')

serie = data['ventas']


# DIVIDIR DATOS

train_size = int(len(serie) * 0.8)
train, test = serie[:train_size], serie[train_size:]


# MODELO ARIMA

model_arima = ARIMA(train, order=(2,1,2))
model_arima_fit = model_arima.fit()

# Predicción ARIMA sobre test
pred_arima = model_arima_fit.forecast(steps=len(test))


# RESIDUOS (USAR TRAIN)

residuos_train = model_arima_fit.resid


# FEATURES PARA RANDOM FOREST

def crear_features(series, lags=2):
    df = pd.DataFrame(series)
    df.columns = ['residuo']
    
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['residuo'].shift(i)
    
    df.dropna(inplace=True)
    return df

lags = 2  # puedes subirlo si tienes más datos
df_rf = crear_features(residuos_train, lags)

# Separar X e y
X = df_rf.drop(columns=['residuo'])
y = df_rf['residuo']

print("Shape RF:", X.shape)


# MODELO RANDOM FOREST

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)


# PREDICCIÓN HÍBRIDA
# Tomar últimos residuos del TRAIN (no del test)
ultimos_residuos = residuos_train[-lags:].values.reshape(1, -1)

# RF
pred_rf = rf.predict(ultimos_residuos)

# Última predicción ARIMA
ultima_pred_arima = pred_arima.iloc[-1]

# Predicción final
pred_final = ultima_pred_arima + pred_rf[0]

print("\n Predicción ARIMA:", ultima_pred_arima)
print(" Corrección RF:", pred_rf[0])
print(" Predicción final híbrida:", pred_final)


# MÉTRICAS (ARIMA base)
mae = mean_absolute_error(test, pred_arima)
rmse = np.sqrt(mean_squared_error(test, pred_arima))

print("\n MAE:", mae)
print(" RMSE:", rmse)


# GRÁFICA

plt.figure(figsize=(10,5))
plt.plot(test.index, test, label="Real")
plt.plot(test.index, pred_arima, label="ARIMA")
plt.legend()
plt.title("Predicción ARIMA vs Real")
plt.show()