import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from main.routes.orden_servicio import get_orden_servicio

# df = pd.read_csv("SOLICITUD_202411301535.csv")
data = get_orden_servicio()
df = pd.DataFrame(data.json())

# Convertir columnas de fechas al formato datetime
df['SOL_FECHA'] = pd.to_datetime(df['SOL_FECHA'], errors='coerce')
df['SOL_FECHA_CIERRE'] = pd.to_datetime(df['SOL_FECHA_CIERRE'], errors='coerce')
df['SOL_FECHA_VENCI'] = pd.to_datetime(df['SOL_FECHA_VENCI'], errors='coerce')

# Eliminar filas con NAs en SOL_FECHA_VENCI
df = df.dropna(subset=['SOL_FECHA_VENCI'])

# Calcular las variables
df['solicitudes'] = 1
df['solicitudes_realizadas_tiempo'] = df['SOL_FECHA_CIERRE'] < df['SOL_FECHA_VENCI']
df['solicitudes_atendidas_vencidas'] = df['SOL_FECHA_CIERRE'] > df['SOL_FECHA_VENCI']
df['solicitudes_sin_atender_vencidas'] = df['SOL_FECHA_CIERRE'].isna() & (df['SOL_FECHA_VENCI'] < datetime.now())

# Calcular la duración y agrupar por día
df['duration'] = (df['SOL_FECHA_CIERRE'] - df['SOL_FECHA']).dt.total_seconds() / 3600  # Duración en horas
df['dia'] = df['SOL_FECHA'].dt.date  # Extraer solo la fecha

# Agrupar y calcular métricas
result = df.groupby('dia').agg(
    num_atenciones=('solicitudes', 'size'),
    num_solicitudes_realizadas_tiempo=('solicitudes_realizadas_tiempo', 'sum'),
    num_solicitudes_atendidas_vencidas=('solicitudes_atendidas_vencidas', 'sum'),
    num_solicitudes_sin_atender_vencidas=('solicitudes_sin_atender_vencidas', 'sum')
).reset_index()

# Calcular la media y la desviación estándar de 'num_atenciones'
mean_atenciones = result['num_atenciones'].mean()
std_atenciones = result['num_atenciones'].std()

# Filtrar los datos que estén dentro de una desviación estándar por encima de la media
result = result[result['num_atenciones'] <= (mean_atenciones + std_atenciones)]

# Asegurarse de que las columnas están ordenadas correctamente
df = result[['dia', 'num_atenciones']]

# Normalización de los datos
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df['num_atenciones'].values.reshape(-1, 1))

# Crear ventanas para LSTM
def create_dfset(df, look_back=10):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df[i:i + look_back, 0])
        y.append(df[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 10
X, y = create_dfset(df_scaled, look_back)

# Redimensionar los datos para LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Dividir en entrenamiento y prueba
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Definir el modelo LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_df=(X_test, y_test), verbose=1)

# Predicciones
y_pred = model.predict(X_test)

# Inversión de la escala para interpretar resultados
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Visualizar predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test_inv)), y_test_inv, label="Real", color='blue')
plt.plot(range(len(y_pred_inv)), y_pred_inv, label="Predicción", color='red')
plt.title("Predicciones vs Valores Reales")
plt.xlabel("Tiempo")
plt.ylabel("Número de Atenciones")
plt.legend()
plt.show()

# Predicción de 10 semanas en el futuro
future_steps = 30
input_seq = df_scaled[-look_back:]  # Usar las últimas `look_back` semanas para predecir

predictions = []
for _ in range(future_steps):
    pred = model.predict(input_seq.reshape(1, look_back, 1))
    predictions.append(pred[0, 0])
    input_seq = np.append(input_seq[1:], pred, axis=0)

# Inversión de la escala para los valores futuros
predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Visualizar predicciones futuras
# plt.figure(figsize=(30, 6))
# plt.plot(range(len(df)), df['num_atenciones'], label="Datos Históricos", color='blue')
# plt.plot(range(len(df), len(df) + future_steps), predictions_inv, label="Predicciones Futuras", color='green')
# plt.title("Predicción de Número de Atenciones - Próximas 30 Semanas")
# plt.xlabel("Semanas")
# plt.ylabel("Número de Atenciones")
# plt.legend()
# plt.show()

# Mostrar las predicciones futuras
# print("Predicciones para las próximas 30 semanas:")
# print(predictions_inv.flatten())


# Crear tabla con valores reales, predichos y fechas
observed_vs_predicted = pd.DataFrame({
    'fechas': df['dia'][-len(y_test_inv):].reset_index(drop=True),  # Fechas de los datos de prueba
    'valores_reales': y_test_inv.flatten(),
    'valores_predichos': y_pred_inv.flatten()
})

# Crear tabla con predicciones futuras
future_predictions = pd.DataFrame({
    'numero_dia_predicho': range(1, future_steps + 1),  # Día predicho a futuro
    'valores_predichos': predictions_inv.flatten()
})

# Exportar valores reales y predichos con fechas a JSON
observed_vs_predicted.to_json('observed_vs_predicted_numerosolicitudes.json', orient='records', date_format='iso', indent=4)

# Exportar predicciones futuras a JSON
future_predictions.to_json('future_predictions_numerosolicitudes.json', orient='records', indent=4)


# plt.figure(figsize=(12, 6))
# plt.plot(observed_vs_predicted['fechas'], observed_vs_predicted['valores_reales'], label="Valores Reales", color='blue')
# plt.plot(observed_vs_predicted['fechas'], observed_vs_predicted['valores_predichos'], label="Valores Predichos", color='red')
# plt.title("Valores Reales vs Predichos")
# plt.xlabel("Fechas")
# plt.ylabel("Número de Atenciones")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# Gráfico 2: Predicciones futuras con número de días predichos
# plt.figure(figsize=(12, 6))
# plt.plot(future_predictions['numero_dia_predicho'], future_predictions['valores_predichos'], label="Valores Predichos Futuro", color='green')
# plt.title("Predicciones Futuras")
# plt.xlabel("Número de Día Predicho")
# plt.ylabel("Número de Atenciones")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.show()