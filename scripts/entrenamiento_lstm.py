import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import re

print("📘 Entrenamiento del modelo LSTM (4 variables) para predicción de pobreza en el Perú\n")

archivos = [
    "data/Pobreza_2022_CORREGIDO.xlsx",
    "data/Pobreza_2023_CORREGIDO.xlsx",
    "data/Pobreza_2024_CORREGIDO.xlsx"
]

# ==========================
# CARGA DE DATOS
# ==========================
df_list = []
for archivo in archivos:
    if os.path.exists(archivo):
        df_temp = pd.read_excel(archivo, header=0)
        df_list.append(df_temp)
    else:
        print(f"⚠️ No se encontró el archivo: {archivo}")

df = pd.concat(df_list, ignore_index=True)

# ==========================
# NORMALIZAR Y LIMPIAR NOMBRES DE COLUMNAS
# ==========================
df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c.lower().strip()) for c in df.columns]
print(f"✅ Datos cargados correctamente: {df.shape[0]} registros totales.\n")

# --- VARIABLES NUMÉRICAS REQUERIDAS ---
features = [
    "pobreza_extrema__",    # X1
    "empleo_informal__",    # X2
    "sin_internet__",       # X3
    "umbral_zona_pobreza"   # X4 -> ESTA ES LA COLUMNA CRÍTICA
]
target = ["pobreza_total__"] # Y (Variable a predecir)

columnas_requeridas = features + target

# ==========================
# VERIFICACIÓN Y PREPARACIÓN DE DATOS
# ==========================
df_final = df.copy()

print("-" * 50)
print("🔎 DIAGNÓSTICO DE COLUMNAS:")
print(f"Columnas disponibles en tus datos (NORMALIZADAS):\n{df.columns.tolist()}")
print("\nColumnas que el script intenta usar:")
print(columnas_requeridas)
print("-" * 50)

registros_totales = df_final.shape[0]
print(f"\n🛑 Iniciando limpieza y transformación de las {registros_totales} filas...")

for col in columnas_requeridas:
    if col in df_final.columns:
        
        # 1. Asegurarse de que sea string y limpiar espacios
        df_final[col] = df_final[col].astype(str).str.strip()
        
        # 2. Reemplazar comas por puntos y eliminar signos de porcentaje
        df_final[col] = df_final[col].str.replace(',', '.', regex=True).str.replace('%', '', regex=False)
        
        # 3. Forzar la conversión a numérico. Cualquier otro texto se convierte a NaN.
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        
        # 4. TRATAMIENTO ESPECÍFICO PARA EL PROBLEMA DE 'umbral_zona_pobreza'
        if col == "umbral_zona_pobreza" and df_final[col].isna().sum() == registros_totales:
            df_final[col] = 0
            print(f"✅ Solución aplicada: Columna '{col}' (30 NaNs) rellenada con 0 para evitar pérdida de datos.")

        # 5. Diagnóstico: Contar NaNs después de la limpieza
        nan_count = df_final[col].isna().sum()
        
        if nan_count > 0:
            print(f"⚠️ Columna '{col}' tiene {nan_count} valores no numéricos convertidos a NaN que serán eliminados.")
        else:
            print(f"✅ Columna '{col}' es totalmente numérica.")
    else:
        print(f"⚠️ Advertencia: La columna requerida '{col}' no se encontró en los datos.")

# Elimina filas con NaNs después de la conversión forzada (solo si quedan)
df_final = df_final.dropna(subset=[col for col in columnas_requeridas if col in df_final.columns])

print("\nPrimeras 5 filas del DataFrame después de la limpieza (DEBERÍAN SER NUMÉRICAS):")
print(df_final[columnas_requeridas].head())
print("-" * 50)


# ==========================
# VERIFICACIÓN FINAL
# ==========================
if df_final.shape[0] == 0:
    print("\n❌ ERROR CRÍTICO: No quedan registros válidos después de la limpieza. Algo inesperado falló.")
    exit()

print(f"✅ Datos listos para entrenamiento: {df_final.shape[0]} registros limpios.\n")
print(f"➡️ El modelo se entrenará con {len(features)} variables de entrada.")

# ==========================
# ESCALADO Y ENTRENAMIENTO 
# ==========================
X = df_final[features].values
y = df_final[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# La forma (samples, timesteps, features) es correcta para un timestep=1
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# ==========================
# MODELO LSTM
# ==========================
model = Sequential([
    LSTM(128, input_shape=(1, len(features))), 
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1) 
])
model.compile(optimizer='adam', loss='mse')

print("🚀 Entrenando modelo LSTM con StandardScaler y arquitectura mejorada...\n")
history = model.fit(X_lstm, y_scaled, epochs=300, batch_size=8, verbose=1) 

# ==========================
# EVALUACIÓN Y GUARDADO
# ==========================
y_pred_scaled = model.predict(X_lstm, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_real = scaler_y.inverse_transform(y_scaled)

r2 = r2_score(y_real, y_pred)
mae = mean_absolute_error(y_real, y_pred)
mse = mean_squared_error(y_real, y_pred)

print("\n✅ Resultados del modelo LSTM:")
print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")

# Guardar modelos y scalers
os.makedirs("models", exist_ok=True)
model.save("models/modelo_pobreza_lstm.h5")
joblib.dump(scaler_X, "models/scaler_X_lstm.pkl")
joblib.dump(scaler_y, "models/scaler_y_lstm.pkl")

print("\n💾 Modelos y escaladores guardados en carpeta 'models/'")

resultados = pd.DataFrame([{
    "Modelo": "LSTM",
    "R2": round(r2, 3),
    "MAE": round(mae, 3),
    "MSE": round(mse, 3),
    "Registros": df_final.shape[0]
}])
resultados.to_csv("resultados_lstm.csv", index=False)
print("📊 Resultados exportados a 'resultados_lstm.csv'")
print("✅ Entrenamiento completado con éxito.")