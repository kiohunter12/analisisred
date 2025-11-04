# scripts/entrenamiento.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from modelo_pobreza import crear_modelo

# === 1️⃣ Cargar dataset consolidado ===
df = pd.read_excel("data/Pobreza_2022_2024.xlsx")
print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

# === 2️⃣ Preparar variables (X, y) ===
# Usaremos como etiqueta si está en 'Alta pobreza'
y = (df["umbral_zona_pobreza"] == "Alta pobreza").astype(int)

# Quitamos columnas no numéricas que no aportan al modelo
X = df.drop(columns=[
    "umbral_zona_pobreza",
    "departamento",
    "fuente_datos",
    "año"
], errors="ignore")

# Eliminar columnas con más del 50% de ceros o nulos
col_filtradas = [c for c in X.columns if (X[c] != 0).sum() > len(X) * 0.5]
X = X[col_filtradas]

# Reemplazar nulos por la media
X = X.fillna(X.mean())

print(f"📊 Variables finales: {X.shape[1]} características.")

# === 3️⃣ Normalizar variables ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

# === 4️⃣ Separar datos ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === 5️⃣ Crear modelo ===
modelo = crear_modelo(input_dim=X_train.shape[1])

# === 6️⃣ Entrenar modelo ===
stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

historial = modelo.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    callbacks=[stop]
)

# === 7️⃣ Guardar modelo ===
modelo.save("models/modelo_pobreza.h5")
print("\n💾 Modelo guardado en: models/modelo_pobreza.h5")

# === 8️⃣ Guardar métricas finales ===
loss, acc = modelo.evaluate(X_test, y_test)
print(f"\n📈 Evaluación final:")
print(f"   Pérdida (loss): {loss:.4f}")
print(f"   Exactitud (accuracy): {acc*100:.2f}%")

# === 9️⃣ Exportar histórico de entrenamiento ===
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(historial.history["loss"], label="Entrenamiento")
plt.plot(historial.history["val_loss"], label="Validación")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.title("Evolución del entrenamiento")
plt.legend()
plt.tight_layout()
plt.savefig("visualizaciones/entrenamiento_loss.png", dpi=120)
print("📊 Gráfico guardado en: visualizaciones/entrenamiento_loss.png")
