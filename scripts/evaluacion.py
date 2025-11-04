# scripts/evaluacion.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

# === 1️⃣ Cargar el dataset y los recursos ===
print("📂 Cargando datos y modelo...")

df = pd.read_excel("data/Pobreza_2022_2024.xlsx")
scaler = joblib.load("models/scaler.pkl")
modelo = load_model("models/modelo_pobreza.h5")

# === 2️⃣ Preparar los datos ===
y = (df["umbral_zona_pobreza"] == "Alta pobreza").astype(int)

# Quitamos columnas no numéricas que no aportan al modelo
X = df.drop(columns=["umbral_zona_pobreza", "departamento", "fuente_datos", "año"], errors="ignore")

# Reemplazar nulos por la media
X = X.fillna(X.mean())

# 🔹 Asegurar que las columnas coincidan con las del scaler
if hasattr(scaler, "feature_names_in_"):
    scaler_features = scaler.feature_names_in_
    # Agregar columnas faltantes con valor 0
    for col in scaler_features:
        if col not in X.columns:
            X[col] = 0
    # Eliminar columnas extra
    X = X[scaler_features]
else:
    print("⚠️ El scaler no contiene nombres de columnas. Se usará directamente con las columnas actuales.")

# === 3️⃣ Normalizar los datos ===
X_scaled = scaler.transform(X)

# === 4️⃣ Dividir en conjunto de prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === 5️⃣ Predicciones ===
print("🤖 Evaluando modelo...")
y_pred_prob = modelo.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()

# === 6️⃣ Reporte de métricas ===
print("\n📈 Reporte de clasificación:\n")
print(classification_report(y_test, y_pred, target_names=["No pobre", "Alta pobreza"]))

# === 7️⃣ Matriz de confusión ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["No pobre", "Alta pobreza"],
            yticklabels=["No pobre", "Alta pobreza"])
plt.title("Matriz de Confusión - Modelo de Pobreza")
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.tight_layout()
plt.savefig("visualizaciones/matriz_confusion.png", dpi=120)
print("📊 Matriz de confusión guardada en: visualizaciones/matriz_confusion.png")

# === 8️⃣ Curva ROC ===
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0,1], [0,1], 'r--')
plt.title("Curva ROC - Modelo de Pobreza")
plt.xlabel("Falsos positivos (FPR)")
plt.ylabel("Verdaderos positivos (TPR)")
plt.legend()
plt.tight_layout()
plt.savefig("visualizaciones/curva_ROC.png", dpi=120)
print("📈 Curva ROC guardada en: visualizaciones/curva_ROC.png")

# === 9️⃣ Guardar resumen en CSV ===
resultados = {
    "accuracy": np.mean(y_pred == y_test),
    "auc": roc_auc,
    "total_pruebas": len(y_test)
}
pd.DataFrame([resultados]).to_csv("visualizaciones/resumen_evaluacion.csv", index=False)
print("💾 Resumen guardado en: visualizaciones/resumen_evaluacion.csv")

print("\n✅ Evaluación completada correctamente.")
