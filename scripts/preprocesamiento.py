# scripts/preprocesamiento.py
import pandas as pd
import numpy as np
import os

# === 1️⃣ Cargar los tres archivos ===
rutas = [
    "data/Pobreza_2022.xlsx",
    "data/Pobreza_2023.xlsx",
    "data/Pobreza_2024.xlsx"
]

dataframes = []
for ruta in rutas:
    if os.path.exists(ruta):
        df = pd.read_excel(ruta)
        df["año"] = int(ruta.split("_")[-1].split(".")[0])  # extraer año del nombre del archivo
        dataframes.append(df)
        print(f"✅ Cargado: {ruta} ({df.shape[0]} filas)")
    else:
        print(f"⚠️ No se encontró: {ruta}")

# === 2️⃣ Unir los tres DataFrames ===
df_total = pd.concat(dataframes, ignore_index=True)
print(f"\n🔹 Total combinado: {df_total.shape[0]} filas, {df_total.shape[1]} columnas")

# === 3️⃣ Normalizar nombres de columnas ===
df_total.columns = (
    df_total.columns.str.lower()
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("á", "a")
    .str.replace("é", "e")
    .str.replace("í", "i")
    .str.replace("ó", "o")
    .str.replace("ú", "u")
)

# === 4️⃣ Asegurar columnas esenciales ===
columnas_necesarias = [
    "departamento", "pobreza_total", "pobreza_extrema",
    "empleo_informal", "subempleo", "internet", "piso_tierra",
    "anemia", "agua_potable", "desague", "energia_electrica",
    "umbral_zona_pobreza", "fuente_datos", "año"
]

for col in columnas_necesarias:
    if col not in df_total.columns:
        df_total[col] = np.nan

# === 5️⃣ Completar valores faltantes y limpieza ===
df_total = df_total.dropna(subset=["departamento"]).fillna(0)
df_total["departamento"] = df_total["departamento"].str.strip().str.title()

# === 6️⃣ Asignar umbral de pobreza automática (por si falta) ===
def asignar_umbral(pobreza):
    if pobreza >= 35:
        return "Alta pobreza"
    elif pobreza >= 20:
        return "Media pobreza"
    else:
        return "Baja pobreza"

if "umbral_zona_pobreza" not in df_total or df_total["umbral_zona_pobreza"].isna().all():
    df_total["umbral_zona_pobreza"] = df_total["pobreza_total"].apply(asignar_umbral)

# === 7️⃣ Fuente de datos ===
df_total["fuente_datos"] = df_total["año"].apply(
    lambda x: f"INEI - Cifras de Pobreza Monetaria {x}"
)

# === 8️⃣ Guardar archivo consolidado ===
salida = "data/Pobreza_2022_2024.xlsx"
df_total.to_excel(salida, index=False)
print(f"\n💾 Archivo generado correctamente: {salida}")

# === 9️⃣ Vista previa ===
print("\n📊 Vista previa de datos combinados:")
print(df_total.head())
