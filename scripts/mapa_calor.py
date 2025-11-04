import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
import json
import os

# === Configuración general ===
data_dir = "data"
output = "visualizaciones/mapa_pobreza_dashboard.html"
geojson_path = os.path.join(data_dir, "peru_departamental.geojson")

# === Archivos Excel ===
archivos_excel = [
    os.path.join(data_dir, "Pobreza_2022_CORREGIDO.xlsx"),
    os.path.join(data_dir, "Pobreza_2023_CORREGIDO.xlsx"),
    os.path.join(data_dir, "Pobreza_2024_CORREGIDO.xlsx"),
]

# === Columnas principales ===
col_departamento = "departamento"
col_año = "año"
col_pobreza = "pobreza_total_%"
col_extrema = "pobreza_extrema_%"
col_informal = "empleo_informal_%"
col_internet = "sin_internet_%"
col_umbral = "umbral_zona_pobreza"

print("📂 Cargando datos...")

# === Cargar todos los Excel ===
dfs = []
for archivo in archivos_excel:
    if os.path.exists(archivo):
        df = pd.read_excel(archivo)
        dfs.append(df)
    else:
        print(f"⚠️ No se encontró: {archivo}")

# Combinar todos los años
df = pd.concat(dfs, ignore_index=True)
print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")

# Normalizar nombres de columnas y departamentos
df.columns = df.columns.str.strip().str.lower()
df[col_departamento] = df[col_departamento].str.upper().str.strip()

# Cargar GeoJSON
geo = gpd.read_file(geojson_path)
geo["NOMBDEP"] = geo["NOMBDEP"].str.upper().str.strip()

# Validar coincidencias
deptos_excel = set(df[col_departamento].unique())
deptos_geo = set(geo["NOMBDEP"].unique())
coinciden = deptos_excel & deptos_geo
print(f"✅ Departamentos coincidentes: {len(coinciden)} de {len(deptos_excel)}")

# === Crear mapa base ===
m = folium.Map(location=[-9.2, -75.0], zoom_start=5, tiles="cartodb dark_matter")

# === Añadir capas por año ===
años = sorted(df[col_año].unique())

for año in años:
    print(f"🗺️ Generando capa para el año {año}...")
    datos = df[df[col_año] == año]
    merged = geo.merge(datos, left_on="NOMBDEP", right_on=col_departamento, how="left")

    # Mapa de calor (choropleth)
    choropleth = folium.Choropleth(
        geo_data=merged,
        data=merged,
        columns=["NOMBDEP", col_pobreza],
        key_on="feature.properties.NOMBDEP",
        fill_color="YlOrRd",
        nan_fill_color="gray",
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name=f"Pobreza total (%) - {año}",
        name=f"Año {año}",
    )
    choropleth.add_to(m)

    # Tooltip
    tooltip = GeoJsonTooltip(
        fields=["NOMBDEP", col_pobreza, col_extrema, col_informal, col_internet, col_umbral],
        aliases=[
            "Departamento:",
            "Pobreza total (%):",
            "Pobreza extrema (%):",
            "Empleo informal (%):",
            "Sin internet (%):",
            "Umbral de pobreza:",
        ],
        localize=True,
        sticky=True,
        labels=True,
        style="background-color: #222; color: #fff; font-family: Arial; font-size: 13px; padding: 8px;",
    )

    # Añadir tooltip encima de cada departamento
    folium.GeoJson(
        merged,
        name=f"{año}",
        style_function=lambda x: {
            "fillColor": "#00000000",
            "color": "white",
            "weight": 0.5,
            "fillOpacity": 0,
        },
        tooltip=tooltip,
    ).add_to(m)

# === Añadir control de capas ===
folium.LayerControl(collapsed=False).add_to(m)

# === Guardar resultado ===
os.makedirs("visualizaciones", exist_ok=True)
m.save(output)
print(f"✅ Dashboard generado: {output}")
print("👉 Abre ese archivo en tu navegador.")
