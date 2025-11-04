import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium
import os

# === Configuración general de la app ===
st.set_page_config(page_title="Dashboard de Pobreza en el Perú (2022–2024)", layout="wide")

data_dir = "data"
geojson_path = os.path.join(data_dir, "peru_departamental.geojson")

# === Archivos Excel (usa los corregidos) ===
archivos_excel = [
    os.path.join(data_dir, "Pobreza_2022_CORREGIDO.xlsx"),
    os.path.join(data_dir, "Pobreza_2023_CORREGIDO.xlsx"),
    os.path.join(data_dir, "Pobreza_2024_CORREGIDO.xlsx"),
]

# === Cargar y combinar los Excel ===
dfs = []
for archivo in archivos_excel:
    if os.path.exists(archivo):
        dfs.append(pd.read_excel(archivo))
df = pd.concat(dfs, ignore_index=True)

# Normalizar columnas y nombres
df.columns = df.columns.str.strip().str.lower()
df["departamento"] = df["departamento"].str.upper().str.strip()

# Cargar GeoJSON
geo = gpd.read_file(geojson_path)
geo["NOMBDEP"] = geo["NOMBDEP"].str.upper().str.strip()

# === Barra lateral (filtros) ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/893/893292.png", width=60)
st.sidebar.title("📊 Dashboard de Pobreza en el Perú")
año = st.sidebar.selectbox("Selecciona un año:", sorted(df["año"].unique()))
st.sidebar.markdown("---")

# Filtrar por año
datos_año = df[df["año"] == año]

# === Crear mapa ===
m = folium.Map(location=[-9.2, -75.0], zoom_start=5, tiles="cartodb dark_matter")

merged = geo.merge(datos_año, left_on="NOMBDEP", right_on="departamento", how="left")

# Capa de color por pobreza total
folium.Choropleth(
    geo_data=merged,
    data=merged,
    columns=["NOMBDEP", "pobreza_total_%"],
    key_on="feature.properties.NOMBDEP",
    fill_color="YlOrRd",
    nan_fill_color="gray",
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name=f"Pobreza total (%) - {año}",
).add_to(m)

# Tooltip con datos
tooltip = GeoJsonTooltip(
    fields=["NOMBDEP", "pobreza_total_%", "pobreza_extrema_%", "empleo_informal_%", "sin_internet_%", "umbral_zona_pobreza"],
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

folium.GeoJson(
    merged,
    name="Datos",
    style_function=lambda x: {"color": "white", "weight": 0.3, "fillOpacity": 0},
    tooltip=tooltip,
).add_to(m)

# === Layout con columnas ===
col1, col2 = st.columns([2.5, 1.2])

with col1:
    st.markdown(f"### 🗺️ Mapa de Pobreza {año}")
    st_data = st_folium(m, width=800, height=600)

with col2:
    st.markdown(f"### 📈 Indicadores Nacionales {año}")
    promedio_total = datos_año["pobreza_total_%"].mean()
    promedio_extrema = datos_año["pobreza_extrema_%"].mean()
    promedio_informal = datos_año["empleo_informal_%"].mean()
    promedio_internet = datos_año["sin_internet_%"].mean()

    st.metric("💰 Pobreza Total Promedio", f"{promedio_total:.1f}%")
    st.metric("⚠️ Pobreza Extrema Promedio", f"{promedio_extrema:.1f}%")
    st.metric("👷 Empleo Informal Promedio", f"{promedio_informal:.1f}%")
    st.metric("🌐 Sin Internet Promedio", f"{promedio_internet:.1f}%")

    st.markdown("---")
    st.dataframe(
        datos_año[["departamento", "pobreza_total_%", "pobreza_extrema_%", "empleo_informal_%", "sin_internet_%", "umbral_zona_pobreza"]]
        .sort_values("pobreza_total_%", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=450
    )

st.success(f"✅ Dashboard interactivo actualizado para el año {año}")
