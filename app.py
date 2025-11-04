import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ============================
# CONFIGURACIÓN INICIAL
# ============================
st.set_page_config(page_title="Dashboard de Pobreza en el Perú", layout="wide")
st.sidebar.title("📊 Dashboard de Pobreza en el Perú")

modo = st.sidebar.radio(
    "Selecciona modo de vista:",
    ["🕰️ Histórico", "🤖 Predicción 2025"]
)

# ============================
# CARGA DE DATOS Y MODELOS
# ============================
@st.cache_data
def cargar_datos():
    df_2022 = pd.read_excel("data/Pobreza_2022_CORREGIDO.xlsx")
    df_2023 = pd.read_excel("data/Pobreza_2023_CORREGIDO.xlsx")
    df_2024 = pd.read_excel("data/Pobreza_2024_CORREGIDO.xlsx")
    geo = gpd.read_file("data/peru_departamental.geojson")
    df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
    return df, geo

@st.cache_resource
def cargar_modelos():
    modelo = load_model("models/modelo_pobreza.h5")
    scaler = joblib.load("models/scaler.pkl")
    return modelo, scaler

df, geo = cargar_datos()
modelo, scaler = cargar_modelos()

# ============================
# FUNCIONES AUXILIARES
# ============================
def pintar_mapa(df_anio: pd.DataFrame, titulo: str = ""):
    m = folium.Map(location=[-9.19, -75.0152], zoom_start=5, tiles="cartodb dark_matter")
    merged = geo.merge(df_anio, left_on="NOMBDEP", right_on="departamento", how="left")
    folium.Choropleth(
        geo_data=merged,
        name="choropleth",
        data=merged,
        columns=["departamento", "pobreza_total_%"],
        key_on="feature.properties.NOMBDEP",
        fill_color="YlOrRd",
        fill_opacity=0.85,
        line_opacity=0.3,
        nan_fill_color="#444444",
        legend_name=f"Pobreza total (%) — {titulo}",
    ).add_to(m)
    return m

# ============================
# MODO HISTÓRICO
# ============================
if modo == "🕰️ Histórico":
    st.title("📘 Mapa de Pobreza (2022–2024)")
    años = sorted(df["año"].unique())
    año_sel = st.sidebar.selectbox("Selecciona un año", años, index=max(0, len(años) - 1))
    df_año = df[df["año"] == año_sel].copy()
    m = pintar_mapa(df_año, titulo=str(año_sel))
    st_folium(m, width=780, height=520)
    st.dataframe(df_año.reset_index(drop=True))

# ============================
# MODO PREDICCIÓN 2025
# ============================
else:
    st.title("🤖 Predicción de Pobreza 2025")
    x1 = st.sidebar.number_input("pobreza_total_%_2023", 0.0, 100.0, 28.10)
    x2 = st.sidebar.number_input("empleo_informal_%", 0.0, 100.0, 37.90)
    x3 = st.sidebar.number_input("subempleo_%", 0.0, 100.0, 22.50)
    x4 = st.sidebar.number_input("uso_internet_no_%", 0.0, 100.0, 52.10)
    x5 = st.sidebar.number_input("viviendas_piso_tierra_%", 0.0, 100.0, 30.50)
    x6 = st.sidebar.number_input("anemia_infantil_%", 0.0, 100.0, 40.20)
    x7 = st.sidebar.number_input("poblacion_sin_servicios_%", 0.0, 100.0, 18.70)

    X = np.array([[x1, x2, x3, x4, x5, x6, x7]], dtype=float)
    Xs = scaler.transform(X)
    y_pred = float(modelo.predict(Xs, verbose=0)[0][0])

    st.metric("🔮 Pobreza total proyectada (2025)", f"{y_pred:.2f}%")

    prom = df.groupby("año", as_index=False)["pobreza_total_%"].mean().sort_values("año")
    prom_2025 = pd.DataFrame({"año": [2025], "pobreza_total_%": [y_pred]})
    tendencia = pd.concat([prom, prom_2025], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(tendencia["año"], tendencia["pobreza_total_%"], marker="o", linewidth=2)
    ax.scatter(2025, y_pred, s=120)
    ax.set_title("Evolución de la pobreza total promedio (2022–2025)")
    ax.set_xlabel("Año")
    ax.set_ylabel("Pobreza total (%)")
    st.pyplot(fig)