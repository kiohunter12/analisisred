import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from tensorflow.keras.models import load_model
from tensorflow.keras import losses, metrics
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import re

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
    # Asume que los archivos están en una carpeta 'data' al lado de 'app.py'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    try:
        df_2022 = pd.read_excel(os.path.join(data_dir, "Pobreza_2022_CORREGIDO.xlsx"))
        df_2023 = pd.read_excel(os.path.join(data_dir, "Pobreza_2023_CORREGIDO.xlsx"))
        df_2024 = pd.read_excel(os.path.join(data_dir, "Pobreza_2024_CORREGIDO.xlsx"))
        
        # Carga del GeoJSON
        geo = gpd.read_file(os.path.join(data_dir, "peru_departamental.geojson"))

        df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
        # Normalizar nombres de columna para consistencia con la carga y el entrenamiento
        df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c.lower().strip()) for c in df.columns]

        # Aplicar la misma lógica de limpieza que en el script de entrenamiento
        # Esto es crucial para asegurar que el gráfico histórico y la predicción usen datos limpios
        
        # --- VARIABLES NUMÉRICAS REQUERIDAS ---
        # Asegúrate de que esta lista coincida con el script de entrenamiento
        features_all = [
            "pobreza_extrema__",    
            "empleo_informal__",    
            "sin_internet__",       
            "umbral_zona_pobreza",   
            "pobreza_total__"
        ]

        for col in features_all:
            if col in df.columns:
                # Limpieza de comas, espacios y porcentajes
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(',', '.', regex=True).str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Tratar la columna problemática (rellenar con 0 si todos son NaN)
                if col == "umbral_zona_pobreza" and df[col].isna().sum() == df.shape[0]:
                    df[col] = 0
            
        # Eliminar filas con NaNs (si queden)
        df = df.dropna(subset=[col for col in features_all if col in df.columns])

        st.success(f"✅ Datos cargados y limpios: {df.shape[0]} registros totales.")
    except Exception as e:
        st.error(f"❌ Error al cargar o limpiar los datos: {e}")
        df, geo = pd.DataFrame(), None

    return df, geo


@st.cache_resource
def cargar_modelos():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    # === Modelo Denso ===
    modelo_denso, scaler_denso = None, None
    try:
        modelo_denso = load_model(os.path.join(models_dir, "modelo_pobreza.h5"))
        # Asumiendo que el scaler_denso es el mismo que el denso original (scaler.pkl)
        scaler_denso = joblib.load(os.path.join(models_dir, "scaler.pkl")) 
        st.success("✅ Modelo Denso cargado correctamente.")
    except Exception as e:
        st.warning(f"⚠️ Modelo Denso no cargado o puede tener un número incorrecto de features (espera 4 ahora): {e}") 

    # === Modelo LSTM ===
    modelo_lstm, scaler_X_lstm, scaler_y_lstm = None, None, None
    try:
        # Cargamos el modelo LSTM reentrenado
        modelo_lstm = load_model(
            os.path.join(models_dir, "modelo_pobreza_lstm.h5"),
            compile=False
        )
        # Recompilamos para compatibilidad
        modelo_lstm.compile(
            optimizer="adam",
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
        )
        scaler_X_lstm = joblib.load(os.path.join(models_dir, "scaler_X_lstm.pkl"))
        scaler_y_lstm = joblib.load(os.path.join(models_dir, "scaler_y_lstm.pkl"))
        st.success("✅ Modelo LSTM cargado correctamente.")
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo LSTM: {e}")

    return modelo_denso, scaler_denso, modelo_lstm, scaler_X_lstm, scaler_y_lstm


# === Cargar datos y modelos ===
df, geo = cargar_datos()
modelo_denso, scaler_denso, modelo_lstm, scaler_X_lstm, scaler_y_lstm = cargar_modelos()

if df.empty:
    st.stop()

# ============================
# FUNCIONES AUXILIARES
# ============================
def pintar_mapa(df_anio: pd.DataFrame, titulo: str = ""):
    m = folium.Map(location=[-9.19, -75.0152], zoom_start=5, tiles="cartodb dark_matter")
    merged = geo.merge(df_anio, left_on="NOMBDEP", right_on="departamento", how="left")
    
    # Columna objetivo real
    columna_pobreza = 'pobreza_total__' 

    folium.Choropleth(
        geo_data=merged,
        name="choropleth",
        data=merged,
        columns=["departamento", columna_pobreza],
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
    años = sorted(df["a_o"].unique()) 
    # Aseguramos que el año sea un entero para el selectbox
    años = [int(a) for a in años]
    año_sel = st.sidebar.selectbox("Selecciona un año", años, index=max(0, len(años) - 1))
    df_año = df[df["a_o"] == año_sel].copy()
    
    m = pintar_mapa(df_año, titulo=str(año_sel))
    st_folium(m, width=780, height=520)
    st.dataframe(df_año.reset_index(drop=True))


# ============================
# MODO PREDICCIÓN 2025
# ============================
else:
    st.title("🤖 Predicción de Pobreza 2025")

    tipo_modelo = st.sidebar.selectbox(
        "Selecciona el modelo a usar:",
        ["Red LSTM (Temporal)", "Red Neuronal Densa (Actual)"] 
    )
    
    # Entradas del usuario (AJUSTADAS A SOLO 4 FEATURES del entrenamiento)
    st.sidebar.subheader("Variables de entrada (4 Features)")
    # Valores por defecto basados en la última imagen que enviaste (27.7%)
    x1 = st.sidebar.number_input("1. Pobreza Extrema (%)", 0.0, 100.0, 5.60, key="x1_input") 
    x2 = st.sidebar.number_input("2. Empleo Informal (%)", 0.0, 100.0, 40.0, key="x2_input") 
    x3 = st.sidebar.number_input("3. Población sin Internet (%)", 0.0, 100.0, 40.0, key="x3_input") 
    x4 = st.sidebar.number_input("4. Umbral Zona Pobreza", 0.0, 100.0, 11.0, key="x4_input") 
    
    # Vector de entrada (4 elementos en el orden correcto)
    X = np.array([[x1, x2, x3, x4]], dtype=float)

    # ============================================
    # PREDICCIÓN CON MODELO DENSO
    # ============================================
    if tipo_modelo == "Red Neuronal Densa (Actual)":
        if modelo_denso is not None:
            try:
                # El modelo denso espera un input escalado con 4 features
                Xs = scaler_denso.transform(X)
                y_pred = float(modelo_denso.predict(Xs, verbose=0)[0][0])
                st.success("✅ Usando modelo Denso entrenado.")
            except Exception as e:
                st.error(f"❌ Error: El modelo Denso espera un número diferente de features ({X.shape[1]}). Reentrénalo o usa LSTM.")
                y_pred = np.nan
        else:
            st.error("⚠️ Modelo Denso no encontrado.")
            y_pred = np.nan

    # ============================================
    # PREDICCIÓN CON MODELO LSTM
    # ============================================
    else: # Red LSTM (Temporal)
        if modelo_lstm is not None:
            try:
                # Escalar y ajustar la forma (1 muestra, 1 timestep, 4 features)
                Xs = scaler_X_lstm.transform(X)
                
                # X_seq tiene la forma (1, 1, 4)
                X_seq = Xs.reshape((1, 1, Xs.shape[1]))

                # Predicción
                y_pred_scaled = modelo_lstm.predict(X_seq, verbose=0)
                
                # Desnormalización
                y_pred = scaler_y_lstm.inverse_transform(y_pred_scaled)[0][0]

                st.info(f"🔁 Usando modelo LSTM entrenado con {X.shape[1]} variables.")
            except Exception as e:
                st.error(f"❌ Error al predecir con LSTM: {e}. Asegúrate de que los archivos 'scaler' y 'modelo' sean los correctos.")
                y_pred = np.nan
        else:
            st.error("⚠️ No se encontró el modelo LSTM entrenado.")
            y_pred = np.nan

    # ============================================
    # MOSTRAR RESULTADOS Y GRÁFICO (USANDO STREAMLIT LINE CHART)
    # ============================================
    if not np.isnan(y_pred):
        st.metric("🔮 Pobreza total proyectada (2025)", f"{y_pred:.2f}%")

        # Usamos la columna real 'pobreza_total__'
        pobreza_col = 'pobreza_total__' 
            
        prom = df.groupby("a_o", as_index=False)[pobreza_col].mean().sort_values("a_o")
        
        # Preparamos el DataFrame para el gráfico
        tendencia_df = prom.rename(columns={'a_o': 'Año', pobreza_col: 'Pobreza Total (%)'})
        
        # Añadir la predicción de 2025
        prediccion_2025 = pd.DataFrame({'Año': [2025], 'Pobreza Total (%)': [y_pred]})
        tendencia_df = pd.concat([tendencia_df, prediccion_2025], ignore_index=True)
        
        # 💡 CAMBIO CLAVE para mayor dinamismo:
        # Revertimos los cambios de indexación y usamos la columna 'Año' como categoría.
        # Streamlit lo interpretará mejor forzando el rango Y por los valores, 
        # y usamos la opción use_container_width.

        # ====== 📈 GRÁFICO DE TENDENCIA CON st.line_chart (DINÁMICO) ======
        st.markdown("### 📊 Evolución de la pobreza total promedio (2022–2025)")

        st.line_chart(
            tendencia_df,
            x='Año', # Eje X: Columna 'Año'
            y='Pobreza Total (%)', # Eje Y: Columna 'Pobreza Total (%)'
            # 💡 Agregamos etiquetas y activamos el ancho completo
            x_label='Año',
            y_label='Pobreza Total (%)',
            use_container_width=True # Hace que el gráfico ocupe todo el ancho
        )

        st.markdown(f"""
        <div style="text-align: right; color: #FF595E; font-weight: bold;">
            <small>Predicción 2025: {y_pred:.2f}% (Modelo: {tipo_modelo})</small>
        </div>
        """, unsafe_allow_html=True)
        