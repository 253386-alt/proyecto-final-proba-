import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
import numpy as np
import requests

# 1. Configuración de página
st.set_page_config(page_title="IA Estadístico", layout="wide")

# 2. Barra lateral para la API Key
with st.sidebar:
    st.title("⚙️ Configuración")
    api_key = st.text_input("Introduce tu OpenRouter API Key", type="password")

    if api_key:
        st.success("✅ API Conectada")

# 3. Título Principal y Carga de Archivos
st.title("📊 Asistente Estadístico Interactivo")
uploaded_file = st.file_uploader("Sube tu archivo CSV para analizar", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    tab1, tab2, tab3 = st.tabs(["📈 Distribución", "🧪 Hipótesis", "🤖 Asistente IA"])

    columnas_num = df.select_dtypes(include=[np.number]).columns

    if len(columnas_num) > 0:

        # PESTAÑA 1: VISUALIZACIÓN
        with tab1:
            col_sel = st.selectbox("Selecciona una columna para graficar", columnas_num, key="visual")
            fig = px.histogram(df, x=col_sel, marginal="box", title=f"Distribución de {col_sel}")
            st.plotly_chart(fig, width="stretch")

        # PESTAÑA 2: ESTADÍSTICAS
        with tab2:
            st.subheader("Prueba de Normalidad (Shapiro-Wilk)")
            col_sel2 = st.selectbox("Selecciona una columna para analizar", columnas_num, key="hipotesis")
            datos = df[col_sel2].dropna()

            if len(datos) >= 3:
                stat, p_val = stats.shapiro(datos)
                st.metric("P-Value", f"{p_val:.4f}")

                if p_val > 0.05:
                    st.success("✅ Los datos parecen seguir una distribución normal.")
                else:
                    st.warning("⚠️ Los datos NO siguen una distribución normal.")
            else:
                st.error("Se necesitan al menos 3 datos.")

        # PESTAÑA 3: ASISTENTE IA
        with tab3:
            st.subheader("🤖 Consulta al experto con IA")

            if not api_key:
                st.warning("Ingresa la API Key en la barra lateral.")
            else:
                col_sel3 = st.selectbox("Selecciona una columna para consultar", columnas_num, key="ia")
                datos3 = df[col_sel3].dropna()

                if len(datos3) >= 3:
                    _, p_val3 = stats.shapiro(datos3)
                else:
                    p_val3 = None

                pregunta = st.text_input("Hazle una pregunta a la IA sobre tus resultados:")

                if st.button("Consultar a IA"):
                    if p_val3 is not None:
                        contexto = (
                            f"Eres un experto en estadística. "
                            f"Analiza este resultado: Variable '{col_sel3}', "
                            f"p-value de Shapiro-Wilk = {p_val3:.4f}. "
                            f"Pregunta del usuario: {pregunta}"
                        )
                    else:
                        contexto = f"Eres un experto en estadística. Pregunta del usuario: {pregunta}"

                    with st.spinner("La IA está analizando..."):
                        try:
                            response = requests.post(
                                url="https://openrouter.ai/api/v1/chat/completions",
                                headers={
                                    "Authorization": f"Bearer {api_key}",
                                    "Content-Type": "application/json"
                                },
                                json={
                                    "model": "mistralai/mistral-7b-instruct",
                                    "messages": [
                                        {"role": "user", "content": contexto}
                                    ]
                                }
                            )

                            result = response.json()

                            if "choices" in result:
                                respuesta = result["choices"][0]["message"]["content"]
                                st.info(respuesta)
                            else:
                                st.error(f"Error de la API: {result}")

                        except Exception as e:
                            st.error(f"Error de conexión: {e}")
    else:
        st.error("No hay columnas numéricas en el archivo.")
else:
    st.info("👋 Sube un archivo CSV para comenzar.")
