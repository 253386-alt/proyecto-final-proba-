import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.figure_factory as ff
import plotly.graph_objects as go
import google.generativeai as genai
import streamlit.components.v1 as components

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Hypothesis Lab Pro: Ocre & AI", layout="wide")

# --- 2. ESTÉTICA PERSONALIZADA (CSS) ---
st.markdown("""
    <style>
    /* Fondo Ocre Negro */
    .stApp {
        background-color: #1a1915;
        color: #e0d9c5;
    }
    
    /* Headers en Oro Viejo */
    h1, h2, h3, h4 {
        color: #d4af37 !important;
        font-family: 'Georgia', serif;
    }
    
    /* Tarjetas de métricas */
    .stMetric {
        background-color: #26241e !important;
        border: 1px solid #443f33 !important;
        border-radius: 10px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #26241e;
        border-right: 1px solid #443f33;
    }

    /* Botones Dorados */
    .stButton>button {
        background-color: #d4af37;
        color: #1a1915;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #e0d9c5;
        color: #1a1915;
    }

    /* Tarjeta de interpretación */
    .advice-card {
        background-color: #26241e;
        padding: 20px;
        border-left: 5px solid #d4af37;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .stMarkdown { color: #e0d9c5; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. EFECTO SEGUIDOR DE CURSOR (JS) ---
components.html("""
    <div id="cursor-follower" style="
        position: fixed;
        width: 15px;
        height: 15px;
        background: rgba(212, 175, 55, 0.5);
        border-radius: 50%;
        pointer-events: none;
        z-index: 9999;
        transition: transform 0.08s ease-out;
        filter: blur(3px);
    "></div>
    <script>
        const follower = document.getElementById('cursor-follower');
        document.addEventListener('mousemove', (e) => {
            follower.style.left = e.clientX - 7 + 'px';
            follower.style.top = e.clientY - 7 + 'px';
        });
    </script>
    """, height=0)

# --- 4. BARRA LATERAL (DATOS Y API) ---
with st.sidebar:
    st.header("🔑 Configuración")
    api_key = st.text_input("Gemini API Key", type="password", help="Pégala aquí para activar la IA")
    
    st.divider()
    st.header("📁 Carga de Datos")
    origen = st.radio("Fuente de datos:", ("Sintéticos (Normal)", "Archivo CSV"))
    
    if origen == "Sintéticos (Normal)":
        n_input = st.slider("Tamaño de muestra (n)", 30, 2000, 500)
        mu_gen = st.number_input("Media Poblacional Real", value=100.0)
        sigma_gen = st.number_input("Desv. Estándar Real", value=15.0)
        df = pd.DataFrame({'Valores': np.random.normal(mu_gen, sigma_gen, n_input)})
    else:
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.DataFrame()

# --- 5. CUERPO PRINCIPAL ---
st.title("🏛️ Laboratorio de Inferencia Estadística")

if not df.empty:
    col_num = df.select_dtypes(include=[np.number]).columns.tolist()
    if not col_num:
        st.error("El archivo no contiene columnas numéricas.")
    else:
        sel = st.selectbox("Seleccione la variable a analizar:", col_num)
        data = df[sel].dropna()

        # --- SECCIÓN: DIAGNÓSTICO ---
        st.header("🔍 1. Exploración y Diagnóstico")
        
        mean_s, std_s, n_s = data.mean(), data.std(), len(data)
        k2, p_norm = stats.normaltest(data)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Media Muestral (x̄)", f"{mean_s:.2f}")
        c2.metric("Desviación Est. (s)", f"{std_s:.2f}")
        c3.metric("Normalidad (p-valor)", f"{p_norm:.4f}", "Normal" if p_norm > 0.05 else "No Normal")

        with st.expander("📊 Ver Tabla de Frecuencias Agrupadas"):
            counts, bins = np.histogram(data, bins='sturges')
            freq_df = pd.DataFrame({
                'Intervalo Inferior': bins[:-1].round(2),
                'Intervalo Superior': bins[1:].round(2),
                'Frecuencia Absoluta': counts,
                'Frecuencia Relativa (%)': (counts / len(data) * 100).round(2)
            })
            st.table(freq_df)

        st.divider()

        # --- SECCIÓN: PRUEBA DE HIPÓTESIS ---
        st.header("🧪 2. Prueba Z de una Muestra")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        h0_val = col_in1.number_input("H₀: μ =", value=100.0)
        tipo_h1 = col_in2.selectbox("H₁ (Hipótesis Alternativa):", 
                                   ["Bilateral (μ ≠ μ₀)", "Cola Izquierda (μ < μ₀)", "Cola Derecha (μ > μ₀)"])
        alpha = col_in3.select_slider("Nivel de Significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)

        z_calc = (mean_s - h0_val) / (std_s / np.sqrt(n_s))
        
        if "Bilateral" in tipo_h1:
            p_val = stats.norm.sf(abs(z_calc)) * 2
            z_crit_inf, z_crit_sup = stats.norm.ppf(alpha/2), stats.norm.ppf(1 - alpha/2)
            h1_sym, alt_ia = "≠", "bilateral"
        elif "Izquierda" in tipo_h1:
            p_val = stats.norm.cdf(z_calc)
            z_crit_inf, z_crit_sup = stats.norm.ppf(alpha), None
            h1_sym, alt_ia = "<", "unilateral izquierda"
        else:
            p_val = stats.norm.sf(z_calc)
            z_crit_inf, z_crit_sup = None, stats.norm.ppf(1 - alpha)
            h1_sym, alt_ia = ">", "unilateral derecha"

        st.subheader("🎯 Resultado de la Inferencia")
        decision = "RECHAZAR H₀" if p_val < alpha else "NO RECHAZAR H₀"
        
        st.markdown(f"""
        <div class="advice-card">
            <h3>Conclusión: {decision}</h3>
            <p>Z-Calculado: <b>{z_calc:.4f}</b> | P-Valor: <b>{p_val:.4f}</b> | Alpha: <b>{alpha}</b></p>
        </div>
        """, unsafe_allow_html=True)

        x_plot = np.linspace(-4, 4, 1000)
        y_plot = stats.norm.pdf(x_plot, 0, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', line=dict(color='#e0d9c5'), name='Distribución Z'))
        
        if "Bilateral" in tipo_h1:
            fig.add_trace(go.Scatter(x=x_plot[x_plot<=z_crit_inf], y=y_plot[x_plot<=z_crit_inf], fill='tozeroy', marker_color='#ff4b4b', name='R. Rechazo'))
            fig.add_trace(go.Scatter(x=x_plot[x_plot>=z_crit_sup], y=y_plot[x_plot>=z_crit_sup], fill='tozeroy', marker_color='#ff4b4b', showlegend=False))
        elif "Izquierda" in tipo_h1:
            fig.add_trace(go.Scatter(x=x_plot[x_plot<=z_crit_inf], y=y_plot[x_plot<=z_crit_inf], fill='tozeroy', marker_color='#ff4b4b', name='R. Rechazo'))
        else:
            fig.add_trace(go.Scatter(x=x_plot[x_plot>=z_crit_sup], y=y_plot[x_plot>=z_crit_sup], fill='tozeroy', marker_color='#ff4b4b', name='R. Rechazo'))

        fig.add_vline(x=z_calc, line_width=3, line_dash="dot", line_color="#d4af37", annotation_text=f"Tu Z ({z_calc:.2f})")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0d9c5"), height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- 6. MÓDULO DE IA (CON CORRECCIÓN 404) ---
        st.header("🤖 3. Consultoría con IA")
        col_ia_info, col_ia_run = st.columns([2, 1])
        
        with col_ia_info:
            st.write("Interpretación avanzada de resultados y validación de supuestos estadísticos.")
            if p_norm < 0.05:
                st.warning("⚠️ Nota: La normalidad ha fallado. La IA analizará la validez del test.")

        with col_ia_run:
            if api_key:
                if st.button("✨ Generar Reporte con IA"):
                    with st.spinner("Gemini analizando resultados..."):
                        try:
                            genai.configure(api_key=api_key)
                            # CORRECCIÓN CLAVE: Usar el nombre completo del modelo
                            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
                            
                            prompt = f"""
                            Eres un experto estadístico. Analiza este Z-test:
                            - Variable: {sel} (n={n_s})
                            - H0: mu={h0_val} | H1: mu {h1_sym} {h0_val}
                            - Estadístico Z: {z_calc:.4f} | P-valor: {p_val:.4f} | Alpha: {alpha}
                            - Normalidad: p={p_norm:.4f}
                            - Decisión tomada: {decision}
                            
                            Responde:
                            1. ¿Es estadísticamente correcta la decisión dada la normalidad y n?
                            2. Explica el p-valor de forma sencilla.
                            3. ¿Qué implicación práctica tiene esto para un negocio?
                            """
                            
                            response = model.generate_content(prompt)
                            st.info(f"**Análisis de Gemini IA:**\n\n{response.text}")
                        except Exception as e:
                            st.error(f"Error con la API: {e}")
            else:
                st.info("Introduce tu API Key para activar.")

else:
    st.info("🌙 Esperando datos... Usa el panel de la izquierda.")