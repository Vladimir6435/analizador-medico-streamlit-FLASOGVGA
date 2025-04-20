import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# Configurar cliente OpenAI
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Configuración segura de longitud
MAX_CARACTERES_ENTRADA = 250000  # ~80k tokens
MAX_OUTPUT_TOKENS = 6000

# --- Función para extraer texto desde PDF ---
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# --- Función para análisis clínico por IA ---
def generar_analisis_clinico(texto_total, seccion_objetivo):
    if seccion_objetivo == "Todo el artículo":
        objetivo_prompt = "analiza el artículo completo"
    else:
        objetivo_prompt = f"analiza exclusivamente la sección de {seccion_objetivo.lower()}"

    prompt = f"""

    # --- DEBUG: Mostrar parte del prompt en pantalla ---
st.text("------ PROMPT DEBUG ------")
st.text_area("Prompt que será enviado al modelo:", prompt[:4000])

Actúa como médico especialista en medicina materno-fetal.

Tienes a continuación el contenido de un artículo científico extraído de un PDF:

{texto_total}

Por favor, {objetivo_prompt} y genera un informe profesional para revisión por especialistas clínicos. El informe debe estar estructurado, enfocado en evidencia médica clara, y ser útil para discusión académica o aplicación clínica.

Incluye solo lo que corresponda a la sección seleccionada si así se indica.
"""
    respuesta = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_OUTPUT_TOKENS
    )
    return respuesta.choices[0].message.content

# --- Función para responder preguntas clínicas ---
def responder_pregunta(texto_total, pregunta):
    prompt = f"""
Eres un asistente clínico experto en medicina materno-fetal.

Con base en el siguiente contenido médico extraído de varios artículos científicos:

{texto_total}

Responde la siguiente pregunta del usuario con base en la evidencia presentada:

PREGUNTA: {pregunta}
"""
    respuesta = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=3000
    )
    return respuesta.choices[0].message.content

# --- Interfaz Streamlit ---
st.set_page_config(page_title="FLASOG 2025 - Análisis de Literatura Médica", layout="centered")

st.title("📘 Análisis de Literatura Médica FLASOG 2025")
st.markdown("### 🧠 IA para análisis estructurado de artículos médicos")
st.info("Modelo: gpt-4-turbo (128k tokens) – Máx. 350,000 caracteres por análisis")

uploaded_files = st.file_uploader("📄 Sube uno o más artículos médicos en PDF", type="pdf", accept_multiple_files=True)

seccion_objetivo = st.radio("Selecciona la sección a analizar:",
                            ["Todo el artículo", "Metodología", "Resultados", "Conclusiones"], index=0)

texto_total = ""

if uploaded_files:
    for archivo in uploaded_files:
        texto_total += extract_text_from_pdf(archivo) + "\n\n"

    st.info(f"📏 Caracteres cargados: {len(texto_total)}")

    if len(texto_total) > MAX_CARACTERES_ENTRADA:
        st.warning("⚠️ El texto fue recortado a 350,000 caracteres para ajustarse al límite del modelo.")
        texto_total = texto_total[:MAX_CARACTERES_ENTRADA]
        st.info(f"✂️ Texto recortado a {len(texto_total)} caracteres.")

    if st.button("📑 Generar análisis clínico"):
        with st.spinner("🧠 Analizando con IA..."):
            resultado = generar_analisis_clinico(texto_total, seccion_objetivo)
            st.subheader("📝 Informe clínico generado:")
            st.write(resultado)
            st.download_button("💾 Descargar informe como .txt", resultado, file_name="informe_clinico.txt")

    st.markdown("---")
    st.subheader("💬 Realiza una pregunta personalizada sobre los artículos")

    pregunta = st.text_input("Escribe tu pregunta aquí:")
    if st.button("❓ Obtener respuesta"):
        if pregunta.strip():
            with st.spinner("🤖 Generando respuesta..."):
                respuesta = responder_pregunta(texto_total, pregunta)
                st.markdown("### ✅ Respuesta basada en el contenido:")
                st.write(respuesta)
        else:
            st.warning("Por favor, escribe una pregunta válida.")


