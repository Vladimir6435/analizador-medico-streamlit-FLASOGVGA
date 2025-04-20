import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# Inicializar cliente OpenAI
client = OpenAI(api_key=st.secrets["openai_api_key"])

MAX_CARACTERES_ENTRADA = 250000  # Seguro hasta ~70k tokens
MAX_OUTPUT_TOKENS = 6000

# --- LIMPIEZA Y EXTRACCIÓN DESDE PDF ---
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            raw = page.get_text("text")  # Modo explícito
            clean = raw.replace('\x00', ' ').replace('\u2028', ' ')
            clean = clean.encode("utf-8", "ignore").decode("utf-8", "ignore")
            clean = ' '.join(clean.split())  # Quita saltos múltiples / palabras pegadas
            text += clean + "\n\n"
    return text

# --- GENERAR ANÁLISIS CLÍNICO ---
def generar_analisis_clinico(texto_total, seccion_objetivo):
    if seccion_objetivo == "Todo el artículo":
        objetivo_prompt = "analiza el artículo completo"
    else:
        objetivo_prompt = f"analiza exclusivamente la sección de {seccion_objetivo.lower()}"

    prompt = f"""
Actúa como médico especialista en medicina materno-fetal.

Tienes a continuación el contenido de un artículo científico extraído de un PDF:

{texto_total}

Por favor, {objetivo_prompt} y genera un informe profesional para revisión por especialistas clínicos. El informe debe estar estructurado, enfocado en evidencia médica clara, y ser útil para discusión académica o aplicación clínica.
"""

    # 🔍 Mostrar parte del prompt para debug
    st.text("------ DEBUG DEL PROMPT ------")
    st.text_area("Prompt enviado al modelo (recorte):", prompt[:4000])

    respuesta = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_OUTPUT_TOKENS
    )
    return respuesta.choices[0].message.content

# --- RESPUESTA A PREGUNTAS PERSONALIZADAS ---
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

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="FLASOG 2025 - Análisis de Literatura Médica", layout="centered")

st.title("📘 Análisis de Literatura Médica FLASOG 2025")
st.markdown("### 🧠 IA para análisis clínico automatizado")
st.info("Modelo activo: gpt-4-turbo · Límite de entrada: 250,000 caracteres (~70k tokens)")

uploaded_files = st.file_uploader("📄 Sube uno o más artículos médicos en PDF", type="pdf", accept_multiple_files=True)

seccion_objetivo = st.radio("¿Qué sección deseas analizar?", 
                            ["Todo el artículo", "Metodología", "Resultados", "Conclusiones"], index=0)

texto_total = ""

if uploaded_files:
    for archivo in uploaded_files:
        texto_total += extract_text_from_pdf(archivo) + "\n\n"

    st.info(f"📏 Caracteres totales cargados: {len(texto_total)}")

    if len(texto_total) > MAX_CARACTERES_ENTRADA:
        st.warning("⚠️ El texto fue recortado a 250,000 caracteres para cumplir el límite del modelo.")
        texto_total = texto_total[:MAX_CARACTERES_ENTRADA]
        st.info(f"✂️ Texto recortado a {len(texto_total)} caracteres.")

    if st.button("📑 Generar análisis clínico"):
        with st.spinner("🧠 Analizando con IA..."):
            resultado = generar_analisis_clinico(texto_total, seccion_objetivo)
            st.subheader("📝 Informe clínico generado:")
            st.write(resultado)
            st.download_button("💾 Descargar informe (.txt)", resultado, file_name="informe_clinico.txt")

    st.markdown("---")
    st.subheader("💬 Haz una pregunta personalizada sobre los artículos")

    pregunta = st.text_input("Escribe tu pregunta:")
    if st.button("❓ Obtener respuesta"):
        if pregunta.strip():
            with st.spinner("💬 Generando respuesta..."):
                respuesta = responder_pregunta(texto_total, pregunta)
                st.markdown("### ✅ Respuesta basada en el contenido:")
                st.write(respuesta)
        else:
            st.warning("Por favor, escribe una pregunta válida.")

