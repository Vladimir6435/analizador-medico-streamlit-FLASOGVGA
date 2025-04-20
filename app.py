import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# Configurar cliente OpenAI
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Límite conservador en caracteres (~60,000–80,000 tokens máximo)
MAX_CARACTERES_ENTRADA = 250000
MAX_OUTPUT_TOKENS = 6000

# --- Limpiar y extraer texto desde PDF ---
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            raw = page.get_text()
            clean = raw.replace('\x00', ' ').replace('\u2028', ' ')
            clean = clean.encode("utf-8", "ignore").decode("utf-8", "ignore")
            text += clean
    return text

# --- Generar análisis clínico por IA ---
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

    # 🔍 Debug del prompt antes de enviarlo
    st.text("------ DEBUG DEL PROMPT ------")
    st.text_area("Contenido enviado al modelo (recorte):", prompt[:4000])

    # Enviar a OpenAI
    respuesta = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_OUTPUT_TOKENS
    )
    return respuesta.choices[0].message.content

# --- Responder preguntas personalizadas ---
def responder_pregunta(texto_total, pregunta):
    prompt = f"""
Eres un asistente clínico experto en medicina materno-fetal.

Con base en el siguiente contenido médico extraído de artículos científicos:

{texto_total}

Responde con precisión y lenguaje médico claro:

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
st.markdown("### 🧠 Inteligencia Artificial para artículos clínicos")
st.info("Modelo: gpt-4-turbo · Máximo 250,000 caracteres de entrada (~70k tokens)")

uploaded_files = st.file_uploader("📄 Sube uno o más artículos médicos en PDF", type="pdf", accept_multiple_files=True)

seccion_objetivo = st.radio("¿Qué sección deseas que la IA analice?",
                            ["Todo el artículo", "Metodología", "Resultados", "Conclusiones"], index=0)

texto_total = ""

if uploaded_files:
    for archivo in uploaded_files:
        texto_total += extract_text_from_pdf(archivo) + "\n\n"

    st.info(f"📏 Caracteres cargados: {len(texto_total)}")

    if len(texto_total) > MAX_CARACTERES_ENTRADA:
        st.warning("⚠️ El texto fue recortado a 250,000 caracteres para evitar errores.")
        texto_total = texto_total[:MAX_CARACTERES_ENTRADA]
        st.info(f"✂️ Texto final con {len(texto_total)} caracteres.")

    if st.button("📑 Generar análisis clínico"):
        with st.spinner("🧠 Analizando con IA..."):
            resultado = generar_analisis_clinico(texto_total, seccion_objetivo)
            st.subheader("📝 Informe generado:")
            st.write(resultado)
            st.download_button("💾 Descargar informe (.txt)", resultado, file_name="informe_clinico.txt")

    st.markdown("---")
    st.subheader("💬 Realiza una pregunta personalizada")

    pregunta = st.text_input("Escribe tu pregunta:")
    if st.button("❓ Responder con IA"):
        if pregunta.strip():
            with st.spinner("💡 Procesando..."):
                respuesta = responder_pregunta(texto_total, pregunta)
                st.markdown("### ✅ Respuesta basada en el contenido:")
                st.write(respuesta)
        else:
            st.warning("Por favor, escribe una pregunta válida.")



