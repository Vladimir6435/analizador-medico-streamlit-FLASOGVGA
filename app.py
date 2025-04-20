import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# Cliente OpenAI
client = OpenAI(api_key=st.secrets["openai_api_key"])

MAX_OUTPUT_TOKENS = 6000
MAX_PALABRAS_ENTRADA = 60000  # estimado para no pasar los 100k tokens

# --- Extraer texto desde PDF ---
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# --- Análisis clínico IA ---
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

Incluye solo lo que corresponda a la sección seleccionada si así se indica.
"""
    respuesta = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_OUTPUT_TOKENS
    )
    return respuesta.choices[0].message.content

# --- Pregunta personalizada ---
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

# --- Interfaz principal ---
st.set_page_config(page_title="FLASOG 2025 - Análisis de Literatura Médica", layout="centered")

st.title("📘 Análisis de Literatura Médica FLASOG 2025")
st.markdown("### 🧠 IA aplicada al análisis clínico de artículos extensos")
st.info("Modelo en uso: gpt-4-turbo (128k tokens) · Se recorta texto a máx. 60,000 palabras")

uploaded_files = st.file_uploader("📄 Sube uno o más artículos médicos en PDF", type="pdf", accept_multiple_files=True)

seccion_objetivo = st.radio("Selecciona la sección que deseas analizar:",
                            ["Todo el artículo", "Metodología", "Resultados", "Conclusiones"], index=0)

texto_total = ""

if uploaded_files:
    for archivo in uploaded_files:
        texto_total += extract_text_from_pdf(archivo) + "\n\n"

    palabras = texto_total.split()
    num_palabras = len(palabras)
    st.info(f"📏 Palabras cargadas: {num_palabras}")

    if num_palabras > MAX_PALABRAS_ENTRADA:
        st.warning("⚠️ El texto fue recortado a 60,000 palabras para evitar errores por exceso de tokens.")
        palabras = palabras[:MAX_PALABRAS_ENTRADA]
        texto_total = " ".join(palabras)
        st.info(f"✂️ Texto recortado a {len(texto_total.split())} palabras.")

    if st.button("📑 Generar análisis clínico"):
        with st.spinner("🧠 Procesando..."):
            resultado = generar_analisis_clinico(texto_total, seccion_objetivo)
            st.subheader("📝 Informe clínico generado:")
            st.write(resultado)
            st.download_button("💾 Descargar informe (.txt)", resultado, file_name="informe_clinico.txt")

    st.markdown("---")
    st.subheader("💬 Haz una pregunta personalizada sobre los artículos")

    pregunta = st.text_input("Escribe tu pregunta aquí:")
    if st.button("❓ Responder con IA"):
        if pregunta.strip():
            with st.spinner("💡 Generando respuesta..."):
                respuesta = responder_pregunta(texto_total, pregunta)
                st.markdown("### ✅ Respuesta basada en el contenido:")
                st.write(respuesta)
        else:
            st.warning("Por favor, escribe una pregunta válida.")

