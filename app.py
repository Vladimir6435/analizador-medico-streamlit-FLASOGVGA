import streamlit as st
import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI

# Cliente OpenAI con clave segura desde Streamlit Cloud
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Tokenizador GPT-4 Turbo (usamos cl100k_base que es compatible)
encoding = tiktoken.get_encoding("cl100k_base")

# Límite real del modelo gpt-4-turbo (entrada + salida)
MAX_TOTAL_TOKENS = 128000
MAX_OUTPUT_TOKENS = 6000  # tokens reservados para respuesta
MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - MAX_OUTPUT_TOKENS

# --- Función para contar tokens ---
def contar_tokens(texto):
    return len(encoding.encode(texto))

# --- Función para extraer texto de PDF ---
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# --- Generar análisis clínico por sección ---
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
st.markdown("### 🧠 IA aplicada al análisis de artículos clínicos extensos")
st.info("Modelo en uso: gpt-4-turbo · Límite: 128k tokens")

uploaded_files = st.file_uploader("📄 Sube uno o más artículos médicos en PDF", type="pdf", accept_multiple_files=True)

seccion_objetivo = st.radio("Selecciona la sección a analizar:",
                            ["Todo el artículo", "Metodología", "Resultados", "Conclusiones"], index=0)

texto_total = ""

if uploaded_files:
    for archivo in uploaded_files:
        texto_total += extract_text_from_pdf(archivo) + "\n\n"

    token_count = contar_tokens(texto_total)
    st.info(f"🔢 Tokens estimados: {token_count}")

    if token_count > MAX_INPUT_TOKENS:
        st.warning("⚠️ El texto fue recortado para ajustarse al límite del modelo (máx. 100k tokens de entrada).")
        tokens = encoding.encode(texto_total)
        texto_total = encoding.decode(tokens[:MAX_INPUT_TOKENS])
        st.info(f"✂️ Texto recortado a {contar_tokens(texto_total)} tokens")

    if st.button("📑 Generar análisis clínico"):
        with st.spinner("🧠 Procesando con IA..."):
            resultado = generar_analisis_clinico(texto_total, seccion_objetivo)
            st.subheader("📝 Informe clínico generado:")
            st.write(resultado)
            st.download_button("💾 Descargar informe", resultado, file_name="informe_clinico.txt")

    st.markdown("---")
    st.subheader("💬 Haz una pregunta personalizada")

    pregunta = st.text_input("Escribe tu pregunta aquí:")
    if st.button("❓ Responder"):
        if pregunta.strip():
            with st.spinner("💡 Respondiendo..."):
                respuesta = responder_pregunta(texto_total, pregunta)
                st.markdown("### ✅ Respuesta:")
                st.write(respuesta)
        else:
            st.warning("Escribe una pregunta válida.")
