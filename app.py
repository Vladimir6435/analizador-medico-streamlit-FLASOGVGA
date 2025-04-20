import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# Crear cliente OpenAI
client = OpenAI(api_key=st.secrets["openai_api_key"])

# --- PARÁMETROS GENERALES ---
MAX_CARACTERES = 90000  # Límite aproximado para gpt-4-turbo (~7000 tokens)

# --- EXTRACCIÓN DE TEXTO DE PDF ---
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# --- GENERAR ANÁLISIS CLÍNICO POR SECCIÓN ---
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
        max_tokens=5000
    )
    return respuesta.choices[0].message.content

# --- STREAMLIT UI ---
st.set_page_config(page_title="FLASOG 2025 - Análisis de Literatura Médica", layout="centered")

st.title("📘 Análisis de Literatura Médica FLASOG 2025")
st.markdown("### 🧠 Inteligencia Artificial para apoyo en lectura crítica de artículos médicos")
st.markdown("Este sistema usa el modelo `gpt-4-turbo` para analizar artículos clínicos y generar reportes estructurados.")
st.info("**Prompt clínico activo:** análisis estructurado por sección (Metodología, Resultados, Conclusiones)")

# --- SUBIR ARCHIVOS PDF ---
uploaded_files = st.file_uploader("📄 Sube uno o más artículos médicos en PDF", type="pdf", accept_multiple_files=True)

# --- ELEGIR SECCIÓN A ANALIZAR ---
st.markdown("### 🧪 Selecciona la sección que deseas analizar:")
seccion_objetivo = st.radio(
    "¿Qué sección deseas que la IA analice?", 
    ["Todo el artículo", "Metodología", "Resultados", "Conclusiones"], 
    index=0
)

# --- PROCESAMIENTO ---
if uploaded_files:
    st.markdown("---")
    st.subheader("🔍 Análisis clínico")

    texto_total = ""
    for archivo in uploaded_files:
        texto_total += extract_text_from_pdf(archivo) + "\n\n"

    num_caracteres = len(texto_total)
    st.info(f"📏 Caracteres cargados: {num_caracteres}")

    if num_caracteres > MAX_CARACTERES:
        st.warning("⚠️ El texto fue recortado para ajustarse al límite del modelo.")
        texto_total = texto_total[:MAX_CARACTERES]
        st.info(f"✂️ Texto reducido a {len(texto_total)} caracteres.")

    if st.button("📑 Generar análisis clínico"):
        with st.spinner("🧠 Analizando con IA..."):
            resultado = generar_analisis_clinico(texto_total, seccion_objetivo)
            st.subheader("📝 Informe clínico generado:")
            st.write(resultado)
            st.download_button("💾 Descargar informe como .txt", resultado, file_name="informe_clinico.txt")
