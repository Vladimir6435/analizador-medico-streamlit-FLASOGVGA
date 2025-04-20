import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io
import tiktoken

# Configuración API moderna
client = OpenAI(api_key=st.secrets["openai_api_key"])

MAX_CARACTERES_POR_PDF = 70000
MAX_OUTPUT_TOKENS = 6000
MAX_INPUT_TOKENS = 120000

if "analisis_clinicos" not in st.session_state:
    st.session_state["analisis_clinicos"] = {}

if "historial_respuestas" not in st.session_state:
    st.session_state["historial_respuestas"] = []

def contar_tokens(texto, modelo="gpt-4-turbo"):
    enc = tiktoken.encoding_for_model(modelo)
    return len(enc.encode(texto))

def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            raw = page.get_text("text")
            clean = raw.replace('\x00', ' ').replace('\u2028', ' ')
            clean = clean.encode("utf-8", "ignore").decode("utf-8", "ignore")
            clean = ' '.join(clean.split())
            text += clean + "\n\n"
    return text

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

    if contar_tokens(prompt) > MAX_INPUT_TOKENS:
        st.error("⚠️ El contenido del artículo es demasiado extenso para el modelo. Intenta reducir su tamaño.")
        return "ERROR: Texto muy largo para el modelo."

    respuesta = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_OUTPUT_TOKENS
    )
    return respuesta.choices[0].message.content

def generar_pdf(nombre_archivo, contenido, seccion):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 80

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Análisis de Literatura Médica FLASOG 2025")
    y -= 25

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Título del artículo: {nombre_archivo}")
    y -= 20
    c.drawString(50, y, f"Fecha de análisis: {datetime.today().strftime('%Y-%m-%d')}")
    y -= 20
    c.drawString(50, y, f"Sección analizada: {seccion}")
    y -= 30

    c.setFont("Helvetica", 10)
    for linea in contenido.split('\n'):

        for fragmento in [linea[i:i+100] for i in range(0, len(linea), 100)]:
            if y < 40:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 10)
            c.drawString(50, y, fragmento)
            y -= 14

    c.save()
    buffer.seek(0)
    return buffer

st.set_page_config(page_title="FLASOG 2025 - Análisis de Literatura Médica", layout="wide")
st.title("📘 Análisis de Literatura Médica FLASOG 2025")
st.markdown("### Suba uno o más artículos PDF para generar informes clínicos independientes")

uploaded_files = st.file_uploader("📄 Sube tus archivos PDF", type="pdf", accept_multiple_files=True)
seccion_objetivo = st.radio("¿Qué sección deseas analizar?",
                            ["Todo el artículo", "Metodología", "Resultados", "Conclusiones"], index=0)

if uploaded_files:
    for archivo in uploaded_files:
        nombre = archivo.name
        texto = extract_text_from_pdf(archivo)
        st.markdown(f"---
### 📄 Informe para: `{nombre}`")
        st.info(f"📏 Caracteres extraídos: {len(texto)}")

        if len(texto) > MAX_CARACTERES_POR_PDF:
            st.warning("⚠️ Recortando texto a 70,000 caracteres.")
            texto = texto[:MAX_CARACTERES_POR_PDF]

        if st.button(f"🧠 Analizar `{nombre}`"):
            with st.spinner("Generando informe clínico..."):
                resultado = generar_analisis_clinico(texto, seccion_objetivo)
                st.session_state["analisis_clinicos"][nombre] = resultado

        if nombre in st.session_state["analisis_clinicos"]:
            st.subheader("📝 Informe clínico generado:")
            st.write(st.session_state["analisis_clinicos"][nombre])

            pdf_bytes = generar_pdf(nombre, st.session_state["analisis_clinicos"][nombre], seccion_objetivo)
            st.download_button("📄 Descargar informe en PDF", pdf_bytes, file_name=f"{nombre}_informe.pdf")

st.markdown("---")
st.subheader("💬 Preguntas clínicas personalizadas")

pregunta = st.text_input("Haz una pregunta sobre los artículos analizados:")

if st.button("❓ Responder con IA"):
    if pregunta.strip():
        contexto = "

contexto = "\n\n".join(st.session_state["analisis_clinicos"].values())
        with st.spinner("Buscando respuesta..."):
            prompt = f"""Actúa como médico materno-fetal. Usa el siguiente contexto clínico para responder:

{contexto}

PREGUNTA: {pregunta}
"""
            respuesta = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            ).choices[0].message.content
            st.session_state["historial_respuestas"].append((pregunta, respuesta))
    else:
        st.warning("Escribe una pregunta válida.")

if st.session_state["historial_respuestas"]:
    st.subheader("📚 Historial de preguntas y respuestas")
    for i, (q, r) in enumerate(st.session_state["historial_respuestas"]):
        st.markdown(f"**{i+1}. Pregunta:** {q}")
        st.markdown(f"🧠 {r}")
