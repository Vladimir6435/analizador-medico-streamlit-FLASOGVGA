import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
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
Además, si identificas condiciones como preeclampsia, restricción del crecimiento intrauterino (RCIU), diabetes gestacional, muerte fetal intrauterina, hipertensión crónica, anomalías congénitas o cualquier otra complicación materno-fetal significativa, destaca estos hallazgos en negrita, con énfasis clínico y, si es posible, ofrece recomendaciones o alertas para el lector.
"""

    n_tokens = contar_tokens(prompt)
    st.info(f"🧮 Tokens del prompt: {n_tokens}")

    if n_tokens > MAX_INPUT_TOKENS:
        st.error("⚠️ El contenido del artículo es demasiado extenso para el modelo. Intenta reducir su tamaño.")
        return "ERROR: Texto muy largo para el modelo."

    respuesta = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=3000
    )
    return respuesta.choices[0].message.content

def generar_pdf(nombre_archivo, contenido, seccion):
    condiciones_detectadas = []
    condiciones_clave = ["preeclampsia", "rciu", "restricción del crecimiento", "diabetes gestacional", "anomalías fetales", "síndrome hipertensivo", "muerte fetal"]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Análisis de Literatura Médica FLASOG 2025", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Título del artículo: {nombre_archivo}", styles['Normal']))
    story.append(Paragraph(f"Fecha de análisis: {datetime.today().strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Paragraph(f"Sección analizada: {seccion}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Índice clínico de condiciones detectadas:</b>", styles['Heading3']))
    story.append(Spacer(1, 6))
    for palabra in condiciones_clave:
        if palabra in contenido.lower():
            condiciones_detectadas.append(palabra)
            story.append(Paragraph(f"- {palabra.capitalize()}", styles['Normal']))
    story.append(Spacer(1, 12))

    for linea in contenido.split('\n'):
        if any(palabra in linea.lower() for palabra in condiciones_clave):
            story.append(Paragraph(f"<b>{linea}</b>", styles['BodyText']))
        else:
            story.append(Paragraph(linea, styles['BodyText']))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

def generar_pdf_combinado(informes):
    condiciones_clave = ["preeclampsia", "rciu", "restricción del crecimiento", "diabetes gestacional", "anomalías fetales", "síndrome hipertensivo", "muerte fetal"]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Análisis Combinado - FLASOG 2025", styles['Title']), Spacer(1, 12)]

    story.append(Paragraph("<b>Índice clínico global:</b>", styles['Heading3']))
    for nombre_archivo, contenido in informes.items():
        condiciones_detectadas = [cond for cond in condiciones_clave if cond in contenido.lower()]
        if condiciones_detectadas:
            story.append(Paragraph(f'<a href="#{nombre_archivo}"><u>{nombre_archivo}</u></a>', styles['Normal']))
            for cond in condiciones_detectadas:
                story.append(Paragraph(f"- {cond.capitalize()}", styles['Normal']))
    story.append(Spacer(1, 12))

    for nombre_archivo, contenido in informes.items():
        story.append(Paragraph(f'<a name="{nombre_archivo}"></a><b>Artículo: {nombre_archivo}</b>', styles['Heading2']))
        story.append(Spacer(1, 6))
        for linea in contenido.split('\n'):
            story.append(Paragraph(linea, styles['BodyText']))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer
