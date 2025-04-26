import streamlit as st
import os
import tiktoken
import openai
from PyPDF2 import PdfReader, PdfWriter
from fpdf import FPDF
import tempfile
import io

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Literatura Médica FLASOG 2025 v3",
    layout="wide"
)

# Inicialización de estado
def init_state():
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}        # {filename: summary}
    if 'history' not in st.session_state:
        st.session_state.history = {}          # {filename: [(user_msg, bot_resp), ...]}
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'char_limit': 70000,
            'model': 'gpt-4-turbo',
            'sections': ['Objetivos', 'Metodología', 'Resultados', 'Conclusiones']
        }

init_state()

# Funciones auxiliares
def extract_text(file) -> str:
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += (page.extract_text() or '') + '\n'
    return text

def count_tokens(text: str, model: str) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def call_openai(messages) -> str:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    resp = openai.ChatCompletion.create(
        model=st.session_state.settings['model'],
        messages=messages,
        temperature=0.0,
        max_tokens=1500
    )
    return resp.choices[0].message.content.strip()

def generate_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 8, line)
    return pdf.output(dest='S').encode('latin-1')

def merge_pdfs(pdf_bytes_list) -> bytes:
    writer = PdfWriter()
    for pdf_bytes in pdf_bytes_list:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()

# Interfaz
st.title("📚 Análisis de Literatura Médica FLASOG 2025 v3")
st.markdown(
    "Sube archivos PDF y obtén resúmenes por sección y chat de seguimiento."
)

# Subida de archivos
uploaded = st.file_uploader("Selecciona PDFs", type='pdf', accept_multiple_files=True)

if uploaded:
    for f in uploaded:
        name = f.name
        if name not in st.session_state.summaries:
            raw = extract_text(f)
            trimmed = raw[:st.session_state.settings['char_limit']]
            system_prompt = (
                "Eres un asistente especializado en medicina materno-fetal. "
                "Divide el resumen en Objetivos, Metodología, Resultados y Conclusiones."
            )
            # Resumen inicial
            summary = call_openai([
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': trimmed}
            ])
            st.session_state.summaries[name] = summary
            st.session_state.history[name] = []

    # Selector de sección
    tab1, tab2 = st.tabs(["Resúmenes", "Informe Combinado"])

    with tab1:
        for fname, summary in st.session_state.summaries.items():
            st.subheader(f"{fname}")
            # Mostrar filtro
            opts = st.multiselect(
                f"Secciones a mostrar ({fname})",
                st.session_state.settings['sections'],
                default=st.session_state.settings['sections']
            )
            # Mostrar solo secciones filtradas
            for section in opts:
                if section in summary:
                    st.markdown(f"**{section}:**")
                    part = summary.split(section + ":")[1].split("\n##")[0].strip()
                    st.write(part)

            # Chat de seguimiento
            q = st.text_input(f"Pregunta de seguimiento para {fname}")
            if q:
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': trimmed}
                ]
                # Incluir historial
                for uq, ur in st.session_state.history[fname]:
                    messages.append({'role': 'user', 'content': uq})
                    messages.append({'role': 'assistant', 'content': ur})
                messages.append({'role': 'user', 'content': q})
                resp = call_openai(messages)
                st.session_state.history[fname].append((q, resp))
                st.write(f"**Bot:** {resp}")

    with tab2:
        st.markdown("### Descargar Informe Combinado")
        if st.button("Generar PDF combinado"):
            bytes_list = []
            for fname, text in st.session_state.summaries.items():
                pdf_b = generate_pdf(text)
                bytes_list.append(pdf_b)
            combined = merge_pdfs(bytes_list)
            st.download_button(
                label="Descargar todo en un PDF",
                data=combined,
                file_name="informe_combinado.pdf",
                mime="application/pdf"
            )
