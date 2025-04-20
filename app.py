import streamlit as st
import fitz  # PyMuPDF
import openai

openai.api_key = st.secrets["openai_api_key"]

def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def generar_analisis_clinico(texto_total):
    prompt = f"""
Actúa como médico especialista en medicina materno-fetal.

A continuación tienes el contenido completo de varios artículos médicos en PDF sobre temas clínicos relevantes:

{texto_total}

Realiza un análisis clínico estructurado:
1. Resumen de hallazgos clínicos clave.
2. Nivel de evidencia según el diseño metodológico.
3. Evaluación de utilidad práctica para medicina materno-fetal.
4. Limitaciones metodológicas.
5. Recomendaciones clínicas aplicables.

Usa lenguaje técnico claro dirigido a médicos gineco-obstetras.
"""
    respuesta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2500
    )
    return respuesta['choices'][0]['message']['content']

def responder_pregunta(texto_total, pregunta):
    prompt = f"""
Eres un asistente clínico experto en medicina materno-fetal.

Con base en el siguiente contenido médico extraído de varios artículos científicos:

{texto_total}

Responde la siguiente pregunta del usuario de forma clara y con base en la evidencia del contenido anterior:

PREGUNTA: {pregunta}
"""
    respuesta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )
    return respuesta['choices'][0]['message']['content']

# Interfaz de usuario
st.title("📊 Analizador Médico - Múltiples PDFs + Preguntas")

uploaded_files = st.file_uploader("📄 Sube artículos médicos en PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔍 Extrayendo texto de los artículos..."):
        texto_total = ""
        for archivo in uploaded_files:
            texto_total += extract_text_from_pdf(archivo) + "\n\n"

        st.success(f"✅ Texto extraído de {len(uploaded_files)} archivos.")
        if st.button("📑 Generar análisis clínico automático"):
            with st.spinner("🧠 Analizando con IA..."):
                resultado = generar_analisis_clinico(texto_total)
                st.subheader("📝 Informe clínico generado:")
                st.write(resultado)
                st.download_button("💾 Descargar informe", resultado, file_name="informe_clinico.txt")

        st.markdown("---")
        st.subheader("🔎 Haz una pregunta personalizada sobre los artículos")
        pregunta = st.text_input("Escribe tu pregunta aquí")
        if st.button("💬 Responder pregunta"):
            if pregunta.strip() != "":
                with st.spinner("Pensando como un experto clínico..."):
                    respuesta = responder_pregunta(texto_total, pregunta)
                    st.markdown("### ✅ Respuesta basada en los artículos:")
                    st.write(respuesta)
            else:
                st.warning("Por favor escribe una pregunta válida.")
