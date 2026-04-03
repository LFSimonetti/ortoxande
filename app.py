import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fpdf import FPDF

# Configuração da Página
st.set_page_config(page_title="OrtoXande - Rockwood AI", page_icon="🦴")

def generate_pdf(text, query):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Relatorio: {query}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    return pdf.output(dest='S').encode('latin-1')

st.title("🛡️ OrtoXande: Rockwood & Green Search")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuração")
    groq_key = st.text_input("Cole sua Groq API Key aqui:", type="password")
    uploaded_file = st.file_uploader("Suba o PDF do Rockwood & Green", type="pdf")
    st.info("A API Key é gratuita e necessária para o cérebro da IA funcionar.")

if uploaded_file and groq_key:
    if not os.path.exists("temp"): os.makedirs("temp")
    file_path = os.path.join("temp", uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    @st.cache_resource
    def init_engine(_file_path):
        loader = PyPDFLoader(_file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = loader.load_and_split(text_splitter)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)

    with st.spinner("🚀 Analisando o livro... Isso leva cerca de 1 minuto."):
        try:
            vector_db = init_engine(file_path)
            st.success("Livro pronto para consulta!")
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

    query = st.text_input("Sobre qual fratura ou região deseja pesquisar?")

    if query:
        llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_key, temperature=0.1)
        search_results = vector_db.similarity_search(query, k=4)
        context = "\n".join([doc.page_content for doc in search_results])
        
        prompt = f"Baseado no Rockwood e Green: {context}\n\nAnalise profundamente: {query}. Foque em Classificação, Indicações e Complicações."

        with st.spinner("🧠 Pesquisando no Rockwood..."):
            response = llm.invoke(prompt)
            answer = response.content
            st.markdown("### 📋 Resultado:")
            st.write(answer)
            
            pdf_bytes = generate_pdf(answer, query)
            st.download_button(label="📥 Baixar em PDF", data=pdf_bytes, file_name=f"Resumo_Rockwood.pdf")
else:
    st.warning("👈 Por favor, cole a API Key e suba o PDF na barra lateral para começar.")
