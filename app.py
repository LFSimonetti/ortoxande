import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fpdf import FPDF

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="OrtoXande Pro", layout="centered", page_icon="🦴")

# 2. PEGAR CHAVE DOS SECRETS (PERMANENTE)
api_key = st.secrets.get("GROQ_API_KEY")

def generate_pdf(text, query, fonte):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Consulta: {fonte}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    return pdf.output(dest='S').encode('latin-1')

@st.cache_resource
def get_vector_db(pasta_livro):
    if not os.path.exists(pasta_livro):
        return None
    
    from langchain_community.document_loaders import TextLoader
    
    docs = []
    # Lista todos os arquivos .md da pasta de forma manual e segura
    for arquivo in os.listdir(pasta_livro):
        if arquivo.endswith(".md"):
            caminho_completo = os.path.join(pasta_livro, arquivo)
            # Usamos o TextLoader que é imune ao erro de download do spaCy
            loader = TextLoader(caminho_completo, encoding="utf-8")
            docs.extend(loader.load())
    
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embeddings)

# --- TELA DE PESQUISA ---
else:
    nome = "Rockwood" if "rockwood" in st.session_state.livro_path else "Campbell"
    st.title(f"🔍 Pesquisando no {nome}")
    if st.button("← Trocar Livro"):
        st.session_state.livro_path = None
        st.rerun()

    if not api_key:
        st.error("ERRO: Configure a GROQ_API_KEY nos Secrets do Streamlit.")
    else:
        with st.spinner(f"Indexando {nome}..."):
            db = get_vector_db(st.session_state.livro_path)
        
        if db:
            query = st.text_input("Digite sua dúvida ortopédica:")
            if query:
                with st.spinner("🧠 IA Analisando..."):
                    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key, temperature=0.1)
                    res = db.similarity_search(query, k=5)
                    context = "\n".join([d.page_content for d in res])
                    prompt = f"Aja como um Ortopedista Senior. Baseado no livro {nome}, responda profundamente: {query}\n\nContexto:\n{context}"
                    resposta = llm.invoke(prompt).content
                    st.markdown(resposta)
                    
                    pdf_data = generate_pdf(resposta, query, nome)
                    st.download_button("📥 Baixar PDF", pdf_data, f"{query}.pdf")
                    
                    txt_wa = f"*Consulta OrtoXande*\n\n{resposta[:800]}..."
                    link_wa = f"https://wa.me/?text={txt_wa.replace(' ', '%20')}"
                    st.markdown(f'<a href="{link_wa}" target="_blank"><button style="background-color:#25D366; color:white; border:none; padding:10px 20px; border-radius:5px; width:100%; cursor:pointer;">📲 Enviar WhatsApp</button></a>', unsafe_allow_html=True)
        else:
            st.warning(f"Atenção: A pasta {st.session_state.livro_path} ainda não contém arquivos .md")
