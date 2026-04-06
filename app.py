import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fpdf import FPDF

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="OrtoXande Pro", layout="centered", page_icon="🦴")

# 2. PEGAR CHAVE DOS SECRETS
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

# 3. ENGINE DE BUSCA OTIMIZADA (COM SALVAMENTO EM DISCO)
@st.cache_resource
def get_vector_db(pasta_livro):
    if not os.path.exists(pasta_livro):
        return None
    
    # Nome da pasta onde o índice pronto será salvo
    pasta_indice = pasta_livro.replace("/", "_") + "_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Se o índice já existir no servidor, carrega ele instantaneamente
    if os.path.exists(pasta_indice):
        return FAISS.load_local(pasta_indice, embeddings, allow_dangerous_deserialization=True)
    
    # Se não existir, processa os arquivos (Apenas 1 vez)
    docs = []
    arquivos = [f for f in os.listdir(pasta_livro) if f.endswith(".md")]
    
    if not arquivos:
        return None

    for arquivo in arquivos:
        caminho_completo = os.path.join(pasta_livro, arquivo)
        loader = TextLoader(caminho_completo, encoding="utf-8")
        docs.extend(loader.load())
    
    # Splitter mais eficiente para poupar memória
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    
    # Cria o banco e SALVA no disco do servidor
    vector_db = FAISS.from_documents(split_docs, embeddings)
    vector_db.save_local(pasta_indice)
    
    return vector_db

# Controle de estado
if "livro_path" not in st.session_state:
    st.session_state.livro_path = None

# --- TELA INICIAL ---
if st.session_state.livro_path is None:
    st.title("🛡️ OrtoXande Pro")
    st.subheader("Selecione a base de conhecimento:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 Rockwood & Green", use_container_width=True):
            st.session_state.livro_path = "livros/rockwood"
            st.rerun()
    with col2:
        if st.button("📖 Campbell's Operative", use_container_width=True):
            st.session_state.livro_path = "livros/campbell"
            st.rerun()

# --- TELA DE PESQUISA ---
else:
    nome_livro = "Rockwood" if "rockwood" in st.session_state.livro_path else "Campbell"
    st.title(f"🔍 Pesquisando no {nome_livro}")
    
    if st.button("← Trocar Livro"):
        st.session_state.livro_path = None
        st.rerun()

    if not api_key:
        st.error("ERRO: Configure a GROQ_API_KEY nos Secrets do Streamlit.")
    else:
        # Carrega ou cria o índice
        db = get_vector_db(st.session_state.livro_path)
        
        if db:
            query = st.text_input(f"Dúvida sobre {nome_livro}:", placeholder="Ex: Fratura de rádio distal...")
            
            if query:
                with st.spinner("🧠 IA analisando base científica..."):
                    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key, temperature=0.1)
                    # Busca os 4 trechos mais relevantes
                    res = db.similarity_search(query, k=4)
                    context = "\n".join([d.page_content for d in res])
                    
                    prompt = f"Aja como um Ortopedista Senior. Baseado no livro {nome_livro}, responda profundamente: {query}\n\nContexto:\n{context}"
                    resposta = llm.invoke(prompt).content
                    st.markdown(resposta)
                    
                    colA, colB = st.columns(2)
                    with colA:
                        pdf_data = generate_pdf(resposta, query, nome_livro)
                        st.download_button("📥 Baixar PDF", pdf_data, f"{query}.pdf")
                    with colB:
                        txt_wa = f"*Consulta OrtoXande*\n\n{resposta[:800]}..."
                        link_wa = f"https://wa.me/?text={txt_wa.replace(' ', '%20')}"
                        st.markdown(f'<a href="{link_wa}" target="_blank"><button style="background-color:#25D366; color:white; border:none; padding:10px 20px; border-radius:5px; width:100%; cursor:pointer;">📲 Enviar WhatsApp</button></a>', unsafe_allow_html=True)
        else:
            st.warning("Pasta de arquivos não encontrada ou processando...")
