import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
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
    pdf.cell(0, 10, f"Consulta Ortopedica: {fonte}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    return pdf.output(dest='S').encode('latin-1')

# 3. ENGINE DE BUSCA ULTRA-LEVE (BM25)
@st.cache_resource
def get_search_engine(pasta_livro):
    if not os.path.exists(pasta_livro):
        return None
    
    docs = []
    arquivos = [f for f in os.listdir(pasta_livro) if f.endswith(".md")]
    
    if not arquivos:
        return None

    for arquivo in arquivos:
        caminho = os.path.join(pasta_livro, arquivo)
        loader = TextLoader(caminho, encoding="utf-8")
        docs.extend(loader.load())
    
    # Divide o texto em pedaços para a IA ler
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    # Cria o motor de busca ultra-rápido que não pesa no servidor
    return BM25Retriever.from_documents(split_docs)

# Controle de estado
if "livro" not in st.session_state:
    st.session_state.livro = None

# --- TELA INICIAL ---
if st.session_state.livro is None:
    st.title("🛡️ OrtoXande Pro")
    st.subheader("Selecione a base de conhecimento:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 Rockwood & Green", use_container_width=True):
            st.session_state.livro = "livros/rockwood"
            st.rerun()
    with col2:
        if st.button("📖 Campbell's Operative", use_container_width=True):
            st.session_state.livro = "livros/campbell"
            st.rerun()

# --- TELA DE PESQUISA ---
else:
    nome_exibicao = "Rockwood" if "rockwood" in st.session_state.livro else "Campbell"
    st.title(f"🔍 Pesquisando: {nome_exibicao}")
    
    if st.button("← Trocar Livro"):
        st.session_state.livro = None
        st.rerun()

    if not api_key:
        st.error("ERRO: Configure a GROQ_API_KEY nos Secrets do Streamlit.")
    else:
        # Carrega o motor de busca (Rápido e leve)
        with st.spinner(f"Preparando biblioteca do {nome_exibicao}..."):
            retriever = get_search_engine(st.session_state.livro)
        
        if retriever:
            query = st.text_input(f"O que deseja saber no {nome_exibicao}?")
            
            if query:
                with st.spinner("🧠 IA analisando capítulos..."):
                    # Busca os 5 trechos mais importantes usando palavras-chave
                    results = retriever.get_relevant_documents(query)[:5]
                    context = "\n".join([d.page_content for d in results])
                    
                    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key, temperature=0.1)
                    prompt = f"Aja como um Ortopedista Senior. Baseado no livro {nome_exibicao}, responda PROFUNDAMENTE e com detalhes técnicos: {query}\n\nContexto:\n{context}"
                    
                    resposta = llm.invoke(prompt).content
                    st.markdown(resposta)
                    
                    # Botões finais
                    colA, colB = st.columns(2)
                    with colA:
                        pdf_bytes = generate_pdf(resposta, query, nome_exibicao)
                        st.download_button("📥 Baixar PDF", pdf_bytes, f"{query}.pdf")
                    with colB:
                        txt_wa = f"*Consulta OrtoXande*\n\n{resposta[:800]}..."
                        link_wa = f"https://wa.me/?text={txt_wa.replace(' ', '%20')}"
                        st.markdown(f'<a href="{link_wa}" target="_blank"><button style="background-color:#25D366; color:white; border:none; padding:10px 20px; border-radius:5px; width:100%; cursor:pointer;">📲 Enviar WhatsApp</button></a>', unsafe_allow_html=True)
        else:
            st.error("Arquivos não encontrados. Verifique a pasta no GitHub.")
