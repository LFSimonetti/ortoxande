import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fpdf import FPDF

# 1. CONFIGURAÇÃO DE ELITE
st.set_page_config(page_title="OrtoXande Pro", layout="centered", page_icon="🦴")

# Estilo para o botão de WhatsApp
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; }
    .wa-button {
        background-color: #25D366;
        color: white;
        padding: 10px;
        text-align: center;
        text-decoration: none;
        display: block;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. FUNÇÕES DE SUPORTE
def generate_pdf(text, query, fonte):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, f"Consulta Ortopedica: {fonte}", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    # Limpa caracteres para o PDF
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    return pdf.output()

@st.cache_resource
def get_search_engine(pasta):
    if not os.path.exists(pasta):
        return None
    
    docs = []
    # Lê todos os arquivos .md da pasta de forma eficiente
    files = [f for f in os.listdir(pasta) if f.endswith(".md")]
    if not files:
        return None
        
    for f in files:
        loader = TextLoader(os.path.join(pasta, f), encoding="utf-8")
        docs.extend(loader.load())
    
    # Divide em blocos menores para análise da IA
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)
    
    # Motor BM25: Ultra-leve (não crasha o servidor)
    return BM25Retriever.from_documents(final_docs)

# 3. LÓGICA DE NAVEGAÇÃO
if "livro_ativo" not in st.session_state:
    st.session_state.livro_ativo = None

# MENU INICIAL
if st.session_state.livro_ativo is None:
    st.title("🛡️ OrtoXande Pro")
    st.subheader("Selecione a fonte da pesquisa:")
    
    c1, c2 = st.columns(2)
    if c1.button("📚 Rockwood & Green", use_container_width=True):
        st.session_state.livro_ativo = "livros/rockwood"
        st.rerun()
    if c2.button("📖 Campbell's Operative", use_container_width=True):
        st.session_state.livro_ativo = "livros/campbell"
        st.rerun()

# TELA DE CONSULTA
else:
    label = "Rockwood" if "rockwood" in st.session_state.livro_ativo else "Campbell"
    st.title(f"🔍 Consultando {label}")
    
    if st.button("← Voltar e Trocar Livro"):
        st.session_state.livro_ativo = None
        st.rerun()

    # Verifica Chave API nos Secrets
    api_key = st.secrets.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("ERRO: GROQ_API_KEY não configurada nos Secrets do Streamlit.")
    else:
        with st.spinner(f"Carregando biblioteca {label}..."):
            retriever = get_search_engine(st.session_state.livro_ativo)
        
        if retriever:
            query = st.text_input(f"O que deseja saber no {label}?", placeholder="Ex: Classificacao de Gustilo...")
            
            if query:
                with st.spinner("🧠 IA Analisando capítulos..."):
                    # Busca os 5 trechos mais relevantes
                    context_docs = retriever.get_relevant_documents(query)[:5]
                    context_text = "\n\n".join([d.page_content for d in context_docs])
                    
                    # Chama o Llama 3 (Groq)
                    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key, temperature=0.1)
                    prompt = f"Aja como um Ortopedista Senior. Baseado no livro {label}, responda profundamente: {query}\n\nContexto extraido:\n{context_text}"
                    
                    resposta = llm.invoke(prompt).content
                    st.markdown("---")
                    st.markdown(resposta)
                    
                    # Botões de Saída
                    col_pdf, col_wa = st.columns(2)
                    with col_pdf:
                        pdf_data = generate_pdf(resposta, query, label)
                        st.download_button("📥 Baixar Relatorio PDF", pdf_data, f"{query}.pdf")
                    with col_wa:
                        wa_msg = f"*OrtoXande - {label}*\n\n{resposta[:800]}..."
                        link_wa = f"https://wa.me/?text={wa_msg.replace(' ', '%20')}"
                        st.markdown(f'<a href="{link_wa}" target="_blank" class="wa-button">📲 Enviar via WhatsApp</a>', unsafe_allow_html=True)
        else:
            st.warning(f"A pasta {st.session_state.livro_ativo} não contém arquivos .md válidos.")
