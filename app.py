import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from fpdf import FPDF

# 1. CONFIGURAÇÃO INICIAL (DEVE SER A PRIMEIRA COISA)
st.set_page_config(page_title="OrtoXande Pro", layout="centered", page_icon="🦴")

# 2. PEGAR CHAVE DOS SECRETS
api_key = st.secrets.get("GROQ_API_KEY")

# 3. FUNÇÕES DE SUPORTE
def generate_pdf(text, query, fonte):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, f"Consulta Ortopedica: {fonte}", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    return pdf.output()

@st.cache_resource
def get_retriever(pasta):
    if not os.path.exists(pasta): return None
    docs = []
    files = [f for f in os.listdir(pasta) if f.endswith(".md")]
    if not files: return None
    for f in files:
        loader = TextLoader(os.path.join(pasta, f), encoding="utf-8")
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return BM25Retriever.from_documents(splitter.split_documents(docs))

# 4. LÓGICA DE NAVEGAÇÃO
if "livro" not in st.session_state:
    st.session_state.livro = None

# TELA DE SELEÇÃO
if st.session_state.livro is None:
    st.title("🛡️ OrtoXande Pro")
    st.subheader("Selecione a fonte da pesquisa:")
    c1, c2 = st.columns(2)
    if c1.button("📚 Rockwood & Green", use_container_width=True):
        st.session_state.livro = "livros/rockwood"
        st.rerun()
    if c2.button("📖 Campbell's Operative", use_container_width=True):
        st.session_state.livro = "livros/campbell"
        st.rerun()

# TELA DE CONSULTA
else:
    label = "Rockwood" if "rockwood" in st.session_state.livro else "Campbell"
    st.title(f"🔍 Consultando {label}")
    if st.button("← Trocar de Livro"):
        st.session_state.livro = None
        st.rerun()

    if not api_key:
        st.error("ERRO: GROQ_API_KEY não configurada nos Secrets do Streamlit.")
    else:
        with st.spinner(f"Acessando biblioteca {label}..."):
            retriever = get_retriever(st.session_state.livro)
        
        if retriever:
            query = st.text_input(f"O que deseja saber no {label}?")
            if query:
                with st.spinner("🧠 IA Analisando base científica..."):
                    context_docs = retriever.get_relevant_documents(query)[:5]
                    context_text = "\n\n".join([d.page_content for d in context_docs])
                    
                    # O "CÉREBRO" (LLM) CONFIGURADO CORRETAMENTE
                    llm = ChatGroq(
                        model="llama-3.1-70b-versatile", 
                        groq_api_key=api_key,
                        temperature=0.1
                    )
                    
                    messages = [
                        SystemMessage(content=f"Aja como um Ortopedista Senior. Baseado no livro {label}, responda profundamente."),
                        HumanMessage(content=f"Pergunta: {query}\n\nUse os trechos abaixo:\n{context_text}")
                    ]
                    
                    try:
                        resposta = llm.invoke(messages).content
                        st.markdown("---")
                        st.markdown(resposta)
                        
                        col_pdf, col_wa = st.columns(2)
                        with col_pdf:
                            pdf_data = generate_pdf(resposta, query, label)
                            st.download_button("📥 Baixar PDF", pdf_data, f"{query}.pdf")
                        with col_wa:
                            msg_wa = f"*OrtoXande - {label}*\n\n{resposta[:700]}..."
                            link_wa = f"https://wa.me/?text={msg_wa.replace(' ', '%20')}"
                            st.markdown(f'<a href="{link_wa}" target="_blank" style="background-color:#25D366;color:white;padding:10px;text-align:center;text-decoration:none;display:block;border-radius:5px;font-weight:bold;">📲 WhatsApp</a>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erro na consulta: {e}")
        else:
            st.warning(f"Atenção: A pasta {st.session_state.livro} está vazia no GitHub.")
