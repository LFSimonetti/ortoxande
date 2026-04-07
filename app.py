import streamlit as st
import os
import urllib.parse
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from fpdf import FPDF

# 1. CONFIGURAÇÃO INICIAL
st.set_page_config(page_title="OrtoXande Pro - Modo Literal", layout="centered", page_icon="🦴")

# 2. PEGAR CHAVE DOS SECRETS
api_key = st.secrets.get("GROQ_API_KEY")

# 3. FUNÇÃO DE PDF
def generate_pdf(text, query, fonte):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, f"Extração Técnica: {fonte}", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    return bytes(pdf.output())

@st.cache_resource
def get_retriever(pasta):
    if not os.path.exists(pasta): return None
    docs = []
    files = [f for f in os.listdir(pasta) if f.endswith(".md")]
    if not files: return None
    for f in files:
        loader = TextLoader(os.path.join(pasta, f), encoding="utf-8")
        # Mantemos o metadado do nome do arquivo (página)
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return BM25Retriever.from_documents(splitter.split_documents(docs))

# 4. LÓGICA DE NAVEGAÇÃO
if "livro" not in st.session_state:
    st.session_state.livro = None

if st.session_state.livro is None:
    st.title("🛡️ OrtoXande Pro")
    st.subheader("Selecione o livro para extração literal:")
    c1, c2 = st.columns(2)
    if c1.button("📚 Rockwood & Green", use_container_width=True):
        st.session_state.livro = "livros/rockwood"
        st.rerun()
    if c2.button("📖 Campbell's Operative", use_container_width=True):
        st.session_state.livro = "livros/campbell"
        st.rerun()

else:
    label = "Rockwood" if "rockwood" in st.session_state.livro else "Campbell"
    st.title(f"🔍 Fonte: {label}")
    if st.button("← Trocar de Livro"):
        st.session_state.livro = None
        st.rerun()

    if not api_key:
        st.error("ERRO: GROQ_API_KEY não configurada.")
    else:
        with st.spinner(f"Sincronizando capítulos do {label}..."):
            retriever = get_retriever(st.session_state.livro)
        
        if retriever:
            query = st.text_input(f"O que deseja extrair do {label}?")
            if query:
                with st.spinner("🧠 Localizando informações exatas..."):
                    # Busca trechos relevantes
                    context_docs = retriever.invoke(query)[:5]
                    
                    # Monta o contexto incluindo explicitamente a fonte (nome do arquivo/página)
                    context_text = ""
                    for d in context_docs:
                        fonte_origem = os.path.basename(d.metadata.get('source', 'Desconhecida'))
                        context_text += f"\n[FONTE: {fonte_origem}]\n{d.page_content}\n---"
                    
                    # LLM com Temperatura ZERO (sem criatividade)
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile", 
                        groq_api_key=api_key,
                        temperature=0.0  # Rigidez absoluta
                    )
                    
                    # PROMPT DE RIGIDEZ MÁXIMA
                    system_prompt = (
                        f"Você é um transcritor técnico médico do livro {label}. "
                        "Sua resposta deve ser baseada EXCLUSIVAMENTE nos trechos fornecidos. "
                        "NÃO use conhecimentos externos. NÃO interprete. "
                        "Transcreva ou resuma os fatos exatamente como estão no texto. "
                        "Ao final de cada parágrafo ou explicação, cite obrigatoriamente entre parênteses "
                        "o nome do arquivo fonte fornecido no contexto (ex: página ou capítulo)."
                    )
                    
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Pergunta: {query}\n\nTrechos do Livro:\n{context_text}")
                    ]
                    
                    try:
                        resposta = llm.invoke(messages).content
                        st.markdown("---")
                        st.markdown(resposta)
                        
                        col_pdf, col_wa = st.columns(2)
                        with col_pdf:
                            pdf_bytes = generate_pdf(resposta, query, label)
                            st.download_button("📥 Baixar Relatório Técnico", pdf_bytes, f"{query}.pdf")
                        with col_wa:
                            msg_wa = urllib.parse.quote(f"*Extração OrtoXande - {label}*\n\n{resposta[:800]}")
                            st.markdown(f'<a href="https://wa.me/?text={msg_wa}" target="_blank" style="text-decoration:none;"><div style="background-color:#25D366;color:white;padding:10px;text-align:center;border-radius:8px;font-weight:bold;margin-top:10px;">📲 Enviar WhatsApp</div></a>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erro na extração: {e}")
