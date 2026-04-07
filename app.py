import streamlit as st
import os
import urllib.parse  # Biblioteca para codificar o link do WhatsApp com segurança
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from fpdf import FPDF

# 1. CONFIGURAÇÃO INICIAL
st.set_page_config(page_title="OrtoXande Pro", layout="centered", page_icon="🦴")

# 2. PEGAR CHAVE DOS SECRETS
api_key = st.secrets.get("GROQ_API_KEY")

# 3. FUNÇÃO DE PDF
def generate_pdf(text, query, fonte):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, f"Consulta Ortopedica: {fonte}", ln=True)
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
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return BM25Retriever.from_documents(splitter.split_documents(docs))

# 4. LÓGICA DE NAVEGAÇÃO
if "livro" not in st.session_state:
    st.session_state.livro = None

# TELA DE SELEÇÃO
if st.session_state.livro is None:
    st.title("🛡️ OrtoXande Pro")
    st.subheader("Selecione o livro para pesquisa:")
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
        st.error("ERRO: GROQ_API_KEY não configurada nos Secrets.")
    else:
        with st.spinner(f"Acessando capítulos do {label}..."):
            retriever = get_retriever(st.session_state.livro)
        
        if retriever:
            query = st.text_input(f"O que deseja saber no {label}?", key="input_query")
            if query:
                with st.spinner("🧠 IA Analisando base científica..."):
                    context_docs = retriever.invoke(query)[:5]
                    context_text = "\n\n".join([d.page_content for d in context_docs])
                    
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile", 
                        groq_api_key=api_key,
                        temperature=0.1
                    )
                    
                    messages = [
                        SystemMessage(content=f"Aja como um Ortopedista Senior. Responda profundamente baseado no livro {label}."),
                        HumanMessage(content=f"Pergunta: {query}\n\nUse os trechos abaixo:\n{context_text}")
                    ]
                    
                    try:
                        resposta = llm.invoke(messages).content
                        st.markdown("---")
                        st.markdown(resposta)
                        
                        col_pdf, col_wa = st.columns(2)
                        with col_pdf:
                            pdf_bytes = generate_pdf(resposta, query, label)
                            st.download_button(
                                label="📥 Baixar PDF",
                                data=pdf_bytes,
                                file_name=f"{query.replace(' ', '_')}.pdf",
                                mime="application/pdf"
                            )
                        with col_wa:
                            # CODIFICAÇÃO PROFISSIONAL PARA WHATSAPP
                            prefixo = f"*OrtoXande - {label}*\n\n"
                            # Pegamos os primeiros 800 caracteres para não exceder o limite do link
                            mensagem_final = prefixo + resposta[:800] + "..."
                            # O comando quote codifica aspas, espaços e símbolos corretamente
                            msg_encoded = urllib.parse.quote(mensagem_final)
                            link_wa = f"https://wa.me/?text={msg_encoded}"
                            
                            # Botão HTML Blindado
                            st.markdown(
                                f"""
                                <a href="{link_wa}" target="_blank" style="text-decoration: none;">
                                    <div style="
                                        background-color: #25D366;
                                        color: white;
                                        padding: 10px;
                                        text-align: center;
                                        border-radius: 8px;
                                        font-weight: bold;
                                        display: block;
                                        margin-top: 10px;
                                    ">
                                        📲 Enviar WhatsApp
                                    </div>
                                </a>
                                """,
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        st.error(f"Erro na consulta à IA: {e}")
        else:
            st.warning(f"A pasta {st.session_state.livro} está vazia ou não existe.")
