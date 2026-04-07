import streamlit as st
import os
import urllib.parse
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from fpdf import FPDF

# 1. CONFIGURAÇÃO DE ALTA PERFORMANCE
st.set_page_config(page_title="OrtoXande Pro - Decisão Clínica", layout="wide", page_icon="🦴")

# 2. PEGAR CHAVE DOS SECRETS
api_key = st.secrets.get("GROQ_API_KEY")

# 3. FUNÇÃO DE GERAÇÃO DE PDF TÉCNICO
def generate_pdf(text, query, fonte):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, f"RELATÓRIO TÉCNICO: {query.upper()}", ln=True, align='C')
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(0, 10, f"Fonte Consultada: {fonte} - Extração Literal", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=11)
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, clean_text)
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
    
    # Chunk maior para preservar tabelas e algoritmos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    return BM25Retriever.from_documents(splitter.split_documents(docs))

# 4. LÓGICA DE NAVEGAÇÃO
if "livro" not in st.session_state:
    st.session_state.livro = None

if st.session_state.livro is None:
    st.title("🛡️ OrtoXande Pro - Sistema de Extração Técnica")
    st.info("Configurado para seguir as diretrizes da 8ª edição do Rockwood & Green e Campbell's Operative.")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📚 Rockwood & Green", use_container_width=True):
            st.session_state.livro = "livros/rockwood"
            st.rerun()
    with c2:
        if st.button("📖 Campbell's Operative", use_container_width=True):
            st.session_state.livro = "livros/campbell"
            st.rerun()

else:
    label = "Rockwood" if "rockwood" in st.session_state.livro else "Campbell"
    st.sidebar.title(f"📖 {label}")
    if st.sidebar.button("← Trocar Livro"):
        st.session_state.livro = None
        st.rerun()

    if not api_key:
        st.error("ERRO: GROQ_API_KEY não configurada.")
    else:
        with st.spinner("Sincronizando base de dados..."):
            retriever = get_retriever(st.session_state.livro)
        
        if retriever:
            query = st.text_input("Descreva a fratura para análise (ex: Fratura diafisária da tíbia):")
            
            if query:
                with st.spinner("🧠 Localizando evidências e algoritmos..."):
                    context_docs = retriever.invoke(query)[:6]
                    context_text = ""
                    for d in context_docs:
                        # Extrai o nome do arquivo para citação
                        pag_ref = os.path.basename(d.metadata.get('source', 'Fonte')).replace('.md', '')
                        context_text += f"\n[TRECHO DA FONTE: {pag_ref}]\n{d.page_content}\n---"
                    
                    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.0)
                    
                    # SYSTEM PROMPT - O RIGOR DO CHECKLIST
                    system_prompt = f"""
                    Você é um motor de extração técnica médica fiel ao livro {label}.
                    Sua resposta deve ser 100% baseada nos trechos fornecidos. Se uma informação não estiver presente, escreva 'Informação não detalhada nos trechos consultados'.
                    
                    ESTRUTURA OBRIGATÓRIA DA RESPOSTA:

                    1. IDENTIFICAÇÃO E LOCALIZAÇÃO ANATÔMICA: Osso, região (terço proximal/médio/distal) e se é intra ou extra-articular.
                    
                    2. MORFOLOGIA E PADRÃO DA FRATURA: Traço (transversal, oblíqua, espiral, cominutiva) e tipo de desvio descrito.
                    
                    3. CLASSIFICAÇÃO ESPECÍFICA: Citar AO/OTA (ex: 32-B2) e classificações epônimas ou específicas mencionadas (ex: Gustilo, Neer, Garden).
                    
                    4. AVALIAÇÃO DE PARTES MOLES: Integridade da pele e escalas de lesão tecidual (ex: Tscherne).
                    
                    5. MECANISMO E CONDIÇÕES DO PACIENTE: Etiologia (alta/baixa energia) e qualidade óssea (osteoporose/patológica).
                    
                    6. ESTRUTURA ACADÊMICA:
                       - Título da Fratura
                       - Mecanismo de Lesão
                       - Classificação Clínica detalhada
                       - Avaliação Radiográfica necessária
                       - Algoritmo de Tratamento (Conservador vs Cirúrgico)
                       - Pérolas e Armadilhas (Pearls and Pitfalls)

                    REGRAS CRÍTICAS:
                    - Use linguagem técnica médica formal.
                    - Ao final de CADA uma das 6 seções, cite entre parênteses a [FONTE: nome_do_arquivo] de onde o dado foi extraído.
                    - Proibido adicionar recomendações que não estejam no texto.
                    """
                    
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Pergunta Clínica: {query}\n\nBase de Dados:\n{context_text}")
                    ]
                    
                    try:
                        resposta = llm.invoke(messages).content
                        st.markdown("---")
                        st.markdown(resposta)
                        
                        col_pdf, col_wa = st.columns(2)
                        with col_pdf:
                            pdf_bytes = generate_pdf(resposta, query, label)
                            st.download_button("📥 Baixar Relatório Clínico (PDF)", pdf_bytes, f"Analise_{query}.pdf")
                        with col_wa:
                            # Limite de segurança para link de WhatsApp
                            msg_wa = urllib.parse.quote(f"*Consulta OrtoXande Pro*\n\n{resposta[:900]}")
                            st.markdown(f'''
                                <a href="https://wa.me/?text={msg_wa}" target="_blank" style="text-decoration:none;">
                                    <div style="background-color:#25D366;color:white;padding:12px;text-align:center;border-radius:10px;font-weight:bold;">
                                        📲 Enviar via WhatsApp
                                    </div>
                                </a>
                            ''', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erro na extração: {e}")
        else:
            st.warning("Base de dados não encontrada. Verifique os arquivos .md no GitHub.")
