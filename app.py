# CONFIGURAÇÃO DO MODELO (MODERNO E ESTÁVEL)
llm = ChatGroq(
    model="llama-3.1-70b-versatile", 
    groq_api_key=api_key,
    temperature=0.1
)
