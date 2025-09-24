# chatbot_mentor.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Configuração Inicial (Etapa 1) ---

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Instancia o modelo
modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=api_key
)

# Lista de perguntas para teste
lista_perguntas = [
    "Eu sou geofísico e quero migrar para a área de dados. Qual linguagem de programação devo aprender primeiro?",
    "E que tipo de projeto de portfólio eu poderia criar usando essa linguagem?"
]

# --- Personalização e Cadeia Sem Memória (Etapa 2) ---

# Cria o template de prompt com a persona e placeholders para histórico e query
prompt_mentor = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é o 'GeoAI Mentor', um assistente especializado em ajudar geocientistas a migrar para a área de Ciência de Dados e IA. Sua base é sólida em Geociências, e você orienta sobre a transição para dados, programação e projetos. Seja amigável e didático."),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

# Cria a cadeia (Prompt | Modelo | Parser)
cadeia = prompt_mentor | modelo | StrOutputParser()

# --- Adição de Memória (Etapa 3) ---

# Dicionário para armazenar o histórico de diferentes sessões
memoria_sessoes = {}
sessao_id_teste = "geo_mentor_session"

# Função para obter/criar o histórico de uma sessão (padrão singleton)
def obter_historico_por_sessao(session_id : str):
    """Retorna o histórico de mensagens para um determinado session_id.
    Cria um novo histórico se o session_id não existir."""
    if session_id not in memoria_sessoes:
        memoria_sessoes[session_id] = InMemoryChatMessageHistory()
    return memoria_sessoes[session_id]

# Envelopa a cadeia com o RunnableWithMessageHistory para adicionar a memória
cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=obter_historico_por_sessao,
    input_messages_key="query",      # Variável que contém a nova pergunta do usuário
    history_messages_key="historico" # Variável no template onde o histórico será injetado
)

# --- Execução e Teste ---

print("--- Iniciando o GeoAI Mentor ---")
print(f"Sessão ID: {sessao_id_teste}\n")

# Loop para processar as perguntas com memória
for uma_pergunta in lista_perguntas:
    print(f"Usuário: {uma_pergunta}")
    
    # Invocação da cadeia com memória e configuração da sessão
    resposta = cadeia_com_memoria.invoke(
        # O input é o valor para a chave "query" no template
        {"query" : uma_pergunta},
        
        # O config é essencial para informar qual histórico usar
        config={"session_id": sessao_id_teste}
    )
    
    print(f"GeoAI Mentor: {resposta}\n")

print("--- Fim da Conversa ---")
# Você pode verificar o histórico após a execução, se desejar:
# print(memoria_sessoes[sessao_id_teste].messages)