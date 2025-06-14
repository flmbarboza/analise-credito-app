import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("📚 Menu da Disciplina")  # Título do menu

    opcao = st.sidebar.selectbox(
        "Navegue pelos tópicos:",
        ("Introdução", "Conteúdo", "Exercícios", "Sobre")
    )
