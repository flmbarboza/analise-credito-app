import streamlit as st

# Configuração global (APENAS AQUI)
st.set_page_config(
    page_title="App Básico",
    page_icon="✨",
    layout="centered"
)

# Menu sidebar (opcional)
st.sidebar.title("NAVEGAÇÃO")
pagina = st.sidebar.radio(
    "Ir para:",
    ["🏠 Home", "🚀 Teste"],
    index=0  # Página padrão
)

# Rodapé (opcional)
st.sidebar.markdown("---")
st.sidebar.caption("v1.0 • Feito com Streamlit")
