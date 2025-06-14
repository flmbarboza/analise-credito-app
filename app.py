import streamlit as st

# Configuração global (único lugar necessário)
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

if pagina == "Home":
    st.switch_page("pages/1_🏠_Home.py")
elif pagina == "Teste":
    st.switch_page("pages/2_🚀_Teste.py")
    
# Conteúdo opcional da página principal
st.write("""
Este é o arquivo principal. Use o menu lateral **automático** do Streamlit 
para navegar entre as páginas (`pages/1_🏠_Home.py` e `pages/2_🚀_Teste.py`).
""")
