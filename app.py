import streamlit as st

# Configuração global (único lugar necessário)
st.set_page_config(
    page_title="App Básico",
    page_icon="✨",
    layout="centered"
)

# Conteúdo opcional da página principal
st.write("""
Este é o arquivo principal. Use o menu lateral **automático** do Streamlit 
para navegar entre as páginas (`pages/1_🏠_Home.py` e `pages/2_🚀_Teste.py`).
""")
