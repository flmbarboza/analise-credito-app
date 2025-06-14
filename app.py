import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Análise de Crédito Inteligente",
    page_icon="💳",
    layout="wide"
)

# Menu principal no sidebar
st.sidebar.title("🏦 Menu Principal")
pagina_selecionada = st.sidebar.selectbox(
    "Escolha uma opção:",
    [
        "Página_Inicial"
    ]
)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    💳 Plataforma Financeira Completa<br>
    Versão 2.0
</div>
""", unsafe_allow_html=True)
