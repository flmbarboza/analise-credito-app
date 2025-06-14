import streamlit as st
from streamlit_option_menu import option_menu  # Instale com: pip install streamlit-option-menu

st.set_page_config(
    page_title="Risco de Crédito",  # Título da aba do navegador
    layout="wide"
)

# Menu horizontal ou vertical
pagina = option_menu(
    menu_title=None,
    options=["Home", "Teste"],
    icons=["house", "rocket"],
    orientation="vertical",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f2f6"},
        "nav-link": {"font-size": "18px", "text-align": "center", "margin": "0px"},
    }
)

# Adicione seu próprio título acima do menu (opcional)
st.markdown("<h1 style='text-align: center;'>Risco de Crédito e Credit Scoring</h1>", unsafe_allow_html=True)
