import streamlit as st
from streamlit_option_menu import option_menu  # Instale com: pip install streamlit-option-menu

st.set_page_config(
    page_title="Risco de Crédito",  # Título da aba do navegador
    layout="wide"
)

# Adicione seu próprio título acima do menu (opcional)
st.markdown("<h1 style='text-align: center;'>Risco de Crédito e Credit Scoring</h1>", unsafe_allow_html=True)
