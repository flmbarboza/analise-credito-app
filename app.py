import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    menu_title=None
    page_title="Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded"  # Garante que a sidebar esteja visível
)

# Título principal centralizado
st.markdown("<h1 style='text-align: center;'>Risco de Crédito e Credit Scoring</h1>", unsafe_allow_html=True)
