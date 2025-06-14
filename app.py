import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded"  # Garante que a sidebar esteja visível
)

# Título principal centralizado
st.markdown("<h1 style='text-align: center;'>Risco de Crédito e Credit Scoring</h1>", unsafe_allow_html=True)

# Menu na sidebar esquerda (CORREÇÃO PRINCIPAL)
with st.sidebar:
    selected = option_menu(
        menu_title="Menu Principal",  # Título que aparece acima do menu
        options=["Home", "Teste"],
        icons=["house", "rocket"],
        menu_icon="cast",  # Ícone do menu (opcional)
        default_index=0,
    )
