import streamlit as st
from streamlit_option_menu import option_menu  # Instale com: pip install streamlit-option-menu

st.set_page_config(layout="wide")

# Menu horizontal ou vertical
pagina = option_menu(
    menu_title="Cardápio",
    options=["Home", "Teste"],
    icons=["house", "rocket"],
    orientation="horizontal"
)

if pagina == "Home":
    st.switch_page("pages/1_🏠_Home.py")
elif pagina == "Teste":
    st.switch_page("pages/2_🚀_Teste.py")
