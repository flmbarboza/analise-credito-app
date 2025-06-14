import streamlit as st

# Configuração principal
st.set_page_config(
    page_title="Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PARA REMOVER SOMENTE O TÍTULO "app" ---
st.markdown("""
<style>
    /* Remove apenas o título "app" */
    [data-testid="stSidebarUserContent"] > div:first-child {
        visibility: hidden;
        height: 0px;
    }
    
    /* Mantém o resto da sidebar visível */
    [data-testid="stSidebarNav"] {
        margin-top: -30px;
    }
</style>
""", unsafe_allow_html=True)

# --- MENU PERSONALIZADO ---
with st.sidebar:
    # Seu título customizado
    st.markdown("# 📚 Menu da Disciplina")
    
    # Itens do menu
    pagina = st.radio(
        "Navegação:",
        ["🏠 Home", "🚀 Teste"],
        index=0
    )

# --- REDIRECIONAMENTO ---
if "Home" in pagina:
    st.switch_page("pages/1_🏠_Home.py")
elif "Teste" in pagina:
    st.switch_page("pages/2_🚀_Teste.py")
