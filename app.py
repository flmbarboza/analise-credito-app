import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para customização do sidebar
st.markdown("""
<style>
    /* Remove o título padrão 'app' */
    [data-testid="stSidebar"] > div:first-child > div:first-child > div:first-child {
        display: none;
    }
    
    /* Ajusta o posicionamento do conteúdo customizado */
    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Menu lateral personalizado
with st.sidebar:
    # Título do menu personalizado
    st.markdown("""
    <h1 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>
    📚 Menu da Disciplina
    </h1>
    """, unsafe_allow_html=True)
    
    # Opções de navegação
    pagina = st.radio(
        "Selecione a página:",
        options=["🏠 Home", "🚀 Teste"],
        label_visibility="collapsed"  # Oculta o label padrão
    )

# Lógica de redirecionamento
if "Home" in pagina:
    st.switch_page("pages/1_🏠_Home.py")
elif "Teste" in pagina:
    st.switch_page("pages/2_🚀_Teste.py")
