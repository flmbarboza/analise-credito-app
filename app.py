import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Seu menu personalizado
with st.sidebar:
    # Título customizado (opcional)
    st.markdown("## 📚 Menu da Disciplina")
    
    # Itens de navegação
    pagina = st.radio(
        "Selecione:",
        ["🏠 Home", "🚀 Teste"],
        index=0
    )

# Lógica de redirecionamento
if "Home" in pagina:
    st.switch_page("pages/1_🏠_Home.py")
elif "Teste" in pagina:
    st.switch_page("pages/2_🚀_Teste.py")
