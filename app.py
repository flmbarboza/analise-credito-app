import streamlit as st

# Configuração básica
st.set_page_config(
    page_title="Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SOLUÇÃO DEFINITIVA PARA O MENU ---
with st.sidebar:
    # Container vazio para "engolir" o título padrão
    placeholder = st.empty()
    
    # Seu menu personalizado
    st.title("📚 Menu da Disciplina")  # Título visível
    
    # Opções de navegação
    pagina = st.radio(
        "Navegação:",
        ["🏠 Home", "🚀 Teste"],
        label_visibility="collapsed"  # Remove label desnecessário
    )

# --- LÓGICA DE REDIRECIONAMENTO ---
if "Home" in pagina:
    st.switch_page("pages/1_🏠_Home.py")
elif "Teste" in pagina:
    st.switch_page("pages/2_🚀_Teste.py")
