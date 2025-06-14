import streamlit as st

# Configuração da página (OBRIGATÓRIO para aplicar o CSS)
st.set_page_config(
    page_title="Risco de Crédito",
    layout="wide"
)

# --- INÍCIO: Código para remover o título "app" ---
st.markdown("""
<style>
    /* Esconde o título padrão "app" */
    [data-testid="stSidebar"] > div:first-child {
        display: none !important;
    }
    
    /* Ajusta o espaçamento do menu */
    [data-testid="stSidebar"] {
        padding-top: 0px !important;
    }
</style>
""", unsafe_allow_html=True)
# --- FIM do código de remoção ---

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
