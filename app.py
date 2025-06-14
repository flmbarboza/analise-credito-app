import streamlit as st

# --- SOLUÇÃO DEFINITIVA PARA O MENU ---
with st.sidebar:
    # Container vazio para "engolir" o título padrão
    placeholder = st.empty()
    
    # Seu menu personalizado
    st.title("📚 Menu da Disciplina")  # Título visível
