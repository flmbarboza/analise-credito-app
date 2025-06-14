import streamlit as st

# --- SIDEBAR ---
with st.sidebar:
    st.title("📚 Menu da Disciplina")  # Título do menu

    opcao = st.sidebar.selectbox(
        "Navegue pelos tópicos:",
        ("Introdução", "Conteúdo", "Exercícios", "Sobre")
    )
