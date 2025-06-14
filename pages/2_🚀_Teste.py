import streamlit as st

def main():
    st.title("PÁGINA DE TESTE")
    st.warning("🧪 Área experimental")
    st.balloons()  # Efeito visual para confirmar o carregamento

    # Teste interativo
    if st.button("Clique para confirmar"):
        st.success("🎉 Tudo funcionando!")

if __name__ == "__main__":
    main()
