import streamlit as st

def main():
    st.title("PÁGINA INICIAL")
    st.success("✅ Funcionando corretamente!")
    st.write("""
    Este é um teste básico para verificar se o roteamento está OK.
    - Se você está vendo esta mensagem, a página **Home** carregou.
    - Verifique o menu lateral para ir para a página **Teste**.
    """)

if __name__ == "__main__":
    main()  # Opcional: permite executar o arquivo diretamente
