import streamlit as st
import pandas as pd
import kagglehub
from io import StringIO

def main():
    st.title("🚀 Coleta de Dados")
    st.balloons()  # Efeito visual para confirmar o carregamento
    st.markdown("""
    Esta página permite:
    - Baixar automaticamente o dataset de aprovação de empréstimos do Kaggle
    - Ou carregar seu próprio arquivo de dados
    """)

    # Container para os dados
    if 'dados' not in st.session_state:
        st.session_state.dados = None

    # Opções de coleta
    opcao = st.radio(
        "Selecione a fonte dos dados:",
        ["Baixar dataset do Kaggle", "Carregar arquivo local"],
        horizontal=True
    )

    if opcao == "Baixar dataset do Kaggle":
        if st.button("▶️ Baixar Dados do Kaggle"):
            with st.spinner("Baixando dataset..."):
                try:
                    path = kagglehub.dataset_download('architsharma01/loan-approval-prediction-dataset')
                    dados = pd.read_csv(f'{path}/loan_approval_dataset.csv')
                    st.session_state.dados = dados
                    st.success("Dados baixados com sucesso!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Erro ao baixar dados: {str(e)}")

    else:  # Upload de arquivo
        arquivo = st.file_uploader(
            "Carregue seu arquivo CSV",
            type=["csv"],
            accept_multiple_files=False
        )
        
        if arquivo is not None:
            try:
                dados = pd.read_csv(arquivo)
                st.session_state.dados = dados
                st.success("Arquivo carregado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {str(e)}")

    # Mostrar estrutura dos dados se disponível
    if st.session_state.dados is not None:
        st.divider()
        st.subheader("Estrutura dos Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Primeiras linhas:**")
            st.dataframe(st.session_state.dados.head())
        
        with col2:
            st.markdown("**Últimas linhas:**")
            st.dataframe(st.session_state.dados.tail())
        
        st.divider()
        
        # Análise de estrutura
        st.subheader("Metadados")
        buffer = StringIO()
        st.session_state.dados.info(buf=buffer)
        info_text = buffer.getvalue()
        
        st.text(info_text)
        
        # Estatísticas básicas
        st.subheader("Estatísticas Descritivas")
        st.dataframe(st.session_state.dados.describe(include='all'))

if __name__ == "__main__":
    main()
