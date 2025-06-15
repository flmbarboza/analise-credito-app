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
        st.session_state.colunas_originais = None

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

   # Seção para ajuste de nomes de variáveis
    if st.session_state.dados is not None:
        st.divider()
        
        # 1. OPÇÃO PARA AJUSTAR NOMES DAS VARIÁVEIS
        st.subheader("🔧 Ajuste dos Nomes das Variáveis")
        
        if st.checkbox("Deseja renomear as colunas?"):
            colunas_atuais = st.session_state.dados.columns.tolist()
            novos_nomes = []
            
            for i, coluna in enumerate(colunas_atuais):
                novo_nome = st.text_input(
                    f"Renomear '{coluna}' para:",
                    value=coluna,
                    key=f"nome_{i}"
                )
                novos_nomes.append(novo_nome)
            
            if st.button("Aplicar novos nomes"):
                st.session_state.dados.columns = novos_nomes
                st.success("Nomes das colunas atualizados!")
                st.session_state.colunas_originais = colunas_atuais  # Guarda original

        # 2. ANÁLISE SIMPLIFICADA PARA LEIGOS
        st.subheader("🧐 Entendendo Seus Dados")
        
        buffer = StringIO()
        st.session_state.dados.info(buf=buffer)
        info_text = buffer.getvalue()
        
        # Processa a saída do .info() para leigos
        st.markdown("""
        ### 📋 Resumo das Variáveis
        
        **O que cada número significa:**
        - **Total de registros**: Quantas linhas de dados você tem
        - **Variáveis não-nulas**: Quantos valores preenchidos existem em cada coluna
        - **Tipo de dado**: Como a informação está armazenada (texto, número, etc.)
        """)
        
        # Tabela simplificada
        info_lines = [line for line in info_text.split('\n') if 'non-null' in line]
        
        resumo = []
        for line in info_lines:
            parts = line.split()
            nome = parts[1] if len(parts) > 1 else ""
            tipo = parts[-1] if len(parts) > 2 else ""
            nao_nulos = parts[3] if len(parts) > 3 else ""
            
            resumo.append({
                "Variável": nome,
                "Tipo": tipo,
                "Preenchida (%)": f"{int(nao_nulos)/len(st.session_state.dados)*100:.1f}%"
            })
        
        st.table(pd.DataFrame(resumo))
        
        # Explicação dos tipos de dados
        with st.expander("ℹ️ O que significam os tipos de dados?"):
            st.markdown("""
            - **object**: Texto ou dados categóricos (ex: nomes, categorias)
            - **int64/float64**: Números inteiros ou decimais
            - **bool**: Valores verdadeiro/falso
            - **datetime64**: Datas e horários
            """)

if __name__ == "__main__":
    main()
