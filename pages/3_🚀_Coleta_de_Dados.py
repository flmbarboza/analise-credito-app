import streamlit as st
import numpy as np
import pandas as pd
import kagglehub
import random
from io import StringIO

def gerar_subamostra(base, percentual=0.2, seed=42):
    return base.sample(frac=percentual, random_state=seed).copy()

def simular_instancias_problema(df, n_instancias):
    df_fake = pd.DataFrame(columns=df.columns)
    for _ in range(n_instancias):
        nova_linha = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                if random.random() < 0.2:
                    nova_linha[col] = "Categoria_Inédita_" + str(random.randint(1, 5))
                else:
                    nova_linha[col] = random.choice(df[col].dropna().unique())
            elif np.issubdtype(df[col].dtype, np.number):
                tipo_problema = random.choice(["outlier", "inconsistencia", "faltante", "normal"])
                if tipo_problema == "outlier":
                    nova_linha[col] = df[col].mean() * random.uniform(5, 10)
                elif tipo_problema == "inconsistencia":
                    nova_linha[col] = -abs(df[col].mean())
                elif tipo_problema == "faltante":
                    nova_linha[col] = np.nan
                else:
                    nova_linha[col] = df[col].mean() + np.random.randn()
            else:
                nova_linha[col] = None
        df_fake = pd.concat([df_fake, pd.DataFrame([nova_linha])], ignore_index=True)
    return df_fake

def tratar_categorias(df):
    for col in df.select_dtypes(include='object').columns:
        freq = df[col].value_counts(normalize=True)
        categorias_frequentes = freq[freq > 0.01].index
        df[col] = df[col].apply(lambda x: x if x in categorias_frequentes else 'Outros')
    return df

def executar_pipeline_seed(base, seed):
    # Garante que tudo seja reproduzível
    random.seed(seed)
    np.random.seed(seed)

    sub = gerar_subamostra(base, seed=seed)
    n_instancias_fake = random.randint(60, 120)
    ruído = simular_instancias_problema(sub, n_instancias_fake)
    combinado = pd.concat([sub, ruído], ignore_index=True)
    combinado = tratar_categorias(combinado)
    return combinado

def main():
    st.title("🚀 Coleta de Dados")
    st.balloons()  # Efeito visual para confirmar o carregamento
    st.markdown("""
    Esta página permite:
    - Baixar automaticamente o dataset de aprovação de empréstimos do Kaggle
    - Ou carregar seu próprio arquivo de dados
    """)

    seeds_disponiveis = [42, 7, 13, 21, 99, 123, 456, 789, 1010, 2025]
    seed_escolhida = st.selectbox("Escolha a seed para subamostragem:", seeds_disponiveis)
        
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
                
                    st.success(f"Executando pipeline com seed escolhida: {seed_escolhida}")
                    st.session_state.dados = executar_pipeline_seed(dados, seed_escolhida)
                    st.write(f"Subamostra Selecionada (marcador {seed_escolhida}) - Dimensões: {st.session_state.dados.shape}")
                    st.dataframe(st.session_state.dados.head())
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

        # 2. ANÁLISE SIMPLIFICADA (VERSÃO CORRIGIDA)
        st.subheader("🧐 Entendendo Seus Dados")
        
        # Método mais robusto para análise dos dados
        st.markdown(f"""
        ### 📋 Resumo do Dataset
        - **Total de registros**: {len(st.session_state.dados):,}
        - **Número de variáveis**: {len(st.session_state.dados.columns)}
        """)

        # Explicação dos tipos
        with st.expander("ℹ️ Legenda dos Tipos de Dados e Descrição das Variáveis"):
            st.markdown("""
            - **object**: Texto ou categorias
            - **int/float**: Números
            - **bool**: Verdadeiro/Falso
            - **datetime**: Datas
            """)

            st.subheader("Descrição das Variáveis")
            
            # Criação da tabela como um DataFrame do Pandas
            data = {
                "Nome da Variável": [
                    "loan_id", "no_of_dependents", "education", "self_employed",
                    "income_annum", "loan_amount", "loan_term", "cibil_score",
                    "residential_assets_value", "commercial_assets_value",
                    "luxury_assets_value", "bank_asset_value", "loan_status"
                ],
                "Descrição": [
                    "Um ID único para cada solicitação de empréstimo. Um exemplo de ID seria o CPF do solicitante.",
                    "O número de dependentes que o solicitante do empréstimo possui.",
                    'O nível de educação do solicitante, indicando se ele é "Graduado" ou "Não Graduado".',
                    'Indica se o solicitante é autônomo ("Sim" ou "Não").',
                    "A renda anual do solicitante.",
                    "O valor do empréstimo solicitado (em moeda corrente).",
                    "O prazo do empréstimo (em meses).",
                    "A pontuação de crédito do solicitante feita pela agência CIBIL.",
                    "O valor dos ativos imobiliários do solicitante para fins de moradia (em moeda corrente).",
                    "O valor dos ativos comerciais do solicitante para fins comerciais (em moeda corrente).",
                    "O valor dos ativos do solicitante para fins de lazer (em moeda corrente).",
                    "O valor dos ativos financeiros do solicitante (em moeda corrente).",
                    'O status da aprovação do empréstimo, indicando se foi "Aprovado" ou "Rejeitado".'
                ]
            }
            
            df = pd.DataFrame(data)
            
            # Exibe a tabela com formatação profissional
            st.dataframe(
                df,
                column_config={
                    "Nome da Variável": st.column_config.TextColumn(width="medium"),
                    "Descrição": st.column_config.TextColumn(width="large")
                },
                hide_index=True,
                use_container_width=True
            )

        # Tabela resumida
        resumo = []
        for coluna in st.session_state.dados.columns:
            nao_nulos = st.session_state.dados[coluna].count()
            percent_preenchido = (nao_nulos / len(st.session_state.dados)) * 100
            
            resumo.append({
                "Variável": coluna,
                "Tipo": str(st.session_state.dados[coluna].dtype),
                "Valores únicos": st.session_state.dados[coluna].nunique(),
                "Preenchida (%)": f"{percent_preenchido:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(resumo))
        
        with st.expander("Salvar a Amostra", expanded=False):
            # 3. SALVAR DATAFRAME COMO CSV
            st.subheader("💾 Salvar Subamostra em CSV")
            nome_csv = st.text_input("Nome do arquivo para download:", value="subamostra_credito.csv")
    
            csv = st.session_state.dados.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name=nome_csv,
                mime='text/csv'
            )
    # 🚀 Link para a próxima página
    st.page_link("pages/4_📊_Analise_Univariada.py", label="➡️ Ir para a próxima página: Análise Univariada", icon="📊")

if __name__ == "__main__":
    main()
