import streamlit as st
import numpy as np
import pandas as pd
import kagglehub
import random
from io import BytesIO
from io import StringIO
from datetime import datetime

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

        # 2. ANÁLISE SIMPLIFICADA
        st.subheader("🧐 Entendendo Seus Dados")
        
        # Método mais robusto para análise dos dados
        st.markdown(f"""
        ### 📋 Resumo do Dataset
        - **Total de registros**: {len(st.session_state.dados):,}
        - **Número de variáveis**: {len(st.session_state.dados.columns)}
        """)

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

    # Configuração inicial
    st.set_page_config(layout="wide")
    st.title("🔍 Pré-Análise de Dados Interativa")
    

    # Opções para carregar dados
    with st.expander("📤 Carregar Dados", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Opção 1: Usar dados existentes na session_state
            if 'dados' in st.session_state:
                st.write("Dados existentes carregados:")
                st.write(f"Shape: {st.session_state.dados.shape}")
                if st.button("Continuar usando estes dados"):
                    df = st.session_state.dados
                    st.success("Continuando com os dados existentes!")
            else:
                st.warning("Nenhum dado carregado na sessão atual")
        
        with col2:
            # Opção 2: Fazer novo upload
            uploaded_file = st.file_uploader("Ou carregue novo arquivo CSV", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.dados = df  # Armazena na session_state
                st.success("Novo dataset carregado com sucesso!")

        # Mostra preview dos dados
        st.subheader("Visualização dos Dados")
        st.dataframe(df.head(), use_container_width=True)
        
    # Verifica se temos dados para trabalhar
    if 'df' not in locals():
        st.warning("Por favor, carregue dados para continuar")
        st.stop()
    
    # Função para download do DataFrame
    def convert_df_to_csv(df):
        output = BytesIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        return output.getvalue()
    
    # Análise Interativa
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔎 Identificar Problemas", 
        "✏️ Corrigir Dados", 
        "📊 Resumo das Ações", 
        "💾 Exportar Dados"
    ])
    
    with tab1:
        st.header("Identificação de Problemas nos Dados")
        
        analysis_type = st.radio("Selecione o tipo de análise:", [
            "Dados Faltantes", 
            "Dados Inconsistentes", 
            "Outliers/Valores Extremos"
        ], horizontal=True)
        
        if analysis_type == "Dados Faltantes":
            missing = st.session_state.dados.isnull().sum()
            missing = missing[missing > 0]
            
            if not missing.empty:
                st.warning("⚠️ Dados Faltantes Detectados:")
                st.dataframe(missing.rename("Quantidade"), use_container_width=True)
                
                # Visualização gráfica
                st.subheader("Mapa de Dados Faltantes")
                st.bar_chart(missing)
            else:
                st.success("✅ Nenhum dado faltante encontrado!")
        
        elif analysis_type == "Dados Inconsistentes":
            st.subheader("Análise de Inconsistências")
            
            # Identificar colunas com possíveis inconsistências
            text_cols = st.session_state.dados.select_dtypes(include=['object']).columns
            num_cols = st.session_state.dados.select_dtypes(include=np.number).columns
            
            col_to_analyze = st.selectbox("Selecione a coluna para análise:", text_cols.union(num_cols))
            
            if col_to_analyze in text_cols:
                # Análise para colunas textuais
                value_counts = st.session_state.dados[col_to_analyze].value_counts()
                st.dataframe(value_counts, use_container_width=True)
                
                # Identificar valores únicos para possível padronização
                st.write("Valores únicos encontrados:")
                st.write(st.session_state.dados[col_to_analyze].unique())
            else:
                # Análise para colunas numéricas
                st.write(f"Estatísticas descritivas para {col_to_analyze}:")
                st.write(st.session_state.dados[col_to_analyze].describe())
                
                # Identificar possíveis outliers
                q1 = st.session_state.dados[col_to_analyze].quantile(0.25)
                q3 = st.session_state.dados[col_to_analyze].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = st.session_state.dados[
                    (st.session_state.dados[col_to_analyze] < lower_bound) | 
                    (st.session_state.dados[col_to_analyze] > upper_bound)
                ]
                
                if not outliers.empty:
                    st.warning(f"⚠️ Possíveis outliers detectados em {col_to_analyze}:")
                    st.dataframe(outliers, use_container_width=True)
                else:
                    st.success(f"✅ Nenhum outlier detectado em {col_to_analyze}")
    
    with tab2:
        st.header("Correção de Dados")
        
        correction_type = st.radio("Tipo de correção:", [
            "Remover Dados", 
            "Substituir Valores", 
            "Preencher Valores Faltantes"
        ], horizontal=True)
        
        col_to_correct = st.selectbox(
            "Selecione a coluna para correção:", 
            st.session_state.dados.columns
        )
        
        if correction_type == "Remover Dados":
            remove_option = st.radio("Remover:", [
                "Linhas com valores faltantes", 
                "Linhas com valores específicos"
            ])
            
            if remove_option == "Linhas com valores específicos":
                if st.session_state.dados[col_to_correct].dtype == 'object':
                    values_to_remove = st.multiselect(
                        "Selecione os valores a remover:", 
                        st.session_state.dados[col_to_correct].unique()
                    )
                else:
                    min_val = float(st.session_state.dados[col_to_correct].min())
                    max_val = float(st.session_state.dados[col_to_correct].max())
                    values_to_remove = st.slider(
                        "Selecione o intervalo de valores a remover:", 
                        min_val, max_val, (min_val, max_val)
                    )
            
            if st.button("Aplicar Remoção"):
                if remove_option == "Linhas com valores faltantes":
                    initial_count = len(st.session_state.dados)
                    st.session_state.dados = st.session_state.dados.dropna(subset=[col_to_correct])
                    removed_count = initial_count - len(st.session_state.dados)
                    
                    # Registrar ação
                    action = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': f"Removidas {removed_count} linhas com valores faltantes na coluna {col_to_correct}",
                        'type': "Remoção"
                    }
                    st.session_state.actions_log.append(action)
                    st.success(f"Removidas {removed_count} linhas com valores faltantes!")
                
                elif remove_option == "Linhas com valores específicos" and values_to_remove:
                    initial_count = len(st.session_state.dados)
                    
                    if isinstance(values_to_remove, tuple):
                        mask = (st.session_state.dados[col_to_correct] >= values_to_remove[0]) & \
                               (st.session_state.dados[col_to_correct] <= values_to_remove[1])
                    else:
                        mask = st.session_state.dados[col_to_correct].isin(values_to_remove)
                    
                    st.session_state.dados = st.session_state.dados[~mask]
                    removed_count = initial_count - len(st.session_state.dados)
                    
                    # Registrar ação
                    action = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': f"Removidas {removed_count} linhas com valores específicos na coluna {col_to_correct}",
                        'type': "Remoção"
                    }
                    st.session_state.actions_log.append(action)
                    st.success(f"Removidas {removed_count} linhas com valores específicos!")
        
        elif correction_type == "Substituir Valores":
            st.subheader("Substituição de Valores")
            
            if st.session_state.dados[col_to_correct].dtype == 'object':
                old_value = st.selectbox(
                    "Valor a ser substituído:", 
                    st.session_state.dados[col_to_correct].unique()
                )
                new_value = st.text_input("Novo valor:")
            else:
                old_value = st.number_input("Valor a ser substituído:", 
                    value=float(st.session_state.dados[col_to_correct].iloc[0]))
                new_value = st.number_input("Novo valor:")
            
            if st.button("Aplicar Substituição") and str(new_value):
                count = (st.session_state.dados[col_to_correct] == old_value).sum()
                st.session_state.dados[col_to_correct] = st.session_state.dados[col_to_correct].replace(old_value, new_value)
                
                # Registrar ação
                action = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': f"Substituídos {count} valores '{old_value}' por '{new_value}' na coluna {col_to_correct}",
                    'type': "Substituição"
                }
                st.session_state.actions_log.append(action)
                st.success(f"Substituídos {count} valores com sucesso!")
        
        elif correction_type == "Preencher Valores Faltantes":
            st.subheader("Preenchimento de Valores Faltantes")
            
            fill_method = st.radio("Método de preenchimento:", [
                "Valor Fixo", 
                "Média/Moda", 
                "Interpolação"
            ])
            
            if fill_method == "Valor Fixo":
                if st.session_state.dados[col_to_correct].dtype == 'object':
                    fill_value = st.text_input("Valor para preenchimento:")
                else:
                    fill_value = st.number_input("Valor para preenchimento:")
                
                if st.button("Aplicar Preenchimento") and fill_value is not None:
                    count = st.session_state.dados[col_to_correct].isnull().sum()
                    st.session_state.dados[col_to_correct] = st.session_state.dados[col_to_correct].fillna(fill_value)
                    
                    # Registrar ação
                    action = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': f"Preenchidos {count} valores faltantes com '{fill_value}' na coluna {col_to_correct}",
                        'type': "Preenchimento"
                    }
                    st.session_state.actions_log.append(action)
                    st.success(f"Preenchidos {count} valores faltantes!")
            
            elif fill_method == "Média/Moda":
                if st.button("Aplicar Preenchimento"):
                    count = st.session_state.dados[col_to_correct].isnull().sum()
                    
                    if st.session_state.dados[col_to_correct].dtype == 'object':
                        fill_value = st.session_state.dados[col_to_correct].mode()[0]
                    else:
                        fill_value = st.session_state.dados[col_to_correct].mean()
                    
                    st.session_state.dados[col_to_correct] = st.session_state.dados[col_to_correct].fillna(fill_value)
                    
                    # Registrar ação
                    action = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': f"Preenchidos {count} valores faltantes com {fill_method} ({fill_value:.2f}) na coluna {col_to_correct}",
                        'type': "Preenchimento"
                    }
                    st.session_state.actions_log.append(action)
                    st.success(f"Preenchidos {count} valores faltantes com {fill_method}!")
        
        # Visualização após correções
        st.subheader("Visualização após Correções")
        st.dataframe(st.session_state.dados.head(), use_container_width=True)
    
    with tab3:
        st.header("Resumo das Ações Realizadas")
        
        if st.session_state.actions_log:
            st.subheader("Histórico de Modificações")
            actions_df = pd.DataFrame(st.session_state.actions_log)
            st.dataframe(actions_df.sort_values('timestamp', ascending=False), use_container_width=True)
            
            # Estatísticas resumidas
            st.subheader("Estatísticas das Ações")
            action_counts = actions_df['type'].value_counts()
            st.bar_chart(action_counts)
            
            # Seleção de ações para manter
            st.subheader("Selecionar Ações para Manter")
            selected_actions = st.multiselect(
                "Selecione as ações que deseja manter no relatório final:",
                options=actions_df['action'].unique(),
                default=actions_df['action'].unique()
            )
            
            if st.button("Confirmar Seleção"):
                st.session_state.selected_actions = selected_actions
                st.success("Seleção confirmada! Estas ações serão incluídas no relatório.")
        else:
            st.info("Nenhuma ação registrada ainda.")
    
    with tab4:
        st.header("Exportar Dados Limpos")
        
        st.subheader("Dataset Modificado")
        st.dataframe(st.session_state.dados.head(), use_container_width=True)
        
        st.download_button(
            label="📥 Baixar Dataset Limpo como CSV",
            data=convert_df_to_csv(st.session_state.dados),
            file_name='dados_limpos.csv',
            mime='text/csv'
        )
        
        if st.session_state.get('actions_log'):
            st.subheader("Relatório de Ações")
            actions_csv = convert_df_to_csv(pd.DataFrame(st.session_state.actions_log))
            
            st.download_button(
                label="📥 Baixar Relatório de Ações",
                data=actions_csv,
                file_name='relatorio_acoes.csv',
                mime='text/csv'
            )
    
    # Rodapé
    st.divider()
    st.caption("🔧 Ferramenta de Pré-Análise de Dados | v1.0")

if __name__ == "__main__":
    main()
