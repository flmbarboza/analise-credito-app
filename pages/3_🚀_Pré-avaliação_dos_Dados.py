
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
            if 'actions_log' not in st.session_state:
                st.session_state.actions_log = []
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
            file_name='base_de_dados-emprestimo.csv',
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
