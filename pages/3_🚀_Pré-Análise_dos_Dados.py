import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Título da página
st.set_page_config(page_title="Tratamento de Dados", layout="wide")
st.title("🔍 Pré-Análise de Dados Interativa")

# Inicialização do session_state
if 'dados' not in st.session_state:
    st.warning("Dados não carregados! Acesse a página de Coleta primeiro.")
    st.page_link("pages/3_🚀_Coleta_de_Dados.py", label="→ Ir para Coleta")
    return

dados = st.session_state.dados

if dados is None or dados.empty:
    st.error("Os dados estão vazios ou não foram carregados corretamente.")
    return

if 'actions_log' not in st.session_state:
    st.session_state.actions_log = []

# Variável auxiliar
df = st.session_state.dados.copy()

# Variáveis de estado para seleção de linhas
for key in ['faltantes_indices', 'inconsistentes_indices', 'outlier_indices']:
    if key not in st.session_state:
        st.session_state[key] = []

# Função auxiliar para registrar ações
def registrar_acao(tipo, acao, removidos):
    st.session_state.actions_log.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': tipo,
        'action': acao,
        'removed': removidos
    })

# Expander: Identificar e Remover Linhas Duplicadas
with st.expander("🔍 Identificar e Remover Linhas Duplicadas", expanded=False):
    st.markdown("### Resumo de Dados Duplicados")

    df = st.session_state.dados.copy()
    
    # Identificar linhas duplicadas (considerando todas as colunas)
    duplicatas = df[df.duplicated(keep=False)]  # keep=False marca todas as duplicatas
    total_linhas = len(df)
    total_duplicatas = len(duplicatas)

    if not duplicatas.empty:
        # Agrupar duplicatas para melhor visualização
        duplicatas_agrupadas = duplicatas.groupby(list(df.columns)).size().reset_index(name='Contagem')
        
        st.warning(f"⚠️ Foram encontradas **{total_duplicatas} linhas duplicadas** (grupos repetidos) em **{total_linhas} linhas totais**.")
        
        st.markdown("### Visualização dos Grupos de Duplicatas")
        st.write("Cada grupo mostra linhas idênticas e quantas vezes aparecem:")
        st.dataframe(duplicatas_agrupadas.sort_values(by='Contagem', ascending=False), use_container_width=True)

        st.markdown("### Ação: Remover Linhas Duplicadas")
        
        col1, col2 = st.columns(2)
        with col1:
            # Opção para manter a primeira ou última ocorrência
            keep_option = st.radio(
                "Manter:",
                options=('Primeira ocorrência', 'Última ocorrência'),
                index=0,
                help="Qual versão da duplicata deve ser mantida?"
            )
        
        with col2:
            # Opção para considerar apenas algumas colunas
            colunas_considerar = st.multiselect(
                "Considerar apenas estas colunas:",
                options=df.columns.tolist(),
                default=df.columns.tolist(),
                help="Selecione as colunas que definem uma duplicata"
            )
        
        if st.button("🗑️ Remover Linhas Duplicadas"):
            # Determinar qual ocorrência manter
            keep = 'first' if keep_option == 'Primeira ocorrência' else 'last'
            
            # Remover duplicatas
            df_sem_duplicatas = df.drop_duplicates(subset=colunas_considerar if colunas_considerar else None, keep=keep)
            
            # Calcular quantas linhas foram removidas
            linhas_removidas = len(df) - len(df_sem_duplicatas)
            
            # Atualizar dados na sessão
            st.session_state.dados = df_sem_duplicatas.reset_index(drop=True)

            # Registrar ação
            action = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'action': f"Removidas {linhas_removidas} linhas duplicadas (mantendo {keep_option.lower()})",
                'type': "Remoção de Duplicatas",
                'removed': linhas_removidas
            }
            st.session_state.actions_log.append(action)

            st.success(f"{linhas_removidas} linhas duplicadas foram removidas com sucesso!")
            st.rerun()
    else:
        st.success("✅ Nenhuma linha duplicada encontrada.")
        
# Expander: Identificar e Remover Linhas com Dados Faltantes
with st.expander("🔎 Identificar e Remover Linhas com Dados Faltantes", expanded=False):
    st.markdown("### Resumo de Dados Faltantes")

    df = st.session_state.dados.copy()
    
    # Identificar linhas com dados faltantes
    mask_linhas_faltantes = df.isnull().any(axis=1)
    linhas_com_faltantes = df[mask_linhas_faltantes]

    total_linhas = len(df)
    total_com_faltantes = len(linhas_com_faltantes)

    if total_com_faltantes > 0:
        st.warning(f"⚠️ Foram encontradas **{total_com_faltantes} linhas** com dados faltantes em **{total_linhas} linhas totais**.")

        st.markdown("### Visualização das Linhas com Dados Faltantes")
        st.dataframe(linhas_com_faltantes, use_container_width=True)

        st.markdown("### Ação: Remover Linhas com Dados Faltantes")
        if st.button("🗑️ Excluir Todas as Linhas com Dados Faltantes"):
            df = df.dropna(how='any')
            st.session_state.dados = df.reset_index(drop=True)

            # Registrar ação
            action = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'action': f"Excluídas {total_com_faltantes} linhas com dados faltantes (todas as colunas)",
                'type': "Remoção"
            }
            st.session_state.actions_log.append(action)

            st.success(f"{total_com_faltantes} linhas com dados faltantes foram excluídas com sucesso!")
            st.rerun()
    else:
        st.success("✅ Nenhuma linha com dados faltantes encontrada.")
        
# Expander 2: Dados Inconsistentes
with st.expander("✏️ Identificar Dados Inconsistentes", expanded=False):
    text_cols = df.select_dtypes(include='object').columns
    if len(text_cols) == 0:
        st.info("Nenhuma coluna textual disponível.")
    else:
        coluna_inconsistente = st.selectbox(
            "Selecione a coluna para verificar inconsistências:",
            text_cols,
            key="inconsistente_coluna"
        )
        if coluna_inconsistente:
            value_counts = df[coluna_inconsistente].value_counts()
            st.dataframe(value_counts.rename("Frequência"), use_container_width=True)

            valores_unicos = df[coluna_inconsistente].unique()
            st.session_state.inconsistentes_indices = st.multiselect(
                "Selecione os valores inconsistentes para excluir:",
                options=valores_unicos,
                default=[],
                key="inconsistentes_multiselect"
            )

            if st.button("Excluir Valores Selecionados", key="excluir_inconsistentes") and st.session_state.inconsistentes_indices:
                initial_count = len(df)
                df = df[~df[coluna_inconsistente].isin(st.session_state.inconsistentes_indices)]
                removed_count = initial_count - len(df)
                st.session_state.dados = df.reset_index(drop=True)
                registrar_acao(
                    "Remoção",
                    f"Excluídos valores: {st.session_state.inconsistentes_indices} na coluna '{coluna_inconsistente}'",
                    removed_count
                )
                st.session_state.inconsistentes_indices = []
                st.success(f"{removed_count} linhas removidas com sucesso!")
                st.rerun()

# Expander 3: Outliers
with st.expander("📊 Identificar Outliers", expanded=False):
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        st.info("Nenhuma coluna numérica disponível.")
    else:
        coluna_outlier = st.selectbox(
            "Selecione a coluna para análise de outliers:",
            numeric_cols,
            key="outlier_coluna"
        )
        if coluna_outlier:
            q1 = df[coluna_outlier].quantile(0.25)
            q3 = df[coluna_outlier].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[coluna_outlier] < lower_bound) | (df[coluna_outlier] > upper_bound)]

            if not outliers.empty:
                st.warning(f"⚠️ Outliers detectados em '{coluna_outlier}':")
                st.dataframe(outliers, use_container_width=True)

                indices = outliers.index.tolist()
                st.session_state.outlier_indices = st.multiselect(
                    "Selecione os índices dos outliers para excluir:",
                    options=indices,
                    default=indices,
                    key="outlier_multiselect"
                )

                if st.button("Excluir Outliers Selecionados", key="excluir_outliers"):
                    initial_count = len(df)
                    df = df.drop(index=st.session_state.outlier_indices)
                    removed_count = initial_count - len(df)
                    st.session_state.dados = df.reset_index(drop=True)
                    registrar_acao(
                        "Remoção",
                        f"Excluídos {removed_count} outliers na coluna '{coluna_outlier}'",
                        removed_count
                    )
                    st.session_state.outlier_indices = []
                    st.success(f"{removed_count} linhas removidas com sucesso!")
                    st.rerun()
            else:
                st.success("✅ Nenhum outlier detectado nessa coluna.")

#Expander 4: Manual
with st.expander("🧹 Exclusão Manual de Linhas", expanded=False):
    # Garantir que os dados existam
    if 'dados' not in st.session_state or st.session_state.dados.empty:
        st.warning("⚠️ Nenhum dado carregado.")
        st.stop()

    # Adicionar coluna de seleção
    df.insert(0, "Selecionar", False)

    # Permitir edição com checkboxes
    df = st.data_editor(
        df,
        hide_index=False,
        column_config={
            "Selecionar": st.column_config.CheckboxColumn("Selecionar", default=False)
        },
        disabled=df.columns[1:].tolist(),  # Desativa edição das outras colunas
        use_container_width=True
    )

    # Campo para justificativa da exclusão
    reason = st.text_input("Informe o motivo da exclusão:")

    # Botão para aplicar a exclusão
    if st.button("🗑️ Excluir Linhas Selecionadas", key="excluir_linhas_selecionadas"):
        if not reason.strip():
            st.warning("⚠️ Por favor, informe o motivo da exclusão.")
        else:
            # Filtrar linhas selecionadas
            selected_rows = df[df["Selecionar"]]
            if not selected_rows.empty:
                indices_to_remove = selected_rows.index.tolist()
                count_removed = len(indices_to_remove)

                # Atualizar dados
                st.session_state.dados = st.session_state.dados[~st.session_state.dados.index.isin(indices_to_remove)]

                # Registrar ação com motivo personalizado
                action = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': f"Exclusão manual de {count_removed} linha(s): {reason}",
                    'type': "Remoção"
                }
                st.session_state.actions_log.append(action)

                st.success(f"{count_removed} linha(s) removida(s) com sucesso!")
                st.rerun()
            else:
                st.warning("⚠️ Nenhuma linha foi selecionada para exclusão.")
                
# Expander 5: Resumo das Ações
with st.expander("📝 Resumo das Ações Realizadas", expanded=False):
    if st.session_state.actions_log:
        st.subheader("Histórico de Modificações")

        # Criar DataFrame
        actions_df = pd.DataFrame(st.session_state.actions_log)
        
        # Verificar se todas as colunas esperadas existem
        available_columns = actions_df.columns.tolist()
        expected_columns = ['timestamp', 'type', 'action', 'removed']
        
        # Criar dicionário de mapeamento de colunas
        column_mapping = {
            'timestamp': 'Quando',
            'type': 'Ação',
            'action': 'Detalhes',
            'removed': 'Quantidade'
        }
        
        # Selecionar apenas as colunas disponíveis
        columns_to_keep = [col for col in expected_columns if col in available_columns]
        actions_df = actions_df[columns_to_keep]
        
        # Renomear as colunas que existem
        actions_df = actions_df.rename(columns={
            col: column_mapping[col] for col in columns_to_keep if col in column_mapping
        })

        # Adicionar coluna de seleção
        actions_df.insert(0, "Selecionar", False)

        # Permitir edição (seleção de linhas)
        edited_df = st.data_editor(
            actions_df,
            hide_index=False,
            column_config={
                "Selecionar": st.column_config.CheckboxColumn("Selecionar", default=False)
            },
            disabled=[col for col in actions_df.columns if col != "Selecionar"],
            use_container_width=True
        )

        # Botão para excluir linhas selecionadas
        if st.button("🗑️ Excluir Linhas Selecionadas", key="excluir"):
            # Filtrar linhas NÃO selecionadas
            selected_rows = edited_df[edited_df["Selecionar"]]
            if not selected_rows.empty:
                indices_to_remove = selected_rows.index.tolist()
                # Atualizar log
                st.session_state.actions_log = [
                    action for i, action in enumerate(st.session_state.actions_log)
                    if i not in indices_to_remove
                ]
                st.success(f"{len(indices_to_remove)} ações removidas do histórico.")
                st.rerun()
            else:
                st.warning("⚠️ Nenhuma linha selecionada para excluir.")

    else:
        st.info("Nenhuma ação registrada ainda.")        
# Expander 6: Exportar Dados Limpos
with st.expander("💾 Exportar Dados Limpos", expanded=True):
    st.markdown("### Exportar os dados tratados como CSV")
    st.markdown("Clique no botão abaixo para baixar o dataset atualizado:")

    # Preparar CSV
    csv = st.session_state.dados.to_csv(index=False).encode('utf-8')

    # Botão de download
    st.download_button(
        label="📥 Baixar Dados Limpos (CSV)",
        data=csv,
        file_name='dados_limpos.csv',
        mime='text/csv',
    )

    st.info("✔️ Este arquivo contém os dados após todas as correções realizadas até agora.\n⚙️ IMPORTANTE! Com isso, CONFIRA SE ESTA ATIVIDADE ESTÁ COERENTE COM A LIMPEZA MANUAL QUE SUA EQUIPE FEZ ANTERIORMENTE.")

        
# Botão para ir para a próxima página
if st.button("Ir para Análise Univariada 📊"):
    st.switch_page("pages/4_📊_Analise_Univariada.py")
