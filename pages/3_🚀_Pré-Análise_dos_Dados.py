import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Título da página
st.set_page_config(page_title="Tratamento de Dados", layout="wide")
st.title("🔍 Pré-Análise de Dados Interativa")

# Inicialização do session_state
if 'dados' not in st.session_state or st.session_state.dados.empty:
    st.warning("⚠️ Nenhum dado carregado. Volte para a página de coleta.")
    st.stop()

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

# Expander 4: Resumo das Ações
with st.expander("💾 Resumo das Ações Realizadas", expanded=False):
    if st.session_state.actions_log:
        st.subheader("Histórico de Modificações")

        # Cria o DataFrame e renomeia as colunas
        actions_df = pd.DataFrame(st.session_state.actions_log)
        actions_df.rename(columns={
            'timestamp': 'Quando',
            'type': 'Ação',
            'action': 'Detalhes',
            'removed': 'Quantidade'
        }, inplace=True)

        # Exibe apenas as colunas renomeadas
        st.dataframe(actions_df[['Quando', 'Ação', 'Detalhes', 'Quantidade']], use_container_width=True)
    else:
        st.info("Nenhuma ação registrada ainda.")

# Expander 5: Exportar Dados Limpos
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

    st.info("✔️ Este arquivo contém os dados após todas as correções realizadas até agora.")

        
# Botão para ir para a próxima página
if st.button("Ir para Análise Univariada"):
    st.switch_page("pages/4_📊_Analise_Univariada.py")
