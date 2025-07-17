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

# Expander 1: Dados Faltantes
with st.expander("🔎 Identificar Dados Faltantes", expanded=False):
    coluna_faltantes = st.selectbox(
        "Selecione a coluna para verificar dados faltantes:",
        df.columns,
        key="faltantes_coluna"
    )
    if coluna_faltantes:
        mask = df[coluna_faltantes].isnull()
        linhas_com_faltantes = df[mask]

        if not linhas_com_faltantes.empty:
            st.warning("⚠️ Linhas com dados faltantes encontradas:")
            st.dataframe(linhas_com_faltantes[[coluna_faltantes]], use_container_width=True)

            indices = linhas_com_faltantes.index.tolist()
            st.session_state.faltantes_indices = st.multiselect(
                "Selecione os índices para excluir:",
                options=indices,
                default=indices,
                key="faltantes_multiselect"
            )

            if st.button("Excluir Linhas Selecionadas", key="excluir_faltantes"):
                df = df.drop(index=st.session_state.faltantes_indices)
                st.session_state.dados = df.reset_index(drop=True)
                removed_count = len(st.session_state.faltantes_indices)
                registrar_acao(
                    "Remoção",
                    f"Excluídas {removed_count} linhas com dados faltantes na coluna '{coluna_faltantes}'",
                    removed_count
                )
                st.session_state.faltantes_indices = []
                st.success(f"{removed_count} linhas removidas com sucesso!")
                st.rerun()
        else:
            st.success("✅ Nenhum dado faltante encontrado nessa coluna.")

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

# Visualização atual dos dados
st.subheader("📊 Dados Atuais")
st.dataframe(df.head(10), use_container_width=True)

# Expander 4: Resumo das Ações
with st.expander("💾 Resumo das Ações Realizadas", expanded=False):
    if st.session_state.actions_log:
        actions_df = pd.DataFrame(st.session_state.actions_log)
        st.dataframe(actions_df[['timestamp', 'type', 'action', 'removed']], use_container_width=True)
    else:
        st.info("Nenhuma ação registrada ainda.")
        
# Botão para ir para a próxima página
if st.button("Ir para Análise Univariada"):
    st.switch_page("pages/4_📊_Analise_Univariada.py")
