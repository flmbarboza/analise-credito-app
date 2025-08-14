import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

def calcular_iv(dados, coluna, target):
    """Calcula o Information Value (IV) de uma variável categórica ou numérica binned."""
    df = dados[[coluna, target]].dropna()
    df['bin'] = pd.cut(df[coluna], bins=10) if df[coluna].dtype != 'object' else df[coluna]
    tmp = pd.crosstab(df['bin'], df[target], margins=True)
    tmp.columns = ['não_default', 'default', 'total']
    tmp['%_default'] = tmp['default'] / tmp['default'].sum()
    tmp['%_não_default'] = tmp['não_default'] / tmp['não_default'].sum()
    tmp['woe'] = np.log(tmp['%_não_default'] / tmp['%_default'])
    tmp['iv'] = (tmp['%_não_default'] - tmp['%_default']) * tmp['woe']
    return tmp['iv'].sum()

def calcular_ks(dados, coluna, target):
    """Calcula a estatística KS entre bons e maus."""
    bons = dados[dados[target] == 0][coluna].dropna()
    maus = dados[dados[target] == 1][coluna].dropna()
    ks_stat, _ = ks_2samp(bons, maus)
    return ks_stat

def main():
    st.title("📈 Análise Bivariada e Pré-Seleção de Variáveis")
    st.markdown("""
    Relacione variáveis e selecione as mais relevantes para o modelo de *credit scoring*.  
    Defina a variável-alvo e explore correlações, IV, WOE e KS.
    """)

    if 'dados' not in st.session_state:
        st.warning("Dados não carregados! Acesse a página de Coleta primeiro.")
        st.page_link("pages/2_📊_Coleta_de_Dados.py", label="→ Ir para Coleta")
        return

    dados = st.session_state.dados.copy()

    # --- 1. SELEÇÃO DA VARIÁVEL-ALVO ---
    st.markdown("### 🔍 Defina a Variável-Alvo")
    variaveis_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
    variaveis_binarias = [col for col in variaveis_numericas if set(dados[col].dropna().unique()) <= {0, 1}]

    target_sugeridas = variaveis_binarias if variaveis_binarias else variaveis_numericas[:3]

    target = st.selectbox(
        "Selecione a variável que indica **Default (1 = inadimplente)**:",
        options=dados.columns,
        index=dados.columns.get_loc(target_sugeridas[0]) if target_sugeridas else 0
    )

    if target not in dados.columns:
        st.error("Variável-alvo inválida.")
        return

    if dados[target].dtype not in ['int64', 'float64'] or not set(dados[target].dropna().unique()) <= {0, 1}:
        st.warning(f"A variável `{target}` não parece binária (0/1). Deseja continuar assim mesmo?")
        if not st.checkbox("Confirmar uso como variável-alvo"):
            return

    # Armazenar target no session_state
    st.session_state.target = target

    # --- 2. ANÁLISE BIVARIADA ---
    st.markdown("### 📊 Análise Bivariada")
    col1, col2 = st.columns(2)

    with col1:
        var_x = st.selectbox("Variável X:", dados.columns, key="x")
    with col2:
        var_y = st.selectbox("Variável Y:", dados.columns, key="y")

    tipo_grafico = st.radio("Tipo de gráfico:",
                           ["Dispersão", "Boxplot", "Barras"], horizontal=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if tipo_grafico == "Dispersão":
            sns.scatterplot(data=dados, x=var_x, y=var_y, hue=target, palette="Set1", ax=ax)
            ax.set_title(f"Dispersão: {var_x} vs {var_y} (cor por {target})")
        elif tipo_grafico == "Boxplot":
            sns.boxplot(data=dados, x=var_x, y=var_y, ax=ax)
            ax.set_title(f"Boxplot: {var_x} vs {var_y}")
        else:
            # Para barras, usar média de Y por categoria de X
            agg = dados.groupby(var_x)[var_y].mean().reset_index()
            sns.barplot(data=agg, x=var_x, y=var_y, ax=ax)
            ax.set_title(f"Média de {var_y} por {var_x}")
    except:
        st.error("Não foi possível gerar o gráfico com essas variáveis.")
        return

    st.pyplot(fig)

    # Correlação (se ambas forem numéricas)
    if dados[var_x].dtype in ['int64', 'float64'] and dados[var_y].dtype in ['int64', 'float64']:
        correlacao = dados[[var_x, var_y]].corr().iloc[0, 1]
        st.metric("Coeficiente de Correlação", f"{correlacao:.3f}")

    # --- 3. PRÉ-SELEÇÃO DE VARIÁVEIS (com expander) ---
    st.markdown("---")
    with st.expander("🔧 Pré-seleção de Variáveis", expanded=True):
        st.markdown("### Etapas para seleção de variáveis preditivas")

        # Armazenar variáveis ativas
        if 'variaveis_ativas' not in st.session_state:
            st.session_state.variaveis_ativas = dados.columns.tolist()

        variaveis_ativas = st.session_state.variaveis_ativas
        numericas = dados[variaveis_ativas].select_dtypes(include=[np.number]).columns.tolist()
        categoricas = dados[variaveis_ativas].select_dtypes(include=['object']).columns.tolist()

        # --- 2.1 Correlação ---
        st.markdown("#### 🧩 1. Análise de Correlação (evitar multicolinearidade)")
        st.info("A correlação identifica variáveis redundantes. Alta correlação (>0.7) pode indicar multicolinearidade, prejudicando o modelo.")

        corr_threshold = st.slider("Defina o limite de correlação para alerta:", 0.1, 0.95, 0.7, 0.05)
        
        if len(numericas) > 1:
            corr_matrix = dados[numericas].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = [(i, j) for i in upper.columns for j in upper.columns if upper.loc[i, j] > corr_threshold]

            if high_corr:
                st.warning(f"⚠️ {len(high_corr)} pares com correlação > {corr_threshold}")
                for i, j in high_corr[:10]:
                    st.caption(f"`{i}` vs `{j}`: {upper.loc[i, j]:.2f}")

                fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax_corr)
                ax_corr.set_title("Mapa de Calor de Correlação")
                st.pyplot(fig_corr)

                remover_corr = st.multiselect(
                    "Aponte as variáveis que deseja remover por alta correlação:",
                    options=[f"{i} vs {j}" for i, j in high_corr],
                    key="remove_corr"
                )
                if st.button("✅ Remover selecionadas"):
                    vars_para_remover = set()
                    for par in remover_corr:
                        i, j = par.split(" vs ")
                        vars_para_remover.add(i.strip())
                    st.session_state.variaveis_ativas = [v for v in st.session_state.variaveis_ativas if v not in vars_para_remover]
                    st.success(f"Variáveis removidas: {list(vars_para_remover)}")
                    st.rerun()
            else:
                st.success("✅ Nenhuma correlação alta encontrada.")
        else:
            st.info("Nenhuma variável numérica suficiente para análise de correlação.")

        # --- 2.2 IV ---
        st.markdown("#### 📈 2. Information Value (IV)")
        st.info("IV mede o poder preditivo. Regra: IV > 0.1 é útil; > 0.3 é forte. IV > 0.5 pode indicar vazamento.")

        iv_data = []
        for col in numericas + categoricas:
            if col == target:
                continue
            try:
                iv = calcular_iv(dados, col, target)
                iv_data.append({'Variável': col, 'IV': iv})
            except:
                continue

        iv_df = pd.DataFrame(iv_data).sort_values("IV", ascending=False)
        st.dataframe(iv_df.style.format({"IV": "{:.3f}"}).background_gradient(cmap="RdYlGn_r", subset=["IV"]))

        if not iv_df.empty:
            fig_iv, ax_iv = plt.subplots(figsize=(6, 4))
            top_iv = iv_df.head(10)
            sns.barplot(data=top_iv, x="IV", y="Variável", ax=ax_iv, palette="viridis")
            ax_iv.set_title("Top 10 Variáveis por IV")
            st.pyplot(fig_iv)

        # --- 2.3 WOE ---
        st.markdown("#### 🔎 3. Weight of Evidence (WOE)")
        st.info("WOE transforma variáveis em escores de risco. Útil para modelos logísticos e scorecards.")

        var_para_woe = st.selectbox("Selecione uma variável para análise de WOE:", numericas + categoricas)
        if var_para_woe != target:
            try:
                df_temp = dados[[var_para_woe, target]].dropna()
                if df_temp[var_para_woe].dtype != 'object':
                    df_temp['bin'] = pd.cut(df_temp[var_para_woe], bins=10, duplicates='drop')
                else:
                    df_temp['bin'] = df_temp[var_para_woe]

                woe_table = pd.crosstab(df_temp['bin'], df_temp[target])
                woe_table.columns = ['não_default', 'default']
                woe_table['%_não_default'] = woe_table['não_default'] / woe_table['não_default'].sum()
                woe_table['%_default'] = woe_table['default'] / woe_table['default'].sum()
                woe_table['woe'] = np.log(woe_table['%_não_default'] / woe_table['%_default'])
                st.dataframe(woe_table.style.format({"woe": "{:.3f}"}))
            except Exception as e:
                st.error(f"Erro ao calcular WOE: {e}")

        # --- 2.4 KS ---
        st.markdown("#### 📊 4. Teste de Kolmogorov-Smirnov (KS)")
        st.info("KS mede a separação entre bons e maus. KS > 0.3 é bom; > 0.4 é excelente.")

        ks_data = []
        for col in numericas:
            if col == target:
                continue
            try:
                ks = calcular_ks(dados, col, target)
                ks_data.append({'Variável': col, 'KS': ks})
            except:
                continue

        ks_df = pd.DataFrame(ks_data).sort_values("KS", ascending=False)
        st.dataframe(ks_df.style.format({"KS": "{:.3f}"}).background_gradient(cmap="Blues", subset=["KS"]))

    # --- 4. RELATÓRIO ---
    st.markdown("---")
    with st.expander("📋 Relatório de Análise"):
        st.markdown("### Resumo das Etapas Realizadas")
        st.write(f"- **Variável-alvo:** `{target}`")
        st.write(f"- **Total de variáveis iniciais:** {len(dados.columns)}")
        st.write(f"- **Variáveis ativas após pré-seleção:** {len(st.session_state.variaveis_ativas)}")
        if 'remove_corr' in st.session_state and st.session_state.get('remove_corr'):
            st.write(f"- **Variáveis removidas por correlação:** {', '.join([p.split(' vs ')[0] for p in st.session_state.remove_corr])}")
        st.write(f"- **Top 3 por IV:** {', '.join(iv_df.head(3)['Variável'].tolist()) if not iv_df.empty else 'N/A'}")
        st.write(f"- **Top 3 por KS:** {', '.join(ks_df.head(3)['Variável'].tolist()) if not ks_df.empty else 'N/A'}")

        st.download_button(
            "⬇️ Baixar relatório (txt)",
            data=f"Relatório de Pré-seleção\nVariável-alvo: {target}\nVariáveis ativas: {st.session_state.variaveis_ativas}\nTop IV: {iv_df.head(3)['Variável'].tolist()}\nTop KS: {ks_df.head(3)['Variável'].tolist()}",
            file_name="relatorio_pre_selecao.txt",
            mime="text/plain"
        )

    # --- 5. EXPORTAÇÃO ---
    st.markdown("---")
    with st.expander("💾 Exportar Outputs"):
        st.markdown("Salve dados, gráficos e resultados para uso futuro.")

        if st.button("💾 Salvar dados ativos e resultados"):
            # Dados limpos e ativos
            dados_salvos = dados[st.session_state.variaveis_ativas + [target]]
            st.session_state.dados_modelagem = dados_salvos
            st.session_state.iv_df = iv_df
            st.session_state.ks_df = ks_df

            st.success("✅ Dados e resultados salvos para modelagem!")

        st.download_button("📥 Exportar dados ativos (CSV)", 
                           data=dados[st.session_state.variaveis_ativas + [target]].to_csv(index=False),
                           file_name="dados_preparados.csv",
                           mime="text/csv")

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/6_🤖_Modelagem.py", label="➡️ Ir para Modelagem", icon="🤖")

if __name__ == "__main__":
    main()
