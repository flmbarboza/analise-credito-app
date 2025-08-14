import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

def calcular_iv(dados, coluna, target):
    df = dados[[coluna, target]].dropna()
    if df[coluna].dtype != 'object':
        df['bin'] = pd.cut(df[coluna], bins=10, duplicates='drop')
    else:
        df['bin'] = df[coluna]
    tmp = pd.crosstab(df['bin'], df[target], margins=True)
    if tmp.shape[0] < 2 or 0 not in tmp.columns or 1 not in tmp.columns:
        return np.nan
    tmp.columns = ['não_default', 'default', 'total']
    tmp = tmp[tmp['total'] > 0]
    tmp['%_default'] = np.where(tmp['default'] > 0, tmp['default'] / tmp['default'].sum(), 0.001)
    tmp['%_não_default'] = np.where(tmp['não_default'] > 0, tmp['não_default'] / tmp['não_default'].sum(), 0.001)
    tmp['woe'] = np.log(tmp['%_não_default'] / tmp['%_default'])
    tmp['iv'] = (tmp['%_não_default'] - tmp['%_default']) * tmp['woe']
    return tmp['iv'].sum()

def calcular_ks(dados, coluna, target):
    bons = dados[dados[target] == 0][coluna].dropna()
    maus = dados[dados[target] == 1][coluna].dropna()
    if len(bons) == 0 or len(maus) == 0:
        return np.nan
    ks_stat, _ = ks_2samp(bons, maus)
    return ks_stat

def main():
    st.title("📈 Análise Bivariada e Pré-Seleção de Variáveis")
    st.markdown("""
    Defina a variável-alvo, corrija seu formato, e realize análises preditivas:  
    **IV, WOE, KS** – tudo em um só lugar.
    """)

    if 'dados' not in st.session_state:
        st.warning("Dados não carregados! Acesse a página de Coleta primeiro.")
        st.page_link("pages/2_📊_Coleta_de_Dados.py", label="→ Ir para Coleta")
        return

    dados = st.session_state.dados.copy()

    # --- 1. SELEÇÃO E VALIDAÇÃO DA VARIÁVEL-ALVO (Y) ---
    st.markdown("### 🔍 Defina a Variável-Alvo (Default)")
    target = st.selectbox(
        "Selecione a coluna que indica **inadimplência**:",
        options=dados.columns,
        index=None,
        placeholder="Escolha a variável de default"
    )

    if target not in dados.columns:
        st.error("ALERTA: variável-alvo inválida ou indefinida.")
        return

    y_data = dados[target].dropna()
    if len(y_data) == 0:
        st.error(f"A coluna `{target}` está vazia.")
        return

    valores_unicos = pd.Series(y_data.unique()).dropna().tolist()
    try:
        valores_unicos = sorted([x for x in valores_unicos if isinstance(x, (int, float))])
    except:
        pass

    # Verificar se é binária (0/1)
    if set(valores_unicos) != {0, 1}:
        st.warning(f"""
        ⚠️ A variável `{target}` não está no formato 0/1.  
        Valores encontrados: {valores_unicos}
        """)
        st.markdown("#### 🔧 Mapeie os valores para 0 (adimplente) e 1 (inadimplente)")
        col1, col2 = st.columns(2)
        with col1:
            valor_bom = st.selectbox("Valor que representa **adimplente (0)**", options=valores_unicos, key="bom")
        with col2:
            valor_mau = st.selectbox("Valor que representa **inadimplente (1)**", options=[v for v in valores_unicos if v != valor_bom], key="mau")
        if st.button("✅ Aplicar Mapeamento"):
            try:
                y_mapped = dados[target].map({valor_bom: 0, valor_mau: 1})
                dados[target] = y_mapped
                st.success(f"✅ `{target}` convertida para 0 (adimplente) e 1 (inadimplente).")
                st.session_state.dados = dados
            except Exception as e:
                st.error(f"Erro ao mapear: {e}")
    else:
        st.success(f"✅ `{target}` já está no formato 0/1.")

    st.session_state.target = target

    # --- DEFINIÇÃO INICIAL DE VARIÁVEIS ATIVAS ---
    if 'variaveis_ativas' not in st.session_state:
        st.session_state.variaveis_ativas = [col for col in dados.columns if col != target]

    variaveis_ativas = st.session_state.variaveis_ativas
    numericas = dados[variaveis_ativas].select_dtypes(include=[np.number]).columns.tolist()
    categoricas = dados[variaveis_ativas].select_dtypes(include='object').columns.tolist()
    features = [c for c in (numericas + categoricas) if c != target]

    # --- 2. ANÁLISE BIVARIADA ---
    st.markdown("### 📊 Análise Bivariada")
    col1, col2 = st.columns(2)
    with col1:
        var_x = st.selectbox("Variável X:", features, key="x_biv")
    with col2:
        var_y = st.selectbox("Variável Y:", features, key="y_biv")

    tipo_grafico = st.radio("Tipo de gráfico:", ["Dispersão", "Boxplot", "Barras"], horizontal=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        if tipo_grafico == "Dispersão":
            sns.scatterplot(data=dados, x=var_x, y=var_y, hue=target, palette="Set1", ax=ax)
            ax.set_title(f"{var_x} vs {var_y} por {target}")
        elif tipo_grafico == "Boxplot":
            sns.boxplot(data=dados, x=var_x, y=var_y, ax=ax)
            ax.set_title(f"Distribuição de {var_y} por {var_x}")
        else:
            agg = dados.groupby(var_x)[var_y].mean().reset_index()
            sns.barplot(data=agg, x=var_x, y=var_y, ax=ax)
            ax.set_title(f"Média de {var_y} por {var_x}")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except:
        st.error("Não foi possível gerar o gráfico com essas variáveis.")

    if pd.api.types.is_numeric_dtype(dados[var_x]) and pd.api.types.is_numeric_dtype(dados[var_y]):
        corr = dados[[var_x, var_y]].corr().iloc[0, 1]
        st.metric("Correlação", f"{corr:.3f}")

    # --- CORRELAÇÃO: REMOÇÃO ANTES DAS OUTRAS ANÁLISES ---
    with st.expander("🧩 Análise de Correlação e Remoção", expanded=False):
        st.markdown("#### Evite multicolinearidade")
        st.info("Alta correlação entre variáveis pode prejudicar o modelo. Defina um limite e remova variáveis redundantes.")

        corr_threshold = st.slider("Limite de correlação para alerta:", 0.1, 0.95, 0.7, 0.05)

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
                    "Aponte os pares que deseja remover (a primeira variável do par será removida):",
                    options=[f"{i} vs {j}" for i, j in high_corr],
                    key="remove_corr"
                )
                if st.button("✅ Remover selecionadas"):
                    vars_para_remover = set()
                    for par in remover_corr:
                        i, j = par.split(" vs ")
                        vars_para_remover.add(i.strip())
                    # Atualiza variáveis ativas
                    st.session_state.variaveis_ativas = [v for v in st.session_state.variaveis_ativas if v not in vars_para_remover]
                    st.success(f"Variáveis removidas: {list(vars_para_remover)}")
                    st.rerun()
            else:
                st.success("✅ Nenhuma correlação alta encontrada.")
        else:
            st.info("Nenhuma variável numérica suficiente para análise.")

    # --- ATUALIZAR LISTAS APÓS REMOÇÃO ---
    variaveis_ativas = st.session_state.variaveis_ativas
    numericas = dados[variaveis_ativas].select_dtypes(include=[np.number]).columns.tolist()
    categoricas = dados[variaveis_ativas].select_dtypes(include='object').columns.tolist()
    features = [c for c in (numericas + categoricas) if c != target]

    # --- PRÉ-SELEÇÃO DE VARIÁVEIS (com IV, WOE, KS usando apenas variáveis ativas) ---
    with st.expander("🔧 Pré-seleção de Variáveis", expanded=True):
        st.markdown("### Etapas com base nas variáveis **ativas** (após remoção por correlação)")

        # --- IV ---
        st.markdown("#### 📈 Information Value (IV)")
        st.info("IV > 0.1: útil | > 0.3: forte | > 0.5: suspeito (vazamento).")

        iv_data = []
        for col in features:
            try:
                iv = calcular_iv(dados, col, target)
                iv_data.append({'Variável': col, 'IV': iv})
            except:
                iv_data.append({'Variável': col, 'IV': np.nan})

        iv_df = pd.DataFrame(iv_data).dropna().sort_values("IV", ascending=True)
        st.session_state.iv_df = iv_df

        if not iv_df.empty:
            fig_iv, ax_iv = plt.subplots(figsize=(6, 0.35 * len(iv_df)))
            bars = ax_iv.barh(iv_df['Variável'], iv_df['IV'], color='skyblue', edgecolor='darkblue', height=0.7)
            ax_iv.set_title("Information Value (IV) por Variável")
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax_iv.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=9)
            st.pyplot(fig_iv)
        else:
            st.warning("Não foi possível calcular IV para nenhuma variável.")

        # --- WOE ---
        st.markdown("#### 🔎 Weight of Evidence (WOE)")
        st.info("WOE transforma variáveis numéricas em escores de risco. Ajuste o número de faixas.")

        n_bins = st.slider("Número de faixas (bins) para WOE:", min_value=2, max_value=20, value=10, key="n_bins_woe")

        woe_tables = {}
        for col in numericas:
            if col == target:
                continue
            try:
                df_temp = dados[[col, target]].dropna()
                df_temp['bin'] = pd.cut(df_temp[col], bins=n_bins, duplicates='drop')
                tmp = pd.crosstab(df_temp['bin'], df_temp[target])
                tmp.columns = ['não_default', 'default']
                total_bons = tmp['não_default'].sum()
                total_maus = tmp['default'].sum()
                tmp['%_não_default'] = tmp['não_default'] / total_bons
                tmp['%_default'] = tmp['default'] / total_maus
                tmp['%_default'] = tmp['%_default'].replace(0, 1e-6)
                tmp['%_não_default'] = tmp['%_não_default'].replace(0, 1e-6)
                tmp['woe'] = np.log(tmp['%_não_default'] / tmp['%_default'])
                tmp['iv'] = (tmp['%_não_default'] - tmp['%_default']) * tmp['woe']
                woe_tables[col] = tmp
            except Exception as e:
                woe_tables[col] = pd.DataFrame({'erro': [str(e)]})

        for var, table in woe_tables.items():
            with st.expander(f"WOE – {var}", expanded=False):
                if 'erro' in table.columns:
                    st.error(table['erro'].iloc[0])
                else:
                    st.dataframe(table.style.format({
                        'não_default': '{:.0f}',
                        'default': '{:.0f}',
                        '%_não_default': '{:.4f}',
                        '%_default': '{:.4f}',
                        'woe': '{:.3f}',
                        'iv': '{:.3f}'
                    }))
                    fig, ax = plt.subplots(figsize=(5, 2))
                    table['woe'].plot(kind='barh', ax=ax, color='teal')
                    ax.set_title(f"WOE por faixa – {var}")
                    st.pyplot(fig)

        # --- KS ---
        st.markdown("#### 📊 Kolmogorov-Smirnov (KS)")
        st.info("KS > 0.3: bom | > 0.4: excelente. Mede a separação entre bons e maus.")

        ks_data = []
        for col in numericas:
            if col == target:
                continue
            try:
                ks = calcular_ks(dados, col, target)
                ks_data.append({'Variável': col, 'KS': ks})
            except:
                ks_data.append({'Variável': col, 'KS': np.nan})

        ks_df = pd.DataFrame(ks_data).dropna().sort_values("KS", ascending=True)
        st.session_state.ks_df = ks_df

        if not ks_df.empty:
            fig_ks, ax_ks = plt.subplots(figsize=(6, 0.35 * len(ks_df)))
            bars = ax_ks.barh(ks_df['Variável'], ks_df['KS'], color='lightcoral', edgecolor='darkred', height=0.7)
            ax_ks.set_title("KS por Variável")
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax_ks.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=9)
            st.pyplot(fig_ks)
        else:
            st.warning("Não foi possível calcular KS.")

    # --- RELATÓRIO ---
    with st.expander("📋 Relatório de Análise"):
        st.markdown("### ✅ Variáveis Ativas Após Pré-Seleção")
        st.write(f"- **Variável-alvo:** `{target}`")
        st.write(f"- **Variáveis ativas:** {len(variaveis_ativas)}")
        st.write(f"- **Numéricas:** {len(numericas)} | **Categóricas:** {len(categoricas)}")
        if 'iv_df' in st.session_state and not st.session_state.iv_df.empty:
            top_iv = st.session_state.iv_df.sort_values("IV", ascending=False).head(3)['Variável'].tolist()
            st.write(f"- **Top 3 por IV:** {', '.join(top_iv)}")

    # --- EXPORTAÇÃO ---
    with st.expander("💾 Exportar Outputs"):
        st.markdown("Salve os dados e resultados para a modelagem.")
        if st.button("💾 Salvar dados ativos"):
            dados_modelagem = dados[variaveis_ativas + [target]]
            st.session_state.dados_modelagem = dados_modelagem
            st.session_state.woe_tables = woe_tables
            st.success("✅ Dados salvos para modelagem!")

        st.download_button(
            "📥 Exportar dados ativos (CSV)",
            data=dados[variaveis_ativas + [target]].to_csv(index=False),
            file_name="dados_ativos.csv",
            mime="text/csv"
        )

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/6_🤖_Modelagem.py", label="➡️ Ir para Modelagem", icon="🤖")

if __name__ == "__main__":
    main()
