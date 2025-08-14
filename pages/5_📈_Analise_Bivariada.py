import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

def calcular_iv(dados, coluna, target):
    """Calcula o Information Value (IV) de uma variável."""
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
    """Calcula KS entre bons (0) e maus (1)."""
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
        options=dados.columns, index=None
    )

    if target not in dados.columns:
        st.error("Erro: variável-alvo inválida.")
        return

    y_data = dados[target].dropna()
    valores_unicos = sorted(y_data.unique())

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
                st.session_state.dados = dados  # Atualiza os dados
            except Exception as e:
                st.error(f"Erro ao mapear: {e}")
    else:
        st.success(f"✅ `{target}` já está no formato 0/1.")

    st.session_state.target = target

    # --- 2. ANÁLISE BIVARIADA ---
    st.markdown("### 📊 Análise Bivariada")
    col1, col2 = st.columns(2)
    with col1:
        var_x = st.selectbox("Variável X:", dados.columns, key="x_biv")
    with col2:
        var_y = st.selectbox("Variável Y:", dados.columns, key="y_biv")

    tipo_grafico = st.radio("Tipo de gráfico:",
                           ["Dispersão", "Boxplot", "Barras"], horizontal=True)

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

    # Correlação
    numericas = dados.select_dtypes(include=[np.number]).columns.tolist()

    if pd.api.types.is_numeric_dtype(dados[var_x]) and pd.api.types.is_numeric_dtype(dados[var_y]):
        corr = dados[[var_x, var_y]].corr().iloc[0, 1]
        st.metric("Correlação", f"{corr:.3f}")

    # --- Correlação ---
    with st.expander("Correlação de Variáveis", expanded=False):
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
            
    # --- 3. PRÉ-SELEÇÃO (com expander) ---
    st.markdown("---")
    with st.expander("🔧 Pré-seleção de Variáveis", expanded=False):
        st.session_state.variaveis_ativas = dados.columns.tolist()
        numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        categoricas = dados.select_dtypes(include='object').columns.tolist()
        features = [c for c in (numericas + categoricas) if c != target]

        # --- IV: Gráfico de barras (todas as variáveis) ---
        st.markdown("#### 📈 Information Value (IV) – Poder Preditivo")
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
            ax_iv.set_xlabel("IV")
            # Rótulos nas barras
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax_iv.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                          f"{width:.3f}", va='center', fontsize=9)
            st.pyplot(fig_iv)
        else:
            st.warning("Não foi possível calcular IV para nenhuma variável.")

        # --- WOE: Todas as numéricas com bining ajustável ---
        st.markdown("#### 🔎 Weight of Evidence (WOE) – Todas as Variáveis Numéricas")
        st.info("WOE transforma variáveis em escores de risco. Faixas uniformes. Ajuste o número de bins abaixo.")

        n_bins = st.slider("Número de faixas (bins) para WOE:", min_value=2, max_value=20, value=10)

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
                tmp['%_default'] = tmp['%_default'].replace(0, 1e-6)  # evitar divisão por zero
                tmp['%_não_default'] = tmp['%_não_default'].replace(0, 1e-6)
                tmp['woe'] = np.log(tmp['%_não_default'] / tmp['%_default'])
                tmp['iv'] = (tmp['%_não_default'] - tmp['%_default']) * tmp['woe']
                woe_tables[col] = tmp
            except Exception as e:
                woe_tables[col] = pd.DataFrame({'erro': [f"Erro: {str(e)}"]})

        # Exibir todas as tabelas WOE
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
                # Gráfico opcional
                if 'woe' in table.columns:
                    fig, ax = plt.subplots(figsize=(5, 2))
                    table['woe'].plot(kind='barh', ax=ax, color='teal', edgecolor='black')
                    ax.set_title(f"WOE por faixa – {var}")
                    st.pyplot(fig)

        # --- KS: Gráfico de barras ---
        st.markdown("#### 📊 Kolmogorov-Smirnov (KS) – Poder de Separação")
        st.info("KS > 0.3: bom | > 0.4: excelente. Mede a separação entre bons e maus pagadores.")

        ks_data = []
        for col in numericas:
            if col == target:
                continue
            try:
                ks = calcular_ks(dados, col, target)
                ks_data.append({'Variável': col, 'KS': ks})
            except Exception as e:
                ks_data.append({'Variável': col, 'KS': np.nan, 'erro': str(e)})

        ks_df = pd.DataFrame([d for d in ks_data if 'erro' not in d]).dropna()
        st.session_state.ks_df = ks_df

        if not ks_df.empty:
            ks_df = ks_df.sort_values("KS", ascending=True)
            fig_ks, ax_ks = plt.subplots(figsize=(6, 0.35 * len(ks_df)))
            bars = ax_ks.barh(ks_df['Variável'], ks_df['KS'], color='lightcoral', edgecolor='darkred', height=0.7)
            ax_ks.set_title("KS por Variável")
            ax_ks.set_xlabel("KS")
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax_ks.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                          f"{width:.3f}", va='center', fontsize=9)
            st.pyplot(fig_ks)
        else:
            st.warning("Não foi possível calcular KS para nenhuma variável. Verifique se há pelo menos um bom e um mau em cada coluna.")

    # --- 4. RELATÓRIO ---
    st.markdown("---")
    with st.expander("📋 Relatório de Análise"):
        st.markdown("### ✅ Etapas Concluídas")
        st.write(f"- **Variável-alvo:** `{target}` (formato 0/1)")
        st.write(f"- **Total de variáveis analisadas:** {len(features)}")
        st.write(f"- **Variáveis numéricas:** {len(numericas)}")
        st.write(f"- **Variáveis categóricas:** {len(categoricas)}")
        top_iv = iv_df.sort_values("IV", ascending=False).head(3)['Variável'].tolist() if 'iv_df' in st.session_state and not st.session_state.iv_df.empty else []
        st.write(f"- **Top 3 por IV:** {', '.join(top_iv) if top_iv else 'N/A'}")
        top_ks = ks_df.sort_values("KS", ascending=False).head(3)['Variável'].tolist() if 'ks_df' in st.session_state and not st.session_state.ks_df.empty else []
        st.write(f"- **Top 3 por KS:** {', '.join(top_ks) if top_ks else 'N/A'}")

        st.download_button(
            "⬇️ Baixar relatório (txt)",
            data=f"""
Relatório de Pré-Análise de Crédito
===================================
Variável-alvo: {target}
Formato: 0/1 (adimplente/inadimplente)
Total de variáveis: {len(features)}
Top IV: {top_iv}
Top KS: {top_ks}
Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
            """.strip(),
            file_name="relatorio_credit_scoring.txt",
            mime="text/plain"
        )

    # --- 5. EXPORTAÇÃO ---
    st.markdown("---")
    with st.expander("💾 Exportar Outputs"):
        st.markdown("Salve dados e resultados para a modelagem.")

        if st.button("💾 Salvar dados e resultados"):
            dados_modelagem = dados[features + [target]]
            st.session_state.dados_modelagem = dados_modelagem
            st.session_state.iv_df = iv_df
            st.session_state.ks_df = ks_df
            st.session_state.woe_tables = woe_tables
            st.success("✅ Dados e resultados salvos para a próxima etapa!")

        st.download_button(
            "📥 Exportar dados limpos (CSV)",
            data=dados[features + [target]].to_csv(index=False),
            file_name="dados_limpos_modelagem.csv",
            mime="text/csv"
        )

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/6_🤖_Modelagem.py", label="➡️ Ir para Modelagem", icon="🤖")

if __name__ == "__main__":
    main()
