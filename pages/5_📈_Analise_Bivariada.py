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
        st.info("Alta correlação entre variáveis pode prejudicar o modelo. Escolha como deseja remover variáveis redundantes.")
    
        corr_threshold = st.slider(
            "Limite de correlação para detecção:",
            0.1, 0.95, 0.7, 0.05,
            key="corr_slider_bivariada"
        )
    
        if len(numericas) > 1:
            corr_matrix = dados[numericas].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = [(i, j) for i in upper.columns for j in upper.columns if upper.loc[i, j] > corr_threshold]
    
            if not high_corr_pairs:
                st.success("✅ Nenhuma correlação acima do limite encontrada.")
            else:
                st.warning(f"⚠️ {len(high_corr_pairs)} pares com correlação > {corr_threshold}")
    
                # Exibir pares
                corr_list = "\n".join([f"- `{i}` vs `{j}`: {upper.loc[i, j]:.2f}" for i, j in high_corr_pairs[:10]])
                st.markdown(f"**Pares com alta correlação:**\n{corr_list}")
    
                # Extrair todas as variáveis envolvidas em pares de alta correlação
                vars_envolvidas = list(set([i for i, j in high_corr_pairs] + [j for i, j in high_corr_pairs]))
                vars_envolvidas = [v for v in vars_envolvidas if v in st.session_state.variaveis_ativas]
    
                if not vars_envolvidas:
                    st.info("Nenhuma variável disponível para remoção.")
                else:
                    st.markdown("##### 🧾 Selecione quais variáveis deseja remover (pode escolher qualquer uma dos pares acima):")
                    vars_para_remover = st.multiselect(
                        "Variáveis a remover",
                        options=sorted(vars_envolvidas),
                        default=[],
                        key="multiselect_vars_correlacao"
                    )
    
                    if st.button("✅ Aplicar Remoção"):
                        if vars_para_remover:
                            # Remove as selecionadas da lista ativa
                            st.session_state.variaveis_ativas = [
                                v for v in st.session_state.variaveis_ativas if v not in vars_para_remover
                            ]
                            st.success(f"✅ Variáveis removidas: `{vars_para_remover}`")
                            st.rerun()
                        else:
                            st.info("Nenhuma variável selecionada para remoção.")
    
    # --- ATUALIZAR LISTAS APÓS REMOÇÃO ---
    # Isso é essencial: recarregar as listas com base na versão atualizada de variaveis_ativas
    variaveis_ativas = st.session_state.variaveis_ativas
    numericas = dados[variaveis_ativas].select_dtypes(include=[np.number]).columns.tolist()
    categoricas = dados[variaveis_ativas].select_dtypes(include='object').columns.tolist()
    features = [c for c in (numericas + categoricas) if c != target]


    # --- PRÉ-SELEÇÃO DE VARIÁVEIS (com IV, WOE, KS usando apenas variáveis ativas) ---
    with st.expander("🔧 Pré-seleção de Variáveis", expanded=False):
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

        # --- WOE
        st.markdown("#### 🔎 Weight of Evidence (WOE)")
        st.info("WOE transforma variáveis numéricas em escores de risco. Ajuste o número de faixas.")

        st.warning("""
        Edite os valores de **mínimo e máximo** ou os **limites das faixas** para ajustar o binning.  
        A tabela e o gráfico são atualizados automaticamente.
        """)
        
        # Inicializa o estado para configurações de WOE
        if 'woe_config' not in st.session_state:
            st.session_state.woe_config = {}
        
        # Variáveis numéricas ativas
        numericas_ativas = [col for col in numericas if col != target]
        if not numericas_ativas:
            st.warning("Nenhuma variável numérica disponível para WOE.")
        else:
            var_selecionada = st.selectbox(
                "Selecione a variável para análise de WOE:",
                options=numericas_ativas,
                key="woe_var_select"
            )
        
            if var_selecionada:
                dados_var = dados[var_selecionada].dropna()
                dados_clean = dados_var[(dados_var.notna()) & (np.isfinite(dados_var))]
        
                min_val_orig = float(dados_clean.min())
                max_val_orig = float(dados_clean.max())
        
                # Inicializa configuração da variável
                if var_selecionada not in st.session_state.woe_config:
                    st.session_state.woe_config[var_selecionada] = {
                        'min_val': min_val_orig,
                        'max_val': max_val_orig,
                        'n_bins': 10
                    }
        
                config = st.session_state.woe_config[var_selecionada]
        
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_min = st.number_input(
                        "Limite inferior",
                        min_value=-1e10,
                        max_value=max_val_orig,
                        value=config['min_val'],
                        step=0.01,
                        key=f"min_{var_selecionada}"
                    )
                with col2:
                    new_max = st.number_input(
                        "Limite superior",
                        min_value=min_val_orig,
                        max_value=1e10,
                        value=config['max_val'],
                        step=0.01,
                        key=f"max_{var_selecionada}"
                    )
                with col3:
                    n_bins = st.number_input(
                        "Número de faixas (bins)",
                        min_value=2,
                        max_value=20,
                        value=config['n_bins'],
                        step=1,
                        key=f"bins_{var_selecionada}"
                    )
        
                # Atualiza configuração se houver mudança
                if (new_min != config['min_val'] or 
                    new_max != config['max_val'] or 
                    n_bins != config['n_bins']):
                    st.session_state.woe_config[var_selecionada] = {
                        'min_val': new_min,
                        'max_val': new_max,
                        'n_bins': n_bins
                    }
                    st.rerun()  # Atualiza para refletir mudanças
        
                # Aplica os limites
                config = st.session_state.woe_config[var_selecionada]
                min_val = config['min_val']
                max_val = config['max_val']
                n_bins = config['n_bins']
        
                # Cria bins com os limites ajustados
                bins = np.linspace(min_val, max_val, n_bins + 1)
        
                # Exibe os limites atuais
                st.markdown(f"**Faixas atuais:**")
                faixas = []
                for i in range(n_bins):
                    faixas.append(f"`[{bins[i]:.2f}, {bins[i+1]:.2f})`")
                st.write(" → ".join(faixas))
        
                try:
                    df_temp = dados[[var_selecionada, target]].dropna()
                    # Filtra apenas dentro dos limites definidos
                    mask = (df_temp[var_selecionada] >= min_val) & (df_temp[var_selecionada] <= max_val)
                    df_temp = df_temp[mask]
        
                    df_temp['bin'] = pd.cut(df_temp[var_selecionada], bins=bins, include_lowest=True, right=False, duplicates='drop')
        
                    tmp = pd.crosstab(df_temp['bin'], df_temp[target])
                    tmp.columns = ['não_default', 'default']
        
                    total_bons = tmp['não_default'].sum()
                    total_maus = tmp['default'].sum()
        
                    tmp['%_não_default'] = tmp['não_default'] / (total_bons or 1)
                    tmp['%_default'] = tmp['default'] / (total_maus or 1)
        
                    # Evitar divisão por zero
                    tmp['%_default'] = tmp['%_default'].replace(0, 1e-6)
                    tmp['%_não_default'] = tmp['%_não_default'].replace(0, 1e-6)
        
                    tmp['woe'] = np.log(tmp['%_não_default'] / tmp['%_default'])
                    #tmp['iv'] = (tmp['%_não_default'] - tmp['%_default']) * tmp['woe']
        
                    # Armazena tabela
                    woe_tables = {}
                    if 'woe_tables' not in st.session_state:
                        st.session_state.woe_tables = {}
                    st.session_state.woe_tables[var_selecionada] = tmp
        
                    # Exibe tabela
                    st.markdown("##### 📊 Tabela de WOE")
                    st.dataframe(
                        tmp.style.format({
                            'não_default': '{:,.0f}',
                            'default': '{:,.0f}',
                            '%_não_default': '{:.2f}',
                            '%_default': '{:.2f}',
                            'woe': '{:.3f}'
                        }).background_gradient(cmap='RdYlGn', subset=['woe'], low=1, high=1)
                    )
        
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    tmp['woe'].plot(kind='barh', ax=ax, color='teal', edgecolor='black')
                    ax.set_title(f"WOE por Faixa - {var_selecionada}", fontsize=10)
                    ax.set_xlabel("Weight of Evidence", fontsize=9)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
        
                except Exception as e:
                    st.error(f"Erro ao calcular WOE: {e}")
        
            # Armazenar para uso futuro
            st.session_state.woe_tables = woe_tables  
    
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
        else:
            st.write("- **Top 3 por IV:** N/A")
    
        if 'ks_df' in st.session_state and not st.session_state.ks_df.empty:
            top_ks = st.session_state.ks_df.sort_values("KS", ascending=False).head(3)['Variável'].tolist()
            st.write(f"- **Top 3 por KS:** {', '.join(top_ks)}")
        else:
            st.write("- **Top 3 por KS:** N/A")


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
