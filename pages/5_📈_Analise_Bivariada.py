import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
import io, zipfile, base64

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

def criar_zip_exportacao(selecionados, dados, target, iv_df, ks_df, woe_tables, corr_matrix, st):
    """Cria um buffer ZIP com os itens selecionados pelo usuário."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # 1. Mapa de calor de correlação
        if "Mapa de Correlação" in selecionados:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Mapa de Calor de Correlação")
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            zip_file.writestr("mapa_correlacao.png", img_data.getvalue())

        # 2. Gráfico de IV
        if "Gráfico de IV" in selecionados and not iv_df.empty:
            fig, ax = plt.subplots(figsize=(6, 0.35 * len(iv_df)))
            bars = ax.barh(iv_df['Variável'], iv_df['IV'], color='skyblue', edgecolor='darkblue', height=0.7)
            ax.set_title("Information Value (IV)")
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=9)
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            zip_file.writestr("iv.png", img_data.getvalue())

        # 3. Gráfico de KS
        if "Gráfico de KS" in selecionados and not ks_df.empty:
            fig, ax = plt.subplots(figsize=(6, 0.35 * len(ks_df)))
            bars = ax.barh(ks_df['Variável'], ks_df['KS'], color='lightcoral', edgecolor='darkred', height=0.7)
            ax.set_title("Kolmogorov-Smirnov (KS)")
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=9)
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            zip_file.writestr("ks.png", img_data.getvalue())

        # 4. Gráficos de WOE
        if "Gráficos de WOE" in selecionados and woe_tables:
            for var, table in woe_tables.items():
                if 'woe' in table.columns:
                    fig, ax = plt.subplots(figsize=(5, 2))
                    table['woe'].plot(kind='barh', ax=ax, color='teal', edgecolor='black')
                    ax.set_title(f"WOE – {var}")
                    img_data = io.BytesIO()
                    fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    zip_file.writestr(f"woe_{var}.png", img_data.getvalue())

        # 5. Tabelas de WOE
        if "Tabelas de WOE" in selecionados and woe_tables:
            for var, table in woe_tables.items():
                if 'erro' not in table.columns:
                    csv_data = table.to_csv(index=True)
                    zip_file.writestr(f"woe_{var}.csv", csv_data)

        # 6. Relatório de análise
        if "Relatório de Análise" in selecionados:
            top_iv = iv_df.sort_values("IV", ascending=False).head(3)['Variável'].tolist() if not iv_df.empty else []
            top_ks = ks_df.sort_values("KS", ascending=False).head(3)['Variável'].tolist() if not ks_df.empty else []
            relatorio = f"""
                        Relatório de Pré-Seleção de Variáveis
                        =====================================
                        Variável-alvo: {target}
                        
                        Resumo:
                        - Total de variáveis ativas: {len(iv_df)}
                        - Top 3 por IV: {', '.join(top_iv) if top_iv else 'N/A'}
                        - Top 3 por KS: {', '.join(top_ks) if top_ks else 'N/A'}
                        
                        Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
                                    """.strip()
            zip_file.writestr("relatorio_analise.txt", relatorio)

    zip_buffer.seek(0)
    return zip_buffer
    
def main():
    st.title("📈 Análise Bivariada e Pré-Seleção de Variáveis")
    st.markdown("""
    Defina a variável-alvo, corrija seu formato, e realize análises preditivas:  
    **IV, WOE, KS** – tudo em um só lugar.
    """)
    
    if 'dados' not in st.session_state:
        st.warning("Dados não carregados! Acesse a página de Coleta primeiro.")
        st.page_link("pages/3_🚀_Coleta_de_Dados.py", label="→ Ir para Coleta")
        return

    dados = st.session_state.dados.copy()

    # --- 1. SELEÇÃO E VALIDAÇÃO DA VARIÁVEL-ALVO (Y) ---
    st.markdown("### 🔍 Defina a Variável-Alvo (Default)")
    target = st.selectbox(
        "Selecione a coluna que indica **inadimplência**:",
        options=dados.columns,
        index=None,
        placeholder="Escolha a variável de default",
        key="target_select"  # ← mantém estado
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
        # Tenta ordenar apenas valores numéricos
        valores_numericos = [x for x in valores_unicos if isinstance(x, (int, float))]
        valores_unicos = sorted(valores_numericos) if valores_numericos else valores_unicos
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
            valor_bom = st.selectbox(
                "Valor que representa **adimplente (0)**",
                options=valores_unicos,
                key="valor_bom_select"  # ← estado persistente
            )
    
        with col2:
            # Remove o valor escolhido como "bom" das opções para "mau"
            opcoes_maus = [v for v in valores_unicos if v != valor_bom]
            valor_mau = st.selectbox(
                "Valor que representa **inadimplente (1)**",
                options=opcoes_maus,
                key="valor_mau_select"  # ← estado persistente
            )
    
        # Botão para aplicar o mapeamento
        if st.button("✅ Aplicar Mapeamento", key="btn_aplicar_mapeamento"):
            if valor_bom == valor_mau:
                st.error("Erro: os valores para 'bom' e 'mau' devem ser diferentes.")
            else:
                try:
                    # Mapeia os valores
                    y_mapped = dados[target].map({valor_bom: 0, valor_mau: 1})
                    
                    # Verifica se houve falha no mapeamento (valores não mapeados)
                    if y_mapped.isnull().any():
                        st.error(f"Erro: alguns valores não foram mapeados corretamente. Verifique os dados.")
                    else:
                        # Atualiza os dados
                        dados_atualizados = dados.copy()
                        dados_atualizados[target] = y_mapped
                        st.session_state.dados = dados_atualizados
                        st.session_state.target = target
                        st.success(f"✅ `{target}` foi convertida para 0 (adimplente) e 1 (inadimplente).")
                        st.rerun()  # ← recarrega para refletir a mudança
                except Exception as e:
                    st.error(f"Erro ao aplicar mapeamento: {e}")
    
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

                fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax_corr)
                ax_corr.set_title("Mapa de Calor de Correlação")
                st.pyplot(fig_corr)
                
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
        Edite os valores de **mínimo e máximo** ou os **limites das faixas** para ajustar o tamanho do intervalo de cada classe.  
        Obs: A tabela e o gráfico são atualizados automaticamente.
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
        #        st.markdown(f"**Faixas atuais:**")
        #        faixas = []
        #        for i in range(n_bins):
        #            faixas.append(f"`[{bins[i]:.2f}, {bins[i+1]:.2f})`")
        #        st.write(" → ".join(faixas))

                try:
                    df_temp = dados[[var_selecionada, target]].dropna()
                    # Filtra apenas dentro dos limites definidos
                    mask = (df_temp[var_selecionada] >= min_val) & (df_temp[var_selecionada] <= max_val)
                    df_temp = df_temp[mask]
                
                    # Aplica os bins
                    df_temp['bin_interval'] = pd.cut(df_temp[var_selecionada], bins=bins, include_lowest=True, right=False, duplicates='drop')
                
                    # Cria rótulos formatados para exibição
                    bin_labels = []
                    for i in range(len(bins) - 1):
                        left = f"{bins[i]:.2f}"
                        right = f"{bins[i+1]:.2f}"
                        bin_labels.append(f"[{left}, {right})")
                    
                    # Mapeia o intervalo para o rótulo formatado
                    bin_to_label = {interval: label for interval, label in zip(pd.IntervalIndex.from_breaks(bins, closed='left'), bin_labels)}
                    df_temp['Classe'] = df_temp['bin_interval'].map(bin_to_label)
                
                    # Garante que a coluna 'bin' esteja ordenada corretamente
                    df_temp['Classe'] = pd.Categorical(df_temp['Classe'], categories=bin_labels, ordered=True)
                
                    # Cria a tabela de contagem
                    tmp = pd.crosstab(df_temp['Classe'], df_temp[target])
                    tmp.columns = ['não_default', 'default']
                
                    total_bons = tmp['não_default'].sum()
                    total_maus = tmp['default'].sum()
                
                    tmp['%_não_default'] = tmp['não_default'] / (total_bons or 1)
                    tmp['%_default'] = tmp['default'] / (total_maus or 1)
                
                    # Evitar divisão por zero
                    tmp['%_default'] = tmp['%_default'].replace(0, 1e-6)
                    tmp['%_não_default'] = tmp['%_não_default'].replace(0, 1e-6)
                
                    tmp['woe'] = np.log(tmp['%_não_default'] / tmp['%_default'])
                
                    woe_tables = {}
                    # Armazena tabela
                    if 'woe_tables' not in st.session_state:
                        st.session_state.woe_tables = {}
                    st.session_state.woe_tables[var_selecionada] = tmp.copy()
                
                    # Exibe tabela formatada
                    st.markdown("##### 📊 Tabela de WOE")
                
                    # Formatação visual
                    st.dataframe(
                        tmp.style.format({
                            'não_default': '{:,.0f}',
                            'default': '{:,.0f}',
                            '%_não_default': '{:.4f}',
                            '%_default': '{:.4f}',
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

    # --- TRANSFORMAÇÃO DE VARIÁVEIS CATEGÓRICAS ---
    with st.expander("🔄 Transformação de Variáveis Categóricas", expanded=False):
        st.markdown("### 🧠 Por que transformar variáveis categóricas?")
        st.info("""
        Variáveis categóricas com muitas classes ou com baixo poder preditivo (IV < 0.1) podem:
        - Aumentar a complexidade do modelo.
        - Gerar overfitting.
        - Ter classes com pouca população (ruído).
        
        **Soluções:**
        - 🔗 **Fusão de classes**: agrupar categorias semelhantes ou com baixa frequência.
        - ➕ **Variáveis dummy**: converter categorias em indicadores binários (útil para modelos lineares).
        """)

        st.info("""ALERTA! Esses procedimentos ainda não estão devidamente implementados para a fase seguinte. 
                  Assim, está presente aqui para suscitar a curiosidade e a possibilidade de serem realizados. 
                  """)

        # Recupera variáveis categóricas ativas
        if 'variaveis_ativas' not in st.session_state:
            st.warning("Nenhuma variável ativa definida. Volte para a análise de correlação.")
            st.stop()
    
        variaveis_ativas = st.session_state.variaveis_ativas
        categoricas = [col for col in variaveis_ativas if col != target]# and dados[col].dtype == 'object']
    
        if not categoricas:
            st.info("Nenhuma variável categórica disponível para transformação.")
        else:
            # Calcula IV para categóricas
            iv_data = []
            for col in categoricas:
                try:
                    iv = calcular_iv(dados, col, target)
                    iv_data.append({'Variável': col, 'IV': iv})
                except:
                    iv_data.append({'Variável': col, 'IV': np.nan})
            iv_df_cat = pd.DataFrame(iv_data).dropna().sort_values("IV", ascending=True)
    
            # Mostra variáveis com baixo IV
            baixo_iv = iv_df_cat[iv_df_cat['IV'] < 0.1]
            if not baixo_iv.empty:
                st.warning(f"⚠️ {len(baixo_iv)} variável(s) com IV < 0.1 (baixo poder preditivo):")
                st.dataframe(baixo_iv.style.format({"IV": "{:.3f}"}).background_gradient(cmap="Oranges", subset=["IV"]))
            else:
                st.success("✅ Todas as variáveis categóricas têm IV ≥ 0.1.")
    
            # Seleção da variável para transformação
            var_cat = st.selectbox(
                "Selecione uma variável categórica para transformar:",
                options=categoricas,
                key="var_cat_select"
            )
    
            if var_cat:
                serie = dados[var_cat].value_counts().reset_index()
                serie.columns = [var_cat, 'Frequência']
                serie['%'] = (serie['Frequência'] / serie['Frequência'].sum() * 100).round(2)
                st.dataframe(serie)
    
                tab1, tab2 = st.tabs(["🔗 Fusão de Classes", "➕ Criar Dummies"])
    
                with tab1:
                    st.markdown("#### 🔗 Reagrupe classes com critério (ex: 'outros', agrupar por risco)")
                    classes = dados[var_cat].dropna().unique().tolist()
                    st.caption("Selecione as classes que deseja **agrupar em uma nova categoria**.")
    
                    col1, col2 = st.columns(2)
                    with col1:
                        selecao = st.multiselect(
                            "Classes para agrupar:",
                            options=classes,
                            key=f"merge_select_{var_cat}"
                        )
                    with col2:
                        novo_nome = st.text_input(
                            "Nome da nova categoria:",
                            value="Outros",
                            key=f"novo_nome_{var_cat}"
                        )
    
                    if st.button("✅ Aplicar Fusão", key=f"btn_merge_{var_cat}"):
                        if len(selecao) < 2:
                            st.warning("Selecione pelo menos duas classes para fundir.")
                        else:
                            # Cria cópia dos dados
                            dados_transformado = dados.copy()
                            dados_transformado[var_cat] = dados_transformado[var_cat].astype('object')
                            dados_transformado[var_cat] = dados_transformado[var_cat].replace(selecao, novo_nome)
    
                            # Recalcula WOE e IV
                            try:
                                df_temp = dados_transformado[[var_cat, target]].dropna()
                                tmp = pd.crosstab(df_temp[var_cat], df_temp[target])
                                tmp.columns = ['não_default', 'default']
                                tmp['%_não_default'] = tmp['não_default'] / tmp['não_default'].sum()
                                tmp['%_default'] = tmp['default'] / tmp['default'].sum()
                                tmp['%_default'] = tmp['%_default'].replace(0, 1e-6)
                                tmp['%_não_default'] = tmp['%_não_default'].replace(0, 1e-6)
                                tmp['woe'] = np.log(tmp['%_não_default'] / tmp['%_default'])
                                iv_novo = ((tmp['%_não_default'] - tmp['%_default']) * tmp['woe']).sum()
    
                                st.success(f"✅ Fusão aplicada! Novo IV: {iv_novo:.3f}")
    
                                # Mostra tabela
                                st.dataframe(
                                    tmp[['não_default', 'default', '%_não_default', '%_default', 'woe']].style.format({
                                        '%_não_default': '{:.4f}',
                                        '%_default': '{:.4f}',
                                        'woe': '{:.3f}'
                                    }).background_gradient(cmap='RdYlGn', subset=['woe'])
                                )
    
                                # Pergunta ao usuário se deseja salvar
                                st.markdown("### 💾 Deseja incluir esta variável transformada no banco de dados?")
                                incluir = st.radio(
                                    "Incluir no conjunto de dados?",
                                    options=["Não", "Sim"],
                                    key=f"incluir_merge_{var_cat}"
                                )
                                if incluir == "Sim":
                                    nome_nova = st.text_input(
                                        "Como deseja identificar essa nova variável?",
                                        value=f"{var_cat}_agrupado",
                                        key=f"nome_merge_{var_cat}"
                                    )
                                    if st.button("💾 Salvar Variável Transformada", key=f"save_merge_{var_cat}"):
                                        if nome_nova in dados.columns:
                                            st.warning(f"Já existe uma coluna chamada `{nome_nova}`. Escolha outro nome.")
                                        else:
                                            # Salva a nova coluna
                                            if 'dados_transformados' not in st.session_state:
                                                st.session_state.dados_transformados = dados.copy()
                                            st.session_state.dados_transformados[nome_nova] = dados_transformado[var_cat]
                                            st.success(f"✅ Variável `{nome_nova}` salva com sucesso!")
                                            st.session_state.get('variaveis_ativas', []).append(nome_nova)  # Opcional: adiciona à lista ativa
    
                            except Exception as e:
                                st.error(f"Erro ao calcular novo WOE/IV: {e}")
    
                with tab2:
                    st.markdown("#### ➕ Criar Variáveis Dummy (One-Hot Encoding)")
                    st.info("Cria uma coluna binária para cada categoria (útil para modelos lineares).")
    
                    if st.button("✅ Gerar Dummies", key=f"btn_dummy_{var_cat}"):
                        try:
                            dummies = pd.get_dummies(dados[var_cat], prefix=var_cat)
                            st.success(f"✅ Criadas {dummies.shape[1]} variáveis dummy a partir de `{var_cat}`")
                            st.dataframe(dummies.head())
    
                            # Pergunta ao usuário se deseja salvar
                            st.markdown("### 💾 Deseja incluir essas variáveis dummy no banco de dados?")
                            incluir = st.radio(
                                "Incluir dummies no conjunto de dados?",
                                options=["Não", "Sim"],
                                key=f"incluir_dummy_{var_cat}"
                            )
                            if incluir == "Sim":
                                prefixo = st.text_input(
                                    "Prefixo para identificar as variáveis dummy:",
                                    value=var_cat,
                                    key=f"prefixo_dummy_{var_cat}"
                                )
                                if st.button("💾 Salvar Variáveis Dummy", key=f"save_dummy_{var_cat}"):
                                    dados_com_dummies = dados.copy()
                                    dummies_renomeadas = dummies.add_prefix(f"{prefixo}_")
                                    colisoes = [col for col in dummies_renomeadas.columns if col in dados_com_dummies.columns]
                                    if colisoes:
                                        st.warning(f"Conflito de nomes: {colisoes}. Remova ou renomeie primeiro.")
                                    else:
                                        # Salva no session_state
                                        if 'dados_transformados' not in st.session_state:
                                            st.session_state.dados_transformados = dados.copy()
                                        st.session_state.dados_transformados = pd.concat([
                                            st.session_state.dados_transformados, dummies_renomeadas
                                        ], axis=1)
                                        st.session_state.get('variaveis_ativas', []).extend(dummies_renomeadas.columns.tolist())
                                        st.success(f"✅ {len(dummies_renomeadas.columns)} variáveis dummy salvas com o prefixo `{prefixo}_`")
    
                        except Exception as e:
                            st.error(f"Erro ao gerar dummies: {e}")

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


    # --- EXPORTAÇÃO PERSONALIZADA ---
    st.markdown("---")
    with st.expander("💾 Exportar Outputs", expanded=False):
        st.markdown("### 📥 Escolha o que deseja incluir no relatório")
    
        # Opções de seleção
        incluir_graficos = st.checkbox("✅ Incluir gráficos (IV, KS, WOE)")
        incluir_tabelas = st.checkbox("✅ Incluir tabelas de WOE")
        incluir_relatorio = st.checkbox("✅ Incluir relatório de análise (txt)")
    
        if st.button("📦 Gerar Relatório ZIP"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                # 1. Incluir gráficos
                if incluir_graficos:
                    # Gráfico de IV
                    if 'iv_df' in st.session_state and not st.session_state.iv_df.empty:
                        fig_iv, ax_iv = plt.subplots(figsize=(6, 0.35 * len(st.session_state.iv_df)))
                        iv_df = st.session_state.iv_df.sort_values("IV", ascending=True)
                        bars = ax_iv.barh(iv_df['Variável'], iv_df['IV'], color='skyblue', edgecolor='darkblue')
                        ax_iv.set_title("Information Value (IV)")
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax_iv.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=9)
                        img_data = io.BytesIO()
                        fig_iv.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
                        plt.close(fig_iv)
                        zip_file.writestr("grafico_iv.png", img_data.getvalue())
    
                    # Gráfico de KS
                    if 'ks_df' in st.session_state and not st.session_state.ks_df.empty:
                        fig_ks, ax_ks = plt.subplots(figsize=(6, 0.35 * len(st.session_state.ks_df)))
                        ks_df = st.session_state.ks_df.sort_values("KS", ascending=True)
                        bars = ax_ks.barh(ks_df['Variável'], ks_df['KS'], color='lightcoral', edgecolor='darkred')
                        ax_ks.set_title("Kolmogorov-Smirnov (KS)")
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax_ks.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=9)
                        img_data = io.BytesIO()
                        fig_ks.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
                        plt.close(fig_ks)
                        zip_file.writestr("grafico_ks.png", img_data.getvalue())
    
                    # Gráficos de WOE
                    if 'woe_tables' in st.session_state:
                        for var, table in st.session_state.woe_tables.items():
                            if 'woe' in table.columns:
                                fig, ax = plt.subplots(figsize=(6, 3))
                                table['woe'].plot(kind='barh', ax=ax, color='teal', edgecolor='black')
                                ax.set_title(f"WOE - {var}")
                                img_data = io.BytesIO()
                                fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
                                plt.close(fig)
                                zip_file.writestr(f"woe_{var}.png", img_data.getvalue())
    
                # 2. Incluir tabelas de WOE
                if incluir_tabelas and 'woe_tables' in st.session_state:
                    for var, table in st.session_state.woe_tables.items():
                        if 'erro' not in table.columns:
                            csv_data = table.to_csv(index=True)
                            zip_file.writestr(f"woe_{var}.csv", csv_data)
    
                # 3. Incluir relatório de análise
                if incluir_relatorio:
                    # Recuperar top IV e KS
                    top_iv = []
                    if 'iv_df' in st.session_state and not st.session_state.iv_df.empty:
                        top_iv = st.session_state.iv_df.sort_values("IV", ascending=False).head(3)['Variável'].tolist()
    
                    top_ks = []
                    if 'ks_df' in st.session_state and not st.session_state.ks_df.empty:
                        top_ks = st.session_state.ks_df.sort_values("KS", ascending=False).head(3)['Variável'].tolist()

    
                    relatorio_txt = f"""
                            Relatório de Análise Bivariada
                            ==============================
                            Variável-alvo: {target}
                            
                            Resumo:
                            - Total de variáveis ativas: {len(st.session_state.variaveis_ativas)}
                            - Top 3 por IV: {', '.join(top_iv) if top_iv else 'N/A'}
                            - Top 3 por KS: {', '.join(top_ks) if top_ks else 'N/A'}
                            
                            Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
                                            """.strip()
                    zip_file.writestr("relatorio_analise.txt", relatorio_txt)
    
            # Finaliza o buffer e cria o link de download
            zip_buffer.seek(0)
            b64 = base64.b64encode(zip_buffer.getvalue()).decode()
            href = f'<a href="data:application/zip;base64,{b64}" download="relatorio_analise_bivariada.zip">📥 Baixar Relatório ZIP</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Relatório personalizado gerado com sucesso!")
    
        
    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/6_🤖_Modelagem.py", label="➡️ Ir para Modelagem", icon="🤖")

if __name__ == "__main__":
    main()
