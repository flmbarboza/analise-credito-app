import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import zipfile
import base64

def main():
    st.title("📊 Análise Univariada")

    # --- 1. VALIDAÇÃO DE DADOS (fora de qualquer expander) ---
    if 'dados' not in st.session_state:
        st.warning("Carregue os dados na página de Coleta primeiro!")
        st.page_link("pages/3_🚀_Coleta_de_Dados.py", label=" → Retornar para Coleta de dados")
   
    dados = st.session_state.dados
    
    if dados is None or dados.empty:
        st.error("""Os dados estão vazios ou inválidos.
                Neste caso, retorne a página de coleta de dados e revise o procedimento.""")
        st.page_link("pages/3_🚀_Coleta_de_Dados.py", label=" → Retornar para Coleta de dados")
        st.stop()
        
    # Se os dados não estão carregados, não vamos prosseguir com análises
    dados_disponiveis = 'dados' in st.session_state and not st.session_state.dados.empty

    # --- 2. SELEÇÃO DE VARIÁVEL (única, fora de expander) ---
    if dados_disponiveis:
        variavel = st.selectbox(
            "Selecione a variável para análise:",
            options=dados.columns,
            index=None,
            placeholder="Clique aqui para escolher",
            key="variavel_uni"
        )
    else:
        variavel = None

    # ========================================================
    # ✅ EXPANDER 1: ANÁLISE INDIVIDUAL
    # ========================================================
    with st.expander("🔍 Explore cada variável individualmente para entender suas características básicas", expanded=True):
        if not dados_disponiveis:
            st.warning("Dados não carregados. Acesse a página de Coleta.")
            st.page_link("pages/2_📊_Coleta_de_Dados.py", label="→ Ir para Coleta de Dados")
        elif variavel is None:
            st.info("👆 Por favor, escolha uma variável acima para iniciar a análise.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Estatísticas Descritivas")
                st.write(dados[variavel].describe())
            with col2:
                st.subheader("Gráfico de Barras/Histograma")
                fig, ax = plt.subplots()
                ax.set_ylabel("Quantidade")
                if pd.api.types.is_numeric_dtype(dados[variavel]):
                    sns.histplot(data=dados, x=variavel, ax=ax, kde=False)
                else:
                    dados[variavel].value_counts().plot(kind='barh', ax=ax)
                st.pyplot(fig)

    # ========================================================
    # ✅ EXPANDER 2: SUGESTÕES DE ANÁLISE
    # ========================================================
    with st.expander("🔍 Sugestões de Informações que pode extrair destes dados:", expanded=False):
        if not dados_disponiveis:
            st.warning("Carregue os dados para ver as sugestões.")
        elif variavel is None:
            st.info("Selecione uma variável para ver a análise detalhada.")
        else:
            st.markdown(f"##### 📊 Análise da variável selecionada: `{variavel}`")

            # 1. Tipo da variável
            if dados[variavel].dtype == 'object':
                st.write(f"🔹 **Tipo:** Variável categórica")
                st.write(f"🔹 **Categorias únicas:** {dados[variavel].nunique()}")
            else:
                st.write(f"🔹 **Tipo:** Variável numérica")
                st.write(f"🔹 **Média:** {dados[variavel].mean():.2f}")
                st.write(f"🔹 **Mediana:** {dados[variavel].median():.2f}")
                st.write(f"🔹 **Desvio padrão:** {dados[variavel].std():.2f}")

            # 2. Valores faltantes
            nulos = dados[variavel].isnull().sum()
            if nulos > 0:
                st.warning(f"⚠️ **Valores faltantes:** {nulos} registros ausentes")
            else:
                st.success(f"✅ **Valores faltantes:** Nenhum")

            # 3. Duplicados
            duplicados = dados[variavel].duplicated().sum()
            if duplicados > 0:
                st.info(f"🔁 **Valores duplicados:** {duplicados} registros repetidos")
            else:
                st.info(f"✅ **Valores duplicados:** Nenhum")

            # 4. Moda
            moda = dados[variavel].mode()
            if len(moda) > 0:
                st.write(f"🔹 **Valor mais frequente (moda):** {moda.iloc[0]}")

            # 5. Faixa (numéricas)
            if pd.api.types.is_numeric_dtype(dados[variavel]):
                st.write(f"🔹 **Valor mínimo:** {dados[variavel].min():.2f}")
                st.write(f"🔹 **Valor máximo:** {dados[variavel].max():.2f}")
                st.write(f"🔹 **Amplitude:** {dados[variavel].max() - dados[variavel].min():.2f}")

            # 6. Outliers (numéricas)
            if pd.api.types.is_numeric_dtype(dados[variavel]):
                q1 = dados[variavel].quantile(0.25)
                q3 = dados[variavel].quantile(0.75)
                iqr = q3 - q1
                limite_inferior = q1 - 1.5 * iqr
                limite_superior = q3 + 1.5 * iqr
                outliers = dados[(dados[variavel] < limite_inferior) | (dados[variavel] > limite_superior)].shape[0]
                if outliers > 0:
                    st.warning(f"⚠️ **Possíveis outliers:** {outliers} registros fora do padrão")
                else:
                    st.success(f"✅ **Outliers:** Nenhum valor fora do padrão detectado")

            # 7. Assimetria
            if pd.api.types.is_numeric_dtype(dados[variavel]):
                media = dados[variavel].mean()
                mediana = dados[variavel].median()
                if abs(media - mediana) > 0.5 * dados[variavel].std():
                    st.info(f"📈 **Assimetria:** A média ({media:.2f}) e a mediana ({mediana:.2f}) são diferentes → provável assimetria")
                else:
                    st.info(f"⚖️ **Simetria:** Média e mediana próximas → distribuição aparentemente simétrica")

    # ========================================================
    # ✅ EXPANDER 3: O QUE É ANÁLISE UNIVARIADA?
    # ========================================================
    with st.expander("🔍 Como explorar, analisar e extrair insights de variáveis individuais?", expanded=False):
        st.markdown("##### 📘 O que é Análise Univariada?")
        st.markdown("""
            A **Análise Univariada** é a análise de **uma variável por vez**, com o objetivo de:
            - Entender sua distribuição
            - Identificar possíveis problemas na base (dados faltantes, inconsistências e outliers)
            - Verificar qualidade dos dados
            - Tomar decisões sobre transformações ou tratamentos
            Essa é a primeira etapa em qualquer análise de dados!
            """)

    # ========================================================
    # ✅ EXPANDER 4: EXEMPLOS DE INSIGHTS
    # ========================================================
    with st.expander("🧠 Exemplos de Informações/Curiosidades/Achados que pode ter", expanded=False):
        st.markdown("##### 📈 Possibilidades para Variáveis Numéricas")
        st.markdown("""
            - A distribuição é simétrica ou assimétrica?
            - Há valores extremos (outliers)?
            - A média está próxima da mediana?
            - A variável tem muitos valores nulos?
            """)
        st.markdown("##### 📊 Possibilidades para Variáveis Categóricas")
        st.markdown("""
            - Qual categoria é mais frequente?
            - Há categorias raras ou inconsistentes?
            - A variável precisa ser padronizada?
            - Há necessidade de agrupar categorias?
            """)
        st.markdown("##### **Tente! Investigue! Navegue pelos dados! Be curious!!!**")

    # ========================================================
    # ✅ FUNÇÕES DE INSIGHTS E GRÁFICOS
    # ========================================================
    def gerar_insights(variavel):
        serie = dados[variavel]
        insights = []
        if pd.api.types.is_numeric_dtype(serie):
            insights.append("🔹 **Tipo:** Variável numérica")
            insights.append(f"🔹 **Média:** {serie.mean():.2f}")
            insights.append(f"🔹 **Desvio padrão:** {serie.std():.2f}")
            insights.append(f"🔹 **Mínimo:** {serie.min():.2f}")
            insights.append(f"🔹 **Máximo:** {serie.max():.2f}")
            insights.append(f"🔹 **Amplitude:** {serie.max() - serie.min():.2f}")
            q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
            iqr = q3 - q1
            lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = serie[(serie < lim_inf) | (serie > lim_sup)].shape[0]
            if outliers > 0:
                insights.append(f"⚠️ **Outliers detectados:** {outliers} valores")
            else:
                insights.append("✅ **Outliers:** Nenhum")
        else:
            insights.append("🔹 **Tipo:** Variável categórica")
            insights.append(f"🔹 **Categorias únicas:** {serie.nunique()}")
            insights.append(f"🔹 **Moda:** {serie.mode().iloc[0] if not serie.mode().empty else 'N/A'}")
        nulos = serie.isnull().sum()
        insights.append(f"{'⚠️' if nulos > 0 else '✅'} **Valores faltantes:** {nulos}")
        return "\n".join(insights)

    def gerar_graficos(variavel):
        figs = []
        serie = dados[variavel]
        if pd.api.types.is_numeric_dtype(serie):
            fig1, ax1 = plt.subplots()
            sns.histplot(data=dados, x=variavel, ax=ax1)
            ax1.set_ylabel("Quantidade")
            figs.append((fig1, f"{variavel}_histograma.png"))
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=dados[variavel], ax=ax2)
            figs.append((fig2, f"{variavel}_boxplot.png"))
        else:
            fig3, ax3 = plt.subplots()
            serie.value_counts().plot(kind='bar', ax=ax3)
            figs.append((fig3, f"{variavel}_barras.png"))
        return figs

    # ========================================================
    # ✅ EXPANDER 5: EXPORTAR RELATÓRIO
    # ========================================================
    with st.expander("📤 Exportar Análises"):
        # ========================================================
        # ✅ SELEÇÃO DE MÚLTIPLAS VARIÁVEIS PARA RELATÓRIO
        # ========================================================
        if dados_disponiveis:
            st.markdown("### 🔍 Selecione as variáveis para gerar um relatório preliminar:")
            colunas_selecionadas = st.multiselect(
                "Selecione as colunas",
                dados.columns.tolist(),
                default=dados.columns.tolist()
            )
        else:
            colunas_selecionadas = []

        
        if not dados_disponiveis:
            st.warning("Carregue os dados para habilitar exportação.")
        else:
            st.markdown("### Escolha o que deseja salvar:")
            save_insights = st.checkbox("Salvar Sugestões de Informações")
            save_hist = st.checkbox("Salvar Histogramas")
            save_boxplot = st.checkbox("Salvar Boxplots")
            save_bars = st.checkbox("Salvar Gráficos de Barras")

            if st.button("Gerar Relatório ZIP"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for var in colunas_selecionadas:
                        if save_insights:
                            insights = gerar_insights(var)
                            zip_file.writestr(f"insights_{var}.txt", insights)
                        figs = gerar_graficos(var)
                        for fig, nome in figs:
                            if ("boxplot" in nome and save_boxplot) or \
                               ("histograma" in nome and save_hist) or \
                               ("barras" in nome and save_bars):
                                img_data = io.BytesIO()
                                fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
                                plt.close(fig)
                                zip_file.writestr(nome, img_data.getvalue())
                zip_buffer.seek(0)
                b64 = base64.b64encode(zip_buffer.getvalue()).decode()
                href = f'<a href="data:application/zip;base64,{b64}" download="relatorio_analise.zip">📥 Baixar Relatório ZIP</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("✅ Relatório gerado com sucesso!")

    # ========================================================
    # ✅ NAVEGAÇÃO
    # ========================================================
    st.markdown("---")
    st.page_link("pages/5_📈_Analise_Bivariada.py", label="➡️ Ir para a próxima página: Análise Bivariada", icon="📈")

if __name__ == "__main__":
    main()
