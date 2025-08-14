import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import zipfile
import base64

def main():
    st.title("📊 Análise Univariada")

    with st.expander("🔍 Explore cada variável individualmente para entender suas características básicas", expanded=True):

        st.markdown("Explore cada variável individualmente para entender suas características básicas")
    
        if 'dados' not in st.session_state:
            st.warning("Carregue os dados na página de Coleta primeiro!")
            st.page_link("pages/2_📊_Coleta_de_Dados.py", label="→ Ir para Coleta de Dados")
            return
    
        dados = st.session_state.dados
        
        # Seletor de variável
        variavel = st.selectbox("Selecione a variável para análise:", dados.columns,
                               index=None, placeholder="Clique aqui para escolher")
        # ✅ Validação: só continua se o usuário escolheu algo
        if variavel is None:
        st.info("👆 Por favor, escolha uma variável acima para iniciar a análise.")
        return

        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Estatísticas Descritivas")
            st.write(dados[variavel].describe())
            
        with col2:
            st.subheader("Gráfico de Barras/Histograma")
            fig, ax = plt.subplots()
            ax.set_ylabel("Quantidade")
            sns.histplot(data=dados, x=variavel, ax=ax)
            st.pyplot(fig)
    
    # Análise automática
    with st.expander("🔍 Sugestões de Informações que pode extrair destes dados:", expanded=False):
    # Análise automática
        st.markdown("##### 📊 Análise da variável selecionada: `{}`".format(variavel))
    
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
    
        # 3. Valores duplicados
        duplicados = dados[variavel].duplicated().sum()
        if duplicados > 0:
            st.info(f"🔁 **Valores duplicados:** {duplicados} registros repetidos")
        else:
            st.info(f"✅ **Valores duplicados:** Nenhum")
    
        # 4. Valor mais frequente (moda)
        moda = dados[variavel].mode()[0]
        st.write(f"🔹 **Valor mais frequente (moda):** {moda}")
    
        # 5. Faixa de valores (para numéricas)
        if pd.api.types.is_numeric_dtype(dados[variavel]):
            st.write(f"🔹 **Valor mínimo:** {dados[variavel].min():.2f}")
            st.write(f"🔹 **Valor máximo:** {dados[variavel].max():.2f}")
            st.write(f"🔹 **Amplitude:** {dados[variavel].max() - dados[variavel].min():.2f}")
    
        # 6. Tendência de concentração (para numéricas)
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
    
        # 7. Tendência de distribuição (assimetria)
        if pd.api.types.is_numeric_dtype(dados[variavel]):
            media = dados[variavel].mean()
            mediana = dados[variavel].median()
            if abs(media - mediana) > 0.5 * dados[variavel].std():
                st.info(f"📈 **Assimetria:** A média ({media:.2f}) e a mediana ({mediana:.2f}) são diferentes → provável assimetria")
            else:
                st.info(f"⚖️ **Simetria:** Média e mediana próximas → distribuição aparentemente simétrica")
        
        
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
            - Há categorias raras ou inconsistentes? E os "outros" tem? Pode?
            - A variável precisa ser padronizada?
            - Há necessidade/oportunidade [e consistência racional] de agrupar categorias?
            """)
        st.markdown("##### **Tente! Investigue! Navegue pelos dados! Be curious!!!**")

    # Seleção de múltiplas variáveis
    st.markdown("### 🔍 Selecione as variáveis para que deseja gerar um relatório preliminar:")
    colunas_selecionadas = st.multiselect("Selecione as colunas", dados.columns.tolist(), default=dados.columns.tolist())
    
    # Função para gerar insights automáticos
    def gerar_insights(variavel):
        insights = []
        serie = dados[variavel]
        
        # Tipo da variável
        if pd.api.types.is_numeric_dtype(serie):
            insights.append("🔹 **Tipo:** Variável numérica")
            insights.append(f"🔹 **Média:** {serie.mean():.2f}")
            insights.append(f"🔹 **Desvio padrão:** {serie.std():.2f}")
            insights.append(f"🔹 **Mínimo:** {serie.min():.2f}")
            insights.append(f"🔹 **Máximo:** {serie.max():.2f}")
            insights.append(f"🔹 **Amplitude:** {serie.max() - serie.min():.2f}")
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            limite_inf = q1 - 1.5 * iqr
            limite_sup = q3 + 1.5 * iqr
            outliers = serie[(serie < limite_inf) | (serie > limite_sup)].shape[0]
            if outliers > 0:
                insights.append(f"⚠️ **Outliers detectados:** {outliers} valores fora do padrão")
            else:
                insights.append("✅ **Outliers:** Nenhum valor fora do padrão detectado")
        else:
            insights.append("🔹 **Tipo:** Variável categórica")
            insights.append(f"🔹 **Categorias únicas:** {serie.nunique()}")
            insights.append(f"🔹 **Moda:** {serie.mode()[0]}")
    
        # Valores faltantes
        nulos = serie.isnull().sum()
        if nulos > 0:
            insights.append(f"⚠️ **Valores faltantes:** {nulos} registros")
        else:
            insights.append("✅ **Valores faltantes:** Nenhum")
    
        return "\n".join(insights)
    
    # Função para exportar gráficos
    def gerar_graficos(variavel):
        figs = []
        serie = dados[variavel]
        if pd.api.types.is_numeric_dtype(serie):
            # Histograma
            fig1, ax1 = plt.subplots()
            sns.histplot(data=dados, x=variavel, ax=ax1)
            ax1.set_ylabel("Quantidade")
            ax1.set_xlabel(variavel)
            figs.append((fig1, f"{variavel}_histograma.png"))
            
            # Boxplot
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=dados[variavel], ax=ax2)
            ax2.set_xlabel(variavel)
            figs.append((fig2, f"{variavel}_boxplot.png"))
        else:
            # Gráfico de barras
            fig3, ax3 = plt.subplots()
            serie.value_counts().plot(kind='bar', ax=ax3)
            ax3.set_title(variavel)
            figs.append((fig3, f"{variavel}_barras.png"))
        return figs
    
    # Bloco de exportação
    with st.expander("📤 Exportar Análises"):
        st.markdown("### Escolha o que deseja salvar:")
        save_insights = st.checkbox("Salvar Sugestões de Informações")
        save_hist = st.checkbox("Salvar Histogramas")
        save_boxplot = st.checkbox("Salvar Boxplots")
        save_bars = st.checkbox("Salvar Gráficos de Barras")
    
        if st.button("Gerar Relatório ZIP"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for variavel in colunas_selecionadas:
                    # Salvar insights
                    if save_insights:
                        insights = gerar_insights(variavel)
                        zip_file.writestr(f"insights_{variavel}.txt", insights)
    
                    # Salvar gráficos
                    figs = gerar_graficos(variavel)
                    for fig, nome in figs:
                        img_data = io.BytesIO()
                        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        zip_file.writestr(nome, img_data.getvalue())
    
            zip_buffer.seek(0)
            b64 = base64.b64encode(zip_buffer.getvalue()).decode()
            href = f'<a href="data:application/zip;base64,{b64}" download="relatorio_analise.zip">📥 Baixar Relatório ZIP</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Relatório gerado com sucesso!")

    # 🚀 Link para a próxima página
    st.page_link("pages/5_📈_Analise_Bivariada.py", label="➡️ Ir para a próxima página: Análise Bivariada", icon="📈")

if __name__ == "__main__":
    main()
