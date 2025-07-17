import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("📊 Análise Univariada")
    st.markdown("Explore cada variável individualmente para entender suas características básicas")

    if 'dados' not in st.session_state:
        st.warning("Carregue os dados na página de Coleta primeiro!")
        st.page_link("pages/2_📊_Coleta_de_Dados.py", label="→ Ir para Coleta de Dados")
        return

    dados = st.session_state.dados
    
    # Seletor de variável
    variavel = st.selectbox("Selecione a variável para análise:", dados.columns)
    
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
    with st.expander("🔍 Sugestões de Informações que pode extrair destes dados:"):
    # Análise automática
        st.markdown("##### 📊 Análise da variável selecionada: `{}`".format(variavel))
    
        # 1. Tipo da variável
        if dados[variavel].dtype == 'object':
            st.write(f"🔹 **Tipo:** Variável categórica")
            st.write(f"🔹 **Categorias únicas:** {dados[variavel].nunique()}")
        else:
            st.write(f"🔹 **Tipo:** Variável numérica")
            st.write(f"🔹 **Média:** {dados[variavel].mean():.2f}")
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

    # 🚀 Link para a próxima página
    st.page_link("pages/5_📈_Analise_Bivariada.py", label="➡️ Ir para a próxima página: Análise Bivariada", icon="📈")

if __name__ == "__main__":
    main()
