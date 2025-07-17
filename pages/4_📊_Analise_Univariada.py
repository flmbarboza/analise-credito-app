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
        st.subheader("Distribuição")
        fig, ax = plt.subplots()
        sns.histplot(data=dados, x=variavel, ax=ax)
        st.pyplot(fig)
    
    # Análise automática
    with st.expander("🔍 Insights Automáticos"):
        if dados[variavel].dtype == 'object':
            st.write(f"🔹 Variável categórica com {dados[variavel].nunique()} categorias")
        else:
            st.write(f"🔹 Variável numérica com média {dados[variavel].mean():.2f}")
        
        if dados[variavel].isnull().sum() > 0:
            st.warning(f"⚠️ Contém {dados[variavel].isnull().sum()} valores faltantes")

    with st.expander("🔍 Como explorar, analisar e extrair insights de variáveis individuais?", expanded=False):
        st.markdown("""📘 O que é Análise Univariada?
            A **Análise Univariada** é a análise de **uma variável por vez**, com o objetivo de:
            - Entender sua distribuição
            - Identificar possíveis problemas na base (dados faltantes, inconsistências e outliers)
            - Verificar qualidade dos dados
            - Tomar decisões sobre transformações ou tratamentos
            
            Essa é a **primeira etapa** em qualquer análise de dados!
            """)
    
    with st.expander("🧠 Exemplos de Insights", expanded=True):
        st.markdown("### 📈 Insights para Variáveis Numéricas")
        st.markdown("""
            - A distribuição é simétrica ou assimétrica?
            - Há valores extremos (outliers)?
            - A média está próxima da mediana?
            - A variável tem muitos valores nulos?
            """)
        
        st.markdown("### 📊 Insights para Variáveis Categóricas")
        st.markdown("""
            - Qual categoria é mais frequente?
            - Há categorias raras ou inconsistentes?
            - A variável precisa ser padronizada?
            - Há necessidade de agrupar categorias?
            """)
    

    # 🚀 Link para a próxima página
    st.page_link("pages/5_📈_Analise_Bivariada.py", label="➡️ Ir para a próxima página: Análise Bivariada", icon="📈")

if __name__ == "__main__":
    main()
