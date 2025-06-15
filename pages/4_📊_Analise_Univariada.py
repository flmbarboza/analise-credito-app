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

if __name__ == "__main__":
    main()
