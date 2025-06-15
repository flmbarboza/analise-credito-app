import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    st.title("📈 Análise Bivariada")
    st.markdown("Relacione pares de variáveis para identificar padrões e correlações")

    if 'dados' not in st.session_state:
        st.warning("Dados não carregados! Acesse a página de Coleta primeiro.")
        st.page_link("pages/2_📊_Coleta_de_Dados.py", label="→ Ir para Coleta")
        return

    dados = st.session_state.dados
    
    col1, col2 = st.columns(2)
    
    with col1:
        var_x = st.selectbox("Variável X:", dados.columns)
    
    with col2:
        var_y = st.selectbox("Variável Y:", dados.columns)
    
    # Visualização dinâmica
    tipo_grafico = st.radio("Tipo de gráfico:", 
                           ["Dispersão", "Boxplot", "Barras"], horizontal=True)
    
    fig, ax = plt.subplots()
    
    if tipo_grafico == "Dispersão":
        sns.scatterplot(data=dados, x=var_x, y=var_y, ax=ax)
    elif tipo_grafico == "Boxplot":
        sns.boxplot(data=dados, x=var_x, y=var_y, ax=ax)
    else:
        sns.barplot(data=dados, x=var_x, y=var_y, ax=ax)
    
    st.pyplot(fig)
    
    # Cálculo de correlação
    if dados[var_x].dtype != 'object' and dados[var_y].dtype != 'object':
        correlacao = dados[[var_x, var_y]].corr().iloc[0,1]
        st.metric("Coeficiente de Correlação", f"{correlacao:.2f}")
    # 🚀 Link para a próxima página
    st.page_link("pages/6_🤖_Modelagem.py", label="➡️ Ir para a próxima página: Modelagem", icon="🤖")

if __name__ == "__main__":
    main()
