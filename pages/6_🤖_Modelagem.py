import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    st.title("🤖 Modelagem Preditiva")
    st.markdown("Construa e avalie modelos de credit scoring")

    if 'dados' not in st.session_state:
        st.warning("Dados não encontrados! Complete a coleta primeiro.")
        st.page_link("pages/2_📊_Coleta_de_Dados.py", label="→ Coleta de Dados")
        return

    dados = st.session_state.dados
    
    st.subheader("Configuração do Modelo")
    
    # Seleção de variáveis
    target = st.selectbox("Variável Target:", dados.columns)
    features = st.multiselect("Variáveis Preditivas:", 
                             [col for col in dados.columns if col != target])
    
    if st.button("Treinar Modelo"):
        with st.spinner("Treinando..."):
            try:
                X = dados[features]
                y = dados[target]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                
                st.session_state.modelo = model
                st.success("Modelo treinado com sucesso!")
                
                acuracia = model.score(X_test, y_test)
                st.metric("Acurácia no Teste", f"{acuracia:.1%}")
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
    # 🚀 Link para a próxima página
    st.page_link("pages/7_✅_Analise_e_Validacao.py", label="➡️ Ir para a próxima página: Análise e Validação", icon="✅")

if __name__ == "__main__":
    main()
