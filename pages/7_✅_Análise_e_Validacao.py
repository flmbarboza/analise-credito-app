import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    st.title("✅ Análise e Validação do Modelo")
    st.markdown("Avalie o desempenho do modelo com métricas técnicas")

    if 'modelo' not in st.session_state:
        st.warning("Nenhum modelo treinado! Construa um modelo primeiro.")
        st.page_link("pages/5_🤖_Modelagem.py", label="→ Ir para Modelagem")
        return

    # Dados de exemplo (substituir pelo seu conjunto real)
    X_test = st.session_state.get('X_test', None)
    y_test = st.session_state.get('y_test', None)
    
    if X_test is None or y_test is None:
        st.error("Dados de teste não disponíveis")
        return
    
    model = st.session_state.modelo
    y_pred = model.predict(X_test)
    
    st.subheader("Relatório de Classificação")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("Matriz de Confusão")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
