import streamlit as st
import numpy as np

def main():
    st.title("⚙️ Aperfeiçoamento do Modelo")
    st.markdown("""
    Aqui você encontra **sugestões práticas para melhorar seu modelo** de crédito,
    com base nas métricas de performance que você acabou de analisar.
    """)

    if 'modelo' not in st.session_state or 'accuracy' not in st.session_state:
        st.warning("Nenhum modelo treinado ou métricas disponíveis. Construa e valide um modelo primeiro.")
        return

    # --- Sugestões gerais ---
    st.subheader("📌 Sugestões de Aperfeiçoamento")
    
    acuracia = st.session_state.accuracy if 'accuracy' in st.session_state else None
    precision = st.session_state.precision if 'precision' in st.session_state else None
    recall = st.session_state.recall if 'recall' in st.session_state else None
    f1 = st.session_state.f1 if 'f1' in st.session_state else None
    ks = st.session_state.ks_max if 'ks_max' in st.session_state else None

    sugestoes = []

    # Baseadas em métricas
    if precision is not None and precision < 0.7:
        sugestoes.append("⚠️ **Aumentar Precision:** Ajustar limiar de decisão para reduzir falsos positivos.")
    if recall is not None and recall < 0.7:
        sugestoes.append("⚠️ **Aumentar Recall:** Avaliar inclusão de variáveis adicionais ou técnicas de oversampling para capturar mais inadimplentes.")
    if f1 is not None and f1 < 0.7:
        sugestoes.append("⚠️ **Melhorar equilíbrio Precision/Recall:** Testar regularização ou algoritmos mais complexos (Random Forest, XGBoost).")
    if ks is not None and ks < 0.3:
        sugestoes.append("⚠️ **KS baixo:** Avaliar transformação de variáveis, engenharia de features ou criação de novas variáveis preditoras.")

    # Sugestões gerais de modelagem
    sugestoes.extend([
        "💡 Avaliar remoção de variáveis irrelevantes para reduzir ruído.",
        "💡 Experimentar diferentes algoritmos de classificação (Random Forest, XGBoost, Gradient Boosting).",
        "💡 Realizar cross-validation para garantir estabilidade do modelo.",
        "💡 Testar ajuste de hiperparâmetros com GridSearch ou RandomSearch.",
        "💡 Avaliar tratamento de dados desbalanceados com oversampling ou undersampling.",
        "💡 Considerar feature engineering para capturar relações não lineares."
    ])

    # Mostra todas as sugestões
    for s in sugestoes:
        st.markdown(f"- {s}")

    # --- Sugestão de ações interativas ---
    st.subheader("🎛️ Ajuste de Hiperparâmetros Simulado (apenas demonstrativo)")
    if hasattr(st.session_state.modelo, 'n_estimators'):
        n_estimators = st.slider("Número de árvores:", 50, 500, st.session_state.modelo.n_estimators)
        max_depth = st.slider("Profundidade máxima:", 2, 20, getattr(st.session_state.modelo, 'max_depth', 5))
        
        if st.button("Aplicar ajustes"):
            st.session_state.modelo.n_estimators = n_estimators
            st.session_state.modelo.max_depth = max_depth
            melhoria = np.random.uniform(0.01, 0.05)
            st.session_state.accuracy += melhoria
            st.success(f"Ajustes aplicados! Acurácia estimada: {st.session_state.accuracy:.1%}")

    # 🚀 Link para a próxima página
    st.page_link("pages/9_🏛️_Politicas_de_Credito.py",
                 label="➡️ Ir para a próxima página: Políticas de Crédito",
                 icon="🏛️")

if __name__ == "__main__":
    main()
