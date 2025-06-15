import streamlit as st
import pandas as pd

def main():
    st.title("🏛️ Políticas de Crédito")
    st.markdown("Defina regras de negócio para decisão de crédito")

    if 'modelo' not in st.session_state:
        st.warning("Modelo não disponível")
        return

    st.subheader("Limiares de Decisão")
    
    corte = st.slider("Score mínimo para aprovação:", 0, 1000, 600)
    st.markdown(f"🔹 Clientes com score abaixo de {corte} serão reprovados")
    
    st.subheader("Simulação de Decisão")
    renda = st.number_input("Renda do Cliente:", 1000, 50000, 3000)
    divida = st.number_input("Dívida Atual:", 0, 100000, 5000)
    
    if st.button("Simular Decisão"):
        # Simulação simplificada
        score = (renda * 0.5) - (divida * 0.3)
        aprovado = score >= corte
        
        st.metric("Score Calculado", f"{score:.0f}")
        st.success("✅ Aprovado" if aprovado else "❌ Reprovado")
    # 🚀 Link para a próxima página
    st.page_link("pages/10_📊_Entendendo_Análise_de_Crédito.py", label="➡️ Ir para a próxima página: Entendendo Análise de Crédito", icon="📊")

if __name__ == "__main__":
    main()
