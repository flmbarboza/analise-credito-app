# pages/1_📊_Entendendo_Analise_Credito.py
import streamlit as st

def main():
    st.title("📊 Entendendo a Análise de Crédito")
    
    with st.expander("🔍 Conceitos-Chave"):
        st.markdown("""
        - **Credit Scoring**: Métrica que quantifica o risco de inadimplência  
        - **Variáveis Comuns**: Renda, histórico de pagamentos, endividamento  
        - **Fontes de Dados**: SPC, Serasa, bancos de dados internos  
        """)
    
    # Widget interativo
    st.subheader("🧮 Simule um Critério de Aprovação")
    renda = st.slider("Renda Mensal (R$):", 1000, 20000, 3000)
    divida = st.number_input("Dívidas Atuais (R$):", 0.0, 100000.0, 5000.0)
    
    if st.button("Calcular Risco"):
        score = (renda * 0.4) - (divida * 0.6)
        st.success(f"Score Preliminar: {score:.1f}")
    # 🚀 Link para a próxima página
    st.page_link("pages/11_📑_Relatorio.py", label="➡️ Ir para a próxima página: Relatório", icon="📑")

if __name__ == "__main__":
    main()
