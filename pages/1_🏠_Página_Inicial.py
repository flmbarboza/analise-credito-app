import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def main():
    st.title("🏦 Bem-vindo à Plataforma Financeira Completa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        ### 💳 Análise de Crédito
        Avalie seu perfil de crédito de forma rápida e inteligente
        """)
    
    with col2:
        st.success("""
        ### 📊 Dashboard Financeiro
        Visualize suas finanças em tempo real
        """)
    
    with col3:
        st.warning("""
        ### 📈 Simulador de Investimentos
        Simule diferentes cenários de investimento
        """)
    
    st.markdown("---")
    
    # Estatísticas gerais
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        st.metric("Usuários Ativos", "1,234", "+12%")
    
    with col5:
        st.metric("Análises Realizadas", "5,678", "+8%")
    
    with col6:
        st.metric("Taxa de Aprovação", "73%", "+2%")
    
    with col7:
        st.metric("Satisfação", "4.8/5", "+0.2")
    
    # Gráfico de exemplo
    st.subheader("📈 Tendências do Mercado")
    
    # Dados simulados
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    aprovacoes = np.random.normal(100, 15, len(dates)).cumsum()
    
    fig = px.line(
        x=dates, 
        y=aprovacoes,
        title="Evolução de Aprovações de Crédito",
        labels={'x': 'Mês', 'y': 'Número de Aprovações'}
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
