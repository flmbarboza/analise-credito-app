import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Configuração da página
st.set_page_config(
    page_title="Análise de Crédito Inteligente",
    page_icon="💳",
    layout="wide"
)

# Título principal
st.title("🏦 Plataforma de Análise de Crédito")
st.markdown("### Análise inteligente de risco de crédito baseada em suas respostas")

# Sidebar para informações do usuário
st.sidebar.header("📋 Informações do Solicitante")

# Coletando dados básicos
nome = st.sidebar.text_input("Nome completo:")
idade = st.sidebar.slider("Idade:", 18, 80, 30)
renda = st.sidebar.number_input("Renda mensal (R$):", min_value=0.0, value=3000.0, step=100.0)

# Seção principal - Perguntas interativas
st.header("🤔 Responda as perguntas abaixo:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Situação Profissional")
    emprego_estavel = st.radio(
        "Você tem emprego estável há mais de 2 anos?",
        ["Sim", "Não"],
        key="emprego"
    )
    
    carteira_assinada = st.radio(
        "Trabalha com carteira assinada?",
        ["Sim", "Não"],
        key="carteira"
    )
    
    renda_extra = st.radio(
        "Possui renda extra?",
        ["Sim", "Não"],
        key="renda_extra"
    )

with col2:
    st.subheader("Histórico Financeiro")
    nome_limpo = st.radio(
        "Seu nome está limpo nos órgãos de proteção (SPC/Serasa)?",
        ["Sim", "Não"],
        key="nome_limpo"
    )
    
    conta_bancaria = st.radio(
        "Possui conta bancária há mais de 1 ano?",
        ["Sim", "Não"],
        key="conta"
    )
    
    cartao_credito = st.radio(
        "Usa cartão de crédito regularmente sem atrasos?",
        ["Sim", "Não"],
        key="cartao"
    )

# Mais perguntas
st.subheader("Situação Patrimonial e Compromissos")

col3, col4 = st.columns(2)

with col3:
    imovel_proprio = st.radio(
        "Possui imóvel próprio?",
        ["Sim", "Não"],
        key="imovel"
    )
    
    veiculo_proprio = st.radio(
        "Possui veículo próprio?",
        ["Sim", "Não"],
        key="veiculo"
    )

with col4:
    dividas_pendentes = st.radio(
        "Possui dívidas pendentes?",
        ["Sim", "Não"],
        key="dividas"
    )
    
    dependentes = st.radio(
        "Possui dependentes financeiros?",
        ["Sim", "Não"],
        key="dependentes"
    )

# Valor solicitado
st.subheader("💰 Crédito Solicitado")
valor_credito = st.number_input(
    "Valor do crédito solicitado (R$):",
    min_value=100.0,
    max_value=100000.0,
    value=5000.0,
    step=500.0
)

# Botão para análise
if st.button("🔍 Realizar Análise de Crédito", type="primary"):
    
    # Calculando score baseado nas respostas
    score = 300  # Score base
    
    # Pontuação baseada em renda
    if renda >= 5000:
        score += 150
    elif renda >= 3000:
        score += 100
    elif renda >= 1500:
        score += 50
    
    # Pontuação baseada na idade
    if 25 <= idade <= 55:
        score += 50
    elif idade > 55:
        score += 30
    
    # Pontuação baseada nas respostas
    respostas_positivas = {
        emprego_estavel: 80,
        carteira_assinada: 60,
        renda_extra: 40,
        nome_limpo: 100,
        conta_bancaria: 50,
        cartao_credito: 70,
        imovel_proprio: 60,
        veiculo_proprio: 30
    }
    
    respostas_negativas = {
        dividas_pendentes: -80,
        dependentes: -20
    }
    
    for resposta, pontos in respostas_positivas.items():
        if resposta == "Sim":
            score += pontos
    
    for resposta, pontos in respostas_negativas.items():
        if resposta == "Sim":
            score += pontos
    
    # Ajuste baseado no valor solicitado vs renda
    if valor_credito > renda * 5:
        score -= 50
    elif valor_credito > renda * 3:
        score -= 30
    
    # Limitando o score entre 300 e 850
    score = max(300, min(850, score))
    
    # Determinando aprovação
    if score >= 650:
        status = "APROVADO"
        cor_status = "green"
        taxa_juros = max(0.8, 3.5 - (score - 650) / 100)
    elif score >= 500:
        status = "PRÉ-APROVADO"
        cor_status = "orange"
        taxa_juros = max(1.5, 5.0 - (score - 500) / 100)
    else:
        status = "NEGADO"
        cor_status = "red"
        taxa_juros = 0
    
    # Exibindo resultados
    st.markdown("---")
    st.header("📊 Resultado da Análise")
    
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        st.metric(
            "Score de Crédito",
            f"{score}",
            delta=f"{score - 600}" if score > 600 else f"{score - 600}"
        )
    
    with col_result2:
        st.markdown(f"**Status:** <span style='color: {cor_status}; font-size: 24px;'>{status}</span>", 
                   unsafe_allow_html=True)
    
    with col_result3:
        if status != "NEGADO":
            st.metric("Taxa de Juros", f"{taxa_juros:.1f}% a.m.")
    
    # Gráfico do score
    fig_score = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score de Crédito"},
        delta = {'reference': 600},
        gauge = {
            'axis': {'range': [None, 850]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [300, 500], 'color': "lightgray"},
                {'range': [500, 650], 'color': "yellow"},
                {'range': [650, 850], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 600
            }
        }
    ))
    
    fig_score.update_layout(height=400)
    st.plotly_chart(fig_score, use_container_width=True)
    
    # Detalhamento da análise
    st.subheader("📋 Detalhamento da Análise")
    
    # Fatores positivos e negativos
    col_pos, col_neg = st.columns(2)
    
    with col_pos:
        st.success("**Fatores Positivos:**")
        fatores_pos = []
        if emprego_estavel == "Sim":
            fatores_pos.append("• Emprego estável")
        if nome_limpo == "Sim":
            fatores_pos.append("• Nome limpo")
        if renda >= 3000:
            fatores_pos.append("• Boa renda mensal")
        if cartao_credito == "Sim":
            fatores_pos.append("• Bom histórico com cartão")
        if imovel_proprio == "Sim":
            fatores_pos.append("• Possui imóvel próprio")
        
        for fator in fatores_pos:
            st.write(fator)
    
    with col_neg:
        st.error("**Pontos de Atenção:**")
        fatores_neg = []
        if dividas_pendentes == "Sim":
            fatores_neg.append("• Possui dívidas pendentes")
        if nome_limpo == "Não":
            fatores_neg.append("• Nome negativado")
        if emprego_estavel == "Não":
            fatores_neg.append("• Emprego instável")
        if valor_credito > renda * 3:
            fatores_neg.append("• Valor alto vs renda")
        
        for fator in fatores_neg:
            st.write(fator)
        
        if not fatores_neg:
            st.write("• Nenhum ponto crítico identificado")
    
    # Simulação de parcelas (se aprovado)
    if status != "NEGADO":
        st.subheader("💳 Simulação de Financiamento")
        
        parcelas_opcoes = [12, 24, 36, 48, 60]
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.write("**Opções de Parcelamento:**")
            for parcelas in parcelas_opcoes:
                valor_parcela = (valor_credito * (1 + taxa_juros/100) ** (parcelas/12)) / parcelas
                st.write(f"• {parcelas}x de R$ {valor_parcela:.2f}")
        
        with col_sim2:
            # Gráfico de evolução do valor
            meses = list(range(1, 61))
            valores = [(valor_credito * (1 + taxa_juros/100) ** (m/12)) / m for m in meses]
            
            fig_parcelas = px.line(
                x=meses, 
                y=valores,
                title="Valor da Parcela por Prazo",
                labels={'x': 'Número de Parcelas', 'y': 'Valor da Parcela (R$)'}
            )
            st.plotly_chart(fig_parcelas, use_container_width=True)
    
    # Recomendações
    st.subheader("💡 Recomendações")
    
    if score < 600:
        st.warning("""
        **Para melhorar seu score:**
        - Quite as dívidas pendentes
        - Mantenha seu nome limpo
        - Use produtos bancários com responsabilidade
        - Comprove renda estável
        """)
    else:
        st.success("""
        **Parabéns! Seu perfil está aprovado.**
        - Mantenha seus dados sempre atualizados
        - Continue honrando seus compromissos
        - Considere produtos de relacionamento
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    💳 Plataforma de Análise de Crédito | Desenvolvido com Streamlit<br>
    ⚠️ Esta é uma simulação para fins educacionais
</div>
""", unsafe_allow_html=True)