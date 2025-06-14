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

# Menu principal no sidebar
st.sidebar.title("🏦 Menu Principal")
pagina_selecionada = st.sidebar.selectbox(
    "Escolha uma opção:",
    [
        "🏠 Página Inicial",
        "💳 Análise de Crédito",
        "📊 Dashboard Financeiro",
        "📈 Simulador de Investimentos",
        "🎯 Planejamento Financeiro",
        "📚 Educação Financeira",
        "⚙️ Configurações"
    ]
)

# ===== PÁGINA INICIAL =====
def pagina_inicial():
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

# ===== ANÁLISE DE CRÉDITO (página original) =====
def analise_credito():
    st.title("💳 Análise de Crédito Inteligente")
    st.markdown("### Análise inteligente de risco de crédito baseada em suas respostas")

    # Informações do usuário
    st.subheader("📋 Informações do Solicitante")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        nome = st.text_input("Nome completo:")
        idade = st.slider("Idade:", 18, 80, 30)
    
    with col_info2:
        renda = st.number_input("Renda mensal (R$):", min_value=0.0, value=3000.0, step=100.0)
        estado_civil = st.selectbox("Estado Civil:", ["Solteiro(a)", "Casado(a)", "Divorciado(a)", "Viúvo(a)"])

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
        
        # [Código da análise permanece o mesmo...]
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
        
        # [Resto do código da análise...]

# ===== DASHBOARD FINANCEIRO =====
def dashboard_financeiro():
    st.title("📊 Dashboard Financeiro")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Saldo Total", "R$ 12.450", "+R$ 850")
    with col2:
        st.metric("Gastos do Mês", "R$ 3.200", "-R$ 150")
    with col3:
        st.metric("Economia", "R$ 1.800", "+R$ 200")
    with col4:
        st.metric("Investimentos", "R$ 8.500", "+R$ 350")
    
    # Gráficos
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Gráfico de gastos por categoria
        categorias = ['Alimentação', 'Transporte', 'Lazer', 'Saúde', 'Educação']
        valores = [800, 450, 300, 200, 150]
        
        fig_pizza = px.pie(
            values=valores, 
            names=categorias,
            title="Gastos por Categoria"
        )
        st.plotly_chart(fig_pizza, use_container_width=True)
    
    with col_chart2:
        # Gráfico de evolução mensal
        meses = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        receitas = [4000, 4200, 3800, 4500, 4100, 4300]
        gastos = [3200, 3100, 2900, 3400, 3000, 3200]
        
        fig_linha = go.Figure()
        fig_linha.add_trace(go.Scatter(x=meses, y=receitas, name='Receitas'))
        fig_linha.add_trace(go.Scatter(x=meses, y=gastos, name='Gastos'))
        fig_linha.update_layout(title="Evolução Financeira")
        
        st.plotly_chart(fig_linha, use_container_width=True)
    
    # Tabela de transações recentes
    st.subheader("💸 Transações Recentes")
    
    transacoes = pd.DataFrame({
        'Data': ['2024-06-14', '2024-06-13', '2024-06-12', '2024-06-11'],
        'Descrição': ['Supermercado XYZ', 'Combustível', 'Netflix', 'Salário'],
        'Categoria': ['Alimentação', 'Transporte', 'Lazer', 'Receita'],
        'Valor': [-250.80, -120.00, -29.90, 4500.00]
    })
    
    st.dataframe(transacoes, use_container_width=True)

# ===== SIMULADOR DE INVESTIMENTOS =====
def simulador_investimentos():
    st.title("📈 Simulador de Investimentos")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        st.subheader("Parâmetros da Simulação")
        
        valor_inicial = st.number_input("Valor inicial (R$):", min_value=100.0, value=1000.0)
        aporte_mensal = st.number_input("Aporte mensal (R$):", min_value=0.0, value=200.0)
        taxa_anual = st.slider("Taxa de juros anual (%):", 1.0, 20.0, 8.0, 0.5)
        periodo_anos = st.slider("Período (anos):", 1, 30, 10)
        
        tipo_investimento = st.selectbox(
            "Tipo de investimento:",
            ["Poupança", "CDB", "Tesouro Direto", "Ações", "Fundos"]
        )
    
    with col_sim2:
        # Simulação
        meses = periodo_anos * 12
        taxa_mensal = (1 + taxa_anual/100) ** (1/12) - 1
        
        valores = [valor_inicial]
        for mes in range(1, meses + 1):
            valor_anterior = valores[-1]
            novo_valor = (valor_anterior + aporte_mensal) * (1 + taxa_mensal)
            valores.append(novo_valor)
        
        # Gráfico de crescimento
        meses_lista = list(range(0, meses + 1))
        
        fig_investimento = px.line(
            x=meses_lista,
            y=valores,
            title=f"Projeção de Investimento - {tipo_investimento}",
            labels={'x': 'Meses', 'y': 'Valor (R$)'}
        )
        st.plotly_chart(fig_investimento, use_container_width=True)
        
        # Resultados
        valor_final = valores[-1]
        valor_investido = valor_inicial + (aporte_mensal * meses)
        rendimento = valor_final - valor_investido
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("Valor Final", f"R$ {valor_final:,.2f}")
        with col_res2:
            st.metric("Total Investido", f"R$ {valor_investido:,.2f}")
        with col_res3:
            st.metric("Rendimento", f"R$ {rendimento:,.2f}")

# ===== PLANEJAMENTO FINANCEIRO =====
def planejamento_financeiro():
    st.title("🎯 Planejamento Financeiro")
    
    st.subheader("💰 Definir Metas Financeiras")
    
    tab1, tab2, tab3 = st.tabs(["Metas de Curto Prazo", "Metas de Médio Prazo", "Metas de Longo Prazo"])
    
    with tab1:
        st.write("### Objetivos para os próximos 2 anos")
        
        col1, col2 = st.columns(2)
        with col1:
            meta_curto = st.text_input("Descrição da meta:", placeholder="Ex: Comprar um carro")
            valor_meta_curto = st.number_input("Valor necessário (R$):", min_value=0.0, value=30000.0)
        
        with col2:
            prazo_curto = st.slider("Prazo (meses):", 1, 24, 12)
            valor_mensal_curto = valor_meta_curto / prazo_curto if prazo_curto > 0 else 0
            st.metric("Valor mensal necessário", f"R$ {valor_mensal_curto:.2f}")
    
    with tab2:
        st.write("### Objetivos para 2-10 anos")
        
        col1, col2 = st.columns(2)
        with col1:
            meta_medio = st.text_input("Descrição da meta:", placeholder="Ex: Comprar um imóvel", key="meta_medio")
            valor_meta_medio = st.number_input("Valor necessário (R$):", min_value=0.0, value=200000.0, key="valor_medio")
        
        with col2:
            prazo_medio = st.slider("Prazo (anos):", 2, 10, 5, key="prazo_medio")
            valor_mensal_medio = valor_meta_medio / (prazo_medio * 12) if prazo_medio > 0 else 0
            st.metric("Valor mensal necessário", f"R$ {valor_mensal_medio:.2f}")
    
    with tab3:
        st.write("### Objetivos para aposentadoria")
        
        col1, col2 = st.columns(2)
        with col1:
            idade_atual = st.number_input("Idade atual:", 18, 65, 30)
            idade_aposentadoria = st.number_input("Idade para aposentar:", idade_atual + 1, 80, 60)
            
        with col2:
            renda_desejada = st.number_input("Renda mensal desejada (R$):", min_value=1000.0, value=5000.0)
            anos_restantes = idade_aposentadoria - idade_atual
            
            # Cálculo simplificado
            valor_total_necessario = renda_desejada * 12 * 25  # 25 anos de aposentadoria
            valor_mensal_aposentadoria = valor_total_necessario / (anos_restantes * 12) if anos_restantes > 0 else 0
            
            st.metric("Anos restantes", f"{anos_restantes} anos")
            st.metric("Valor mensal necessário", f"R$ {valor_mensal_aposentadoria:.2f}")

# ===== EDUCAÇÃO FINANCEIRA =====
def educacao_financeira():
    st.title("📚 Educação Financeira")
    
    # Dicas financeiras
    st.subheader("💡 Dicas de Educação Financeira")
    
    dicas = [
        {
            "titulo": "📝 Controle seus gastos",
            "conteudo": "Anote todos os seus gastos durante um mês para entender para onde vai seu dinheiro."
        },
        {
            "titulo": "🎯 Estabeleça metas",
            "conteudo": "Defina objetivos claros e prazos realistas para suas conquistas financeiras."
        },
        {
            "titulo": "🏦 Construa uma reserva de emergência",
            "conteudo": "Mantenha de 3 a 6 meses de gastos guardados para imprevistos."
        },
        {
            "titulo": "📈 Invista regularmente",
            "conteudo": "Mesmo pequenos valores investidos mensalmente fazem diferença no longo prazo."
        }
    ]
    
    for dica in dicas:
        with st.expander(dica["titulo"]):
            st.write(dica["conteudo"])
    
    # Calculadora de juros compostos
    st.subheader("🧮 Calculadora de Juros Compostos")
    
    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
        principal = st.number_input("Capital inicial (R$):", value=1000.0, min_value=0.0)
        taxa = st.number_input("Taxa de juros anual (%):", value=10.0, min_value=0.0)
        tempo = st.number_input("Tempo (anos):", value=5, min_value=1)
    
    with col_calc2:
        montante = principal * (1 + taxa/100) ** tempo
        juros = montante - principal
        
        st.metric("Montante final", f"R$ {montante:,.2f}")
        st.metric("Juros ganhos", f"R$ {juros:,.2f}")
        st.metric("Crescimento", f"{((montante/principal - 1) * 100):.1f}%")

# ===== CONFIGURAÇÕES =====
def configuracoes():
    st.title("⚙️ Configurações")
    
    st.subheader("👤 Perfil do Usuário")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        nome_usuario = st.text_input("Nome:", value="João Silva")
        email_usuario = st.text_input("E-mail:", value="joao@email.com")
        telefone_usuario = st.text_input("Telefone:", value="(11) 99999-9999")
    
    with col_config2:
        notificacoes = st.checkbox("Receber notificações por e-mail", value=True)
        newsletter = st.checkbox("Receber newsletter semanal", value=False)
        modo_escuro = st.checkbox("Modo escuro", value=False)
    
    st.subheader("🔐 Segurança")
    
    if st.button("Alterar senha"):
        st.info("Funcionalidade em desenvolvimento")
    
    if st.button("Salvar configurações", type="primary"):
        st.success("Configurações salvas com sucesso!")

# ===== ROTEAMENTO DAS PÁGINAS =====
if pagina_selecionada == "🏠 Página Inicial":
    pagina_inicial()
elif pagina_selecionada == "💳 Análise de Crédito":
    analise_credito()
elif pagina_selecionada == "📊 Dashboard Financeiro":
    dashboard_financeiro()
elif pagina_selecionada == "📈 Simulador de Investimentos":
    simulador_investimentos()
elif pagina_selecionada == "🎯 Planejamento Financeiro":
    planejamento_financeiro()
elif pagina_selecionada == "📚 Educação Financeira":
    educacao_financeira()
elif pagina_selecionada == "⚙️ Configurações":
    configuracoes()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    💳 Plataforma Financeira Completa<br>
    Versão 2.0
</div>
""", unsafe_allow_html=True)
