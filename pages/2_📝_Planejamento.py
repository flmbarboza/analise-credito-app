# pages/1_📝_Planejamento.py
import streamlit as st
import time

def mostrar_texto_pausado(texto, velocidade=0.03):
    """Exibe texto letra por letra com efeito de digitação"""
    placeholder = st.empty()
    texto_completo = ""
    for char in texto:
        texto_completo += char
        placeholder.markdown(texto_completo + "▌")
        time.sleep(velocidade)
    placeholder.markdown(texto_completo)

def seção_interativa(titulo, conteudo, quiz=None, opcoes_expansão=None):
    with st.expander(f"📌 {titulo}", expanded=True):
        if isinstance(conteudo, str):
            mostrar_texto_pausado(conteudo)
        else:
            for item in conteudo:
                st.write(item)
                
        if quiz:
            resposta = st.radio(
                quiz["pergunta"],
                options=quiz["opcoes"],
                index=None
            )
            if resposta:
                if resposta == quiz["resposta_correta"]:
                    st.success("✅ Correto! " + quiz["feedback_positivo"])
                else:
                    st.error("❌ Tente novamente. " + quiz["feedback_negativo"])
        
        if opcoes_expansão:
            opcao = st.selectbox(
                "O que você gostaria de explorar sobre este tópico?",
                ["Selecione..."] + list(opcoes_expansão.keys())
            )
            if opcao != "Selecione...":
                st.info(opcoes_expansão[opcao])

def main():
    st.title("📝 Planejamento: Fundamentos de Credit Scoring")
    
    # Menu de navegação interno
    st.sidebar.header("Navegação do Capítulo")
    topico = st.sidebar.radio(
        "Ir para:",
        ["Introdução", "Conceitos Básicos", "Modelagem", "Roteiro Prático"]
    )
    
    if topico == "Introdução":
        st.header("🧭 Introdução ao Risco de Crédito")
        
        seção_interativa(
            "O que é Risco de Crédito?",
            "Risco de crédito é a possibilidade de uma parte não cumprir suas obrigações financeiras conforme acordado. É como avaliar 'quão arriscado' é emprestar dinheiro para alguém.",
            quiz={
                "pergunta": "Qual destes NÃO é um exemplo de risco de crédito?",
                "opcoes": [
                    "Atraso no pagamento de um empréstimo",
                    "Variação nas taxas de juros do mercado",
                    "Inadimplência em cartão de crédito"
                ],
                "resposta_correta": "Variação nas taxas de juros do mercado",
                "feedback_positivo": "Exato! Variação de juros é risco de mercado, não de crédito.",
                "feedback_negativo": "Essa opção refere-se a outro tipo de risco financeiro."
            }
        )
        
        seção_interativa(
            "Por que medir o Risco de Crédito?",
            [
                "✔️ Reduz perdas financeiras para instituições",
                "✔️ Oferece taxas justas conforme o perfil do cliente",
                "✔️ Permite acesso mais democrático ao crédito",
                "✔️ Aumenta a estabilidade do sistema financeiro"
            ],
            opcoes_expansão={
                "Exemplo Prático": "Bancos usam scoring para definir limites de cartão: clientes com baixo risco recebem limites maiores e taxas menores.",
                "Impacto Social": "Um bom sistema de scoring pode incluir pessoas sem histórico creditício tradicional."
            }
        )
    
    elif topico == "Conceitos Básicos":
        st.header("📊 Conceitos Fundamentais")
        
        seção_interativa(
            "Credit Scores: Seu Número Mágico",
            "Um credit score é como uma nota que resume seu risco de crédito. Varia tipicamente de 0 a 1000 - quanto maior, melhor!",
            quiz={
                "pergunta": "Qual fator geralmente NÃO afeta seu credit score?",
                "opcoes": [
                    "Histórico de pagamentos",
                    "Cor do seu cartão de crédito",
                    "Endividamento atual"
                ],
                "resposta_correta": "Cor do seu cartão de crédito",
                "feedback_positivo": "Isso mesmo! Características físicas não influenciam.",
                "feedback_negativo": "Reveja os fatores que compõem um score tradicional."
            }
        )
        
        seção_interativa(
            "Probabilidade e Erros de Decisão",
            "Nenhum modelo é perfeito. Sempre há:\n\n"
            "- Falsos positivos: clientes 'bons' classificados como ruins\n"
            "- Falsos negativos: clientes 'ruins' classificados como bons\n\n"
            "O desafio é balancear esses erros."
        )
    
    elif topico == "Modelagem":
        st.header("🤖 Como os Modelos Funcionam?")
        
        with st.expander("Premissas Básicas", expanded=True):
            st.markdown("""
            **1. Padronização:** Todos os clientes são avaliados pelos mesmos critérios  
            **2. Objetividade:** Decisões baseadas apenas em dados, não em opiniões  
            **3. Atualização:** Modelos são revisados periodicamente  
            """)
            st.image("https://i.imgur.com/JQH90yl.png", width=300)
        
        with st.expander("Cálculo dos Escores", expanded=False):
            st.latex(r'''
            Score = 600 + 
            \begin{cases} 
            50 \times \text{(idade > 30)} \\
            -30 \times \text{(dívidas > renda)} \\
            20 \times \text{(tem conta bancária)}
            \end{cases}
            ''')
            st.caption("Exemplo simplificado de como variáveis são ponderadas")
    
    elif topico == "Roteiro Prático":
        st.header("🗺️ Roteiro para Desenvolver um Modelo")
        
        passos = st.session_state.get("passos", [False]*6)
        
        with st.form("roteiro_form"):
            st.write("**Marque os passos já completados:**")
            passos[0] = st.checkbox("1. Coleta de dados históricos", value=passos[0])
            passos[1] = st.checkbox("2. Análise exploratória", value=passos[1])
            passos[2] = st.checkbox("3. Seleção de variáveis", value=passos[2])
            passos[3] = st.checkbox("4. Desenvolvimento do modelo", value=passos[3])
            passos[4] = st.checkbox("5. Validação", value=passos[4])
            passos[5] = st.checkbox("6. Implementação", value=passos[5])
            
            if st.form_submit_button("Salvar Progresso"):
                st.session_state.passos = passos
                st.success("Progresso atualizado!")
        
        progresso = sum(passos)/len(passos)
        st.progress(progresso)
        st.caption(f"Você completou {int(progresso*100)}% do roteiro")
    # 🚀 Link para a próxima página
    st.page_link("pages/3_🚀_Coleta_de_Dados.py", label="➡️ Ir para a próxima página: Coleta de Dados", icon="📝")


if __name__ == "__main__":
    main()
