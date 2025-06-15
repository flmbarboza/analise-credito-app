import streamlit as st

def main():
    st.title("🚀 Desafio da Disciplina: Risco de Crédito e Credit Scoring")

    st.subheader("🕹️ Bora começar? Você precisa desbloquear o desafio.")
    
    # Primeira interação - botão para revelar a introdução
    if st.button("🔓 Clique para desbloquear o primeiro desafio"):
        st.markdown("""
        ## 🚩 **Desafio lançado: Você aprovaria esse crédito?**  
        Imagine que você trabalha no setor financeiro de uma empresa, de um banco ou de uma fintech.  
        Um cliente chega solicitando crédito. A proposta parece boa…  
        **Mas e se ele não pagar?** Quem arca com esse prejuízo?  

        Como separar quem é bom pagador de quem traz risco real?  
        E mais: como fazer isso de forma **rápida, precisa e baseada em dados?**  

        **Essa não é só uma pergunta acadêmica.** É uma decisão que acontece **todos os dias em milhares de empresas, bancos e plataformas digitais.**  
        E quem sabe fazer isso bem, **domina uma das habilidades mais valorizadas no mercado.**  
        """)

        st.divider()

        # Segunda interação - caixas expansíveis com conteúdos
        with st.expander("🔥 Clique aqui para descobrir como vamos trabalhar"):
            st.markdown("""
            ## 🔥 **Aqui, a sala vira uma empresa de crédito.**  
            Todos vocês fazem parte de uma grande empresa simulada.  
            O nosso trabalho, a partir de hoje, é **construir juntos um modelo de análise de risco de crédito**, capaz de responder:  
            - Quem merece crédito?  
            - Quanto vale o risco?  
            - Como transformar dados em decisões que geram lucro — e evitam prejuízo?  

            **Todos terão os mesmos dados.**  
            **Todos enfrentarão os mesmos desafios.**  
            **Todos irão desenvolver, testar, errar e melhorar… juntos.**  
            """)

        with st.expander("🚀 Como será o jogo?"):
            st.markdown("""
            ## 🚀 **O jogo é real.**  
            Cada etapa da disciplina será uma fase desse desafio:  
            1️⃣ Entender como funciona uma decisão de crédito.  
            2️⃣ Escolher as variáveis que realmente importam.  
            3️⃣ Analisar dados — e descobrir padrões que ninguém vê.  
            4️⃣ Construir e testar modelos de scoring.  
            5️⃣ Validar, ajustar e, no final, **implantar o modelo vencedor da turma.**  
            """)

        with st.expander("🎯 O que você leva disso?"):
            st.markdown("""
            ## 🎯 **O que você leva disso?**  
            - Uma habilidade que o mercado paga muito bem.  
            - Capacidade real de transformar dados em decisão.  
            - Um raciocínio mais analítico, mais lógico e mais preparado pra qualquer área da gestão — não só finanças.  

            👉 **Se você acha que essa disciplina é só mais uma… prepare-se para se surpreender.**  
            """)

        st.divider()

        # Terceira interação - mini quiz reflexivo
        st.subheader("🧠 Mini Desafio Rápido")

        resposta = st.radio(
            "Por que empresas se preocupam tanto em analisar risco de crédito?",
            [
                "Porque é uma exigência legal apenas.",
                "Porque precisam proteger seu dinheiro e tomar melhores decisões.",
                "Porque é uma formalidade burocrática sem impacto real.",
                "Porque é uma moda recente trazida pela tecnologia."
            ]
        )

        if resposta:
            if resposta == "Porque precisam proteger seu dinheiro e tomar melhores decisões.":
                st.success("✅ Perfeito! Você já entendeu o ponto central da disciplina!")
            else:
                st.error("❌ Não exatamente... Tente pensar no impacto de inadimplência para qualquer negócio.")

    else:
        st.info("👆 Clique no botão acima para desbloquear o desafio e começar a jornada!")
        

if __name__ == "__main__":
    main()
