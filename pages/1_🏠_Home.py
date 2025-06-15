import streamlit as st

def main():
    st.title("🚀 Desafio da Disciplina: Risco de Crédito e Credit Scoring")
    
    st.subheader("🕹️ Bora começar? Você precisa desbloquear o desafio.")

    # ✅ Controle de estado
    if 'desafio_desbloqueado' not in st.session_state:
        st.session_state.desafio_desbloqueado = False

    if st.button("🔓 Clique para desbloquear o primeiro desafio"):
        st.session_state.desafio_desbloqueado = True

    if st.session_state.desafio_desbloqueado:
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

        st.video("https://www.youtube.com/watch?v=8jzvzRo3Ij0")

        st.divider()

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
            - Uma habilidade que o mercado paga muito bem. De acordo com o site Glassdoor ([clique aqui](https://www.glassdoor.com.br/Sal%C3%A1rios/credit-risk-manager-sal%C3%A1rio-SRCH_KO0,19.htm)), um gestor de risco de crédito ganha entre **R\$ 200 mil e R\$ 400 mil**, além de bonificações.  
            - Capacidade real de transformar dados em decisão.  
            - Um raciocínio mais analítico, mais lógico e mais preparado pra qualquer área da gestão — não só finanças.  

            👉 **Se você acha que essa disciplina é só mais uma… prepare-se para se surpreender.**  
            """)

        st.divider()

        if 'resposta_desafio' not in st.session_state:
            st.session_state.resposta_desafio = None

        with st.expander("🧠 Mini Desafio Rápido"):
            st.subheader("💡 Responda antes de avançar:")

            resposta = st.radio(
                "Por que empresas se preocupam tanto em analisar risco de crédito?",
                [
                    "Porque é uma exigência legal apenas.",
                    "Porque precisam proteger seu dinheiro e tomar melhores decisões.",
                    "Porque é uma formalidade burocrática sem impacto real.",
                    "Porque é uma moda recente trazida pela tecnologia."
                ],
                index=None,
                key="radio_resposta"
            )

            if resposta:
                st.session_state.resposta_desafio = resposta

            if st.session_state.resposta_desafio:
                if st.session_state.resposta_desafio == "Porque precisam proteger seu dinheiro e tomar melhores decisões.":
                    st.success("✅ Perfeito! Você já entendeu o ponto central da disciplina!")
                else:
                    st.error("❌ Não exatamente... Tente pensar no impacto de inadimplência para qualquer negócio.")

        st.divider()

        # 🚀 Link para a próxima página
        st.page_link("pages/2_📝_Planejamento.py", label="➡️ Ir para a próxima página: Planejamento", icon="📝")

    else:
        st.info("👆 Clique no botão acima para desbloquear o desafio e começar a jornada!")

if __name__ == "__main__":
    main()
