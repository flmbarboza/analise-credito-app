# app.py
import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Risco de Crédito Academy",
    page_icon="🏦",
    layout="centered"  # Layout mais clean para a página inicial
)

# Mensagem de boas-vindas (aparece apenas na root URL)
if not st.session_state.get('redirecionado'):
    st.session_state.redirecionado = True
    
    st.title("🏦 Bem-vindo ao Sistema de Análise de Crédito!")
    st.markdown("""
    ## 👋 Olá, Analista de Risco!
    
    Esta plataforma foi desenvolvida para auxiliar na tomada de decisão de crédito
    através de técnicas avançadas de análise de dados.
    
    ### Como começar?
    1. Clique em **🏠 Home** no menu lateral
    2. Siga o fluxo de análise proposto
    3. Explore as ferramentas interativas
    
    """)
    
    st.image("https://i.imgur.com/JQH90yl.png", width=300)  # Imagem ilustrativa
    
    if st.button("➡️ Ir para a Página Inicial", type="primary"):
        st.switch_page("pages/1_🏠_Home.py")
    
    st.stop()  # Impede a execução do resto do código

# Menu principal (só aparece após redirecionamento)
st.sidebar.title("🏦 Menu Principal")
pagina_selecionada = st.sidebar.selectbox(
    "Escolha uma opção:",
    [
        "🏠 Página Inicial",
        "📝 Planejamento", 
        "📊 Coleta de Dados",
        "📈 Análise Univariada",
        "📉 Análise Bivariada",
        "🤖 Modelagem",
        "✅ Validação",
        "⚙️ Aperfeiçoamento",
        "🏛️ Políticas de Crédito",
        "📑 Relatório"
    ]
)

# Footer (aparece em todas as páginas)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Sistema de Análise de Crédito v2.0<br>
    Desenvolvido para a disciplina de Risco de Crédito
</div>
""", unsafe_allow_html=True)

# Roteamento das páginas (
