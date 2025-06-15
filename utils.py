import streamlit as st

def leitor_de_texto(texto: str):
    """
    Botão para leitura do texto informado usando a Web Speech API do navegador.
    """
    st.subheader("🔈 Acessibilidade - Leitura do conteúdo")

    st.markdown(f"""
    <button onclick="lerTexto()">🔊 Clique para ouvir o conteúdo da página</button>

    <script>
    function lerTexto() {{
      const texto = `{texto}`;
      const utterance = new SpeechSynthesisUtterance(texto);
      utterance.lang = "pt-BR";
      speechSynthesis.cancel();
      speechSynthesis.speak(utterance);
    }}
    </script>
    """, unsafe_allow_html=True)
