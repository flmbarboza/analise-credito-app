import streamlit as st

def leitor_de_texto(texto: str):
    texto_js = texto.replace("\n", " ").replace("`", "'").replace('"', "'")  # Sanitiza texto para JS
    st.subheader("🔈 Acessibilidade - Leitura do conteúdo")

    st.markdown(f"""
    <button id="btnAudio" style="padding:10px; font-size:16px;">🔊 Clique para ouvir o conteúdo da página</button>

    <script>
    const btn = document.getElementById('btnAudio');
    btn.addEventListener('click', () => {{
        let texto = `{texto_js}`;
        if ('speechSynthesis' in window) {{
            speechSynthesis.cancel();  // Para qualquer fala atual
            let utterance = new SpeechSynthesisUtterance(texto);
            utterance.lang = 'pt-BR';
            speechSynthesis.speak(utterance);
        }} else {{
            alert('Desculpe, seu navegador não suporta síntese de voz.');
        }}
    }});
    </script>
    """, unsafe_allow_html=True)
