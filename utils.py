import streamlit as st
import pandas as pd
import pickle
import os

DATA_FILE = "session_data.pkl"

def save_session():
    """Salva o session_state e os dados em disco."""
    if 'dados' in st.session_state:
        data_to_save = {
            'dados': st.session_state.dados,
            'target': st.session_state.get('target'),
            'variaveis_ativas': st.session_state.get('variaveis_ativas'),
            'modelo': st.session_state.get('modelo'),
            'iv_df': st.session_state.get('iv_df'),
            'ks_df': st.session_state.get('ks_df'),
            # Adicione outros itens importantes
        }
        with open(DATA_FILE, "wb") as f:
            pickle.dump(data_to_save, f)

def load_session():
    """Carrega o session_state salvo do disco."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
        return data
    return {}
   
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
