import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import ks_2samp
import io
import zipfile
import base64
from datetime import datetime

def calcular_ks(bons, maus):
    """Calcula KS entre bons (0) e maus (1)."""
    if len(bons) == 0 or len(maus) == 0:
        return np.nan
    ks_stat, _ = ks_2samp(bons, maus)
    return ks_stat

def main():
    st.title("📊 Implantação do Credit Scoring")
    st.markdown("""
    Teste o modelo com **novos dados** e valide sua performance na prática.  
    Esta é a etapa final antes da produção.
    """)

    # --- 1. VALIDAÇÃO: Modelo disponível ---
    if 'modelo' not in st.session_state:
        st.warning("Nenhum modelo treinado! Construa um modelo primeiro.")
        st.page_link("pages/6_🤖_Modelagem.py", label="→ Ir para Modelagem", icon="🤖")
        return

    model = st.session_state.modelo
    target = st.session_state.get('target', None)
    threshold = st.session_state.get('threshold', 0.5)  # Limiar definido no aperfeiçoamento

    if not target:
        st.error("Variável-alvo não definida. Volte para a Análise Bivariada.")
        return

    # --- 2. UPLOAD DE NOVOS DADOS ---
    st.markdown("### 📥 Carregar Nova Amostra de Teste")
    st.info("""
    Carregue um conjunto de dados **diferente daquele usado no treinamento** para simular uma implantação real.  
    Os dados devem conter as mesmas variáveis preditoras e a coluna-alvo.
    """)

    uploaded_file = st.file_uploader("Escolha um arquivo CSV com novos dados:", type="csv")

    if not uploaded_file:
        st.info("Aguardando upload de dados...")
        return

    try:
        novos_dados = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return

    if target not in novos_dados.columns:
        st.error(f"A coluna-alvo `{target}` não está presente nos novos dados.")
        return

    # Verifica se as features do modelo estão presentes
    features = st.session_state.get('features', [])
    if not features:
        st.error("Nenhuma lista de variáveis preditoras disponível. Treine o modelo novamente.")
        return

    missing_features = [f for f in features if f not in novos_dados.columns]
    if missing_features:
        st.error(f"Variáveis ausentes nos novos dados: {missing_features}")
        return

    X_novo = novos_dados[features]
    y_novo = novos_dados[target]

    # --- 3. APLICAÇÃO DO MODELO ---
    st.markdown("### 🔧 Aplicando o Modelo aos Novos Dados")
    try:
        y_proba_novo = model.predict_proba(X_novo)[:, 1]
        y_pred_novo = (y_proba_novo >= threshold).astype(int)
    except Exception as e:
        st.error(f"Erro ao aplicar o modelo: {e}")
        return

    st.success("✅ Modelo aplicado com sucesso aos novos dados!")

    # --- 4. APURAÇÃO DE INDICADORES ---
    st.markdown("### 📊 Apuração de Indicadores")

    # Métricas
    acc = accuracy_score(y_novo, y_pred_novo)
    prec = precision_score(y_novo, y_pred_novo)
    rec = recall_score(y_novo, y_pred_novo)
    f1 = f1_score(y_novo, y_pred_novo)

    # KS
    bons_proba = y_proba_novo[y_novo == 0]
    maus_proba = y_proba_novo[y_novo == 1]
    ks = calcular_ks(bons_proba, maus_proba)

    # Matriz de confusão
    cm = confusion_matrix(y_novo, y_pred_novo)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Adimplente', 'Inadimplente'],
                yticklabels=['Adimplente', 'Inadimplente'])
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusão (Novos Dados)')
    st.pyplot(fig)

    # Exibe métricas
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acurácia", f"{acc:.1%}")
    col2.metric("Precision", f"{prec:.1%}")
    col3.metric("Recall", f"{rec:.1%}")
    col4.metric("F1-Score", f"{f1:.1%}")
    col5.metric("KS", f"{ks:.2f}")

    # --- 5. VALIDAÇÃO DAS POLÍTICAS DE CRÉDITO ---
    st.markdown("### 🏛️ Validação das Políticas de Crédito")
    st.info("Verifique se o modelo atende às políticas definidas anteriormente.")

    # Recupera políticas simuladas (ex: do session_state)
    politicas_esperadas = {
        "Precision mínima": 0.7,
        "Recall mínimo": 0.7,
        "KS mínimo": 0.3,
        "Taxa de aprovação esperada": 0.6  # pode vir de st.session_state.policy_aprovacao
    }

    cumpridas = []
    nao_cumpridas = []

    if prec >= politicas_esperadas["Precision mínima"]:
        cumpridas.append(f"✅ Precision ≥ {politicas_esperadas['Precision mínima']:.0%}")
    else:
        nao_cumpridas.append(f"❌ Precision < {politicas_esperadas['Precision mínima']:.0%}")

    if rec >= politicas_esperadas["Recall mínimo"]:
        cumpridas.append(f"✅ Recall ≥ {politicas_esperadas['Recall mínimo']:.0%}")
    else:
        nao_cumpridas.append(f"❌ Recall < {politicas_esperadas['Recall mínimo']:.0%}")

    if ks >= politicas_esperadas["KS mínimo"]:
        cumpridas.append(f"✅ KS ≥ {politicas_esperadas['KS mínimo']:.1f}")
    else:
        nao_cumpridas.append(f"❌ KS < {politicas_esperadas['KS mínimo']:.1f}")

    taxa_aprovacao = y_pred_novo.mean()
    if taxa_aprovacao >= politicas_esperadas["Taxa de aprovação esperada"] * 0.9:  # margem de 10%
        cumpridas.append(f"✅ Taxa de aprovação dentro do esperado (~{politicas_esperadas['Taxa de aprovação esperada']:.0%})")
    else:
        nao_cumpridas.append(f"❌ Taxa de aprovação fora do esperado")

    for item in cumpridas:
        st.markdown(item)
    for item in nao_cumpridas:
        st.markdown(item)

    with st.expander("🔍 Concluiu o teste final?"):
        st.markdown("""
        - **Novos dados**: Testou se realmente a sua Análise de Crédito funciona na prática? 
        - **Apurar resultados**: Os indicadores apresentam números similares ao teste?  
        - **Concluir análise**: Agora é só emitir um documento que sintetize sua proposta, demonstrando claramente os pontos fortes e fracos. 
        """)
    
        st.markdown("""
            Encerrada a jornada, vamos a conclusão do documento que contém todas as partes do processo em detalhes.
        """)

    # --- 6. RELATÓRIO DE IMPLANTAÇÃO ---
    with st.expander("📄 Gerar Relatório de Implantação", expanded=False):
        st.markdown("### 📝 Resumo da validação com novos dados")

        relatorio = f"""
RELATÓRIO DE IMPLANTAÇÃO DO MODELO
==================================

🎯 **Modelo implantado:** {st.session_state.get('modelo_tipo', 'Desconhecido')}
🎯 **Variável-alvo:** {target}
🎯 **Limiar de decisão:** {threshold:.2f}
🎯 **Fonte dos dados:** Upload do usuário

📊 **DESEMPENHO EM NOVOS DADOS**
--------------------------------
Acurácia: {acc:.1%}
Precision: {prec:.1%}
Recall: {rec:.1%}
F1-Score: {f1:.1%}
KS: {ks:.2f}

📋 **POLÍTICAS DE CRÉDITO**
----------------------------
{' | '.join([c[1:] for c in cumpridas])}
{' | '.join([n[1:] for n in nao_cumpridas])}

📌 **Conclusão**
---------------
{'O modelo atende às políticas definidas e está pronto para produção.' if not nao_cumpridas else 'O modelo NÃO atende a todas as políticas. Revisar ajustes.'}

📅 **Data da implantação:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
        """.strip()

        st.text(relatorio)

        # Botão de download
        st.download_button(
            label="⬇️ Baixar Relatório (TXT)",
            data=relatorio,
            file_name="relatorio_implantacao_modelo.txt",
            mime="text/plain"
        )

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/11_📑_Relatorio.py", label="➡️ Ir para Relatório Final", icon="📑")

if __name__ == "__main__":
    main()
