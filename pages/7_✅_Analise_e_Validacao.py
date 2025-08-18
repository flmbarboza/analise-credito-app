import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    st.title("✅ Análise e Validação do Modelo")
    st.markdown("""
    Entenda **como o seu modelo se comporta na prática** com métricas essenciais para credit scoring.  
    Aqui você vai aprender o que cada indicador significa — e por que ele importa.
    """)

    if 'modelo' not in st.session_state:
        st.warning("Nenhum modelo treinado! Construa um modelo primeiro.")
        st.page_link("pages/6_🤖_Modelagem.py", label="→ Ir para Modelagem", icon="🤖")
        return

    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')

    if X_test is None or y_test is None:
        st.error("Dados de teste não disponíveis. Treine o modelo novamente.")
        return

    model = st.session_state.modelo
    y_pred = model.predict(X_test)

    # Probabilidades (necessárias para ROC e KS)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidade de classe 1 (inadimplente)
    except:
        st.warning("O modelo não suporta `predict_proba`. Usando predição direta.")
        y_proba = y_pred

    # --- 1. MATRIZ DE CONFUSÃO ---
    st.markdown("### 📊 Matriz de Confusão")
    st.info("""
    Mostra os acertos e erros do modelo. Ajuda a entender **quem foi classificado corretamente**.
    - **VP (Verdadeiro Positivo)**: Inadimplente → previsto como inadimplente ✅  
    - **VN (Verdadeiro Negativo)**: Adimplente → previsto como adimplente ✅  
    - **FP (Falso Positivo)**: Adimplente → previsto como inadimplente ❌ (rejeição injusta)  
    - **FN (Falso Negativo)**: Inadimplente → previsto como adimplente ❌ (pior erro: risco não detectado)
    """)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(2.5, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Adimplente (0)', 'Inadimplente (1)'],
                yticklabels=['Adimplente (0)', 'Inadimplente (1)'])
    ax.set_ylabel('Real')
    ax.set_xlabel('Previsto')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

    # --- 2. RELATÓRIO DE CLASSIFICAÇÃO AMIGÁVEL ---
    st.markdown("### 🧠 Relatório de Classificação (Explicado)")
    st.info("""
    Abaixo, cada métrica é explicada com base em um cenário de concessão de crédito.  
    Imagine que o modelo decide **a quem emprestar dinheiro**.
    """)

    # Calcula métricas
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Tabela de métricas
    metricas_df = pd.DataFrame({
        "Métrica": ["Precision", "Recall", "F1-Score", "Acurácia"],
        "Valor": [f"{precision:.2%}", f"{recall:.2%}", f"{f1:.2%}", f"{accuracy:.2%}"],
        "Explicação": [
            "**Precision (Precisão)**: Entre todos os clientes marcados como 'inadimplentes', quantos realmente são? Alta precisão = poucos falsos positivos (não rejeitamos bons clientes por engano).",
            "**Recall (Sensibilidade)**: Dos verdadeiros inadimplentes, quantos o modelo conseguiu identificar? Alto recall = poucos falsos negativos (capturamos mais risco).",
            "**F1-Score**: Média harmônica entre Precision e Recall. Ótimo quando queremos equilíbrio entre capturar risco e não rejeitar bons clientes.",
            "**Acurácia**: Proporção total de acertos. Pode ser enganosa se a base for desbalanceada (ex: 95% adimplentes)."
        ]
    })

    for _, row in metricas_df.iterrows():
        st.markdown(f"#### {row['Métrica']} → `{row['Valor']}`")
        st.markdown(row['Explicação'])
        st.markdown("---")

    # --- 3. CURVA ROC e AUROC ---
    st.markdown("### 📈 Curva ROC e AUC")
    st.info("""
    A **Curva ROC** mostra o trade-off entre **verdadeiros positivos (Recall)** e **falsos positivos** em diferentes limiares.
    - **AUC (Área sob a curva)**: Quanto maior, melhor o modelo separa bons e maus pagadores.
    - **AUC > 0.7**: razoável | **> 0.8**: bom | **> 0.9**: excelente
    """)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Recall)')
    ax.set_title('Curva ROC')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.metric("AUC (ROC)", f"{roc_auc:.2f}")

    # --- 4. TESTE KS (Kolmogorov-Smirnov) ---
    st.markdown("### 📊 Estatística KS")
    st.info("""
    O **KS** mede a maior separação entre a distribuição acumulada de **bons** e **maus pagadores**.
    - **KS > 0.3**: bom poder de separação
    - **KS > 0.4**: excelente
    É uma métrica muito usada em crédito porque mostra claramente o poder discriminatório do modelo.
    """)

    # Distribuição acumulada
    thresholds = np.linspace(0, 1, 100)
    tpr_ks = [np.mean(y_proba[y_test == 1] >= th) for th in thresholds]
    fpr_ks = [np.mean(y_proba[y_test == 0] >= th) for th in thresholds]
    ks_values = [t - f for t, f in zip(tpr_ks, fpr_ks)]
    ks_max = max(ks_values)
    best_th = thresholds[np.argmax(ks_values)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, tpr_ks, label='Bons (TPR)', color='green')
    ax.plot(thresholds, fpr_ks, label='Maus (FPR)', color='red')
    ax.plot(thresholds, ks_values, label='KS (diferença)', color='blue', linestyle='--')
    ax.axvline(best_th, color='gray', linestyle=':', label=f'Limiar ótimo (KS): {best_th:.2f}')
    ax.set_xlabel('Limiar de decisão')
    ax.set_ylabel('Taxa acumulada')
    ax.set_title('Análise KS')
    ax.legend()
    st.pyplot(fig)

    st.metric("KS Máximo", f"{ks_max:.2f}")
    st.caption(f"Limiar ótimo para KS: {best_th:.2f}")

    # --- 5. PONTOS PARA DISCUSSÃO (Lúdico e Educacional) ---
    st.markdown("### 💬 Pontos para Reflexão")
    st.markdown("""
    #### 🎯 1. Precision vs Recall: Qual priorizar?
    - Se você é **banco conservador**, quer alto **Recall** (capturar todos os inadimplentes).
    - Se quer **não rejeitar bons clientes**, priorize **Precision**.
    - O **F1-Score** ajuda a equilibrar os dois.

    #### ⚠️ 2. Acurácia pode enganar!
    Se 95% dos clientes são adimplentes, um modelo que diz "todos são bons" terá 95% de acurácia... mas é inútil.

    #### 📊 3. KS > 0.4 é ótimo, mas...
    ...verifique se o poder de separação é estável em subgrupos (idade, renda, região).

    #### 🧩 4. E se o modelo for justo?
    Avalie se ele trata grupos sensíveis (gênero, raça) de forma equitativa. Isso é **ética em IA**.

    #### 🔄 5. E agora?
    Use essas métricas para **ajustar o limiar de aprovação** ou **melhorar o modelo**.
    """)

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/8_⚙️_Aperfeicoamento.py", label="➡️ Ir para Aperfeiçoamento", icon="⚙️")

if __name__ == "__main__":
    main()
