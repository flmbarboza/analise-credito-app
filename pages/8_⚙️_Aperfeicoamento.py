import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import io
import zipfile
import base64

def main():
    st.title("⚙️ Aperfeiçoamento do Modelo")
    st.markdown("""
    Refine o modelo com base nas métricas anteriores.  
    Ajuste o **limiar de decisão**, explore hiperparâmetros e gere um relatório com as melhorias.
    """)

    # --- 1. VALIDAÇÃO: Verifica se há modelo e métricas ---
    if 'modelo' not in st.session_state:
        st.warning("Nenhum modelo treinado! Construa um modelo primeiro.")
        st.page_link("pages/6_🤖_Modelagem.py", label="→ Ir para Modelagem", icon="🤖")
        return

    if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
        st.error("Dados de teste não disponíveis.")
        return

    model = st.session_state.modelo
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # Recupera predições e probabilidades
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        st.error("O modelo não suporta `predict_proba`. Não é possível ajustar o limiar.")
        return

    target = st.session_state.get('target', 'Variável-Alvo')
    features = st.session_state.get('features', [])

    # Recupera métricas do session_state (ou calcula)
    accuracy = st.session_state.get('accuracy')
    precision = st.session_state.get('precision')
    recall = st.session_state.get('recall')
    f1 = st.session_state.get('f1')
    ks_max = st.session_state.get('ks_max', 0)

    # Calcula taxa de inadimplência na amostra de teste
    taxa_inadimplencia = y_test.mean()
    st.info(f"📊 **Taxa de inadimplência na amostra de teste:** {taxa_inadimplencia:.1%}")

    # --- 2. SUGESTÕES DE APRIMORAMENTO ---
    st.subheader("📌 Sugestões de Aperfeiçoamento")
    sugestoes = []

    if precision is not None and precision < 0.7:
        sugestoes.append("⚠️ **Aumentar Precision:** Ajustar limiar de decisão para reduzir falsos positivos.")
    if recall is not None and recall < 0.7:
        sugestoes.append("⚠️ **Aumentar Recall:** Avaliar inclusão de variáveis adicionais ou técnicas de oversampling.")
    if f1 is not None and f1 < 0.7:
        sugestoes.append("⚠️ **Melhorar equilíbrio Precision/Recall:** Testar regularização ou algoritmos mais complexos.")
    if ks_max is not None and ks_max < 0.3:
        sugestoes.append("⚠️ **KS baixo:** Avaliar transformação de variáveis ou engenharia de features.")

    sugestoes.extend([
        "💡 Avaliar remoção de variáveis irrelevantes para reduzir ruído.",
        "💡 Experimentar diferentes algoritmos (Random Forest, XGBoost).",
        "💡 Realizar cross-validation para garantir estabilidade.",
        "💡 Testar ajuste de hiperparâmetros com GridSearch ou RandomSearch.",
        "💡 Avaliar tratamento de dados desbalanceados (SMOTE, undersampling).",
        "💡 Considerar feature engineering para capturar relações não lineares."
    ])

    for s in sugestoes:
        st.markdown(f"- {s}")

    # --- 3. EXPANDER: AJUSTE DE LIMIAR (com base no KS) ---
    with st.expander("🎛️ Ajuste de Limiar de Decisão (Threshold)", expanded=True):
        st.markdown("### 📊 Otimização com base no KS, custo de erro e taxa de inadimplência")
    
        # Cálculo do KS e limiar ótimo
        thresholds = np.linspace(0, 1, 100)
        tpr = [np.mean(y_proba[y_test == 1] >= th) for th in thresholds]
        fpr = [np.mean(y_proba[y_test == 0] >= th) for th in thresholds]
        ks_values = np.array(tpr) - np.array(fpr)
        best_idx = np.argmax(ks_values)
        best_threshold = thresholds[best_idx]
        max_ks = ks_values[best_idx]
    
        st.info(f"""
        - **KS Máximo encontrado:** {max_ks:.2f} no limiar **{best_threshold:.2f}**
        - **Taxa de inadimplência na amostra:** {taxa_inadimplencia:.1%}
        - **Sugestão de limiar inicial:** Próximo de {best_threshold:.2f}.
        """)
    
        # Usuário escolhe o limiar
        threshold = st.slider(
            "Escolha o limiar de aprovação:",
            min_value=0.0,
            max_value=1.0,
            value=float(best_threshold),
            step=0.01,
            format="%.2f"
        )
    
        # --- CÁLCULO DE MÉTRICAS E CUSTO DO ERRO ---
        y_pred_threshold = (y_proba >= threshold).astype(int)
    
        # Métricas com novo limiar
        acc = accuracy_score(y_test, y_pred_threshold)
        prec = precision_score(y_test, y_pred_threshold)
        rec = recall_score(y_test, y_pred_threshold)
        f1_val = f1_score(y_test, y_pred_threshold)
    
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred_threshold)
        tn, fp, fn, tp = cm.ravel()
    
        # --- CUSTO DO ERRO ---
        st.markdown("### 💰 Custo do Erro")
        st.info("""
        O erro do modelo tem custo real:
        - **Falso Negativo (FN)**: Cliente inadimplente aprovado → **prejuízo financeiro** (ex: perda do valor emprestado).
        - **Falso Positivo (FP)**: Cliente adimplente rejeitado → **perda de receita** (juros e tarifas não realizadas).
        """)
    
        # Entrada de custos (valores médios)
        col1, col2 = st.columns(2)
        with col1:
            custo_fn = st.number_input("Custo médio de um Falso Negativo (ex: valor médio de perda):", 
                                       min_value=0.0, value=5000.0, step=100.0)
        with col2:
            custo_fp = st.number_input("Custo médio de um Falso Positivo (ex: lucro médio perdido):", 
                                       min_value=0.0, value=1000.0, step=100.0)
    
        # Custo total do erro
        custo_total = (fn * custo_fn) + (fp * custo_fp)
        custo_por_cliente = custo_total / len(y_test)
    
        # --- MODELO BASE (limiar 0.5) para comparação ---
        y_pred_base = (y_proba >= 0.5).astype(int)
        acc_base = accuracy_score(y_test, y_pred_base)
        prec_base = precision_score(y_test, y_pred_base)
        rec_base = recall_score(y_test, y_pred_base)
        f1_base = f1_score(y_test, y_pred_base)
        cm_base = confusion_matrix(y_test, y_pred_base)
        tn_base, fp_base, fn_base, tp_base = cm_base.ravel()
        custo_total_base = (fn_base * custo_fn) + (fp_base * custo_fp)
        custo_por_cliente_base = custo_total_base / len(y_test)
    
        # --- VARIAÇÃO EM RELAÇÃO AO MODELO BASE ---
        st.markdown("### 📈 Variação em Relação ao Modelo Base (limiar = 0.5)")
    
        variacao = pd.DataFrame({
            "Métrica": ["Acurácia", "Precision", "Recall", "F1-Score", "Custo por Cliente"],
            "Modelo Base (0.5)": [
                f"{acc_base:.1%}",
                f"{prec_base:.1%}",
                f"{rec_base:.1%}",
                f"{f1_base:.1%}",
                f"R$ {custo_por_cliente_base:.2f}"
            ],
            "Novo Modelo": [
                f"{acc:.1%}",
                f"{prec:.1%}",
                f"{rec:.1%}",
                f"{f1_val:.1%}",
                f"R$ {custo_por_cliente:.2f}"
            ],
            "Variação": [
                f"{'🟢' if acc > acc_base else '🔴'} {((acc - acc_base)/acc_base*100):+.1f}%",
                f"{'🟢' if prec > prec_base else '🔴'} {((prec - prec_base)/prec_base*100):+.1f}%",
                f"{'🟢' if rec > rec_base else '🔴'} {((rec - rec_base)/rec_base*100):+.1f}%",
                f"{'🟢' if f1_val > f1_base else '🔴'} {((f1_val - f1_base)/f1_base*100):+.1f}%",
                f"{'🟢' if custo_por_cliente < custo_por_cliente_base else '🔴'} R$ {custo_por_cliente - custo_por_cliente_base:+.2f}"
            ]
        })
    
        st.dataframe(variacao, use_container_width=True)
    
        # Destaque se o custo melhorou
        if custo_por_cliente < custo_por_cliente_base:
            st.success(f"✅ O novo limiar reduziu o custo por cliente em **R$ {custo_por_cliente_base - custo_por_cliente:.2f}**.")
        else:
            st.warning(f"⚠️ O novo limiar aumentou o custo por cliente em **R$ {custo_por_cliente - custo_por_cliente_base:.2f}**.")
    
        # --- SALVAMENTO DAS MÉTRICAS ---
        st.session_state.accuracy = acc
        st.session_state.precision = prec
        st.session_state.recall = rec
        st.session_state.f1 = f1_val
        st.session_state.ks_max = max_ks
        st.session_state.threshold = threshold
        st.session_state.y_pred_final = y_pred_threshold
        st.session_state.custo_por_cliente = custo_por_cliente
        st.session_state.custo_por_cliente_base = custo_por_cliente_base
    
        st.success("✅ Limiar aplicado e métricas atualizadas!")

    # --- 4. AJUSTE DE HIPERPARÂMETROS (simulado) ---
    with st.expander("🛠️ Ajuste de Hiperparâmetros (simulado)", expanded=False):
        if hasattr(model, 'n_estimators'):
            n_estimators = st.slider("Número de árvores:", 50, 500, getattr(model, 'n_estimators', 100))
            max_depth = st.slider("Profundidade máxima:", 2, 20, getattr(model, 'max_depth', 5))
            if st.button("Aplicar ajustes"):
                # Em um cenário real, você re-treinaria o modelo
                st.session_state.modelo.n_estimators = n_estimators
                st.session_state.modelo.max_depth = max_depth
                st.success("✅ Hiperparâmetros atualizados (simulado)!")

    # --- 5. RELATÓRIO DE APRIMORAMENTO ---
    with st.expander("📄 Gerar Relatório de Aperfeiçoamento", expanded=False):
        st.markdown("### 📝 Resumo das melhorias aplicadas")

        relatorio = f"""
RELATÓRIO DE APRIMORAMENTO DO MODELO
=====================================

🎯 **Modelo:** {st.session_state.get('modelo_tipo', 'Desconhecido')}
🎯 **Variável-alvo:** {target}
🎯 **Número de variáveis preditoras:** {len(features)}

📊 **MÉTRICAS FINAIS (com limiar = {threshold:.2f})**
--------------------------------------
Acurácia: {acc:.1%}
Precision: {prec:.1%}
Recall: {rec:.1%}
F1-Score: {f1_val:.1%}
KS Máximo: {max_ks:.2f}
Limiar de decisão: {threshold:.2f}
Custo do Erro Total: R$ {custo_total:.2f}
Custo do Erro por cliente: R$ {custo_por_cliente:.2f}

📉 **Taxa de inadimplência (amostra):** {taxa_inadimplencia:.1%}

🔧 **Ajustes Aplicados**
------------------------
- Limiar de decisão ajustado para maximizar separação (KS).
- Hiperparâmetros atualizados (simulado).

📅 **Data do aprimoramento:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
        """.strip()

        st.text(relatorio)

        # Botão de download
        st.download_button(
            label="⬇️ Baixar Relatório (TXT)",
            data=relatorio,
            file_name="relatorio_aprimoramento_modelo.txt",
            mime="text/plain"
        )

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/9_🏛️_Politicas_de_Credito.py", label="➡️ Ir para Políticas de Crédito", icon="🏛️")

if __name__ == "__main__":
    main()
