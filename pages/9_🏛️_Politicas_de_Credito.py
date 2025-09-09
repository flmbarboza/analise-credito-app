import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
from utils import load_session, save_session

# Carrega sessão salva
if 'dados' not in st.session_state:
    saved = load_session()
    st.session_state.update(saved)
    if saved:
        st.info("✅ Dados recuperados da sessão anterior.")
        
def main():
    st.title("🏛️ Políticas de Crédito")
    st.markdown("""
    Defina regras de negócio para decisão de crédito, **baseadas em evidências do modelo**.  
    Identifique oportunidades de melhoria e gere restrições inteligentes.
    """)

    # --- 1. VALIDAÇÃO: Modelo e dados disponíveis ---
    if 'modelo' not in st.session_state:
        st.warning("Nenhum modelo treinado! Construa um modelo primeiro.")
        st.page_link("pages/6_🤖_Modelagem.py", label="→ Ir para Modelagem", icon="🤖")
        return

    if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
        st.error("Dados de teste não disponíveis. Volte para a etapa de validação.")
        return

    model = st.session_state.modelo
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        st.error("O modelo não suporta `predict_proba`. Não é possível calcular scores.")
        return

    target = st.session_state.get('target', 'default')
    threshold = st.session_state.get('threshold', 0.5)  # Limiar usado no aperfeiçoamento

    # Recupera dados originais (com todas as colunas)
    if 'dados' in st.session_state:
        dados_completos = st.session_state.dados
        # Tenta recuperar apenas as linhas do teste (baseado em índice)
        test_indices = X_test.index
        dados_teste = dados_completos.loc[test_indices].copy()
    else:
        dados_teste = X_test.copy()
        dados_teste[target] = y_test
        st.info("Dados completos não disponíveis. Usando apenas as variáveis do modelo.")

    # Adiciona y_proba
    dados_teste['y_proba'] = y_proba
    dados_teste['y_pred'] = (y_proba >= threshold).astype(int)

    # --- 2. ANÁLISE DE ERROS E PREJUÍZO ---
    st.markdown("### 🔍 Análise de Erros do Modelo")
    st.info("""
    Entenda **onde o modelo erra** para definir políticas de restrição.
    - **Falsos Negativos (FN)**: Inadimplente → previsto como adimplente → **prejuízo financeiro**
    - **Falsos Positivos (FP)**: Adimplente → previsto como inadimplente → **perda de receita**
    """)

    # Matriz de confusão
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, dados_teste['y_pred'])
    tn, fp, fn, tp = cm.ravel()

    col1, col2, col3 = st.columns(3)
    col1.metric("Falsos Negativos (FN)", fn)
    col2.metric("Falsos Positivos (FP)", fp)
    col3.metric("Taxa de Erro (FP + FN)", f"{(fp + fn) / len(y_test):.1%}")

    # Identifica os casos de erro
    dados_teste['erro'] = ''
    dados_teste.loc[(dados_teste[target] == 1) & (dados_teste['y_pred'] == 0), 'erro'] = 'Falso Negativo (Risco não detectado)'
    dados_teste.loc[(dados_teste[target] == 0) & (dados_teste['y_pred'] == 1), 'erro'] = 'Falso Positivo (Cliente bom rejeitado)'

    # Filtra apenas erros
    erros = dados_teste[dados_teste['erro'] != '']
    st.dataframe(erros[[target, 'y_proba', 'y_pred', 'erro'] + X_test.columns.tolist()[:5]].head(10))
    st.caption(f"Mostrando os 10 primeiros de {len(erros)} casos com erro.")

    # --- 3. SUGESTÕES DE POLÍTICAS BASEADAS EM DADOS ---
    st.markdown("### 📌 Sugestões de Políticas de Crédito")

    sugestoes = []

    if fn > 0:
        sugestoes.append("⚠️ **Restrição para Falsos Negativos:** Defina regras adicionais para clientes com alta probabilidade de inadimplência (>0.7) mas que foram aprovados (ex: exigir garantia).")
        sugestoes.append("💡 **Aprofundar análise manual** para clientes com `y_proba > 0.6` e renda baixa ou dívida alta.")

    if fp > 0:
        sugestoes.append("⚠️ **Relaxar políticas para Falsos Positivos:** Clientes com `y_proba < 0.3` mas reprovados podem ser reconsiderados (ex: oferta de crédito menor).")

    if dados_teste['y_proba'].mean() < 0.3:
        sugestoes.append("💡 **Modelo conservador:** A maioria dos clientes tem baixo risco. Considere expandir o mercado com limite inicial menor.")

    sugestoes.extend([
        "📌 Defina **scores mínimos variáveis** por perfil (ex: mais baixo para clientes com garantia).",
        "📌 Crie **regras de sobreposição** (ex: mesmo com score alto, rejeitar se `dti > 0.5`).",
        "📌 Estabeleça **limites de crédito escalonados** com base no score (Score 600-700 → R$5k, 700-800 → R$15k, etc.)."
    ])

    for s in sugestoes:
        st.markdown(f"- {s}")

    # --- 4. DOWNLOAD DA AMOSTRA DE TESTE ---
    st.markdown("### 📥 Baixar Amostra de Teste para Análise Externa")
    st.info("Baixe os dados de teste com `y_real` e `y_proba` para analisar com ferramentas externas (Excel, Python, IA generativa).")

    # Prepara CSV
    export_columns = [target, 'y_proba', 'y_pred'] + X_test.columns.tolist()
    df_export = dados_teste[export_columns].round(4)
    csv_data = df_export.to_csv(index=True)

    st.download_button(
        label="⬇️ Baixar Dados de Teste (CSV)",
        data=csv_data,
        file_name="amostra_teste_com_probabilidades.csv",
        mime="text/csv"
    )

    st.caption("Inclui: variáveis preditoras, valor real, probabilidade prevista e predição.")

    # --- 5. SIMULAÇÃO DE POLÍTICA COM RESTRIÇÕES ---
    st.markdown("### 🛠️ Simular Política de Crédito com Restrições")
    
    corte_score = st.slider("Score mínimo para aprovação:", 0.0, 1.0, float(threshold), step=0.01, format="%.2f")
    exigir_garantia = st.checkbox("Exigir garantia para clientes com score entre 0.4 e 0.6?")
    limite_dti = st.number_input("Limite máximo de DTI (dívida/renda):", 0.1, 1.0, 0.5, 0.05)
    
    if st.button("Simular Política"):
        # Simula aprovação com base no corte
        aprovado = (y_proba >= corte_score)
    
        # Aplica restrição de DTI (se a coluna existir)
        if 'dti' in X_test.columns:
            dti = X_test['dti']
            aprovado = aprovado & (dti <= limite_dti)
        else:
            st.warning("A coluna 'dti' não está disponível. Restrição de DTI ignorada.")
    
        # Aplica restrição de garantia (simulada)
        precisa_garantia = (y_proba >= 0.4) & (y_proba < 0.6)
        # Simula que 70% dos clientes fornecem garantia (aleatório)
        np.random.seed(42)  # Para reprodutibilidade
        com_garantia = np.random.rand(len(aprovado)) < 0.7
        if exigir_garantia:
            aprovado = aprovado & (~precisa_garantia | com_garantia)
    
        aprovacao_rate = aprovado.mean()
    
        # Salva no session_state para exibir depois
        st.session_state.aprovacao_rate = aprovacao_rate
    
        st.success("✅ Política simulada com sucesso!")
    
    # --- EXIBIÇÃO DA TAXA DE APROVAÇÃO (só se já foi simulada)
    if 'aprovacao_rate' in st.session_state:
        st.metric("Taxa de Aprovação", f"{st.session_state.aprovacao_rate:.1%}")
    else:
        st.info("👆 Clique em **'Simular Política'** para calcular a taxa de aprovação com as regras definidas.")
        
# --- 6. RELATÓRIO DA POLÍTICA ---
with st.expander("📄 Gerar Relatório de Política de Crédito", expanded=False):
    st.markdown("### 📝 Resumo da política definida")

    # Verifica se a simulação já foi feita
    if 'aprovacao_rate' not in st.session_state:
        st.info("👆 Primeiro, clique em **'Simular Política'** para gerar o relatório.")
    else:
        # Recupera os valores do session_state
        aprovacao_rate = st.session_state.aprovacao_rate
        fn = st.session_state.get('fn', 0)
        fp = st.session_state.get('fp', 0)
        total = len(y_test)

        relatorio = f"""
RELATÓRIO DE POLÍTICA DE CRÉDITO
=================================

🎯 **Modelo base:** {st.session_state.get('modelo_tipo', 'Desconhecido')}
🎯 **Limiar atual:** {threshold:.2f}
🎯 **Variável-alvo:** {target}

📊 **Desempenho no Teste**
--------------------------
Falsos Negativos (prejuízo): {fn}
Falsos Positivos (perda de receita): {fp}
Taxa de erro total: {(fp + fn) / total:.1%}

🔧 **Política Simulada**
------------------------
Score mínimo para aprovação: {corte_score:.2f}
{'✅ Exigência de garantia para score 0.4–0.6' if exigir_garantia else '❌ Sem exigência de garantia'}
Limite máximo de DTI: {limite_dti:.2f}
Taxa de aprovação simulada: {aprovacao_rate:.1%}

💡 **Sugestões de Melhoria**
----------------------------
{chr(10).join(f"- {s[2:]}" for s in sugestoes[:5])}

📅 **Data da política:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
        """.strip()

        st.text(relatorio)

        st.download_button(
            label="⬇️ Baixar Relatório (TXT)",
            data=relatorio,
            file_name="politica_credito.txt",
            mime="text/plain"
        )

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/10_📊_Implantacao_do_Credit_Scoring.py", label="➡️ Ir para Implantação do Credit Scoring", icon="📊")

if __name__ == "__main__":
    main()
