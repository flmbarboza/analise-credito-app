import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, log_loss
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from utils import load_session, save_session

def main():
    # Carrega sessão salva
    if 'dados' not in st.session_state:
        saved = load_session()
        st.session_state.update(saved)
        if saved:
            st.info("✅ Dados recuperados da sessão anterior.")
    
    st.title("✅ Análise e Validação do Modelo")
    st.markdown("""
    Entenda **como o seu modelo se comporta na prática** com métricas essenciais para credit scoring.  
    Aqui você vai aprender o que cada indicador significa — e por que ele importa.
    """)
    
    modelo_tipo = st.session_state.get('modelo_tipo', 'Desconhecido')
    target = st.session_state.get('target', 'Variável-Alvo')

    if 'modelo' not in st.session_state:
        st.warning("Nenhum modelo treinado! Construa um modelo primeiro.")
        st.page_link("pages/6_🤖_Modelagem.py", label="→ Ir para Modelagem", icon="🤖")
        return

    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')
    features = X_test.columns if isinstance(X_test, pd.DataFrame) else range(X_test.shape[1])

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

    # --- 8. CÁLCULO DO CUSTO DOS ERROS (FALSOS NEGATIVOS) ---
    st.markdown("### 💸 Custo dos Erros: Falsos Negativos")
    st.info("""
    Um **Falso Negativo (FN)** ocorre quando o modelo diz que o cliente é **adimplente (0)**,  
    mas ele **é inadimplente (1)**. Ou seja: **o crédito foi concedido, mas o cliente não pagará**.
    
    Assumimos aqui que o **custo desse erro é o valor do empréstimo concedido (`loan_amount`)**.
    """)
    
    # Recupera os dados originais
    df_original = st.session_state.get('dados')
    if df_original is None:
        st.warning("⚠️ Dados brutos não disponíveis. Não é possível calcular o custo do erro.")
        return
    
    # Lista de colunas numéricas candidatas a "valor do empréstimo"
    colunas_numericas = df_original.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(colunas_numericas) == 0:
        st.warning("Nenhuma coluna numérica disponível para usar como valor do empréstimo.")
        return
    
    # Opção para o usuário escolher a coluna de valor
    coluna_valor = st.selectbox(
        "Selecione a coluna que representa o **valor do empréstimo**:",
        options=colunas_numericas,
        index=colunas_numericas.index('loan_amount') if 'loan_amount' in colunas_numericas else 0,
        help="Essa coluna será usada para calcular o custo dos erros (ex: loan_amount, amount, valor_emprestimo)"
    )
    
    # Garante que y_test e X_test estão alinhados com o df_original
    try:
        # Extrai os valores do empréstimo para o conjunto de teste
        loan_test = df_original.loc[y_test.index, coluna_valor]
    
        # Cria DataFrame com real, predito e valor do empréstimo
        results = pd.DataFrame({
            'y_real': y_test,
            'y_pred': y_pred,
            'valor_emprestimo': loan_test.values
        })
    
        # Filtra os Falsos Negativos
        falsos_negativos = results[(results['y_real'] == 1) & (results['y_pred'] == 0)]
    
        if len(falsos_negativos) == 0:
            st.success("✅ Nenhum Falso Negativo encontrado! Nenhum custo de erro.")
            custo_total = 0.0
        else:
            custo_total = falsos_negativos['valor_emprestimo'].sum()
            media_custo = falsos_negativos['valor_emprestimo'].mean()
    
            st.warning(f"⚠️ Foram identificados **{len(falsos_negativos)} Falsos Negativos**.")
            st.metric("Custo Total dos Falsos Negativos", f"R$ {custo_total:,.2f}")
            st.write(f"Valor médio do empréstimo por FN: R$ {media_custo:,.2f}")
    
            # Mostra uma amostra
            with st.expander("📊 Ver detalhes dos Falsos Negativos"):
                st.dataframe(
                    falsos_negativos[['valor_emprestimo']]
                    .rename(columns={'valor_emprestimo': 'Valor do Empréstimo'})
                    .head(10)
                )
    
        # Armazena para uso futuro (ex: otimização de limiar)
        st.session_state.custo_falsos_negativos = custo_total
        st.session_state.coluna_valor_emprestimo = coluna_valor
    
    except KeyError:
        st.error(f"❌ A coluna `{coluna_valor}` não está disponível no conjunto de teste.")
    except Exception as e:
        st.error(f"Erro ao calcular custo dos erros: {e}")
    
    # --- 9. INDICADOR RELATIVO: Custo do Erro vs. Montante Total Emprestado ---
    st.markdown("### 📊 Indicador Relativo: Custo do Erro")
    st.info("""
    O custo absoluto dos erros é importante, mas ganha mais sentido quando comparado ao **total de crédito concedido**.
    
    Aqui, calculamos a **proporção do valor dos empréstimos que se tornou prejuízo** por causa de Falsos Negativos.
    """)
    
    # Recupera o valor total emprestado no conjunto de teste
    try:
        # Valor total emprestado (soma de todos os loan_amount no conjunto de teste)
        montante_total = df_original.loc[y_test.index, coluna_valor].sum()
    
        if 'custo_falsos_negativos' in st.session_state:
            custo_fn = st.session_state.custo_falsos_negativos
        else:
            # Recalcula se necessário
            results = pd.DataFrame({
                'y_real': y_test,
                'y_pred': y_pred,
                'valor': df_original.loc[y_test.index, coluna_valor].values
            })
            falsos_negativos = results[(results['y_real'] == 1) & (results['y_pred'] == 0)]
            custo_fn = falsos_negativos['valor'].sum()
    
        # Calcula o indicador relativo
        if montante_total > 0:
            percentual_perda = (custo_fn / montante_total) * 100
        else:
            percentual_perda = 0.0
    
        # Exibe os resultados
        col1, col2, col3 = st.columns(3)
        col1.metric("Montante Total Concedido", f"R$ {montante_total:,.2f}")
        col2.metric("Custo dos Falsos Negativos", f"R$ {custo_fn:,.2f}")
        col3.metric("Taxa de Perda por Erro", f"{percentual_perda:.2f}%")
    
        # Interpretação
        if percentual_perda < 1:
            st.success(f"✅ Baixo impacto: apenas **{percentual_perda:.2f}%** do montante concedido foi perdido por erros do modelo.")
        elif percentual_perda < 5:
            st.warning(f"⚠️ Impacto moderado: **{percentual_perda:.2f}%** do crédito concedido resultou em prejuízo.")
        else:
            st.error(f"🚨 Alto impacto: **{percentual_perda:.2f}%** do valor emprestado foi perdido por Falsos Negativos. Reavalie o modelo ou o limiar de aprovação.")
    
        # Armazena para relatório
        st.session_state.indicador_relativo_custo = {
            'montante_total': montante_total,
            'custo_fn': custo_fn,
            'percentual_perda': percentual_perda
        }
    
    except Exception as e:
        st.error(f"Erro ao calcular indicador relativo: {e}")
    save_session()
    
    # --- 5. ANÁLISE DE OVERFITTING (Curva de Perda) ---
    st.markdown("### 📉 Análise de Overfitting: Curva de Perda")
    st.info("""
    **Overfitting** ocorre quando o modelo "decora" os dados de treino, mas falha em generalizar para novos dados.  
    A **curva de perda** mostra o desempenho do modelo no treino e no teste ao longo do tempo (ou de iterações).  
    Se a curva de treino continua melhorando, mas a do teste estabiliza ou piora, é sinal de overfitting.
    """)

    # Simulação de curva de perda (já que sklearn não fornece diretamente)
    try:
        # Usamos um modelo similar para simular a curva (apenas para fins didáticos)
        if modelo_tipo == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model_for_curve = RandomForestClassifier(max_depth=10, random_state=42)
            param_range = np.arange(1, 11)
            train_scores, test_scores = validation_curve(
                model_for_curve, X_test, y_test, param_name="max_depth", param_range=param_range,
                cv=3, scoring="accuracy", n_jobs=-1
            )
            xlabel = "Profundidade da Árvore (max_depth)"
            title = "Curva de Validação - Random Forest"
            # Gráfico
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Treino')
            ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
        
            ax.plot(param_range, test_mean, color='red', marker='s', linestyle='--', markersize=5, label='Validação')
            ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='red')
        
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Acurácia')
            ax.set_title(title)
            ax.legend(loc='lower right')
            ax.grid(True)
            st.pyplot(fig)

        else:
            st.info("""
                ⚠️ Para modelos de **Regressão Logística**, não geramos a curva de perda.
                
                Motivo: O `LogisticRegression` do scikit-learn **não fornece a função de perda por iteração**.  
                Além disso, como o modelo é relativamente simples e convexamente otimizado, o overfitting pode ser melhor avaliado por métricas de teste (acurácia, KS, AUC) do que por curvas de loss.
                """) 

        # Média e desvio
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Interpretação
        if np.argmax(test_mean) < len(test_mean) - 1 and test_mean[-1] < test_mean[np.argmax(test_mean)]:
            st.warning("⚠️ **Possível overfitting**: a performance no conjunto de validação diminui após um certo ponto.")
        else:
            st.success("✅ **Sem sinais de overfitting claro**: a performance no teste acompanha a do treino.")
    
    except Exception as e:
        st.warning("Não foi possível gerar a curva de perda. Modelo não suporta validação direta.")
        
    # --- 6. PONTOS PARA DISCUSSÃO (Lúdico e Educacional) ---
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

    # --- 7. RELATÓRIO DA ANÁLISE ---
    st.markdown("### 📄 Relatório da Análise de Validação")
    st.info("Gere um resumo das métricas e insights desta análise para compartilhar ou documentar.")

    # Detecta overfitting apenas se a curva existir
    if modelo_tipo != "Regressão Logística" and 'test_mean' in locals():
        overfit_msg = ('Possível overfitting detectado.'
                       if np.argmax(test_mean) < len(test_mean) - 1 and 
                          test_mean[-1] < test_mean[np.argmax(test_mean)]
                       else 'Sem sinais claros de overfitting.')
    else:
        overfit_msg = "Não analisado para Regressão Logística."
    # Prepara o conteúdo do relatório
    relatorio_texto = f"""
    RELATÓRIO DE VALIDAÇÃO DO MODELO
    =================================
    
    🎯 Modelo: {modelo_tipo}
    🎯 Variável-alvo: {target}
    🎯 Número de variáveis preditoras: {len(features)}
    
    📊 MÉTRICAS PRINCIPAIS
    ----------------------
    Acurácia no Teste: {accuracy:.1%}
    Precision: {precision:.1%}
    Recall: {recall:.1%}
    F1-Score: {f1:.1%}
    AUC-ROC: {roc_auc:.2f}
    KS Máximo: {ks_max:.2f}
    Custo do Erro: R$ {custo_total:,.2f} ({percentual_perda:.2f}%)
    
    🔍 INTERPRETAÇÃO
    ----------------
    - **Precision**: Entre os clientes classificados como inadimplentes, {precision:.1%} realmente são.
    - **Recall**: O modelo identificou {recall:.1%} dos verdadeiros inadimplentes.
    - **AUC-ROC**: {'Excelente' if roc_auc > 0.9 else 'Bom' if roc_auc > 0.8 else 'Razoável' if roc_auc > 0.7 else 'Fraco'} poder preditivo.
    - **KS**: {'Excelente' if ks_max > 0.4 else 'Bom' if ks_max > 0.3 else 'Moderado'} separação entre bons e maus.
    - **Custo do Erro**: Somatório do montante emprestado que resultou em inadimplência. O percentual é dado em relação ao total do valor emprestado.
    
    📉 Overfitting
    -------------
    {overfit_msg}    
    
    📅 Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
    """
    
    # Opções de exportação
    export_option = st.radio("Escolha o formato de exportação:", ["Texto (.txt)", "Copiar para área de transferência"])
    
    if export_option == "Texto (.txt)":
        st.download_button(
            label="⬇️ Baixar Relatório (TXT)",
            data=relatorio_texto,
            file_name="relatorio_validacao_modelo.txt",
            mime="text/plain"
        )
    else:
        st.code(relatorio_texto, language="text")
        st.info("Você pode copiar o texto acima com o botão no canto superior direito.")

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/8_⚙️_Aperfeicoamento.py", label="➡️ Ir para Aperfeiçoamento", icon="⚙️")

if __name__ == "__main__":
    main()
