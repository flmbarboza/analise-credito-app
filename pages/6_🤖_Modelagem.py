import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import io

def main():
    st.title("🤖 Modelagem Preditiva")
    st.markdown("Construa e avalie modelos de credit scoring com interpretação clara.")

    if 'dados' not in st.session_state:
        st.warning("Dados não encontrados! Complete a coleta primeiro.")
        st.page_link("pages/2_📊_Coleta_de_Dados.py", label="→ Coleta de Dados")
        return

    dados = st.session_state.dados.copy()

    st.subheader("⚙️ Configuração do Modelo")

    # --- 1. SELEÇÃO E VALIDAÇÃO DA VARIÁVEL-ALVO (Y) ---
    st.markdown("### 🔍 Defina a Variável-Alvo (Default)")
    target = st.selectbox(
        "Selecione a coluna que indica **inadimplência**:",
        options=dados.columns,
        index=None,
        placeholder="Escolha a variável de default",
        key="target_select"  # ← mantém estado
    )
    
    if target not in dados.columns:
        st.error("ALERTA: variável-alvo inválida ou indefinida.")
        return
    
    y_data = dados[target].dropna()
    if len(y_data) == 0:
        st.error(f"A coluna `{target}` está vazia.")
        return
    
    valores_unicos = pd.Series(y_data.unique()).dropna().tolist()
    try:
        # Tenta ordenar apenas valores numéricos
        valores_numericos = [x for x in valores_unicos if isinstance(x, (int, float))]
        valores_unicos = sorted(valores_numericos) if valores_numericos else valores_unicos
    except:
        pass
    
    # Verificar se é binária (0/1)
    if set(valores_unicos) != {0, 1}:
        st.warning(f"""
        ⚠️ A variável `{target}` não está no formato 0/1.  
        Valores encontrados: {valores_unicos}
        """)
    
        st.markdown("#### 🔧 Mapeie os valores para 0 (adimplente) e 1 (inadimplente)")
        col1, col2 = st.columns(2)
    
        with col1:
            valor_bom = st.selectbox(
                "Valor que representa **adimplente (0)**",
                options=valores_unicos,
                key="valor_bom_select"  # ← estado persistente
            )
    
        with col2:
            # Remove o valor escolhido como "bom" das opções para "mau"
            opcoes_maus = [v for v in valores_unicos if v != valor_bom]
            valor_mau = st.selectbox(
                "Valor que representa **inadimplente (1)**",
                options=opcoes_maus,
                key="valor_mau_select"  # ← estado persistente
            )
    
        # Botão para aplicar o mapeamento
        if st.button("✅ Aplicar Mapeamento", key="btn_aplicar_mapeamento"):
            if valor_bom == valor_mau:
                st.error("Erro: os valores para 'bom' e 'mau' devem ser diferentes.")
            else:
                try:
                    # Mapeia os valores
                    y_mapped = dados[target].map({valor_bom: 0, valor_mau: 1})
                    
                    # Verifica se houve falha no mapeamento (valores não mapeados)
                    if y_mapped.isnull().any():
                        st.error(f"Erro: alguns valores não foram mapeados corretamente. Verifique os dados.")
                    else:
                        # Atualiza os dados
                        dados_atualizados = dados.copy()
                        dados_atualizados[target] = y_mapped
                        st.session_state.dados = dados_atualizados
                        st.session_state.target = target
                        st.success(f"✅ `{target}` foi convertida para 0 (adimplente) e 1 (inadimplente).")
                        st.rerun()  # ← recarrega para refletir a mudança
                except Exception as e:
                    st.error(f"Erro ao aplicar mapeamento: {e}")
    
    else:
        st.success(f"✅ `{target}` já está no formato 0/1.")
        st.session_state.target = target

    # --- 2. Seleção de variáveis preditoras ---
    st.markdown("### 📊 Dados que serão usados no modelo")
    features = st.multiselect(
        "Variáveis Preditivas:",
        options=[col for col in dados.columns if col != target],
        default=[col for col in dados.columns if col != target][:5]
    )

    if len(features) == 0:
        st.warning("Selecione pelo menos uma variável preditora.")
        st.stop()

    # --- 3. Mostrar DataFrame antes do modelo ---
    st.info("Abaixo estão as variáveis preditoras (X) e a variável-alvo (y) que serão usadas no treinamento.")
    X_preview = dados[features].head(10)
    y_preview = dados[target].head(10)
    preview = pd.concat([X_preview, y_preview], axis=1)
    st.dataframe(preview)

    # --- 4. Escolha do modelo ---
    modelo_tipo = st.radio(
        "Escolha o modelo:",
        options=["Regressão Logística", "Random Forest"],
        horizontal=True
    )
    st.info("""🔹 **Regressão Logística**: Interpretação clara, ideal para modelos regulatórios.  
            🔹 **Random Forest**: Alta performance, menos interpretável.""")

    # --- 5. Botão de treinamento ---
    if st.button("🚀 Treinar Modelo", type="primary"):
        with st.spinner("Preparando dados e treinando o modelo..."):
            try:
                X = dados[features].copy()
                y = dados[target]

                # --- Tratamento de variáveis categóricas (feito aqui, não antes) ---
                cat_vars = X.select_dtypes(include='object').columns.tolist()

                if len(cat_vars) > 0:
                    st.info(f"🔍 Detectadas {len(cat_vars)} variáveis categóricas: `{', '.join(cat_vars)}`. Aplicando tratamento...")
                    
                    # Pergunta como tratar cada uma (pode ser melhorado com interface, mas funcional)
                    encoding_choice = {}
                    for var in cat_vars:
                        choice = st.radio(
                            f"Tratamento para `{var}`:",
                            options=["One-Hot Encoding", "Label Encoding"],
                            key=f"encoding_{var}",
                            horizontal=True
                        )
                        encoding_choice[var] = choice

                    # Aplica tratamento
                    for var in cat_vars:
                        if encoding_choice[var] == "One-Hot Encoding":
                            dummies = pd.get_dummies(X[var], prefix=var, drop_first=True)
                            X = pd.concat([X.drop(columns=[var]), dummies], axis=1)
                            st.success(f"✅ `{var}`: One-Hot Encoding aplicado.")
                        elif encoding_choice[var] == "Label Encoding":
                            X[var] = X[var].astype('category').cat.codes
                            st.success(f"✅ `{var}`: Label Encoding aplicado.")
                else:
                    st.info("✅ Nenhuma variável categórica encontrada. Continuando com variáveis numéricas.")

                # --- Conversão final para numérico ---
                for col in X.columns:
                    if X[col].dtype == 'object':
                        try:
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                            st.warning(f"⚠️ Coluna `{col}` convertida para numérico (com coerção).")
                        except:
                            st.error(f"Erro ao converter `{col}` para numérico.")
                            st.stop()

                # --- Preenche valores faltantes ---
                if X.isnull().any().any():
                    st.warning("⚠️ Dados faltantes encontrados. Preenchendo com média.")
                    X = X.fillna(X.mean(numeric_only=True))

                # --- Garante tipo numérico ---
                X = X.astype(float)

                # --- Divisão treino/teste ---
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # --- Treinamento do modelo ---
                if modelo_tipo == "Regressão Logística":
                    model = LogisticRegression(max_iter=1000, solver='liblinear')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Statsmodels para p-valores
                    X_train_sm = sm.add_constant(X_train)
                    model_sm = sm.Logit(y_train, X_train_sm).fit(disp=False)
                    p_values = model_sm.pvalues[1:]

                    st.session_state.modelo = model
                    st.session_state.model_sm = model_sm
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.features = X.columns.tolist()

                    acuracia = model.score(X_test, y_test)
                    cm = confusion_matrix(y_test, y_pred)

                    st.success("✅ Modelo de Regressão Logística treinado!")

                    # --- MATRIZ DE CONFUSÃO ---
                    st.markdown("### 📊 Matriz de Confusão")
                    st.info("""Ajuda a entender os erros do modelo.
                    Mostra quantos casos foram classificados correta e incorretamente:
                    - **Verdadeiros Positivos (VP)**: Inadimplentes corretamente identificados.
                    - **Falsos Positivos (FP)**: Adimplentes classificados como inadimplentes.
                    - **Verdadeiros Negativos (VN)**: Adimplentes corretamente identificados.
                    - **Falsos Negativos (FN)**: Inadimplentes não detectados (pior erro).
                    """)
                    
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Adimplente', 'Inadimplente'],
                                yticklabels=['Adimplente', 'Inadimplente'])
                    ax.set_xlabel('Previsto')
                    ax.set_ylabel('Real')
                    st.pyplot(fig)

                    # --- EXPRESSÃO ALGÉBRICA ---
                    st.markdown("### 🧮 Expressão do Modelo (Logit)")
                    coef_intercept = model.intercept_[0]
                    terms = [f"{coef_intercept:.4f}"]
                    symbols = [f"X_{i+1}" for i in range(len(X.columns))]
                    # --- EXPRESSÃO ALGÉBRICA COM NOTAÇÃO PADRÃO ---
                    st.info("""
                    A probabilidade de inadimplência é calculada a partir do **logit**, dado por:
                    `logit = β₀ + β₁·X₁ + β₂·X₂ + ... + βₖ·Xₖ`
                    Este score linear é convertido em probabilidade com a função logística:
                    `P(default) = 1 / (1 + e^(-logit))`
                    """)
                                        
                    # Monta os termos com sinais
                    for symbols, coef in zip(symbols, model.coef_[0]):
                        sinal = "+" if coef >= 0 else "-"
                        terms.append(f"{sinal} {abs(coef):.2f} \\cdot {symbols}")
                    
                    # Monta a fórmula em LaTeX
                    formula = " ".join(terms)
                    st.latex(f"\\text{{logit}} = {formula}")
                    
                    # --- TABELA DE LEGENDA DAS VARIÁVEIS ---
                    st.warning("""Cada símbolo $$X_i$$ representa uma variável preditora do modelo. Mais especificamente:
                        # Gera a lista de legenda em LaTeX
                        legenda_latex = []
                        for i, var in enumerate(X.columns):
                            # Escapa caracteres problemáticos (como _)
                            var_escapado = var.replace('_', r'\_')
                            legenda_latex.append(rf"X_{{{i+1}}} = \text{{{var_escapado}}}")
                        
                        # Junta com quebra de linha
                        legenda_str = r" \\ ".join(legenda_latex)
                        st.latex(legenda_str)
                    """)
                    # --- TABELA DE COEFICIENTES ---
                    st.markdown("### 📋 Coeficientes e Significância")
                    st.info("""Coeficiente: impacto no log-odds. P-valor: significância estatística. 
                            Nota: Níveis de Significância são importantes para validar estatisticamente a importância da variável no modelo. No caso, *** é muito alta (praticamente 0%), ** é alta (1%) e * é significante a 5%. """)
                    coef_df = pd.DataFrame({
                        'Variável': X.columns,
                        'Coeficiente': model.coef_[0],
                        'P-valor': p_values.values
                    }).round(4)
                    coef_df['Significância'] = coef_df['P-valor'].apply(
                        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    )
                    st.dataframe(coef_df.style.background_gradient(cmap='RdYlGn', subset=['Coeficiente']))

                    st.metric("Acurácia no Teste", f"{acuracia:.1%}")

                elif modelo_tipo == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.session_state.modelo = model
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.features = X.columns.tolist()

                    acuracia = model.score(X_test, y_test)
                    cm = confusion_matrix(y_test, y_pred)

                    st.success("✅ Modelo Random Forest treinado!")

                    # --- MATRIZ DE CONFUSÃO ---
                    st.markdown("### 📊 Matriz de Confusão")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Adimplente', 'Inadimplente'],
                                yticklabels=['Adimplente', 'Inadimplente'])
                    ax.set_xlabel('Previsto')
                    ax.set_ylabel('Real')
                    st.pyplot(fig)

                    # --- IMPORTÂNCIA DAS VARIÁVEIS ---
                    st.markdown("### 🔍 Importância das Variáveis")
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({'Variável': X.columns, 'Importância': importances}).sort_values('Importância', ascending=True)
                    fig, ax = plt.subplots(figsize=(6, 0.35 * len(importance_df)))
                    ax.barh(importance_df['Variável'], importance_df['Importância'], color='teal')
                    ax.set_title("Importância das Variáveis (Random Forest)")
                    st.pyplot(fig)

                    st.metric("Acurácia no Teste", f"{acuracia:.1%}")

            except Exception as e:
                st.error(f"Erro ao treinar o modelo: {e}")

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/7_✅_Analise_e_Validacao.py", label="➡️ Ir para Análise e Validação", icon="✅")

if __name__ == "__main__":
    main()
