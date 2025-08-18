import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import statsmodels.api as sm  # Para p-valores e significância

def main():
    st.title("🤖 Modelagem Preditiva")
    st.markdown("Construa e avalie modelos de credit scoring com interpretação clara.")

    if 'dados' not in st.session_state:
        st.warning("Dados não encontrados! Complete a coleta primeiro.")
        st.page_link("pages/2_📊_Coleta_de_Dados.py", label="→ Coleta de Dados")
        return

    dados = st.session_state.dados.copy()

    st.subheader("⚙️ Configuração do Modelo")

    # Seleção da variável-alvo
    target = st.selectbox(
        "Variável Target (inadimplência):",
        options=dados.columns,
        index=None,
        placeholder="Escolha a variável de default"
    )

    if target is None or target not in dados.columns:
        st.stop()

    if target not in st.session_state:
        st.session_state.target = target

    # Seleção de variáveis preditoras
    features = st.multiselect(
        "Variáveis Preditivas:",
        options=[col for col in dados.columns if col != target],
        default=[col for col in dados.columns if col != target][:5]  # Sugere até 5
    )

    if len(features) == 0:
        st.warning("Selecione pelo menos uma variável preditora.")
        st.stop()

    # Seleção do modelo
    modelo_tipo = st.radio(
        "Escolha o modelo:",
        options=["Regressão Logística", "Random Forest"],
        horizontal=True
    )
    st.info("🔹 **Regressão Logística**: Interpretação clara, bom para modelos regulatórios.  
            🔹 **Random Forest**: Alta performance, menos interpretável.")

    # Tratamento de variáveis categóricas
    st.markdown("#### 🧱 Tratamento de Variáveis Categóricas")
    cat_vars = dados[features].select_dtypes(include='object').columns.tolist()
    if len(cat_vars) > 0:
        st.write(f"Variáveis categóricas detectadas: `{', '.join(cat_vars)}`")
        encoding = st.radio(
            "Como deseja codificar variáveis categóricas?",
            options=["One-Hot Encoding (dummy)", "Label Encoding (numérico)"],
            horizontal=True
        )
    else:
        encoding = "Nenhuma"
        st.info("Nenhuma variável categórica encontrada.")

    # Botão de treinamento
    if st.button("🚀 Treinar Modelo", type="primary"):
        with st.spinner("Treinando e avaliando o modelo..."):
            try:
                X = dados[features].copy()
                y = dados[target]

                # Aplica encoding
                if encoding == "One-Hot Encoding (dummy)" and len(cat_vars) > 0:
                    X = pd.get_dummies(X, columns=cat_vars, drop_first=True)
                elif encoding == "Label Encoding (numérico)" and len(cat_vars) > 0:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    for col in cat_vars:
                        X[col] = le.fit_transform(X[col].astype(str))

                # Divisão treino/teste
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Treina o modelo
                if modelo_tipo == "Regressão Logística":
                    model = LogisticRegression(max_iter=1000, solver='liblinear')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Usar statsmodels para p-valores
                    X_train_sm = sm.add_constant(X_train)
                    model_sm = sm.Logit(y_train, X_train_sm).fit(disp=False)
                    p_values = model_sm.pvalues
                    significancia = p_values.reindex(X.columns, fill_value=np.nan)

                    # Armazena modelo e resultados
                    st.session_state.modelo = model
                    st.session_state.model_sm = model_sm
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.features = X.columns.tolist()

                    # Métricas
                    acuracia = model.score(X_test, y_test)
                    cm = confusion_matrix(y_test, y_pred)

                    st.success("✅ Modelo de Regressão Logística treinado com sucesso!")

                    # --- MATRIZ DE CONFUSÃO ---
                    st.markdown("### 📊 Matriz de Confusão")
                    st.info("""
                    Mostra quantos casos foram classificados correta e incorretamente:
                    - **Verdadeiros Positivos (VP)**: Inadimplentes corretamente identificados.
                    - **Falsos Positivos (FP)**: Adimplentes classificados como inadimplentes.
                    - **Verdadeiros Negativos (VN)**: Adimplentes corretamente identificados.
                    - **Falsos Negativos (FN)**: Inadimplentes não detectados (pior erro).
                    """)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Adimplente (0)', 'Inadimplente (1)'],
                                yticklabels=['Adimplente (0)', 'Inadimplente (1)'])
                    ax.set_ylabel('Real')
                    ax.set_xlabel('Previsto')
                    ax.set_title('Matriz de Confusão')
                    st.pyplot(fig)

                    # --- EXPRESSÃO ALGÉBRICA ---
                    st.markdown("### 🧮 Expressão do Modelo (Score Linear)")
                    st.info("""
                    O modelo calcula um **logit** (score bruto) com base nos coeficientes:
                    `logit = β₀ + β₁·x₁ + β₂·x₂ + ...`
                    Depois converte para probabilidade com a função logística:
                    `P(default) = 1 / (1 + e^(-logit))`
                    """)
                    coef_intercept = model.intercept_[0]
                    terms = [f"{coef_intercept:.4f}"]
                    for feat, coef in zip(X.columns, model.coef_[0]):
                        sign = "+" if coef >= 0 else "-"
                        terms.append(f"{sign} {abs(coef):.4f}·{feat}")
                    formula = " + ".join(terms)
                    st.latex(f"\\text{{logit}} = {formula}")

                    # --- TABELA DE COEFICIENTES ---
                    st.markdown("### 📋 Tabela de Coeficientes e Significância")
                    st.info("""
                    - **Coeficiente**: impacto da variável no log-odds.
                    - **P-valor**: indica se o efeito é estatisticamente significativo (geralmente < 0.05).
                    - **Significância**: *** (p<0.001), ** (p<0.01), * (p<0.05), . (p<0.1)
                    """)
                    coef_df = pd.DataFrame({
                        'Variável': ['Intercept'] + X.columns.tolist(),
                        'Coeficiente': [model.intercept_[0]] + model.coef_[0].tolist(),
                        'P-valor': [p_values['const']] + significancia.tolist()
                    })
                    coef_df['Significância'] = coef_df['P-valor'].apply(
                        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''
                    )
                    coef_df['Coeficiente'] = coef_df['Coeficiente'].round(4)
                    coef_df['P-valor'] = coef_df['P-valor'].round(4)

                    st.dataframe(
                        coef_df.style.format({
                            'Coeficiente': '{:.4f}',
                            'P-valor': '{:.4f}'
                        }).background_gradient(cmap='RdYlGn', subset=['Coeficiente'], low=1, high=1)
                    )

                    # Métrica final
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

                    st.success("✅ Modelo Random Forest treinado com sucesso!")

                    # --- MATRIZ DE CONFUSÃO ---
                    st.markdown("### 📊 Matriz de Confusão")
                    st.info("""
                    Mesma interpretação que na regressão logística. Avalia a qualidade das previsões.
                    """)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Adimplente (0)', 'Inadimplente (1)'],
                                yticklabels=['Adimplente (0)', 'Inadimplente (1)'])
                    ax.set_ylabel('Real')
                    ax.set_xlabel('Previsto')
                    ax.set_title('Matriz de Confusão')
                    st.pyplot(fig)

                    # --- IMPORTÂNCIA DAS VARIÁVEIS ---
                    st.markdown("### 🔍 Importância das Variáveis")
                    st.info("""
                    Mostra quais variáveis mais contribuíram para as decisões do modelo.
                    Útil para explicabilidade, mesmo que o modelo seja menos interpretável.
                    """)
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Variável': X.columns,
                        'Importância': importances
                    }).sort_values('Importância', ascending=True)

                    fig, ax = plt.subplots(figsize=(6, 0.35 * len(importance_df)))
                    ax.barh(importance_df['Variável'], importance_df['Importância'], color='teal')
                    ax.set_title("Importância das Variáveis (Random Forest)")
                    st.pyplot(fig)

                    st.metric("Acurácia no Teste", f"{acuracia:.1%}")

            except Exception as e:
                st.error(f"Erro ao treinar o modelo: {str(e)}")

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/7_✅_Analise_e_Validacao.py", label="➡️ Ir para Análise e Validação", icon="✅")

if __name__ == "__main__":
    main()
