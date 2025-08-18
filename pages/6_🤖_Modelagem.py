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

    # --- 1. Seleção da variável-alvo ---
    target = st.selectbox(
        "Variável Target (inadimplência):",
        options=dados.columns,
        index=None,
        placeholder="Escolha a variável de default"
    )

    if target is None or target not in dados.columns:
        st.stop()

    # --- 2. Seleção de variáveis preditoras ---
    features = st.multiselect(
        "Variáveis Preditivas:",
        options=[col for col in dados.columns if col != target],
        default=[col for col in dados.columns if col != target][:5]
    )

    if len(features) == 0:
        st.warning("Selecione pelo menos uma variável preditora.")
        st.stop()

    # --- 3. Mostrar DataFrame antes do modelo ---
    st.markdown("### 📊 Dados que serão usados no modelo")
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
                    st.info("Mostra VP, VN, FP, FN. Ajuda a entender os erros do modelo.")
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
                    for feat, coef in zip(X.columns, model.coef_[0]):
                        sign = "+" if coef >= 0 else "-"
                        terms.append(f"{sign} {abs(coef):.4f}·{feat}")
                    formula = " + ".join(terms)
                    st.latex(f"\\text{{logit}} = {formula}")

                    # --- TABELA DE COEFICIENTES ---
                    st.markdown("### 📋 Coeficientes e Significância")
                    st.info("Coeficiente: impacto no log-odds. P-valor: significância estatística.")
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
