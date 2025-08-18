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
    st.info("""🔹 **Regressão Logística**: Interpretação clara, bom para modelos regulatórios.  
            🔹 **Random Forest**: Alta performance, menos interpretável.""")
    
    # --- TRATAMENTO DE VARIÁVEIS CATEGÓRICAS ---
    st.markdown("#### 🧱 Tratamento de Variáveis Categóricas")
    st.info("""
    Defina como cada variável categórica será tratada:
    - **One-Hot Encoding**: cria colunas binárias (recomendado para poucas categorias).
    - **Label Encoding**: converte em números (use com cuidado).
    - **Remover**: exclui a variável.
    """)
    
    # Identifica variáveis categóricas
    cat_vars = [col for col in features if dados[col].dtype == 'object']
    
    if len(cat_vars) == 0:
        st.success("✅ Nenhuma variável categórica encontrada.")
        # Define X diretamente
        X = dados[features]
        st.session_state.X_processed = X
        st.session_state.tratamento_feito = True
    else:
        if 'encoding_choice' not in st.session_state:
            st.session_state.encoding_choice = {}
    
        for var in cat_vars:
            choice = st.session_state.encoding_choice.get(var, "One-Hot Encoding")
            st.markdown(f"**Variável:** `{var}`")
            st.caption(f"Valores únicos: {sorted(dados[var].dropna().unique().astype(str))[:10]}{'...' if dados[var].nunique() > 10 else ''}")
    
            opcao = st.radio(
                f"Tratamento para `{var}`:",
                options=["One-Hot Encoding", "Label Encoding", "Remover"],
                key=f"encoding_{var}",
                horizontal=True,
                index=["One-Hot Encoding", "Label Encoding", "Remover"].index(choice) if choice in ["One-Hot Encoding", "Label Encoding", "Remover"] else 0
            )
            st.session_state.encoding_choice[var] = opcao
            st.markdown("---")
    
        # Botão para aplicar tratamento
        if st.button("✅ Aplicar Tratamento de Variáveis Categóricas"):
            try:
                X = dados[features].copy()
                for var in cat_vars:
                    opcao = st.session_state.encoding_choice[var]
                    if opcao == "One-Hot Encoding":
                        dummies = pd.get_dummies(X[var], prefix=var, drop_first=True)
                        X = pd.concat([X.drop(columns=[var]), dummies], axis=1)
                        st.success(f"✅ `{var}`: One-Hot Encoding aplicado.")
                    elif opcao == "Label Encoding":
                        X[var] = X[var].astype('category').cat.codes
                        st.success(f"✅ `{var}`: Label Encoding aplicado.")
                    elif opcao == "Remover":
                        X = X.drop(columns=[var])
                        st.info(f"ℹ️ `{var}`: Removida do modelo.")
                # Salva no estado
                st.session_state.X_processed = X
                st.session_state.tratamento_feito = True
                st.success("✅ Tratamento concluído! Você pode treinar o modelo agora.")
            except Exception as e:
                st.error(f"Erro ao aplicar tratamento: {e}")
    
    # --- TREINAMENTO DO MODELO ---
    if st.button("🚀 Treinar Modelo", type="primary"):
        # Verifica se o tratamento foi feito
        if 'tratamento_feito' not in st.session_state or not st.session_state.tratamento_feito:
            st.warning("Por favor, clique em 'Aplicar Tratamento' antes de treinar o modelo.")
            st.stop()
    
        try:
            X = st.session_state.X_processed
            y = dados[target]
    
            # Divisão treino/teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
            # Treina o modelo
            if modelo_tipo == "Regressão Logística":
                model = LogisticRegression(max_iter=1000, solver='liblinear')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
    
                # Statsmodels para p-valores
                X_train_sm = sm.add_constant(X_train)
                model_sm = sm.Logit(y_train, X_train_sm).fit(disp=False)
                p_values = model_sm.pvalues[1:]  # remove const
    
                st.session_state.modelo = model
                st.session_state.model_sm = model_sm
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.features = X.columns.tolist()
    
                acuracia = model.score(X_test, y_test)
                cm = confusion_matrix(y_test, y_pred)
    
                st.success("✅ Modelo de Regressão Logística treinado!")
    
                # Matriz de confusão
                st.markdown("### 📊 Matriz de Confusão")
                st.info("Mostra VP, VN, FP, FN. Ajuda a entender os erros do modelo.")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Adimplente', 'Inadimplente'],
                            yticklabels=['Adimplente', 'Inadimplente'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                st.pyplot(fig)
    
                # Expressão algébrica
                st.markdown("### 🧮 Expressão do Modelo")
                st.info("O modelo calcula a probabilidade de inadimplência com base nos coeficientes.")
                coef_intercept = model.intercept_[0]
                terms = [f"{coef_intercept:.4f}"]
                for feat, coef in zip(X.columns, model.coef_[0]):
                    sign = "+" if coef >= 0 else "-"
                    terms.append(f"{sign} {abs(coef):.4f}·{feat}")
                formula = " + ".join(terms)
                st.latex(f"\\text{{logit}} = {formula}")
    
                # Tabela de coeficientes
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
    
                st.markdown("### 📊 Matriz de Confusão")
                st.info("Mesma interpretação que na regressão logística.")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Adimplente', 'Inadimplente'],
                            yticklabels=['Adimplente', 'Inadimplente'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                st.pyplot(fig)
    
                st.markdown("### 🔍 Importância das Variáveis")
                importances = model.feature_importances_
                importance_df = pd.DataFrame({'Variável': X.columns, 'Importância': importances}).sort_values('Importância')
                fig, ax = plt.subplots()
                ax.barh(importance_df['Variável'], importance_df['Importância'], color='teal')
                ax.set_title("Importância das Variáveis")
                st.pyplot(fig)
    
               
    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/7_✅_Analise_e_Validacao.py", label="➡️ Ir para Análise e Validação", icon="✅")

if __name__ == "__main__":
    main()
