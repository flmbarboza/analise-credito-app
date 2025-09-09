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
from utils import load_session, save_session
  
def main():
    # Carrega sessão salva
    if 'dados' not in st.session_state:
        saved = load_session()
        st.session_state.update(saved)
        if saved:
            st.info("✅ Dados recuperados da sessão anterior.")
    
    st.title("🤖 Modelagem Preditiva")
    st.markdown("Construa e avalie modelos de credit scoring com interpretação clara.")
    
    if 'encoding_choice' not in st.session_state:
        st.session_state.encoding_choice = {}
    
     # --- 1. VALIDAÇÃO INICIAL DE DADOS ---
    if 'dados' not in st.session_state or st.session_state.dados is None or st.session_state.dados.empty:
        st.warning("Dados não carregados ou vazios! Acesse a página de Coleta primeiro.")
        st.page_link("pages/3_🚀_Coleta_de_Dados.py", label=" → Retornar para Coleta de dados")
        st.stop()
    
    dados = st.session_state.dados.copy()
    
    # --- 2. VALIDAÇÃO DA VARIÁVEL-ALVO ---
    target = st.session_state.get('target')
    if not target or target not in dados.columns:
        st.warning("⚠️ Variável-alvo não definida ou inválida.")
    
    # --- 3. DEFINIÇÃO SEGURO DE VARIÁVEIS ATIVAS ---
    if 'variaveis_ativas' not in st.session_state or st.session_state.variaveis_ativas is None:
        # st.info(f"ℹ️ A lista de variáveis ativas não foi definida ou está vazia. Usando todas as colunas exceto `{target}`.")
        # Fallback seguro
        st.session_state.variaveis_ativas = [col for col in dados.columns if col != target]
    
    # Recupera a lista
    variaveis_ativas = st.session_state.variaveis_ativas
    
    # --- 4. VALIDAÇÃO FINAL: Garantir que é uma lista válida ---
    if not isinstance(variaveis_ativas, list):
        st.error("❌ A lista de variáveis ativas não foi carregada. Reinicializando...")
        variaveis_ativas = [col for col in dados.columns if col != target]
    
    # Remove colunas que não existem mais nos dados
    variaveis_ativas = [col for col in variaveis_ativas if col in dados.columns]
    
    # Remove a target, se estiver presente
    if target in variaveis_ativas:
        variaveis_ativas.remove(target)
    
    # --- 5. VERIFICAÇÃO DE VAZIO ---
    if not variaveis_ativas:
        st.error("""
        ❌ Nenhuma variável ativa válida encontrada.  
        Isso pode ocorrer se:
        - Todas as variáveis foram removidas.
        - O nome das colunas mudou.
        - A variável-alvo é a única coluna no dataset.
        """)
        st.stop()
    
    # Atualiza o session_state (para garantir consistência)
    st.session_state.variaveis_ativas = variaveis_ativas
    
    # ✅ Confirmação final
    st.success(f"✅ {len(variaveis_ativas)} variáveis ativas carregadas e validadas.")

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
    save_session()
    
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
    
    # --- 4. Análise de variáveis categóricas (antes do treinamento) ---
    st.markdown("#### 🔍 Resumo das Variáveis Preditivas")
    num_vars = [col for col in features if pd.api.types.is_numeric_dtype(dados[col])]
    cat_vars = [col for col in features if dados[col].dtype == 'object']
    
    st.write(f"- **Total de variáveis preditoras:** {len(features)}")
    st.write(f"- **Numéricas:** {len(num_vars)}")
    st.write(f"- **Categóricas:** {len(cat_vars)}")
    
    if len(cat_vars) > 0:
        st.info(f"📌 Variáveis categóricas: `{', '.join(cat_vars)}` serão tratadas durante o treinamento.")
    else:
        st.success("✅ Nenhuma variável categórica encontrada.")
        
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
                    #encoding_choice = {}
                    for var in cat_vars:
                        choice = st.session_state.encoding_choice.get(var, "One-Hot Encoding")
                        opcao = st.radio(
                            f"Tratamento para `{var}`:",
                            options=["One-Hot Encoding", "Label Encoding"],
                            key=f"encoding_{var}",
                            horizontal=True,
                            index=["One-Hot Encoding", "Label Encoding"].index(choice)
                        )
                        st.session_state.encoding_choice[var] = opcao  # Salva no estado

                    # Aplica tratamento
                    for var in cat_vars:
                        opcao = st.session_state.encoding_choice[var]
                        if opcao == "One-Hot Encoding":
                            dummies = pd.get_dummies(X[var], prefix=var, drop_first=True)
                            X = pd.concat([X.drop(columns=[var]), dummies], axis=1)
                            st.success(f"✅ `{var}`: One-Hot Encoding aplicado.")
                        elif opcao == "Label Encoding":
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
                    st.session_state.modelo_tipo = modelo_tipo
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
                    st.latex(f"\\text{{P(inadimplência)}} = {formula}")
                    
                    # --- TABELA DE LEGENDA DAS VARIÁVEIS ---
                    st.warning("Cada símbolo $$X_i$$ representa uma variável preditora do modelo. Mais especificamente:")
                    # Gera a lista de legenda em LaTeX
                    legenda_latex = []
                    for i, var in enumerate(X.columns):
                        # Escapa caracteres problemáticos (como _)
                        var_escapado = var.replace('_', r'\_')
                        legenda_latex.append(rf"X_{{{i+1}}} = \text{{{var_escapado}}}")
                    
                    # Junta com quebra de linha
                    legenda_str = r" \\ ".join(legenda_latex)
                    st.latex(legenda_str)
                
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
                    st.session_state.modelo_tipo = modelo_tipo
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
            save_session()
            
    # --- EXPORTAÇÃO DO RELATÓRIO ---
    with st.expander("📝 Relatório das Ações Realizadas", expanded=False):
        st.info("Veja abaixo um resumo detalhado de todas as etapas executadas nesta modelagem.")
        
        # Verifica se o modelo já foi treinado
        if 'modelo' not in st.session_state:
            st.info("Treine o modelo para gerar o relatório detalhado.")
        else:
            relatorio_acoes = []
        
            # 1. Variável-alvo
            relatorio_acoes.append(f"🎯 **Variável-alvo definida:** `{target}` (formato 0/1)")
            st.markdown(f"**Variável-alvo:** `{target}`")
        
            # 2. Tratamento de variáveis categóricas
            cat_vars = [col for col in features if dados[col].dtype == 'object']
            if len(cat_vars) > 0:
                st.markdown("**Tratamento de variáveis categóricas:**")
                if 'encoding_choice' in st.session_state:
                    tratamentos_aplicados = []
                    for var in cat_vars:
                        choice = st.session_state.encoding_choice.get(var, "Não definido")
                        if choice == "One-Hot Encoding":
                            # Estima número de dummies (baseado no número de categorias únicas)
                            n_cats = dados[var].nunique()
                            n_dummies = n_cats - 1  # drop_first=True
                            desc = f"`{var}` → One-Hot Encoding ({n_dummies} colunas geradas)"
                        else:
                            desc = f"`{var}` → Label Encoding"
                        st.markdown(f"- {desc}")
                        tratamentos_aplicados.append(desc)
                    relatorio_acoes.append("🔧 **Tratamento de categóricas:**")
                    relatorio_acoes.extend([f"   - {t}" for t in tratamentos_aplicados])
                else:
                    st.warning("Nenhum tratamento de categóricas registrado.")
                    relatorio_acoes.append("🔧 **Tratamento de categóricas:** Dados não disponíveis.")
            else:
                st.markdown("**Tratamento de variáveis categóricas:** Nenhuma variável categórica encontrada.")
                relatorio_acoes.append("🔧 **Tratamento de categóricas:** Nenhuma variável categórica presente.")
        
            # 3. Conversão e limpeza
            st.markdown("**Conversão e limpeza de dados:**")
            X = st.session_state.get('X_processed', dados[features])
            if X.isnull().any().any():
                st.markdown("- Preenchimento de valores faltantes com a média")
                relatorio_acoes.append("🧹 **Tratamento de missing:** Valores faltantes preenchidos com a média.")
            else:
                st.markdown("- Nenhum valor faltante encontrado")
                relatorio_acoes.append("🧹 **Tratamento de missing:** Nenhum valor faltante encontrado.")
        
            # 4. Modelo treinado
            modelo_tipo = st.session_state.get('modelo_tipo', 'Não identificado')
            relatorio_acoes.append(f"🧠 **Modelo escolhido:** {modelo_tipo}")
            relatorio_acoes.append(f"📊 **Variáveis preditoras ({len(features)}):** {', '.join(features)}")
            st.markdown(f"**Modelo treinado:** {modelo_tipo}")
            st.markdown(f"**Número de variáveis preditoras:** {len(features)}")
        
            # 5. Métricas
            if 'acuracia' in st.session_state:
                acuracia = st.session_state.acuracia
                st.markdown(f"**Acurácia no teste:** {acuracia:.1%}")
                relatorio_acoes.append(f"📈 **Acurácia no teste:** {acuracia:.1%}")
        
            # Armazena para exportação
            st.session_state.relatorio_acoes = relatorio_acoes

        st.markdown("#### 📤 Exportar Relatório Personalizado.")
        st.caption('Selecione os itens que deseja incluir no relatório final:')
        
        opcoes_relatorio = [
            "Variável-alvo",
            "Tratamento de variáveis categóricas",
            "Conversão e limpeza de dados",
            "Modelo escolhido",
            "Variáveis preditoras",
            "Acurácia no teste",
            "Matriz de Confusão",
            "Expressão do Modelo",
            "Tabela de Coeficientes"
        ]
        
        itens_selecionados = st.multiselect(
            "Itens do relatório",
            options=opcoes_relatorio,
            default=opcoes_relatorio
        )
        
        if st.button("📄 Gerar Relatório"):
            relatorio_final = []
            for item in itens_selecionados:
                if item == "Variável-alvo":
                    relatorio_final.append(f"🎯 Variável-alvo: {target}")
                elif item == "Tratamento de variáveis categóricas":
                    relatorio_final.append("🔧 Tratamento de variáveis categóricas:")
                    if len(cat_vars) > 0:
                        for var in cat_vars:
                            relatorio_final.append(f"   - {var}: {encoding_choice[var]}")
                    else:
                        relatorio_final.append("   - Nenhuma variável categórica.")
                elif item == "Conversão e limpeza de dados":
                    relatorio_final.append("🧹 Conversão e limpeza:")
                    if 'object' in X.dtypes.values:
                        relatorio_final.append("   - Colunas object convertidas para numérico.")
                    if X.isnull().any().any():
                        relatorio_final.append("   - Missing preenchidos com média.")
                    else:
                        relatorio_final.append("   - Nenhum dado faltante ou problema de tipo.")
                elif item == "Modelo escolhido":
                    relatorio_final.append(f"🧠 Modelo: {modelo_tipo}")
                elif item == "Variáveis preditoras":
                    relatorio_final.append(f"📊 Variáveis preditoras ({len(features)}): {', '.join(features)}")
                elif item == "Acurácia no teste" and 'acuracia' in locals():
                    relatorio_final.append(f"📈 Acurácia no teste: {acuracia:.1%}")
                elif item == "Matriz de Confusão" and 'cm' in locals():
                    relatorio_final.append("🔢 Matriz de Confusão:")
                    relatorio_final.append(f"   Verdadeiros Positivos: {cm[1,1]}")
                    relatorio_final.append(f"   Falsos Positivos: {cm[0,1]}")
                    relatorio_final.append(f"   Verdadeiros Negativos: {cm[0,0]}")
                    relatorio_final.append(f"   Falsos Negativos: {cm[1,0]}")
                elif item == "Expressão do Modelo" and modelo_tipo == "Regressão Logística":
                    relatorio_final.append(f"🧮 Expressão do Modelo: logit = {formula}")
                elif item == "Tabela de Coeficientes" and modelo_tipo == "Regressão Logística":
                    relatorio_final.append("📋 Coeficientes:")
                    for var, coef, pval in zip(X.columns, model.coef_[0], p_values.values):
                        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                        relatorio_final.append(f"   - {var}: {coef:.4f} (p={pval:.4f}) {sig}")
            
            # Gera o conteúdo do relatório
            relatorio_texto = "\n".join(relatorio_final)
            
            # Botão de download
            st.download_button(
                label="⬇️ Baixar Relatório (TXT)",
                data=relatorio_texto,
                file_name="relatorio_modelagem.txt",
                mime="text/plain"
            )

    # --- EXPORTAÇÃO DOS DADOS DE TESTE ---
    st.markdown("---")
    with st.expander("📥 Baixar Amostra de Teste para Validação Externa", expanded=False):
        st.markdown("### 📥 Dados de Teste (`X_test` e `y_test`)")
    
        if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
            st.info("Treine o modelo primeiro para gerar a amostra de teste.")
        else:
            st.info("""
            Use este conjunto para:
            - Validar o modelo em outro ambiente (Excel, Python, etc).
            - Testar com ferramentas de IA generativa.
            - Simular políticas de crédito com dados reais.
            """)
    
            # Recupera os dados
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            target_name = st.session_state.get('target', 'target')  # Usa o nome original da coluna
    
            # Junta X_test e y_test, com a coluna-alvo no nome original
            teste_completo = X_test.copy()
            teste_completo[target_name] = y_test.values  # ← Aqui está a correção!
    
            # Converte para CSV
            csv_teste = teste_completo.to_csv(index=False)
    
            # Botão de download
            st.download_button(
                label="⬇️ Baixar Amostra de Teste (CSV)",
                data=csv_teste,
                file_name="amostra_teste_modelo.csv",
                mime="text/csv",
                help=f"Inclui todas as variáveis preditoras e a variável-alvo: `{target_name}`."
            )
    
            # Preview
            st.markdown("#### 🔍 Prévia dos dados (primeiras 10 linhas)")
            st.dataframe(teste_completo.head(10))
    
            st.success("✅ Pronto para validação externa!")
        
    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.page_link("pages/7_✅_Analise_e_Validacao.py", label="➡️ Ir para Análise e Validação", icon="✅")

if __name__ == "__main__":
    main()
