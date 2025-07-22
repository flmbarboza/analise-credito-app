import streamlit as st
import numpy as np
import pandas as pd
import kagglehub
import random
from io import BytesIO
from io import StringIO
from datetime import datetime

def gerar_subamostra(base, percentual=0.2, seed=42):
    return base.sample(frac=percentual, random_state=seed).copy()

import random
import numpy as np

def corrigir_tipos_numericos(df):
    """Garante que todas as colunas numéricas do DataFrame mantenham seu tipo inteiro"""
    for col in df.select_dtypes(include=np.number).columns:
        # Verifica se todos os valores (não nulos) são inteiros
        if (df[col].dropna() % 1 == 0).all():
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df
    
def simular_instancias_problema(df, n_instancias):
    # Primeiro corrige os tipos numéricos
    df = corrigir_tipos_numericos(df.copy())
    
    # Identifica colunas que devem ser tratadas como inteiras
    int_cols = [col for col in df.columns if pd.api.types.is_integer_dtype(df[col])]
    
    # DataFrame para armazenar as instâncias problemáticas
    df_fake = pd.DataFrame(columns=df.columns)
    
    for _ in range(n_instancias):
        nova_linha = {}
        for col in df.columns:
            if col in int_cols:
                # Lógica específica para colunas inteiras
                tipo_problema = random.choice(["duplicata", "novo_valor", "faltante", "inconsistencia"])
                
                if tipo_problema == "duplicata":
                    nova_linha[col] = int(random.choice(df[col].dropna().values))
                elif tipo_problema == "novo_valor":
                    # Gera valores dentro de 3 desvios padrão da média (como inteiro)
                    media = int(df[col].mean())
                    std = int(df[col].std())
                    nova_linha[col] = random.randint(media - 3*std, media + 3*std)
                elif tipo_problema == "inconsistencia":
                    nova_linha[col] = -abs(int(random.choice(df[col].dropna().values)))
                else:  # faltante
                    nova_linha[col] = np.nan
                    
            elif df[col].dtype == 'object':
                # Lógica para colunas categóricas
                if random.random() < 0.2:
                    nova_linha[col] = "Categoria_Inédita_" + str(random.randint(1, 5))
                else:
                    nova_linha[col] = random.choice(df[col].dropna().unique())
            else:
                # Outros tipos (deveria ser apenas colunas já convertidas)
                nova_linha[col] = None
                
        df_fake = pd.concat([df_fake, pd.DataFrame([nova_linha])], ignore_index=True)
    
    # Garante tipos corretos nas colunas numéricas
    df_fake = corrigir_tipos_numericos(df_fake)
    
    # Adiciona duplicatas completas
    n_duplicatas = random.randint(4, 15)
    linhas_originais = df.sample(n=min(n_duplicatas, len(df)), replace=False)
    
    for i in range(n_duplicatas):
        duplicata = linhas_originais.iloc[[i % len(linhas_originais)]].copy()
        df_fake = pd.concat([df_fake, duplicata], ignore_index=True)
    
    return df_fake
    
def tratar_categorias(df):
    # Aplicar apenas em colunas do tipo object que representam categorias
    for col in df.select_dtypes(include='object').columns:
        # Verificar se a coluna parece ser categórica (não é uma string livre)
        if df[col].nunique() < len(df) * 0.5:  # Se menos que 50% dos valores são únicos
            freq = df[col].value_counts(normalize=True)
            categorias_frequentes = freq[freq > 0.01].index
            df[col] = df[col].apply(lambda x: x if x in categorias_frequentes else 'Others')
    return df
    
def executar_pipeline_seed(base, seed):
    # Garante que tudo seja reproduzível
    random.seed(seed)
    np.random.seed(seed)

    # Garante tipos corretos antes de qualquer operação
    base = corrigir_tipos_numericos(base.copy())
    
    sub = gerar_subamostra(base, seed=seed)
    n_instancias_fake = random.randint(60, 120)
    ruido = simular_instancias_problema(sub, n_instancias_fake)
    
    # Combina os dados mantendo os tipos
    combinado = pd.concat([sub, ruido], ignore_index=True)
    combinado = tratar_categorias(combinado)
    
    # Verificação final de tipos
    return corrigir_tipos_numericos(combinado)
    
def main():
    st.title("🚀 Coleta de Dados")
    st.balloons()  # Efeito visual para confirmar o carregamento
    st.markdown("""
    Esta página permite:
    - Baixar automaticamente o dataset de aprovação de empréstimos do Kaggle
    - Ou carregar seu próprio arquivo de dados
    """)

    seeds_disponiveis = [42, 7, 13, 21, 99, 123, 456, 789, 1010, 2025]
    seed_escolhida = st.selectbox("Escolha a seed para subamostragem:", seeds_disponiveis)
        
    # Container para os dados
    if 'dados' not in st.session_state:
        st.session_state.dados = None
        st.session_state.colunas_originais = None

    # Opções de coleta
    opcao = st.radio(
        "Selecione a fonte dos dados:",
        ["Baixar dataset do Kaggle", "Carregar arquivo local"],
        horizontal=True
    )

    if opcao == "Baixar dataset do Kaggle":
        if st.button("▶️ Baixar Dados do Kaggle"):
            with st.spinner("Baixando dataset..."):
                try:
                    path = kagglehub.dataset_download('architsharma01/loan-approval-prediction-dataset')
                    dados = pd.read_csv(f'{path}/loan_approval_dataset.csv')
                    
                    st.session_state.dados = dados
                    st.success("Dados baixados com sucesso!")
                
                    #st.success(f"Executando pipeline com seed escolhida: {seed_escolhida}")
                    st.session_state.dados = executar_pipeline_seed(dados, seed_escolhida)
                    st.write(f"A critério de exemplo, veja um subconjunto dos seus dados - Dimensões (número de amostras, quantidade de variáveis): {st.session_state.dados.shape}")
                    st.dataframe(st.session_state.dados.head())
                    st.balloons()
                except Exception as e:
                    st.error(f"Erro ao baixar dados: {str(e)}")
    

    else:  # Upload de arquivo
        # Adicionar seleção de delimitador
        col1, col2 = st.columns(2)
        with col1:
            delimiter = st.selectbox(
                "Delimitador do arquivo CSV",
                options=[",", ";", "\t", "|", "outro"],
                index=0,
                help="Selecione o caractere usado para separar as colunas no arquivo CSV"
            )
            
            if delimiter == "outro":
                delimiter = st.text_input("Especifique o delimitador", value=",")
        
        with col2:
            # Opção para remover espaços em branco
            auto_trim = st.checkbox(
                "Remover espaços em branco automaticamente",
                value=True,
                help="Remove espaços extras no início/fim de textos e nomes de colunas"
            )
        
        arquivo = st.file_uploader(
            "Carregue seu arquivo CSV",
            type=["csv", "txt"],
            accept_multiple_files=False
        )
        
        if arquivo is not None:
            try:
                # Ler o arquivo com o delimitador especificado
                dados = pd.read_csv(arquivo, delimiter=delimiter)
                
                # Aplicar trim automático se selecionado
                if auto_trim:
                    # Trim em nomes de colunas
                    dados.columns = dados.columns.str.strip()
                    
                    # Trim em valores de texto
                    for col in dados.select_dtypes(include=['object']).columns:
                        dados[col] = dados[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                
                st.session_state.dados = dados
                
                # Mostrar pré-visualização
                st.success("Arquivo carregado com sucesso!")
                st.write("Pré-visualização dos dados (5 primeiras linhas):")
                st.dataframe(dados.head())
                
                # Mostrar estatísticas básicas
                with st.expander("📊 Estatísticas básicas do arquivo"):
                    st.write(f"Total de linhas: {len(dados)}")
                    st.write(f"Total de colunas: {len(dados.columns)}")
                    st.write("Tipos de dados:")
                    st.write(dados.dtypes)
                    
            except pd.errors.ParserError as e:
                st.error(f"Erro ao ler arquivo com o delimitador '{delimiter}'. Tente outro delimitador.")
                st.error(f"Detalhes do erro: {str(e)}")
            except Exception as e:
                st.error(f"Erro inesperado ao processar o arquivo: {str(e)}")
                st.error("Verifique se o arquivo está no formato correto e tente novamente.")
    
   # Seção para ajuste de nomes de variáveis
    if st.session_state.dados is not None:
        st.divider()


        # Explicação dos tipos
        with st.expander("ℹ️ Descrição das Variáveis"):
            # Criação da tabela como um DataFrame do Pandas
            data = {
                "Nome da Variável": [
                    "loan_id", "no_of_dependents", "education", "self_employed",
                    "income_annum", "loan_amount", "loan_term", "cibil_score",
                    "residential_assets_value", "commercial_assets_value",
                    "luxury_assets_value", "bank_asset_value", "loan_status"
                ],
                "Descrição": [
                    "Um ID único para cada solicitação de empréstimo. Um exemplo de ID seria o CPF do solicitante.",
                    "O número de dependentes que o solicitante do empréstimo possui.",
                    'O nível de educação do solicitante, indicando se ele é "Graduado" ou "Não Graduado".',
                    'Indica se o solicitante é autônomo ("Sim" ou "Não").',
                    "A renda anual do solicitante.",
                    "O valor do empréstimo solicitado (em moeda corrente).",
                    "O prazo do empréstimo (em meses).",
                    "A pontuação de crédito do solicitante feita pela agência CIBIL.",
                    "O valor dos ativos imobiliários do solicitante para fins de moradia (em moeda corrente).",
                    "O valor dos ativos comerciais do solicitante para fins comerciais (em moeda corrente).",
                    "O valor dos ativos do solicitante para fins de lazer (em moeda corrente).",
                    "O valor dos ativos financeiros do solicitante (em moeda corrente).",
                    'O status da aprovação do empréstimo, indicando se foi "Aprovado" ou "Rejeitado".'
                ]
            }
            
            df = pd.DataFrame(data)
            
            # Exibe a tabela com formatação profissional
            st.dataframe(
                df,
                column_config={
                    "Nome da Variável": st.column_config.TextColumn(width="medium"),
                    "Descrição": st.column_config.TextColumn(width="large")
                },
                hide_index=True,
                use_container_width=True
            )
        
        # 1. OPÇÃO PARA AJUSTAR NOMES DAS VARIÁVEIS
        st.subheader("🔧 Ajuste dos Nomes das Variáveis")
        
        if st.checkbox("Deseja renomear as colunas?"):
            colunas_atuais = st.session_state.dados.columns.tolist()
            novos_nomes = []
            
            for i, coluna in enumerate(colunas_atuais):
                novo_nome = st.text_input(
                    f"Renomear '{coluna}' para:",
                    value=coluna,
                    key=f"nome_{i}"
                )
                novos_nomes.append(novo_nome)
            
            if st.button("Aplicar novos nomes"):
                st.session_state.dados.columns = novos_nomes
                st.success("Nomes das colunas atualizados!")
                st.session_state.colunas_originais = colunas_atuais  # Guarda original

        # 2. ANÁLISE SIMPLIFICADA
        st.subheader("🧐 Entendendo Seus Dados")
        
        # Método mais robusto para análise dos dados
        st.markdown(f"""
        ### 📋 Resumo do Dataset
        - **Total de registros**: {len(st.session_state.dados):,}
        - **Número de variáveis**: {len(st.session_state.dados.columns)}
        - **Tipo**:
            - **object**: Texto ou categorias
            - **int/float**: Números
            - **bool**: Verdadeiro/Falso
            - **datetime**: Datas
        """)

        # Tabela resumida
        resumo = []
        for coluna in st.session_state.dados.columns:
            nao_nulos = st.session_state.dados[coluna].count()
            percent_preenchido = (nao_nulos / len(st.session_state.dados)) * 100
            
            resumo.append({
                "Variável": coluna,
                "Tipo *": str(st.session_state.dados[coluna].dtype),
                "Valores únicos": st.session_state.dados[coluna].nunique(),
                "Preenchida (%)": f"{percent_preenchido:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(resumo))
        
        with st.expander("Salvar a Amostra", expanded=False):
            # 3. SALVAR DATAFRAME COMO CSV
            st.subheader("💾 Salvar Subamostra em CSV")
            nome_csv = st.text_input("Nome do arquivo para download:", value="subamostra_credito.csv")
    
            csv = st.session_state.dados.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name=nome_csv,
                mime='text/csv'
            )
    # 🚀 Link para a próxima página
    st.page_link("pages/3_🚀_Pré-Análise_dos_Dados.py", label="➡️ Ir para a próxima página: Pré-Análise dos Dados", icon="🚀")

if __name__ == "__main__":
    main()
