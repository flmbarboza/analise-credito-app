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

from faker import Faker

def executar_pipeline_seed(base, seed):
    """
    Pipeline completo para preparação de dados com problemas simulados
    
    Parâmetros:
    base -- DataFrame original
    seed -- Semente para reprodutibilidade
    
    Retorna:
    DataFrame com dados originais + amostras simuladas com problemas
    """
    # Configuração de sementes para reprodutibilidade
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Pré-processamento inicial
    base = base.copy()
    base = corrigir_tipos_numericos(base)  # Converte numéricos para Int64
    
    # 2. Cria subamostra dos dados originais
    sub = gerar_subamostra(base, seed=seed)
    
    # 3. Gera dados problemáticos
    n_instancias_fake = random.randint(60, 120)
    ruido = simular_dados_problematicos(sub, n_instancias_fake)
    
    # 4. Combina os dados
    combinado = pd.concat([sub, ruido], ignore_index=True)
    
    # 5. Pós-processamento
    combinado = tratar_categorias(combinado)  # Agrupa categorias raras em 'Others'
    combinado = corrigir_tipos_numericos(combinado)  # Garante tipos finais
    
    # 6. Verificação de qualidade
    if not verificar_integridade(combinado, sub):
        st.warning("Atenção: Alguma inconsistência foi detectada na geração dos dados")
    
    return combinado

def simular_dados_problematicos(df, n_amostras):
    """Versão adaptada da função de simulação que mantém seus requisitos"""
    df_simulado = pd.DataFrame(columns=df.columns)
    
    for _ in range(n_amostras):
        # Decidir se será uma linha duplicada (20% chance)
        if random.random() < 0.2 and len(df) > 0:
            linha_original = df.iloc[random.randint(0, len(df)-1)].copy()
            df_simulado = pd.concat([df_simulado, linha_original.to_frame().T], ignore_index=True)
            continue
        
        nova_linha = {}
        for coluna in df.columns:
            # 30% chance de problema em cada coluna
            if random.random() < 0.3:
                problema = random.choices(
                    ['faltante', 'inconsistencia', 'outlier', 'categoria_nova'],
                    weights=[0.4, 0.3, 0.2, 0.1],  # Probabilidades ajustadas
                    k=1
                )[0]
                
                if problema == 'faltante':
                    nova_linha[coluna] = np.nan
                
                elif problema == 'inconsistencia':
                    if pd.api.types.is_numeric_dtype(df[coluna]):
                        valor = random.choice(df[coluna].dropna().values)
                        nova_linha[coluna] = -abs(int(valor))
                    else:
                        nova_linha[coluna] = f"INVALID_{random.randint(1, 100)}"
                
                elif problema == 'outlier' and pd.api.types.is_numeric_dtype(df[coluna]):
                    mediana = df[coluna].median()
                    iqr = df[coluna].quantile(0.75) - df[coluna].quantile(0.25)
                    nova_linha[coluna] = int(mediana + (random.uniform(5, 10) * iqr))
                
                elif problema == 'categoria_nova' and pd.api.types.is_string_dtype(df[coluna]):
                    nova_linha[coluna] = 'Others'
                
                else:
                    nova_linha[coluna] = obter_valor_normal(df, coluna)
            else:
                nova_linha[coluna] = obter_valor_normal(df, coluna)
        
        df_simulado = pd.concat([df_simulado, pd.DataFrame([nova_linha])], ignore_index=True)
    
    return df_simulado

def verificar_integridade(df_completo, df_original):
    """Verifica se os dados simulados mantêm a integridade dos originais"""
    # Verifica se todas as colunas originais estão presentes
    if not set(df_original.columns).issubset(set(df_completo.columns)):
        return False
    
    # Verifica se os valores originais não foram alterados
    for col in df_original.columns:
        original_values = set(df_original[col].dropna().unique())
        completo_values = set(df_completo[col].dropna().unique())
        
        if not original_values.issubset(completo_values):
            return False
    
    return True
    
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
