import streamlit as st
import pandas as pd
import io
import zipfile
import base64
from datetime import datetime
import json

def main():
    st.title("📑 Relatório Final de Credit Scoring")
    st.markdown("""
    Gere um **relatório completo e personalizado** para apresentar à área de risco.  
    Combine insights de todas as etapas do projeto.
    """)

    if 'dados' not in st.session_state:
        st.warning("Dados não disponíveis. Complete as etapas anteriores.")
        st.page_link("pages/3_🚀_Coleta_de_Dados.py", label="→ Ir para Coleta de Dados")
        return

    # --- 1. CONFIGURAÇÕES DO RELATÓRIO ---
    st.subheader("📝 Configurações do Relatório")

    col1, col2 = st.columns(2)
    with col1:
        nome_projeto = st.text_input("Nome do Projeto:", "Análise de Risco de Crédito")
        autor = st.text_input("Autor / Equipe:", st.session_state.get("user_id", "Equipe de Risco"))

    with col2:
        data_relatorio = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.info(f"Data de geração: {data_relatorio}")

    # --- 2. SELEÇÃO DE SEÇÕES ---
    st.subheader("🔍 Selecione as seções e detalhes a incluir")

    # Estrutura de seleção por página
    secoes = {
        "1. Coleta de Dados": {
            "Descrição da fonte dos dados": True,
            "Tratamento de dados faltantes": True,
            "Estatísticas da amostra": True
        },
        "2. Análise Univariada": {
            "Distribuição de variáveis numéricas": True,
            "Frequência de variáveis categóricas": True,
            "Identificação de outliers e nulos": True
        },
        "3. Análise Bivariada": {
            "Top variáveis por IV (Information Value)": True,
            "Top variáveis por KS (Kolmogorov-Smirnov)": True,
            "Mapa de calor de correlação": True,
            "Tabelas e gráficos de WOE": True
        },
        "4. Modelagem": {
            "Tipo de modelo (Regressão Logística / Random Forest)": True,
            "Variáveis preditoras utilizadas": True,
            "Equação do modelo (Logit)": True,
            "Tabela de coeficientes e significância": True,
            "Matriz de confusão": True
        },
        "5. Validação do Modelo": {
            "Acurácia, Precision, Recall, F1-Score": True,
            "Curva ROC e AUC": True,
            "Estatística KS": True,
            "Análise de overfitting (curva de perda)": True
        },
        "6. Aperfeiçoamento": {
            "Limiar de decisão ajustado": True,
            "Custo do erro (FN e FP)": True,
            "Comparação com modelo base": True
        },
        "7. Políticas de Crédito": {
            "Score mínimo para aprovação": True,
            "Restrições aplicadas (DTI, garantia)": True,
            "Taxa de aprovação simulada": True
        },
        "8. Implantação": {
            "Desempenho em novos dados": True,
            "Validação das políticas de crédito": True,
            "Conclusão sobre prontidão para produção": True
        }
    }

    # Interface de seleção
    relatorio_final_parts = []

    for secao, itens in secoes.items():
        with st.expander(f"📌 {secao}", expanded=False):
            st.markdown(f"**Incluir nesta seção:**")
            for item, default in itens.items():
                if st.checkbox(item, value=default, key=f"chk_{secao}_{item}"):
                    relatorio_final_parts.append(f"- {item}")

    # --- 3. INCLUIR RELATÓRIOS INDIVIDUAIS (ZIP) ---
    st.subheader("📎 Incluir relatórios parciais emitidos anteriormente")

    incluir_relatorios = st.checkbox("Incluir todos os relatórios gerados nas páginas anteriores", value=True)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Simulação de relatórios (substitua com os reais se salvos no session_state)
        relatorios_existentes = []

        # Exemplo: se você salvou relatórios em session_state
        if 'relatorio_acoes' in st.session_state:
            conteudo = "\n".join(st.session_state.relatorio_acoes)
            zip_file.writestr("modelagem_relatorio.txt", conteudo)
            relatorios_existentes.append("Modelagem")

        if 'dados' in st.session_state:
            dados_csv = st.session_state.dados.to_csv(index=False)
            zip_file.writestr("dados_limpos.csv", dados_csv)
            relatorios_existentes.append("Dados Limpos")

        # Adicione outros se existirem...
        # Ex: análise bivariada, validação, etc.

        if not relatorios_existentes:
            zip_file.writestr("README.txt", "Nenhum relatório parcial foi gerado nas etapas anteriores.\nGere relatórios nas páginas específicas para incluí-los aqui.")

    zip_buffer.seek(0)

    # --- 4. GERAR RELATÓRIO CONSOLIDADO ---
    if st.button("📄 Gerar Relatório Consolidado", type="primary"):
        if not relatorio_final_parts:
            st.warning("Selecione pelo menos uma seção para incluir no relatório.")
        else:
            st.success("✅ Relatório consolidado gerado com sucesso!")

            # Monta o relatório final
            relatorio_completo = f"""
RELATÓRIO FINAL DE CREDIT SCORING
==================================

Projeto: {nome_projeto}
Autor: {autor}
Data: {data_relatorio}

SUMÁRIO
--------
{chr(10).join(relatorio_final_parts[:10])}
{ '...' if len(relatorio_final_parts) > 10 else '' }

DETALHAMENTO
------------

{chr(10).join([f"{item}" for item in relatorio_final_parts])}

CONCLUSÃO
---------
O modelo foi desenvolvido com base em boas práticas de análise de crédito, desde a limpeza dos dados até a validação em novas amostras. Está pronto para implantação com monitoramento contínuo.

Equipe de Risco - {datetime.now().strftime('%Y')}
            """.strip()

            # Mostra o relatório
            st.text(relatorio_completo)

            # Botão de download do relatório completo
            st.download_button(
                label="⬇️ Baixar Relatório (TXT)",
                data=relatorio_completo,
                file_name=f"relatorio_final_{nome_projeto.replace(' ', '_').lower()}.txt",
                mime="text/plain"
            )

    # --- 5. DOWNLOAD DOS RELATÓRIOS PARCIAIS ---
    if incluir_relatorios and relatorios_existentes:
        st.download_button(
            label="📦 Baixar Todos os Relatórios (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="relatorios_parciais.zip",
            mime="application/zip"
        )
        st.caption(f"Contém: {', '.join(relatorios_existentes)}")

    # --- NAVEGAÇÃO ---
    st.markdown("---")
    st.info("✅ Projeto concluído! Agradecemos pela jornada de análise de crédito.")

if __name__ == "__main__":
    main()
