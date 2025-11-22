# # app_evasao.py
# # Dashboard interativo com análise de evasão usando Streamlit

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# csv_path = os.path.join(BASE_DIR, '..', '..', 'dados', 'processado', 'dados_ingresso_evasao_conclusao.csv')
# df = pd.read_csv(csv_path, sep=';')

# st.set_page_config(page_title="Dashboard Evasão IES", layout="wide")

# # Título
# st.title("📊 Dashboard - Taxas de Ingresso, Conclusão e Evasão")

# # Filtro por curso
# cursos = df['nome_curso'].unique()
# curso_selecionado = st.selectbox("Selecione um curso:", sorted(cursos))

# # Filtrar
# df_filtrado = df[df['nome_curso'] == curso_selecionado]

# # Métricas rápidas
# col1, col2, col3 = st.columns(3)
# col1.metric("Taxa de Ingresso (média)", f"{df_filtrado['taxa_ingresso'].mean():.2f}")
# col2.metric("Taxa de Conclusão (média)", f"{df_filtrado['taxa_conclusao'].mean():.2f}")
# col3.metric("Taxa de Evasão (média)", f"{df_filtrado['taxa_evasao'].mean():.2f}")

# # Gráfico de linha
# st.subheader("📈 Evolução das Taxas")
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.lineplot(data=df_filtrado[['taxa_ingresso', 'taxa_conclusao', 'taxa_evasao']])
# st.pyplot(fig)

# # Salvar gráfico como imagem PNG
# output_dir = os.path.join(BASE_DIR, '..', '..', 'acessibilidade_web', 'graficos')
# os.makedirs(output_dir, exist_ok=True)
# fig.savefig(os.path.join(output_dir, 'grafico_taxas.png'))


import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# ===============================
#  LOCALIZAÇÃO DOS ARQUIVOS
# ===============================
BASE_DIR = Path(__file__).resolve().parents[2]
CAMINHO_DADOS = BASE_DIR / "dados" / "processado" / "dados_ingresso_evasao_conclusao.csv"
CAMINHO_MODELO_BASE = BASE_DIR / "modelos" / "modelos_salvos" / "modelo_melhor_evasao.pkl"


# ===============================
#  CONFIGURAÇÃO DO LAYOUT STREAMLIT
# ===============================
st.set_page_config(
    page_title="Predição de Evasão — Ensino Superior",
    page_icon="📉",
    layout="wide",
)

st.sidebar.title("📊 Predição de Evasão")
st.sidebar.markdown("Aplicação prática — **2COP507 (Reconhecimento de Padrões)**")


# ===============================
#  CARREGAR BASE TRATADA
# ===============================
@st.cache_data(show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    df = pd.read_csv(CAMINHO_DADOS, sep=";", encoding="utf-8", low_memory=False)
    return df


df_ref = load_reference_data()


# ===============================
#  CARREGAR MODELO BASE (TREINADO NO PIPELINE)
# ===============================
@st.cache_resource(show_spinner=False)
def load_base_model():
    modelo = joblib.load(CAMINHO_MODELO_BASE)
    return modelo


modelo_base = load_base_model()

# Colunas de entrada usadas no modelo (garante compatibilidade com o pickle)
if hasattr(modelo_base, "feature_names_in_"):
    FEATURE_COLS = list(modelo_base.feature_names_in_)
else:
    # fallback: usa o mesmo conjunto principal do pipeline
    FEATURE_COLS = [
        "numero_cursos",
        "vagas_totais",
        "inscritos_totais",
        "ingressantes",
        "matriculados",
        "concluintes",
    ]


# ===============================
#  FUNÇÃO PARA TREINO CUSTOMIZADO
# ===============================
@st.cache_resource(show_spinner=False)
def treinar_modelo_customizado(
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
):
    """Treina um RandomForestRegressor com hiperparâmetros escolhidos no sidebar."""

    df = df_ref.copy()

    X = df[FEATURE_COLS]
    y = df["taxa_evasao"]

    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(X, y)
    return modelo


# ===============================
#  SIDEBAR — SELEÇÃO DE HIPERPARÂMETROS
# ===============================
st.sidebar.header("Ajuste de Hiperparâmetros")

algoritmo = st.sidebar.selectbox(
    "Algoritmo",
    ["RandomForest — modelo do pipeline", "RandomForest — customizado"],
)

n_estimators = st.sidebar.slider("Número de Árvores (n_estimators)", 50, 500, 200, step=10)
max_depth = st.sidebar.slider("Profundidade Máxima (max_depth)", 2, 30, 15)
min_samples_split = st.sidebar.slider("Mínimo de amostras para dividir o nó (min_samples_split)", 2, 20, 4)

usar_custom = algoritmo == "RandomForest — customizado"

if usar_custom:
    modelo_ativo = treinar_modelo_customizado(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )
    st.sidebar.success("Usando modelo RandomForest **customizado** treinado em tempo real.")
else:
    modelo_ativo = modelo_base
    st.sidebar.info("Usando modelo RandomForest do **pipeline original** (pickle).")


# ===============================
#  FUNÇÃO AUXILIAR DE PRÉ-PROCESSAMENTO
# ===============================
def preprocess_row(row: pd.Series) -> np.ndarray:
    """Recebe uma linha com FEATURES e devolve o array na ordem esperada pelo modelo."""

    row = row.copy()
    row = row[FEATURE_COLS]
    return row.values.reshape(1, -1)


# Estatísticas para sugerir valores padrão na interface
stats = df_ref[FEATURE_COLS].describe()


# ===============================
#  TABS DA INTERFACE
# ===============================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📘 Sobre a Aplicação",
        "📈 Predição Individual",
        "📊 Avaliação do Modelo",
        "📁 Enviar Arquivo",
    ]
)


# ===============================
#  TAB 1 — SOBRE
# ===============================
with tab1:
    st.header("Sobre a Aplicação")
    st.markdown(
        """
Esta aplicação faz parte da disciplina **2COP507 – Reconhecimento de Padrões** e utiliza
um modelo de **Random Forest** para estimar a **taxa de evasão** em cursos da educação superior.

### ✔️ O que você encontrará aqui
- Interface amigável para experimentos com o modelo
- Ajuste interativo de hiperparâmetros do Random Forest
- Métricas de avaliação (MAE, RMSE, R², acurácia binária, F1 etc.)
- Gráficos: dispersão, distribuição e matriz de confusão
- Upload de arquivo CSV para predições em lote

### 🎯 Tecnologias utilizadas
- **Streamlit** — interface web interativa
- **Scikit-Learn** — RandomForestRegressor
- **Pandas / NumPy** — manipulação dos dados
- **Matplotlib / Seaborn** — visualização
"""
    )


# ===============================
#  TAB 2 — PREDIÇÃO INDIVIDUAL
# ===============================
with tab2:
    st.header("📈 Predição Individual de Taxa de Evasão")
    st.markdown("Ajuste os valores das variáveis e clique em **Calcular**.")

    # Cria inputs numéricos com base nas FEATURES usadas no modelo
    cols = st.columns(3)
    valores_usuario = {}

    for idx, col_name in enumerate(FEATURE_COLS):
        col_streamlit = cols[idx % 3]
        desc = stats[col_name]
        default_val = float(desc["50%"])
        min_val = float(max(0, desc["min"]))
        max_val = float(desc["max"])

        with col_streamlit:
            valores_usuario[col_name] = st.number_input(
                f"{col_name}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=max(1.0, (max_val - min_val) / 100),
            )

    if st.button("🔮 Calcular probabilidade de evasão", key="btn_pred_individual"):
        df_user = pd.DataFrame([valores_usuario])
        x = preprocess_row(df_user.iloc[0])

        # RandomForestRegressor retorna valor contínuo de taxa de evasão
        taxa_predita = float(modelo_ativo.predict(x)[0])
        taxa_predita_clipped = float(np.clip(taxa_predita, 0.0, 1.0))

        # Interpretação binária simples (threshold 0.5)
        evasao_flag = int(taxa_predita_clipped >= 0.5)

        if evasao_flag == 1:
            st.error(f"🚨 Probabilidade alta de evasão — **{taxa_predita_clipped:.2%}**")
        else:
            st.success(f"✅ Probabilidade maior de permanência — evasão estimada em **{taxa_predita_clipped:.2%}**")

        st.metric("Taxa de evasão predita", f"{taxa_predita_clipped:.2%}")


# ===============================
#  TAB 3 — AVALIAÇÃO DO MODELO
# ===============================
with tab3:
    st.header("📊 Avaliação do Modelo")
    st.markdown("Avaliação usando todo o conjunto consolidado `dados_ingresso_evasao_conclusao.csv`.")

    X = df_ref[FEATURE_COLS]
    y_continuo = df_ref["taxa_evasao"]

    # Predição contínua
    y_pred_cont = modelo_ativo.predict(X)

    # Métricas de regressão
    mae = mean_absolute_error(y_continuo, y_pred_cont)
    rmse = np.sqrt(mean_squared_error(y_continuo, y_pred_cont))
    r2 = r2_score(y_continuo, y_pred_cont)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("R²", f"{r2:.4f}")

    st.markdown("---")

    # Conversão para classificação binária (threshold 0.5)
    y_true_bin = (y_continuo >= 0.5).astype(int)
    y_pred_bin = (np.clip(y_pred_cont, 0.0, 1.0) >= 0.5).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)
    rec = recall_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin)

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Acurácia (binária)", f"{acc:.4f}")
    col5.metric("F1-score", f"{f1:.4f}")
    col6.metric("Recall", f"{rec:.4f}")
    col7.metric("Precisão", f"{prec:.4f}")

    st.markdown("---")

    # Matriz de confusão
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_xlabel("Predito")
    ax_cm.set_ylabel("Verdadeiro")
    ax_cm.set_title("Matriz de Confusão — classificação binária da evasão")
    st.pyplot(fig_cm)

    st.markdown("---")

    # Dispersão real vs predito (regressão)
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
    ax_scatter.scatter(y_continuo, y_pred_cont, alpha=0.3)
    ax_scatter.plot([0, 1], [0, 1], "r--", label="Linha ideal")
    ax_scatter.set_xlabel("Taxa de evasão real")
    ax_scatter.set_ylabel("Taxa de evasão predita")
    ax_scatter.set_title("Dispersão — real vs predito")
    ax_scatter.legend()
    st.pyplot(fig_scatter)

    st.markdown("---")

    # Importância das features
    if hasattr(modelo_ativo, "feature_importances_"):
        importancias = modelo_ativo.feature_importances_
        df_imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": importancias,
        }).sort_values("importance", ascending=False)

        st.subheader("Importância das variáveis (Random Forest)")
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_imp, x="importance", y="feature", ax=ax_imp)
        ax_imp.set_xlabel("Importância relativa")
        ax_imp.set_ylabel("Variável")
        st.pyplot(fig_imp)

        st.dataframe(df_imp.reset_index(drop=True))


# ===============================
#  TAB 4 — UPLOAD DE CSV
# ===============================
with tab4:
    st.header("📁 Enviar Arquivo CSV para Previsão em Lote")
    st.markdown(
        "O arquivo deve conter **pelo menos** as colunas usadas pelo modelo:\n"
        f"`{', '.join(FEATURE_COLS)}`."
    )

    file = st.file_uploader("Envie um arquivo CSV", type=["csv"])

    if file is not None:
        df_upload = pd.read_csv(file)

        st.subheader("Pré-visualização do arquivo enviado")
        st.dataframe(df_upload.head())

        # Verifica se todas as colunas necessárias existem
        missing = [c for c in FEATURE_COLS if c not in df_upload.columns]
        if missing:
            st.error(
                "As seguintes colunas obrigatórias não foram encontradas no CSV enviado: "
                + ", ".join(missing)
            )
        else:
            X_up = df_upload[FEATURE_COLS]
            y_pred_up = modelo_ativo.predict(X_up)
            y_pred_up_clipped = np.clip(y_pred_up, 0.0, 1.0)
            evasao_flag = (y_pred_up_clipped >= 0.5).astype(int)

            df_result = df_upload.copy()
            df_result["taxa_evasao_predita"] = y_pred_up_clipped
            df_result["evasao_alta"] = evasao_flag

            st.success("Predições geradas com sucesso!")
            st.dataframe(df_result.head())

            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Baixar resultados com predições",
                data=csv_bytes,
                file_name="predicoes_evasao.csv",
                mime="text/csv",
            )

