# # # app_evasao.py
# # # Dashboard interativo com análise de evasão usando Streamlit

# import os
# from pathlib import Path
# import sys

# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     f1_score,
#     mean_absolute_error,
#     mean_squared_error,
#     precision_score,
#     r2_score,
#     recall_score,
# )

# # ===============================
# #  LOCALIZAÇÃO DOS ARQUIVOS
# # ===============================
# BASE_DIR = Path(__file__).resolve().parents[2]

# # Garante que o diretório raiz do projeto esteja no sys.path para permitir import de `scripts.*`
# if str(BASE_DIR) not in sys.path:
#     sys.path.insert(0, str(BASE_DIR))

# from scripts.modelagem.randomforest import treinar_modelos

# CAMINHO_DADOS = BASE_DIR / "dados" / "processado" / "dados_ingresso_evasao_conclusao.csv"
# CAMINHO_MODELO_BASE = BASE_DIR / "modelos" / "modelos_salvos" / "modelo_melhor_evasao.pkl"


# # ===============================
# #  CONFIGURAÇÃO DO LAYOUT STREAMLIT
# # ===============================
# st.set_page_config(
#     page_title="Predição de Evasão — Ensino Superior",
#     page_icon="📉",
#     layout="wide",
# )

# st.sidebar.title("📊 Predição de Evasão")
# st.sidebar.markdown("Aplicação prática — **2COP507 (Reconhecimento de Padrões)**")


# # ===============================
# #  CARREGAR BASE TRATADA
# # ===============================
# @st.cache_data(show_spinner=False)
# def load_reference_data() -> pd.DataFrame:
#     df = pd.read_csv(CAMINHO_DADOS, sep=";", encoding="utf-8", low_memory=False)
#     return df


# df_ref = load_reference_data()


# # ===============================
# #  CARREGAR MODELO BASE (TREINADO NO PIPELINE)
# # ===============================
# @st.cache_resource(show_spinner=False)
# def load_base_model():
#     modelo = joblib.load(CAMINHO_MODELO_BASE)
#     return modelo


# modelo_base = load_base_model()

# # Colunas de entrada usadas no modelo (garante compatibilidade com o pickle)
# if hasattr(modelo_base, "feature_names_in_"):
#     FEATURE_COLS = list(modelo_base.feature_names_in_)
# else:
#     # fallback: usa o mesmo conjunto principal do pipeline
#     FEATURE_COLS = [
#         "numero_cursos",
#         "vagas_totais",
#         "inscritos_totais",
#         "ingressantes",
#         "matriculados",
#         "concluintes",
#     ]


# # ===============================
# #  FUNÇÃO PARA TREINO CUSTOMIZADO
# # ===============================
# @st.cache_resource(show_spinner=False)
# def treinar_modelo_customizado(
#     n_estimators: int,
#     max_depth: int | None,
#     min_samples_split: int,
#     min_samples_leaf: int,
# ):
#     """Treina um RandomForestRegressor com hiperparâmetros escolhidos no sidebar."""

#     df = df_ref.copy()

#     X = df[FEATURE_COLS]
#     y = df["taxa_evasao"]

#     modelo = RandomForestRegressor(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         random_state=42,
#         n_jobs=-1,
#     )

#     modelo.fit(X, y)
#     return modelo


# # ===============================
# #  SIDEBAR — SELEÇÃO DE HIPERPARÂMETROS
# # ===============================
# st.sidebar.header("Ajuste de Hiperparâmetros")

# algoritmo = st.sidebar.selectbox(
#     "Algoritmo",
#     ["RandomForest — modelo do pipeline", "RandomForest — customizado"],
# )

# n_estimators = st.sidebar.slider("Número de Árvores (n_estimators)", 50, 500, 200, step=10)
# max_depth = st.sidebar.slider("Profundidade Máxima (max_depth)", 2, 30, 15)
# min_samples_split = st.sidebar.slider("Mínimo de amostras para dividir o nó (min_samples_split)", 2, 20, 4)
# min_samples_leaf = st.sidebar.slider("Mínimo de amostras na folha (min_samples_leaf)", 1, 50, 1)
# threshold_ui = st.sidebar.slider(
#     "Limiar para evasão alta (threshold)",
#     0.0,
#     1.0,
#     0.50,
#     step=0.01,
# )

# usar_custom = algoritmo == "RandomForest — customizado"

# if usar_custom:
#     modelo_ativo = treinar_modelo_customizado(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#     )
#     st.sidebar.success("Usando modelo RandomForest **customizado** treinado em tempo real.")
# else:
#     modelo_ativo = modelo_base
#     st.sidebar.info("Usando modelo RandomForest do **pipeline original** (pickle).")


# # ===============================
# #  FUNÇÃO AUXILIAR DE PRÉ-PROCESSAMENTO
# # ===============================
# def preprocess_row(row: pd.Series) -> np.ndarray:
#     """Recebe uma linha com FEATURES e devolve o array na ordem esperada pelo modelo."""

#     row = row.copy()
#     row = row[FEATURE_COLS]
#     return row.values.reshape(1, -1)


# # Estatísticas para sugerir valores padrão na interface
# stats = df_ref[FEATURE_COLS].describe()


# # ===============================
# #  TABS DA INTERFACE
# # ===============================

# tab1, tab2, tab3, tab4 = st.tabs(
#     [
#         "📘 Sobre a Aplicação",
#         "📈 Predição Individual",
#         "📊 Avaliação do Modelo",
#         "📁 Enviar Arquivo",
#     ]
# )


# # ===============================
# #  TAB 1 — SOBRE
# # ===============================
# with tab1:
#     st.header("Sobre a Aplicação")
#     st.markdown(
#         """
# Esta aplicação faz parte da disciplina **2COP507 – Reconhecimento de Padrões** e utiliza
# um modelo de **Random Forest** para estimar a **taxa de evasão** em cursos da educação superior.

# ### ✔️ O que você encontrará aqui
# - Interface amigável para experimentos com o modelo
# - Ajuste interativo de hiperparâmetros do Random Forest
# - Métricas de avaliação (MAE, RMSE, R², acurácia binária, F1 etc.)
# - Gráficos: dispersão, distribuição e matriz de confusão
# - Upload de arquivo CSV para predições em lote

# ### 🎯 Tecnologias utilizadas
# - **Streamlit** — interface web interativa
# - **Scikit-Learn** — RandomForestRegressor
# - **Pandas / NumPy** — manipulação dos dados
# - **Matplotlib / Seaborn** — visualização
# """
#     )


# # ===============================
# #  TAB 2 — PREDIÇÃO INDIVIDUAL
# # ===============================
# with tab2:
#     st.header("📈 Predição Individual de Taxa de Evasão")
#     st.markdown("Ajuste os valores das variáveis e clique em **Calcular**.")

#     # Cria inputs numéricos com base nas FEATURES usadas no modelo
#     cols = st.columns(3)
#     valores_usuario = {}

#     for idx, col_name in enumerate(FEATURE_COLS):
#         col_streamlit = cols[idx % 3]
#         desc = stats[col_name]
#         default_val = float(desc["50%"])
#         min_val = float(max(0, desc["min"]))
#         max_val = float(desc["max"])

#         with col_streamlit:
#             valores_usuario[col_name] = st.number_input(
#                 f"{col_name}",
#                 min_value=min_val,
#                 max_value=max_val,
#                 value=default_val,
#                 step=max(1.0, (max_val - min_val) / 100),
#             )

#     if st.button("🔮 Calcular probabilidade de evasão", key="btn_pred_individual"):
#         df_user = pd.DataFrame([valores_usuario])
#         x = preprocess_row(df_user.iloc[0])

#         # RandomForestRegressor retorna valor contínuo de taxa de evasão
#         taxa_predita = float(modelo_ativo.predict(x)[0])
#         taxa_predita_clipped = float(np.clip(taxa_predita, 0.0, 1.0))

#         # Interpretação binária simples (threshold 0.5)
#         evasao_flag = int(taxa_predita_clipped >= 0.5)

#         if evasao_flag == 1:
#             st.error(f"🚨 Probabilidade alta de evasão — **{taxa_predita_clipped:.2%}**")
#         else:
#             st.success(f"✅ Probabilidade maior de permanência — evasão estimada em **{taxa_predita_clipped:.2%}**")

#         st.metric("Taxa de evasão predita", f"{taxa_predita_clipped:.2%}")


# # ===============================
# #  TAB 3 — AVALIAÇÃO DO MODELO
# # ===============================
# with tab3:
#     st.header("📊 Avaliação do Modelo")
#     st.markdown("Avaliação usando todo o conjunto consolidado `dados_ingresso_evasao_conclusao.csv`.")

#     st.subheader("Resultados do pipeline (randomforest.py)")

#     resultados_backend = treinar_modelos(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         threshold_evasao_alta=threshold_ui,
#     )

#     # Métricas de regressão do backend
#     reg_backend = resultados_backend.get("modelo_evasao_regressao", {})
#     col_rb1, col_rb2 = st.columns(2)
#     if reg_backend:
#         col_rb1.metric(
#             "MSE (Random Forest — pipeline)",
#             f"{reg_backend.get('mse_random_forest', float('nan')):.4f}",
#         )
#         col_rb2.metric(
#             "R² (Random Forest — pipeline)",
#             f"{reg_backend.get('r2_random_forest', float('nan')):.4f}",
#         )

#     # Métricas de classificação binária do backend
#     clf_backend = resultados_backend.get("classificacao_evasao_binaria", {})
#     if clf_backend:
#         st.markdown("**Classificação binária da evasão (threshold do sidebar)**")
#         col_cb1, col_cb2, col_cb3, col_cb4 = st.columns(4)
#         col_cb1.metric("Acurácia (pipeline)", f"{clf_backend.get('accuracy', float('nan')):.4f}")
#         col_cb2.metric("Precisão (pipeline)", f"{clf_backend.get('precision', float('nan')):.4f}")
#         col_cb3.metric("Recall (pipeline)", f"{clf_backend.get('recall', float('nan')):.4f}")
#         col_cb4.metric("F1-score (pipeline)", f"{clf_backend.get('f1', float('nan')):.4f}")

#         caminho_cm_rel = clf_backend.get("caminho_matriz_confusao_png")
#         if caminho_cm_rel:
#             # Caminho relativo vindo do backend (ex.: "./modelos/resultados_modelos/matriz_confusao.png")
#             caminho_cm_abs = BASE_DIR / caminho_cm_rel.lstrip("./")
#             st.image(str(caminho_cm_abs), caption="Matriz de confusão — pipeline (randomforest.py)")

#     st.markdown("---")  # separador antes da avaliação local existente

#     X = df_ref[FEATURE_COLS]
#     y_continuo = df_ref["taxa_evasao"]

#     # Predição contínua
#     y_pred_cont = modelo_ativo.predict(X)

#     # Métricas de regressão
#     mae = mean_absolute_error(y_continuo, y_pred_cont)
#     rmse = np.sqrt(mean_squared_error(y_continuo, y_pred_cont))
#     r2 = r2_score(y_continuo, y_pred_cont)

#     col1, col2, col3 = st.columns(3)
#     col1.metric("MAE", f"{mae:.4f}")
#     col2.metric("RMSE", f"{rmse:.4f}")
#     col3.metric("R²", f"{r2:.4f}")

#     st.markdown("---")

#     # Conversão para classificação binária (threshold 0.5)
#     y_true_bin = (y_continuo >= 0.5).astype(int)
#     y_pred_bin = (np.clip(y_pred_cont, 0.0, 1.0) >= 0.5).astype(int)

#     acc = accuracy_score(y_true_bin, y_pred_bin)
#     f1 = f1_score(y_true_bin, y_pred_bin)
#     rec = recall_score(y_true_bin, y_pred_bin)
#     prec = precision_score(y_true_bin, y_pred_bin)

#     col4, col5, col6, col7 = st.columns(4)
#     col4.metric("Acurácia (binária)", f"{acc:.4f}")
#     col5.metric("F1-score", f"{f1:.4f}")
#     col6.metric("Recall", f"{rec:.4f}")
#     col7.metric("Precisão", f"{prec:.4f}")

#     st.markdown("---")

#     # Matriz de confusão
#     cm = confusion_matrix(y_true_bin, y_pred_bin)
#     fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
#     ax_cm.set_xlabel("Predito")
#     ax_cm.set_ylabel("Verdadeiro")
#     ax_cm.set_title("Matriz de Confusão — classificação binária da evasão")
#     st.pyplot(fig_cm)

#     st.markdown("---")

#     # Dispersão real vs predito (regressão)
#     fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
#     ax_scatter.scatter(y_continuo, y_pred_cont, alpha=0.3)
#     ax_scatter.plot([0, 1], [0, 1], "r--", label="Linha ideal")
#     ax_scatter.set_xlabel("Taxa de evasão real")
#     ax_scatter.set_ylabel("Taxa de evasão predita")
#     ax_scatter.set_title("Dispersão — real vs predito")
#     ax_scatter.legend()
#     st.pyplot(fig_scatter)

#     st.markdown("---")

#     # Importância das features
#     if hasattr(modelo_ativo, "feature_importances_"):
#         importancias = modelo_ativo.feature_importances_
#         df_imp = pd.DataFrame({
#             "feature": FEATURE_COLS,
#             "importance": importancias,
#         }).sort_values("importance", ascending=False)

#         st.subheader("Importância das variáveis (Random Forest)")
#         fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
#         sns.barplot(data=df_imp, x="importance", y="feature", ax=ax_imp)
#         ax_imp.set_xlabel("Importância relativa")
#         ax_imp.set_ylabel("Variável")
#         st.pyplot(fig_imp)

#         st.dataframe(df_imp.reset_index(drop=True))


# # ===============================
# #  TAB 4 — UPLOAD DE CSV
# # ===============================
# with tab4:
#     st.header("📁 Enviar Arquivo CSV para Previsão em Lote")
#     st.markdown(
#         "O arquivo deve conter **pelo menos** as colunas usadas pelo modelo:\n"
#         f"`{', '.join(FEATURE_COLS)}`."
#     )

#     file = st.file_uploader("Envie um arquivo CSV", type=["csv"])

#     if file is not None:
#         df_upload = pd.read_csv(file)

#         st.subheader("Pré-visualização do arquivo enviado")
#         st.dataframe(df_upload.head())

#         # Verifica se todas as colunas necessárias existem
#         missing = [c for c in FEATURE_COLS if c not in df_upload.columns]
#         if missing:
#             st.error(
#                 "As seguintes colunas obrigatórias não foram encontradas no CSV enviado: "
#                 + ", ".join(missing)
#             )
#         else:
#             X_up = df_upload[FEATURE_COLS]
#             y_pred_up = modelo_ativo.predict(X_up)
#             y_pred_up_clipped = np.clip(y_pred_up, 0.0, 1.0)
#             evasao_flag = (y_pred_up_clipped >= 0.5).astype(int)

#             df_result = df_upload.copy()
#             df_result["taxa_evasao_predita"] = y_pred_up_clipped
#             df_result["evasao_alta"] = evasao_flag

#             st.success("Predições geradas com sucesso!")
#             st.dataframe(df_result.head())

#             csv_bytes = df_result.to_csv(index=False).encode("utf-8")
#             st.download_button(
#                 "⬇️ Baixar resultados com predições",
#                 data=csv_bytes,
#                 file_name="predicoes_evasao.csv",
#                 mime="text/csv",
#             )

# app_evasao.py
# Dashboard interativo com análise de evasão usando Streamlit
# ---------------------------------------------------------------
# app_evasao.py
# Dashboard interativo com análise de evasão usando Streamlit
# ---------------------------------------------------------------

import os
from pathlib import Path
import sys
from typing import Optional
import time
import datetime

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

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.modelagem.randomforest import treinar_modelos

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
#  ESTADO GLOBAL
# ===============================
if "modo" not in st.session_state:
    st.session_state["modo"] = "pipeline"   # pipeline | custom

if "modelo_custom" not in st.session_state:
    st.session_state["modelo_custom"] = None

# ===============================
#  CARREGAR BASE TRATADA
# ===============================
@st.cache_data(show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    return pd.read_csv(CAMINHO_DADOS, sep=";", encoding="utf-8", low_memory=False)

df_ref = load_reference_data()

# ===============================
#  CARREGAR MODELO BASE
# ===============================
@st.cache_resource(show_spinner=False)
def load_base_model():
    return joblib.load(CAMINHO_MODELO_BASE)

modelo_base = load_base_model()

if hasattr(modelo_base, "feature_names_in_"):
    FEATURE_COLS = list(modelo_base.feature_names_in_)
else:
    FEATURE_COLS = [
        "numero_cursos",
        "vagas_totais",
        "inscritos_totais",
        "ingressantes",
        "matriculados",
        "concluintes",
    ]

stats = df_ref[FEATURE_COLS].describe()

# ===============================
#  TREINO CUSTOMIZADO (sob demanda)
# ===============================
@st.cache_resource(show_spinner=True)
def treinar_custom(
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
):
    df = df_ref.copy()
    X = df[FEATURE_COLS]
    y = df["taxa_evasao"]

    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(X, y)
    return modelo

# ===============================
#  PIPELINE BACKEND (caching)
# ===============================
@st.cache_resource(show_spinner=True)
def pipeline_backend(
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    threshold_evasao_alta: float,
):
    return treinar_modelos(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        threshold_evasao_alta=threshold_evasao_alta,
    )

# ===============================
#  SIDEBAR — BOTÕES E SLIDERS
# ===============================
st.sidebar.subheader("Modo do Modelo")

col_a, col_b = st.sidebar.columns(2)
btn_pipeline = col_a.button("📦 Pipeline")
btn_custom = col_b.button("🧪 Customizado")

if btn_pipeline:
    st.session_state["modo"] = "pipeline"

if btn_custom:
    st.session_state["modo"] = "custom"

modo = st.session_state["modo"]

if modo == "pipeline":
    st.sidebar.info("Usando modelo **pré-treinado** (pickle).")
else:
    if st.session_state["modelo_custom"] is None:
        st.sidebar.warning("Configure os sliders e clique em **Treinar modelo customizado**.")
    else:
        st.sidebar.success("Modelo customizado carregado.")

st.sidebar.markdown("---")

# Threshold global de decisão permanece no sidebar
threshold_ui = st.sidebar.slider("Threshold evasão", 0.0, 1.0, 0.5, 0.01)

# ===============================
#  MODELO ATIVO
# ===============================
if modo == "pipeline":
    modelo_ativo = modelo_base
else:
    modelo_ativo = st.session_state["modelo_custom"] or modelo_base

# ===============================
#  INTERFACE
# ===============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📘 Sobre", "📈 Predição Individual", "📊 Avaliação", "📁 Upload CSV"]
)

# -------------------------------------------------------
# TAB 1 — SOBRE
# -------------------------------------------------------
with tab1:
    st.header("📘 Sobre a Aplicação")
    st.write("""
Aplicação desenvolvida na disciplina **Reconhecimento de Padrões**.
Permite testar **Random Forest** na predição da taxa de evasão em cursos.
""")

# -------------------------------------------------------
# TAB 2 — PREDIÇÃO INDIVIDUAL
# -------------------------------------------------------
with tab2:
    st.header("📈 Predição Individual")

    if modo == "custom" and st.session_state["modelo_custom"] is None:
        st.info("⚠️ Treine o modelo customizado na aba **Avaliação**.")
    
    cols = st.columns(3)
    valores = {}

    for i, col in enumerate(FEATURE_COLS):
        c = cols[i % 3]
        valores[col] = c.number_input(
            col,
            min_value=float(stats[col]["min"]),
            max_value=float(stats[col]["max"]),
            value=float(stats[col]["50%"])
        )

    if st.button("🔮 Calcular evasão"):
        df_user = pd.DataFrame([valores])
        x = df_user[FEATURE_COLS].values.reshape(1, -1)
        y = float(np.clip(modelo_ativo.predict(x)[0], 0, 1))

        if y >= threshold_ui:
            st.error(f"🚨 Alta evasão: **{y:.2%}**")
        else:
            st.success(f"✅ Baixa evasão: **{y:.2%}**")

        # métricas locais individual (debug visual)
        st.subheader("📌 Métricas (execução pontual)")
        st.write(f"Valor previsto contínuo: {y:.6f}")
        st.write(f"Threshold aplicado: {threshold_ui}")
        st.write(f"Classificação binária: {'Alta evasão' if y >= threshold_ui else 'Baixa evasão'}")

# -------------------------------------------------------
# TAB 3 — AVALIAÇÃO
# -------------------------------------------------------
with tab3:
    st.header("📊 Avaliação do Modelo")

    # Estado de qual avaliação foi disparada pelo usuário
    if "ultimo_avaliado" not in st.session_state:
        st.session_state["ultimo_avaliado"] = None  # "pipeline" | "custom" | None

    st.markdown(
        """Nesta aba você pode comparar o comportamento do modelo **pipeline fixo**
        com um modelo **customizado**, retreinado em tempo real.

        - **Pipeline fixo**: usa o modelo salvo em disco (pickle), sem retreinar.
        - **Customizado**: chama o backend (randomforest.py) para treinar novamente
          com os hiperparâmetros escolhidos.
        """
    )

    # Botões para escolher o tipo de avaliação
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📦 Avaliar pipeline (fixo)", key="btn_avaliar_pipeline"):
            st.session_state["ultimo_avaliado"] = "pipeline"
    with col_btn2:
        if st.button("🧪 Treinar e avaliar customizado", key="btn_avaliar_custom"):
            st.session_state["ultimo_avaliado"] = "custom"

    st.markdown("---")

    # Card do modelo ativo + sliders dos hiperparâmetros para o CUSTOM
    with st.expander("🤖 Modelo ativo (pipeline ou customizado)", expanded=True):
        st.markdown(
            """O modelo ativo é aquele usado nas demais abas (**Predição Individual**
            e **Upload CSV**). Ele pode ser **pipeline** (pickle original) ou
            **customizado** (treinado a partir desta aba).
            
            - O threshold global de decisão (evasão alta) é o slider do sidebar.
            - Os hiperparâmetros abaixo valem para o **modelo customizado**.
            """
        )

        # 🔧 Hiperparâmetros do Random Forest (modelo customizado)
        st.markdown("#### Hiperparâmetros do Random Forest (modelo customizado)")
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            n_estimators = st.slider(
                "n_estimators",
                50, 500, 200, step=10,
                key="n_estimators_avaliacao",
            )
            max_depth = st.slider(
                "max_depth",
                2, 30, 15,
                key="max_depth_avaliacao",
            )

        with col_h2:
            min_samples_split = st.slider(
                "min_samples_split",
                2, 20, 4,
                key="min_samples_split_avaliacao",
            )
            min_samples_leaf = st.slider(
                "min_samples_leaf",
                1, 50, 1,
                key="min_samples_leaf_avaliacao",
            )

        # Botão opcional para treinar o modelo customizado que será usado
        # nas abas Predição Individual e Upload CSV (modelo_ativo).
        if modo == "custom":
            if st.button("🚀 Treinar modelo customizado (modelo ativo)", key="btn_train_custom_avaliacao"):
                start_local = time.perf_counter()
                with st.spinner("Treinando modelo customizado para uso nas demais abas..."):
                    modelo = treinar_custom(
                        n_estimators,
                        max_depth,
                        min_samples_split,
                        min_samples_leaf,
                    )
                elapsed_local = time.perf_counter() - start_local
                st.session_state["modelo_custom"] = modelo
                st.session_state["custom_treino_segundos"] = elapsed_local
                st.session_state["custom_ultima_execucao"] = datetime.datetime.now()
                st.success(f"Modelo customizado (ativo) treinado em {elapsed_local:.1f} segundos.")

        # Atualiza a referência local do modelo ativo para exibição nas abas 2 e 4
        if modo == "pipeline":
            modelo_ativo_local = modelo_base
        else:
            modelo_ativo_local = st.session_state["modelo_custom"] or modelo_base

    st.markdown("---")

    # ======================================================
    # 1) MÉTRICAS DO PIPELINE FIXO (sem retreinamento)
    # ======================================================
    if st.session_state["ultimo_avaliado"] == "pipeline":
        st.subheader("📦 Resultados do pipeline fixo (modelo salvo)")

        X_full = df_ref[FEATURE_COLS]
        y_full = df_ref["taxa_evasao"]
        y_pred_full = modelo_base.predict(X_full)

        mae_p = mean_absolute_error(y_full, y_pred_full)
        rmse_p = np.sqrt(mean_squared_error(y_full, y_pred_full))
        r2_p = r2_score(y_full, y_pred_full)

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("MAE (pipeline)", f"{mae_p:.4f}")
        col_p2.metric("RMSE (pipeline)", f"{rmse_p:.4f}")
        col_p3.metric("R² (pipeline)", f"{r2_p:.4f}")

        # Classificação binária (pipeline fixo)
        y_bin_true_p = (y_full >= threshold_ui).astype(int)
        y_bin_pred_p = (np.clip(y_pred_full, 0.0, 1.0) >= threshold_ui).astype(int)

        acc_p = accuracy_score(y_bin_true_p, y_bin_pred_p)
        f1_p = f1_score(y_bin_true_p, y_bin_pred_p)
        rec_p = recall_score(y_bin_true_p, y_bin_pred_p)
        prec_p = precision_score(y_bin_true_p, y_bin_pred_p)

        col_p4, col_p5, col_p6, col_p7 = st.columns(4)
        col_p4.metric("Acurácia (pipeline)", f"{acc_p:.4f}")
        col_p5.metric("F1 (pipeline)", f"{f1_p:.4f}")
        col_p6.metric("Recall (pipeline)", f"{rec_p:.4f}")
        col_p7.metric("Precisão (pipeline)", f"{prec_p:.4f}")

        st.markdown("#### Matriz de confusão — pipeline fixo")
        cm_p = confusion_matrix(y_bin_true_p, y_bin_pred_p)
        fig_p, ax_p = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_p, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_p)
        ax_p.set_xlabel("Predito")
        ax_p.set_ylabel("Verdadeiro")
        st.pyplot(fig_p)

        st.markdown("#### Dispersão — real vs predito (pipeline)")
        fig_p_sc, ax_p_sc = plt.subplots(figsize=(6, 4))
        ax_p_sc.scatter(y_full, y_pred_full, alpha=0.3)
        ax_p_sc.plot([0, 1], [0, 1], "r--", label="Linha ideal")
        ax_p_sc.set_xlabel("Taxa de evasão real")
        ax_p_sc.set_ylabel("Taxa de evasão predita")
        ax_p_sc.legend()
        st.pyplot(fig_p_sc)

        if hasattr(modelo_base, "feature_importances_"):
            st.markdown("#### Importância das variáveis — pipeline")
            imp_p = pd.DataFrame({
                "feature": FEATURE_COLS,
                "importance": modelo_base.feature_importances_,
            }).sort_values("importance", ascending=False)
            fig_imp_p, ax_imp_p = plt.subplots(figsize=(6, 4))
            sns.barplot(data=imp_p, x="importance", y="feature", ax=ax_imp_p)
            ax_imp_p.set_xlabel("Importância relativa")
            ax_imp_p.set_ylabel("Variável")
            st.pyplot(fig_imp_p)
            st.dataframe(imp_p.reset_index(drop=True))

        # Guarda métricas no estado para comparação
        st.session_state["metrics_pipeline"] = {
            "MAE": mae_p,
            "RMSE": rmse_p,
            "R2": r2_p,
            "accuracy": acc_p,
            "f1": f1_p,
            "recall": rec_p,
            "precision": prec_p,
        }

    # ======================================================
    # 2) MÉTRICAS DO CUSTOM (backend randomforest.py)
    # ======================================================
    elif st.session_state["ultimo_avaliado"] == "custom":
        st.subheader("🧪 Resultados do modelo customizado (backend randomforest.py)")

        start_backend = time.perf_counter()
        with st.spinner("Executando backend (treinar_modelos em randomforest.py)..."):
            resultados = pipeline_backend(
                n_estimators,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                threshold_ui,
            )
        elapsed_backend = time.perf_counter() - start_backend
        st.caption(f"⏱ Tempo de execução do backend (custom): {elapsed_backend:.1f} segundos")

        reg = resultados.get("modelo_evasao_regressao", {})
        clf = resultados.get("classificacao_evasao_binaria", {})

        if reg:
            col_c1, col_c2 = st.columns(2)
            col_c1.metric("MSE (custom)", f"{reg.get('mse_random_forest', float('nan')):.4f}")
            col_c2.metric("R² (custom)", f"{reg.get('r2_random_forest', float('nan')):.4f}")

        if clf:
            col_c3, col_c4, col_c5, col_c6 = st.columns(4)
            col_c3.metric("Acurácia (custom)", f"{clf.get('accuracy', float('nan')):.4f}")
            col_c4.metric("Precisão (custom)", f"{clf.get('precision', float('nan')):.4f}")
            col_c5.metric("Recall (custom)", f"{clf.get('recall', float('nan')):.4f}")
            col_c6.metric("F1 (custom)", f"{clf.get('f1', float('nan')):.4f}")

            caminho_cm_rel = clf.get("caminho_matriz_confusao_png")
            if caminho_cm_rel:
                caminho_cm_abs = BASE_DIR / caminho_cm_rel.lstrip("./")
                st.markdown("#### Matriz de confusão — custom (backend)")
                st.image(str(caminho_cm_abs), caption="Matriz de confusão — randomforest.py")

        # Guarda métricas no estado para comparação
        if reg and clf:
            st.session_state["metrics_custom"] = {
                "MAE": reg.get("mae_random_forest", float("nan")) if "mae_random_forest" in reg else float("nan"),
                "RMSE": reg.get("rmse_random_forest", float("nan")) if "rmse_random_forest" in reg else float("nan"),
                "R2": reg.get("r2_random_forest", float("nan")),
                "accuracy": clf.get("accuracy", float("nan")),
                "f1": clf.get("f1", float("nan")),
                "recall": clf.get("recall", float("nan")),
                "precision": clf.get("precision", float("nan")),
            }

    else:
        st.info("Selecione uma das opções acima para visualizar as métricas: **pipeline fixo** ou **customizado**.")

    st.markdown("---")

    # ======================================================
    # 3) COMPARAÇÃO RESUMIDA (se tivermos as duas métricas)
    # ======================================================
    metrics_p = st.session_state.get("metrics_pipeline")
    metrics_c = st.session_state.get("metrics_custom")

    if metrics_p and metrics_c:
        st.subheader("📊 Comparação resumida: Pipeline vs Custom")

        linhas = []
        for nome in ["MAE", "RMSE", "R2", "accuracy", "precision", "recall", "f1"]:
            p_val = metrics_p.get(nome)
            c_val = metrics_c.get(nome)
            if p_val is None or c_val is None:
                continue
            diff = c_val - p_val
            linhas.append({
                "Métrica": nome,
                "Pipeline": p_val,
                "Custom": c_val,
                "Diferença (Custom - Pipeline)": diff,
            })

        if linhas:
            df_comp = pd.DataFrame(linhas)
            st.dataframe(df_comp)

# -------------------------------------------------------
# TAB 4 — CSV
# -------------------------------------------------------
with tab4:
    st.header("📁 Upload CSV")

    file = st.file_uploader("Enviar CSV", type=["csv"])

    if file:
        df_up = pd.read_csv(file)
        st.dataframe(df_up.head())

        missing = [c for c in FEATURE_COLS if c not in df_up.columns]

        if missing:
            st.error(f"Colunas faltando: {missing}")
        else:
            X_up = df_up[FEATURE_COLS]
            y_pred = np.clip(modelo_ativo.predict(X_up), 0, 1)
            df_up["evasao_pred"] = y_pred
            df_up["evasao_alta"] = (y_pred >= threshold_ui).astype(int)

            st.success("Predições concluídas.")
            st.dataframe(df_up.head())

            st.download_button(
                "⬇️ Baixar resultados",
                df_up.to_csv(index=False).encode("utf-8"),
                "predicoes_evasao.csv",
                mime="text/csv"
            )