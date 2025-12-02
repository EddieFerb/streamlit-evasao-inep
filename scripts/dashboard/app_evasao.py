# # app_evasao.py
# # Dashboard interativo com análise de evasão usando Streamlit
# # ---------------------------------------------------------------
# # app_evasao.py
# # Dashboard interativo com análise de evasão usando Streamlit
# # ---------------------------------------------------------------

# import os
# from pathlib import Path
# import sys
# from typing import Optional
# import time
# import datetime

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
# #  ESTADO GLOBAL
# # ===============================
# if "modo" not in st.session_state:
#     st.session_state["modo"] = "pipeline"   # pipeline | custom

# if "modelo_custom" not in st.session_state:
#     st.session_state["modelo_custom"] = None

# # ===============================
# #  CARREGAR BASE TRATADA
# # ===============================
# @st.cache_data(show_spinner=False)
# def load_reference_data() -> pd.DataFrame:
#     return pd.read_csv(CAMINHO_DADOS, sep=";", encoding="utf-8", low_memory=False)

# df_ref = load_reference_data()

# # ===============================
# #  CARREGAR MODELO BASE
# # ===============================
# @st.cache_resource(show_spinner=False)
# def load_base_model():
#     return joblib.load(CAMINHO_MODELO_BASE)

# modelo_base = load_base_model()

# if hasattr(modelo_base, "feature_names_in_"):
#     FEATURE_COLS = list(modelo_base.feature_names_in_)
# else:
#     FEATURE_COLS = [
#         "numero_cursos",
#         "vagas_totais",
#         "inscritos_totais",
#         "ingressantes",
#         "matriculados",
#         "concluintes",
#     ]

# stats = df_ref[FEATURE_COLS].describe()

# # ===============================
# #  TREINO CUSTOMIZADO (sob demanda)
# # ===============================
# @st.cache_resource(show_spinner=True)
# def treinar_custom(
#     n_estimators: int,
#     max_depth: Optional[int],
#     min_samples_split: int,
#     min_samples_leaf: int,
# ):
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
# #  PIPELINE BACKEND (caching)
# # ===============================
# @st.cache_resource(show_spinner=True)
# def pipeline_backend(
#     n_estimators: int,
#     max_depth: Optional[int],
#     min_samples_split: int,
#     min_samples_leaf: int,
#     threshold_evasao_alta: float,
# ):
#     return treinar_modelos(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         threshold_evasao_alta=threshold_evasao_alta,
#     )

# # ===============================
# #  SIDEBAR — BOTÕES E SLIDERS
# # ===============================
# st.sidebar.subheader("Modo do Modelo")

# col_a, col_b = st.sidebar.columns(2)
# btn_pipeline = col_a.button("📦 Pipeline")
# btn_custom = col_b.button("🧪 Customizado")

# if btn_pipeline:
#     st.session_state["modo"] = "pipeline"

# if btn_custom:
#     st.session_state["modo"] = "custom"

# modo = st.session_state["modo"]

# if modo == "pipeline":
#     st.sidebar.info("Usando modelo **pré-treinado** (pickle).")
# else:
#     if st.session_state["modelo_custom"] is None:
#         st.sidebar.warning("Configure os sliders e clique em **Treinar modelo customizado**.")
#     else:
#         st.sidebar.success("Modelo customizado carregado.")

# st.sidebar.markdown("---")

# # Threshold global de decisão permanece no sidebar
# threshold_ui = st.sidebar.slider("Threshold evasão", 0.0, 1.0, 0.5, 0.01)

# # ===============================
# #  MODELO ATIVO
# # ===============================
# if modo == "pipeline":
#     modelo_ativo = modelo_base
# else:
#     modelo_ativo = st.session_state["modelo_custom"] or modelo_base

# # ===============================
# #  INTERFACE
# # ===============================
# tab1, tab2, tab3, tab4 = st.tabs(
#     ["📘 Sobre", "📈 Predição Individual", "📊 Avaliação", "📁 Upload CSV"]
# )

# # -------------------------------------------------------
# # TAB 1 — SOBRE
# # -------------------------------------------------------
# with tab1:
#     st.header("📘 Sobre a Aplicação")
#     st.write("""
# Aplicação desenvolvida na disciplina **Reconhecimento de Padrões**.
# Permite testar **Random Forest** na predição da taxa de evasão em cursos.
# """)

# # -------------------------------------------------------
# # TAB 2 — PREDIÇÃO INDIVIDUAL
# # -------------------------------------------------------
# with tab2:
#     st.header("📈 Predição Individual")

#     if modo == "custom" and st.session_state["modelo_custom"] is None:
#         st.info("⚠️ Treine o modelo customizado na aba **Avaliação**.")
    
#     cols = st.columns(3)
#     valores = {}

#     for i, col in enumerate(FEATURE_COLS):
#         c = cols[i % 3]
#         valores[col] = c.number_input(
#             col,
#             min_value=float(stats[col]["min"]),
#             max_value=float(stats[col]["max"]),
#             value=float(stats[col]["50%"])
#         )

#     if st.button("🔮 Calcular evasão"):
#         df_user = pd.DataFrame([valores])
#         x = df_user[FEATURE_COLS].values.reshape(1, -1)
#         y = float(np.clip(modelo_ativo.predict(x)[0], 0, 1))

#         if y >= threshold_ui:
#             st.error(f"🚨 Alta evasão: **{y:.2%}**")
#         else:
#             st.success(f"✅ Baixa evasão: **{y:.2%}**")

#         # métricas locais individual (debug visual)
#         st.subheader("📌 Métricas (execução pontual)")
#         st.write(f"Valor previsto contínuo: {y:.6f}")
#         st.write(f"Threshold aplicado: {threshold_ui}")
#         st.write(f"Classificação binária: {'Alta evasão' if y >= threshold_ui else 'Baixa evasão'}")

# # -------------------------------------------------------
# # TAB 3 — AVALIAÇÃO
# # -------------------------------------------------------
# with tab3:
#     st.header("📊 Avaliação do Modelo")

#     # Estado de qual avaliação foi disparada pelo usuário
#     if "ultimo_avaliado" not in st.session_state:
#         st.session_state["ultimo_avaliado"] = None  # "pipeline" | "custom" | None

#     st.markdown(
#         """Nesta aba você pode comparar o comportamento do modelo **pipeline fixo**
#         com um modelo **customizado**, retreinado em tempo real.

#         - **Pipeline fixo**: usa o modelo salvo em disco (pickle), sem retreinar.
#         - **Customizado**: chama o backend (randomforest.py) para treinar novamente
#           com os hiperparâmetros escolhidos.
#         """
#     )

#     # Botões para escolher o tipo de avaliação
#     col_btn1, col_btn2 = st.columns(2)
#     with col_btn1:
#         if st.button("📦 Avaliar pipeline (fixo)", key="btn_avaliar_pipeline"):
#             st.session_state["ultimo_avaliado"] = "pipeline"
#     with col_btn2:
#         if st.button("🧪 Treinar e avaliar customizado", key="btn_avaliar_custom"):
#             st.session_state["ultimo_avaliado"] = "custom"

#     st.markdown("---")

#     # Card do modelo ativo + sliders dos hiperparâmetros para o CUSTOM
#     with st.expander("🤖 Modelo ativo (pipeline ou customizado)", expanded=True):
#         st.markdown(
#             """O modelo ativo é aquele usado nas demais abas (**Predição Individual**
#             e **Upload CSV**). Ele pode ser **pipeline** (pickle original) ou
#             **customizado** (treinado a partir desta aba).
            
#             - O threshold global de decisão (evasão alta) é o slider do sidebar.
#             - Os hiperparâmetros abaixo valem para o **modelo customizado**.
#             """
#         )

#         # 🔧 Hiperparâmetros do Random Forest (modelo customizado)
#         st.markdown("#### Hiperparâmetros do Random Forest (modelo customizado)")
#         col_h1, col_h2 = st.columns(2)

#         with col_h1:
#             n_estimators = st.slider(
#                 "n_estimators",
#                 50, 500, 200, step=10,
#                 key="n_estimators_avaliacao",
#             )
#             max_depth = st.slider(
#                 "max_depth",
#                 2, 30, 15,
#                 key="max_depth_avaliacao",
#             )

#         with col_h2:
#             min_samples_split = st.slider(
#                 "min_samples_split",
#                 2, 20, 4,
#                 key="min_samples_split_avaliacao",
#             )
#             min_samples_leaf = st.slider(
#                 "min_samples_leaf",
#                 1, 50, 1,
#                 key="min_samples_leaf_avaliacao",
#             )

#         # Botão opcional para treinar o modelo customizado que será usado
#         # nas abas Predição Individual e Upload CSV (modelo_ativo).
#         if modo == "custom":
#             if st.button("🚀 Treinar modelo customizado (modelo ativo)", key="btn_train_custom_avaliacao"):
#                 start_local = time.perf_counter()
#                 with st.spinner("Treinando modelo customizado para uso nas demais abas..."):
#                     modelo = treinar_custom(
#                         n_estimators,
#                         max_depth,
#                         min_samples_split,
#                         min_samples_leaf,
#                     )
#                 elapsed_local = time.perf_counter() - start_local
#                 st.session_state["modelo_custom"] = modelo
#                 st.session_state["custom_treino_segundos"] = elapsed_local
#                 st.session_state["custom_ultima_execucao"] = datetime.datetime.now()
#                 st.success(f"Modelo customizado (ativo) treinado em {elapsed_local:.1f} segundos.")

#         # Atualiza a referência local do modelo ativo para exibição nas abas 2 e 4
#         if modo == "pipeline":
#             modelo_ativo_local = modelo_base
#         else:
#             modelo_ativo_local = st.session_state["modelo_custom"] or modelo_base

#     st.markdown("---")

#     # ======================================================
#     # 1) MÉTRICAS DO PIPELINE FIXO (sem retreinamento)
#     # ======================================================
#     if st.session_state["ultimo_avaliado"] == "pipeline":
#         st.subheader("📦 Resultados do pipeline fixo (modelo salvo)")

#         X_full = df_ref[FEATURE_COLS]
#         y_full = df_ref["taxa_evasao"]
#         y_pred_full = modelo_base.predict(X_full)

#         mae_p = mean_absolute_error(y_full, y_pred_full)
#         rmse_p = np.sqrt(mean_squared_error(y_full, y_pred_full))
#         r2_p = r2_score(y_full, y_pred_full)

#         col_p1, col_p2, col_p3 = st.columns(3)
#         col_p1.metric("MAE (pipeline)", f"{mae_p:.4f}")
#         col_p2.metric("RMSE (pipeline)", f"{rmse_p:.4f}")
#         col_p3.metric("R² (pipeline)", f"{r2_p:.4f}")

#         # Classificação binária (pipeline fixo)
#         y_bin_true_p = (y_full >= threshold_ui).astype(int)
#         y_bin_pred_p = (np.clip(y_pred_full, 0.0, 1.0) >= threshold_ui).astype(int)

#         acc_p = accuracy_score(y_bin_true_p, y_bin_pred_p)
#         f1_p = f1_score(y_bin_true_p, y_bin_pred_p)
#         rec_p = recall_score(y_bin_true_p, y_bin_pred_p)
#         prec_p = precision_score(y_bin_true_p, y_bin_pred_p)

#         col_p4, col_p5, col_p6, col_p7 = st.columns(4)
#         col_p4.metric("Acurácia (pipeline)", f"{acc_p:.4f}")
#         col_p5.metric("F1 (pipeline)", f"{f1_p:.4f}")
#         col_p6.metric("Recall (pipeline)", f"{rec_p:.4f}")
#         col_p7.metric("Precisão (pipeline)", f"{prec_p:.4f}")

#         st.markdown("#### Matriz de confusão — pipeline fixo")
#         cm_p = confusion_matrix(y_bin_true_p, y_bin_pred_p)
#         fig_p, ax_p = plt.subplots(figsize=(5, 4))
#         sns.heatmap(cm_p, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_p)
#         ax_p.set_xlabel("Predito")
#         ax_p.set_ylabel("Verdadeiro")
#         st.pyplot(fig_p)

#         st.markdown("#### Dispersão — real vs predito (pipeline)")
#         fig_p_sc, ax_p_sc = plt.subplots(figsize=(6, 4))
#         ax_p_sc.scatter(y_full, y_pred_full, alpha=0.3)
#         ax_p_sc.plot([0, 1], [0, 1], "r--", label="Linha ideal")
#         ax_p_sc.set_xlabel("Taxa de evasão real")
#         ax_p_sc.set_ylabel("Taxa de evasão predita")
#         ax_p_sc.legend()
#         st.pyplot(fig_p_sc)

#         if hasattr(modelo_base, "feature_importances_"):
#             st.markdown("#### Importância das variáveis — pipeline")
#             imp_p = pd.DataFrame({
#                 "feature": FEATURE_COLS,
#                 "importance": modelo_base.feature_importances_,
#             }).sort_values("importance", ascending=False)
#             fig_imp_p, ax_imp_p = plt.subplots(figsize=(6, 4))
#             sns.barplot(data=imp_p, x="importance", y="feature", ax=ax_imp_p)
#             ax_imp_p.set_xlabel("Importância relativa")
#             ax_imp_p.set_ylabel("Variável")
#             st.pyplot(fig_imp_p)
#             st.dataframe(imp_p.reset_index(drop=True))

#         # Guarda métricas no estado para comparação
#         st.session_state["metrics_pipeline"] = {
#             "MAE": mae_p,
#             "RMSE": rmse_p,
#             "R2": r2_p,
#             "accuracy": acc_p,
#             "f1": f1_p,
#             "recall": rec_p,
#             "precision": prec_p,
#         }

#     # ======================================================
#     # 2) MÉTRICAS DO CUSTOM (backend randomforest.py)
#     # ======================================================
#     elif st.session_state["ultimo_avaliado"] == "custom":
#         st.subheader("🧪 Resultados do modelo customizado (backend randomforest.py)")

#         start_backend = time.perf_counter()
#         with st.spinner("Executando backend (treinar_modelos em randomforest.py)..."):
#             resultados = pipeline_backend(
#                 n_estimators,
#                 max_depth,
#                 min_samples_split,
#                 min_samples_leaf,
#                 threshold_ui,
#             )
#         elapsed_backend = time.perf_counter() - start_backend
#         st.caption(f"⏱ Tempo de execução do backend (custom): {elapsed_backend:.1f} segundos")

#         reg = resultados.get("modelo_evasao_regressao", {})
#         clf = resultados.get("classificacao_evasao_binaria", {})

#         if reg:
#             col_c1, col_c2 = st.columns(2)
#             col_c1.metric("MSE (custom)", f"{reg.get('mse_random_forest', float('nan')):.4f}")
#             col_c2.metric("R² (custom)", f"{reg.get('r2_random_forest', float('nan')):.4f}")

#         if clf:
#             col_c3, col_c4, col_c5, col_c6 = st.columns(4)
#             col_c3.metric("Acurácia (custom)", f"{clf.get('accuracy', float('nan')):.4f}")
#             col_c4.metric("Precisão (custom)", f"{clf.get('precision', float('nan')):.4f}")
#             col_c5.metric("Recall (custom)", f"{clf.get('recall', float('nan')):.4f}")
#             col_c6.metric("F1 (custom)", f"{clf.get('f1', float('nan')):.4f}")

#             caminho_cm_rel = clf.get("caminho_matriz_confusao_png")
#             if caminho_cm_rel:
#                 caminho_cm_abs = BASE_DIR / caminho_cm_rel.lstrip("./")
#                 st.markdown("#### Matriz de confusão — custom (backend)")
#                 st.image(str(caminho_cm_abs), caption="Matriz de confusão — randomforest.py")

#         # Guarda métricas no estado para comparação
#         if reg and clf:
#             st.session_state["metrics_custom"] = {
#                 "MAE": reg.get("mae_random_forest", float("nan")) if "mae_random_forest" in reg else float("nan"),
#                 "RMSE": reg.get("rmse_random_forest", float("nan")) if "rmse_random_forest" in reg else float("nan"),
#                 "R2": reg.get("r2_random_forest", float("nan")),
#                 "accuracy": clf.get("accuracy", float("nan")),
#                 "f1": clf.get("f1", float("nan")),
#                 "recall": clf.get("recall", float("nan")),
#                 "precision": clf.get("precision", float("nan")),
#             }

#     else:
#         st.info("Selecione uma das opções acima para visualizar as métricas: **pipeline fixo** ou **customizado**.")

#     st.markdown("---")

#     # ======================================================
#     # 3) COMPARAÇÃO RESUMIDA (se tivermos as duas métricas)
#     # ======================================================
#     metrics_p = st.session_state.get("metrics_pipeline")
#     metrics_c = st.session_state.get("metrics_custom")

#     if metrics_p and metrics_c:
#         st.subheader("📊 Comparação resumida: Pipeline vs Custom")

#         linhas = []
#         for nome in ["MAE", "RMSE", "R2", "accuracy", "precision", "recall", "f1"]:
#             p_val = metrics_p.get(nome)
#             c_val = metrics_c.get(nome)
#             if p_val is None or c_val is None:
#                 continue
#             diff = c_val - p_val
#             linhas.append({
#                 "Métrica": nome,
#                 "Pipeline": p_val,
#                 "Custom": c_val,
#                 "Diferença (Custom - Pipeline)": diff,
#             })

#         if linhas:
#             df_comp = pd.DataFrame(linhas)
#             st.dataframe(df_comp)

# # -------------------------------------------------------
# # TAB 4 — CSV
# # -------------------------------------------------------
# with tab4:
#     st.header("📁 Upload CSV")

#     file = st.file_uploader("Enviar CSV", type=["csv"])

#     if file:
#         df_up = pd.read_csv(file)
#         st.dataframe(df_up.head())

#         missing = [c for c in FEATURE_COLS if c not in df_up.columns]

#         if missing:
#             st.error(f"Colunas faltando: {missing}")
#         else:
#             X_up = df_up[FEATURE_COLS]
#             y_pred = np.clip(modelo_ativo.predict(X_up), 0, 1)
#             df_up["evasao_pred"] = y_pred
#             df_up["evasao_alta"] = (y_pred >= threshold_ui).astype(int)

#             st.success("Predições concluídas.")
#             st.dataframe(df_up.head())

#             st.download_button(
#                 "⬇️ Baixar resultados",
#                 df_up.to_csv(index=False).encode("utf-8"),
#                 "predicoes_evasao.csv",
#                 mime="text/csv"
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
#  CARREGAR MODELO BASE (leve, em runtime)
# ===============================
@st.cache_resource(show_spinner=True)
def load_base_model():
    with st.spinner("Treinando modelo base inicial (leve)..."):
        df = load_reference_data()

        feature_cols = [
            "numero_cursos",
            "vagas_totais",
            "inscritos_totais",
            "ingressantes",
            "matriculados",
            "concluintes",
        ]

        # Remove linhas com NaN nas features ou na taxa de evasão
        df_model = df.dropna(subset=feature_cols + ["taxa_evasao"]).copy()

        # ---- AMOSTRAGEM PARA FICAR RÁPIDO NO STREAMLIT CLOUD ----
        if len(df_model) > 50000:
            df_model = df_model.sample(50000, random_state=42)

        X = df_model[feature_cols]
        y = df_model["taxa_evasao"]

        modelo = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        modelo.fit(X, y)

    return modelo, feature_cols

modelo_base, FEATURE_COLS = load_base_model()
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

    st.markdown(
        """
### Contexto: por que olhar para a evasão?

A evasão no ensino superior brasileiro é um problema estrutural. Reportagem recente do
[Jornal da Unesp](https://jornal.unesp.br/2025/11/28/expansao-do-ensino-superior-no-brasil-se-deu-antes-que-se-consolidasse-a-qualidade-da-educacao-basica-e-enfrentamos-os-resultados-desta-opcao-diz-cristovam-buarque/)
indica que cerca de **metade dos estudantes** não conclui a graduação, o que impacta
planejamento acadêmico, orçamento institucional e políticas públicas.

Esse comportamento é **multivariado**: depende de vagas ofertadas, inscritos, ingressantes,
matriculados, concluintes, modalidade (presencial/EaD), rede (pública/privada) e do ano de oferta.
Intuição humana sozinha não é suficiente para entender todos esses padrões.

### O que é o App Evasão?

O **App Evasão** é uma aplicação web em **Streamlit** que:

- estima a **taxa de evasão** (0 a 1) de cursos de graduação em IES brasileiras;
- usa **microdados oficiais do INEP/MEC (2009–2024)** já tratados em um pipeline de dados;
- permite simular cenários “*e se…*” alterando vagas, ingressantes, matriculados e concluintes;
- oferece uma interface simples para **gestores, pesquisadores e estudantes** explorarem os dados.

A aplicação foi desenvolvida como projeto prático da disciplina **2COP507 – Reconhecimento de Padrões**, integrando
um pipeline completo de Aprendizado de Máquina com visualização interativa.

### Como funciona o pipeline de dados e modelos?

O backend do App é composto por um pipeline em Python dividido em múltiplos scripts, que cuidam de:

1. **Coleta automatizada** dos arquivos oficiais do INEP.
2. **Pré-processamento e padronização** das variáveis (limpeza, tipos, nomes, filtros).
3. **Cálculo das taxas educacionais** (ingresso, conclusão, evasão) por curso/ano.
4. **Análise exploratória automatizada (EDA)** com gráficos de séries históricas.
5. **Treinamento de modelos** (Regressão Linear e Random Forest) com *split* temporal:
   - treino: 2009–2018  
   - teste: 2019–2024
6. **Seleção do melhor modelo** (RandomForestRegressor) e salvamento em arquivo `.pkl`.
7. **Geração de gráficos e métricas** para avaliação.
8. **App Streamlit (este painel)** que carrega os artefatos e disponibiliza as predições.

O modelo principal é um **Random Forest regressivo**, que aprende relações entre:

- número de cursos;  
- vagas totais;  
- inscritos;  
- ingressantes;  
- matriculados;  
- concluintes;  

e a **taxa de evasão** histórica. Para evitar *data leakage*, as próprias taxas (ingresso, conclusão, evasão) **não** são usadas como entrada, apenas como alvo na etapa de treino.

### Como usar este painel?

- Use a aba **“📈 Predição Individual”** para testar um curso hipotético ou real e ver a evasão estimada.
- Use a aba **“📊 Avaliação”** para inspecionar métricas, gráficos e comparar modelos.
- Use a aba **“📁 Upload CSV”** para gerar predições em lote para vários cursos ao mesmo tempo.
"""
    )

    st.markdown("### Recursos do projeto")

    col_links1, col_links2 = st.columns(2)
    with col_links1:
        st.link_button("▶️ Vídeo no YouTube", "https://youtu.be/J4HJlpyYT8M")
    with col_links2:
        st.link_button("💻 Código no GitHub", "https://github.com/EddieFerb/streamlit-evasao-inep.git")

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

    st.markdown(
        f"""
Esta aba permite gerar **predições em lote** para vários cursos de uma só vez.

O arquivo deve ser um **CSV** contendo, no mínimo, as seguintes colunas numéricas
(já agregadas por curso/ano), com estes nomes exatos:

- `{", ".join(FEATURE_COLS)}`

Essas colunas devem representar os mesmos conceitos usados no pipeline oficial:

- número de cursos ofertados (`numero_cursos`);
- vagas totais disponíveis (`vagas_totais`);
- total de inscritos (`inscritos_totais`);
- ingressantes (`ingressantes`);
- matriculados (`matriculados`);
- concluintes (`concluintes`).

O modelo estima a **taxa de evasão** para cada linha do CSV (valores entre 0 e 1) e,
a partir do *threshold* configurado na barra lateral, classifica cada curso em
**“evasão alta”** ou **“evasão baixa”**.

**Recomendações de uso:**

1. Gere o CSV a partir do pipeline deste projeto ou de bases com estrutura semelhante
   (por exemplo, microdados do INEP agrupados por curso/ano).
2. Verifique se as colunas obrigatórias existem e têm valores numéricos válidos.
3. Evite incluir informações sensíveis de indivíduos — o app foi pensado para dados
   agregados por curso, não por aluno.

Depois do processamento, você poderá:

- visualizar uma amostra dos resultados diretamente na tela;
- baixar um novo arquivo CSV com as colunas `evasao_pred` (taxa prevista) e
  `evasao_alta` (0/1) adicionadas.
"""
    )

    file = st.file_uploader("Enviar arquivo CSV para predição em lote", type=["csv"])

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