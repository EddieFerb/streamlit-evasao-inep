"""Fine-tuning de modelos para previsão de evasão.

Este script carrega o dataset consolidado
`dados/processado/dados_ingresso_evasao_conclusao.csv` e, opcionalmente,
os subconjuntos (ex.: `final_medicina.csv`, `final_direito.csv` etc.),
realiza comparação automática entre:

- Random Forest (baseline)
- XGBoost (regressão e classificação, com Optuna quando disponível)
- Modelos feature_based já treinados (se os arquivos .pkl existirem)

Saídas principais:
- modelos/melhor_modelo_xgb.pkl        -> dicionário com regressor e classifier
- modelos/metricas_xgb.txt             -> resumo de métricas e comparação
- logs em modelos/logs/fine_tuning_xgb.log

Compatível com o app Streamlit (app_evasao.py):
- Os modelos salvos possuem o atributo `feature_names_in_`,
  permitindo integração direta com o front-end.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Dependências opcionais -----------------------------------------------------
try:  # XGBoost
    import xgboost as xgb

    HAS_XGB = True
except Exception:  # pragma: no cover - ambiente sem xgboost
    xgb = None
    HAS_XGB = False

try:  # Optuna
    import optuna

    HAS_OPTUNA = True
except Exception:  # pragma: no cover - ambiente sem optuna
    optuna = None
    HAS_OPTUNA = False


# Caminhos base --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DADOS_PROCESSADO = BASE_DIR / "dados" / "processado"
MODELOS_DIR = BASE_DIR / "modelos"
MODELOS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = MODELOS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Logging --------------------------------------------------------------------
LOGGER = logging.getLogger("fine_tuning_xgb")
LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:
    # Log em arquivo
    fh = logging.FileHandler(LOGS_DIR / "fine_tuning_xgb.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)

    # Log no console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    LOGGER.addHandler(ch)


# Funções utilitárias --------------------------------------------------------

def carregar_consolidado() -> pd.DataFrame:
    """Carrega o dataset consolidado principal.

    Espera encontrar o arquivo:
    `dados/processado/dados_ingresso_evasao_conclusao.csv`.
    """

    caminho = DADOS_PROCESSADO / "dados_ingresso_evasao_conclusao.csv"
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    # O separador ';' é o padrão usado no pipeline do projeto
    df = pd.read_csv(caminho, sep=";", low_memory=False)
    LOGGER.info("Consolidado carregado com %d linhas e %d colunas.", *df.shape)
    return df


def carregar_subconjuntos() -> Dict[str, pd.DataFrame]:
    """Carrega subconjuntos finais, se existirem.

    Exemplo de arquivos esperados (opcionais):
    - final_medicina.csv
    - final_direito.csv
    - final_eng_civil.csv
    - final_administracao.csv
    - final_ingressantes.csv
    """

    candidatos = [
        "final_medicina.csv",
        "final_direito.csv",
        "final_eng_civil.csv",
        "final_administracao.csv",
        "final_ingressantes.csv",
    ]

    subconjuntos: Dict[str, pd.DataFrame] = {}
    for nome in candidatos:
        caminho = DADOS_PROCESSADO / nome
        if caminho.exists():
            df = pd.read_csv(caminho, sep=";", low_memory=False)
            LOGGER.info("Subconjunto %s carregado com %d linhas.", nome, len(df))
            subconjuntos[nome] = df
        else:
            LOGGER.info("Subconjunto %s não encontrado (ignorado).", nome)

    return subconjuntos


def preparar_matrizes(
    df: pd.DataFrame,
    threshold_evasao_alta: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    """Prepara X (features), y_reg (taxa_evasao) e y_clf (evasao alta).

    - Remove linhas com NA em `taxa_evasao`.
    - Usa todas as colunas numéricas como features, exceto métricas-alvo.
    - Gera alvo binário para classificação (evasão alta) se não existir.
    """

    # Caso o dataset não tenha 'taxa_evasao', mas tenha 'taxa_evasao_XXXX'
    if "taxa_evasao" not in df.columns:
        candidatos = [c for c in df.columns if c.startswith("taxa_evasao_")]
        if candidatos:
            # ordenar pelos anos e pegar o último
            candidatos_ordenados = sorted(
                candidatos,
                key=lambda x: int(x.split("_")[-1])
            )
            df = df.copy()
            df["taxa_evasao"] = df[candidatos_ordenados[-1]]
            LOGGER.info(
                "Coluna 'taxa_evasao' criada automaticamente a partir de %s.",
                candidatos_ordenados[-1]
            )
        else:
            raise KeyError(
                "Coluna 'taxa_evasao' não encontrada no dataset consolidado "
                "e nenhuma taxa_evasao_XXXX disponível."
            )

    df = df.copy()
    df = df.dropna(subset=["taxa_evasao"]).reset_index(drop=True)

    y_reg = df["taxa_evasao"].astype(float)

    # Cria alvo binário para classificação, se necessário
    if "evasao_alta" in df.columns:
        y_clf = df["evasao_alta"].astype(int)
    else:
        if threshold_evasao_alta is None:
            threshold_evasao_alta = float(y_reg.median())
        y_clf = (y_reg >= threshold_evasao_alta).astype(int)
        LOGGER.info(
            "Alvo binário 'evasao_alta' criado com threshold=%.4f (mediana da taxa_evasao).",
            threshold_evasao_alta,
        )

    # Colunas que não entram como features numéricas
    colunas_excluir = {
        "taxa_evasao",
        "taxa_conclusao",
        "taxa_ingresso",
        "evasao_alta",
    }

    features = [
        c
        for c in df.columns
        if c not in colunas_excluir and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not features:
        raise ValueError("Nenhuma feature numérica válida encontrada para modelagem.")

    X = df[features].astype(float)
    LOGGER.info("Total de features numéricas utilizadas: %d", len(features))
    return X, y_reg, y_clf, np.array(features)


# Métricas -------------------------------------------------------------------

def metricas_regressao(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def metricas_classificacao(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


# Random Forest baseline -----------------------------------------------------

def treinar_rf(
    X_train: pd.DataFrame,
    y_train_reg: pd.Series,
    y_train_clf: pd.Series,
) -> Tuple[RandomForestRegressor, RandomForestClassifier]:
    """Treina RF regressão e classificação com hiperparâmetros razoáveis.
    Não faz busca exaustiva (baseline).
    """

    rf_reg = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    rf_reg.fit(X_train, y_train_reg)

    rf_clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_train_clf)

    # Compatibilidade com Streamlit
    rf_reg.feature_names_in_ = np.array(X_train.columns)
    rf_clf.feature_names_in_ = np.array(X_train.columns)

    return rf_reg, rf_clf


# XGBoost + Optuna -----------------------------------------------------------

def _tune_xgb_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
) -> "xgb.XGBRegressor":
    """Executa busca de hiperparâmetros com Optuna para XGBRegressor.

    Caso Optuna não esteja disponível, retorna um modelo com hiperparâmetros
    fixos padrão (degraça elegante).
    """

    if not HAS_XGB:
        raise RuntimeError("XGBoost não está instalado no ambiente.")

    if not HAS_OPTUNA:
        LOGGER.warning(
            "Optuna não encontrado. Usando hiperparâmetros padrão para XGBRegressor."
        )
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train)
        return model

    # Com Optuna -------------------------------------------------------------
    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params,
        )

        # Validação holdout simples dentro do treinamento
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        return rmse

    LOGGER.info("Iniciando Optuna para XGBRegressor (%d trials)...", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    LOGGER.info("Melhor trial reg: rmse=%.5f, params=%s", study.best_value, study.best_params)

    best_params = study.best_params
    best_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    best_model.fit(X_train, y_train)
    return best_model


def _tune_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
) -> "xgb.XGBClassifier":
    """Busca de hiperparâmetros com Optuna para XGBClassifier.

    Se Optuna não estiver disponível, usa hiperparâmetros fixos.
    """

    if not HAS_XGB:
        raise RuntimeError("XGBoost não está instalado no ambiente.")

    if not HAS_OPTUNA:
        LOGGER.warning(
            "Optuna não encontrado. Usando hiperparâmetros padrão para XGBClassifier."
        )
        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="binary:logistic",
        )
        model.fit(X_train, y_train)
        return model

    # Com Optuna -------------------------------------------------------------
    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=42,
            n_jobs=-1,
            **params,
        )

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, zero_division=0)
        # Optuna maximiza se definirmos direction="maximize", mas aqui
        # queremos minimizar, então retornamos 1 - f1.
        return 1.0 - float(f1)

    LOGGER.info("Iniciando Optuna para XGBClassifier (%d trials)...", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    LOGGER.info("Melhor trial clf: loss=%.5f, params=%s", study.best_value, study.best_params)

    best_params = study.best_params
    best_model = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    best_model.fit(X_train, y_train)
    return best_model


# Avaliação de modelos -------------------------------------------------------

def avaliar_regressor(nome: str, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    m = metricas_regressao(y_true, y_pred)
    LOGGER.info(
        "[REG] %-20s MAE=%.6f | RMSE=%.6f | R2=%.6f",
        nome,
        m["MAE"],
        m["RMSE"],
        m["R2"],
    )
    return m


def avaliar_classifier(nome: str, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    m = metricas_classificacao(y_true, y_pred)
    LOGGER.info(
        "[CLF] %-20s ACC=%.4f | F1=%.4f | PREC=%.4f | REC=%.4f",
        nome,
        m["accuracy"],
        m["f1"],
        m["precision"],
        m["recall"],
    )
    return m


def tentar_carregar_feature_based_regressor() -> Optional[object]:
    caminho = BASE_DIR / "modelos" / "modelos_salvos" / "modelo_evasao_reg_feature_based.pkl"
    if not caminho.exists():
        LOGGER.info("Modelo feature_based (regressor) não encontrado em %s", caminho)
        return None
    try:
        modelo = joblib.load(caminho)
        LOGGER.info("Modelo feature_based regressor carregado: %s", caminho)
        return modelo
    except Exception as exc:  # pragma: no cover - erro em artefato antigo
        LOGGER.error("Falha ao carregar modelo feature_based regressor: %s", exc)
        return None


def tentar_carregar_feature_based_classifier() -> Optional[object]:
    caminho = BASE_DIR / "modelos" / "modelos_salvos" / "modelo_evasao_clf_feature_based.pkl"
    if not caminho.exists():
        LOGGER.info("Modelo feature_based (classifier) não encontrado em %s", caminho)
        return None
    try:
        modelo = joblib.load(caminho)
        LOGGER.info("Modelo feature_based classifier carregado: %s", caminho)
        return modelo
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Falha ao carregar modelo feature_based classifier: %s", exc)
        return None


# Geração de relatório em texto ----------------------------------------------

def salvar_metricas_em_txt(conteudo: str) -> None:
    caminho = MODELOS_DIR / "metricas_xgb.txt"
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)
    LOGGER.info("Métricas salvas em %s", caminho)


# Pipeline principal ---------------------------------------------------------

def executar_para_dataset(
    nome_contexto: str,
    df: pd.DataFrame,
    n_trials_xgb: int = 30,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Executa todo o fluxo de comparação RF × XGB × Feature-Based.

    Retorna dicionário aninhado com métricas de regressão e classificação
    para cada tipo de modelo.
    """

    LOGGER.info("\n========== Dataset: %s =========", nome_contexto)

    X, y_reg, y_clf, feature_names = preparar_matrizes(df)

    # Split único compartilhado entre regressão e classificação
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=y_clf,
    )

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train_reg, y_test_reg = y_reg.iloc[idx_train], y_reg.iloc[idx_test]
    y_train_clf, y_test_clf = y_clf.iloc[idx_train], y_clf.iloc[idx_test]

    resultados: Dict[str, Dict[str, Dict[str, float]]] = {
        "regressao": {},
        "classificacao": {},
    }

    # ------------------------------------------------------------------
    # 1) Random Forest baseline
    # ------------------------------------------------------------------
    LOGGER.info("Treinando Random Forest (baseline)...")
    rf_reg, rf_clf = treinar_rf(X_train, y_train_reg, y_train_clf)

    preds_rf_reg = rf_reg.predict(X_test)
    preds_rf_clf = rf_clf.predict(X_test)

    resultados["regressao"]["RF"] = avaliar_regressor(
        f"RF ({nome_contexto})", y_test_reg, preds_rf_reg
    )
    resultados["classificacao"]["RF"] = avaliar_classifier(
        f"RF ({nome_contexto})", y_test_clf, preds_rf_clf
    )

    # ------------------------------------------------------------------
    # 2) XGBoost (regressão + classificação, com Optuna se disponível)
    # ------------------------------------------------------------------
    if HAS_XGB:
        # XGBoost's sklearn wrapper infers feature_names_in_ automatically when fitted on a DataFrame.
        LOGGER.info("Treinando XGBoost (regressão)...")
        xgb_reg = _tune_xgb_regressor(X_train, y_train_reg, n_trials=n_trials_xgb)

        LOGGER.info("Treinando XGBoost (classificação)...")
        xgb_clf = _tune_xgb_classifier(X_train, y_train_clf, n_trials=n_trials_xgb)

        preds_xgb_reg = xgb_reg.predict(X_test)
        preds_xgb_clf = xgb_clf.predict(X_test)

        resultados["regressao"]["XGB"] = avaliar_regressor(
            f"XGB ({nome_contexto})", y_test_reg, preds_xgb_reg
        )
        resultados["classificacao"]["XGB"] = avaliar_classifier(
            f"XGB ({nome_contexto})", y_test_clf, preds_xgb_clf
        )

        # Somente no dataset consolidado salvamos o melhor modelo XGB em disco.
        if nome_contexto == "consolidado":
            melhor_modelos_xgb = {"regressor": xgb_reg, "classifier": xgb_clf}
            caminho_modelo_xgb = MODELOS_DIR / "melhor_modelo_xgb.pkl"
            joblib.dump(melhor_modelos_xgb, caminho_modelo_xgb)
            LOGGER.info("Melhor modelo XGB salvo em %s", caminho_modelo_xgb)
    else:
        LOGGER.warning("XGBoost não está disponível; etapa XGB será ignorada.")

    # ------------------------------------------------------------------
    # 3) Modelos feature_based (se existirem)
    # ------------------------------------------------------------------
    fb_reg = tentar_carregar_feature_based_regressor()
    if fb_reg is not None:
        try:
            if hasattr(fb_reg, "feature_names_in_"):
                cols = list(fb_reg.feature_names_in_)
                X_fb = df.loc[X_test.index, cols]
            else:  # fallback para numéricas
                X_fb = df.loc[X_test.index].select_dtypes(include=[np.number])

            preds_fb_reg = fb_reg.predict(X_fb)
            resultados["regressao"]["Feature-Based"] = avaliar_regressor(
                f"Feature-Based ({nome_contexto})", y_test_reg, preds_fb_reg
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Falha ao avaliar regressor feature_based: %s", exc)

    fb_clf = tentar_carregar_feature_based_classifier()
    if fb_clf is not None:
        try:
            if hasattr(fb_clf, "feature_names_in_"):
                cols = list(fb_clf.feature_names_in_)
                X_fb = df.loc[X_test.index, cols]
            else:
                X_fb = df.loc[X_test.index].select_dtypes(include=[np.number])

            preds_fb_clf = fb_clf.predict(X_fb)
            resultados["classificacao"]["Feature-Based"] = avaliar_classifier(
                f"Feature-Based ({nome_contexto})", y_test_clf, preds_fb_clf
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Falha ao avaliar classifier feature_based: %s", exc)

    return resultados


def gerar_relatorio_textual(
    resultados_consolidado: Dict[str, Dict[str, Dict[str, float]]],
    resultados_subconjuntos: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
) -> str:
    """Gera texto consolidado com as métricas de todos os experimentos."""

    linhas = []
    linhas.append("FINE-TUNING XGBOOST × RANDOM FOREST × FEATURE-BASED")
    linhas.append("=" * 72)
    linhas.append("")

    # Consolidado ------------------------------------------------------------
    linhas.append("[1] Dataset consolidado — dados_ingresso_evasao_conclusao.csv")
    linhas.append("""
Métricas de REGRESSÃO (taxa_evasao):
-----------------------------------""")

    for modelo, m in resultados_consolidado["regressao"].items():
        linhas.append(
            f"- {modelo:15s} | MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | R2={m['R2']:.6f}"
        )

    linhas.append("")
    linhas.append("""
Métricas de CLASSIFICAÇÃO (evasao_alta):
---------------------------------------""")

    for modelo, m in resultados_consolidado["classificacao"].items():
        linhas.append(
            f"- {modelo:15s} | ACC={m['accuracy']:.4f} | F1={m['f1']:.4f} | "
            f"PREC={m['precision']:.4f} | REC={m['recall']:.4f}"
        )

    # Subconjuntos -----------------------------------------------------------
    if resultados_subconjuntos:
        linhas.append("")
        linhas.append("""
[2] Subconjuntos específicos (final_*.csv)
-----------------------------------------""")

        for nome_sub, res in resultados_subconjuntos.items():
            linhas.append("")
            linhas.append(f"Subconjunto: {nome_sub}")
            linhas.append("  REGRESSÃO:")
            for modelo, m in res.get("regressao", {}).items():
                linhas.append(
                    "    - "
                    + f"{modelo:15s} | MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | R2={m['R2']:.6f}"
                )
            linhas.append("  CLASSIFICAÇÃO:")
            for modelo, m in res.get("classificacao", {}).items():
                linhas.append(
                    "    - "
                    + (
                        f"{modelo:15s} | ACC={m['accuracy']:.4f} | F1={m['f1']:.4f} | "
                        f"PREC={m['precision']:.4f} | REC={m['recall']:.4f}"
                    )
                )

    linhas.append("")
    return "\n".join(linhas)


from datetime import datetime

def main(n_trials_xgb: int = 30) -> None:
    LOGGER.info("Iniciando fine_tuning.py — comparação RF × XGB × Feature-Based")
    LOGGER.info("BASE_DIR: %s", BASE_DIR)

    start_time = datetime.now()
    LOGGER.info("Início da execução: %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # 1) Dataset consolidado -------------------------------------------------
    df_consolidado = carregar_consolidado()
    resultados_consolidado = executar_para_dataset(
        nome_contexto="consolidado",
        df=df_consolidado,
        n_trials_xgb=n_trials_xgb,
    )

    # 2) Subconjuntos (opcional) --------------------------------------------
    subconjuntos = carregar_subconjuntos()
    resultados_subconjuntos: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for nome, df_sub in subconjuntos.items():
        try:
            res_sub = executar_para_dataset(
                nome_contexto=nome,
                df=df_sub,
                n_trials_xgb=max(10, n_trials_xgb // 2),  # tuning mais leve
            )
            resultados_subconjuntos[nome] = res_sub
        except Exception as exc:  # pragma: no cover - robustez
            LOGGER.error("Falha ao executar fine_tuning no subconjunto %s: %s", nome, exc)

    # 3) Relatório consolidado ----------------------------------------------
    texto = gerar_relatorio_textual(resultados_consolidado, resultados_subconjuntos)
    salvar_metricas_em_txt(texto)

    end_time = datetime.now()
    duration = end_time - start_time
    LOGGER.info("Fim da execução: %s", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    LOGGER.info("Tempo total de execução: %s", str(duration))
    LOGGER.info("Fine-tuning concluído.")


if __name__ == "__main__":
    # Execução padrão com 30 trials para XGB.
    # Pode ser ajustado via CLI se desejado, por exemplo:
    #   python scripts/modelagem/fine_tuning.py 50
    import sys as _sys

    if len(_sys.argv) > 1:
        try:
            trials = int(_sys.argv[1])
        except ValueError:
            trials = 30
    else:
        trials = 30

    main(n_trials_xgb=trials)