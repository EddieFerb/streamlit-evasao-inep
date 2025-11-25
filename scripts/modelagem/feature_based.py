# /scripts/modelagem/feature_based.py.py
# Este script treina modelos de aprendizado de máquina combinado com Transfer Learning baseado em features para prever a taxa de evasão,
# bem como explorar as relações entre taxa de ingresso, taxa de evasão e taxa de conclusão,
# e realizar previsões para ingressantes e concluintes.
# 3º Modelo Feature Based - Random Forest


"""
Modelagem de evasão — abordagem feature_based com Random Forest
---------------------------------------------------------------

- Lê dados consolidados de ingresso, conclusão e evasão
- Faz split temporal: treino (<= 2018) e teste (>= 2019)
- Regressão: predizer taxa_evasao (0–1)
- Classificação: evasao_alta (taxa_evasao > 0.5)
- Usa Pipeline com:
    - StandardScaler (numéricas)
    - OneHotEncoder (categóricas)
    - RandomForest (regressor e classificador)
- Faz um tuning leve de hiperparâmetros com TimeSeriesSplit no treino
- Salva modelos e métricas em ./modelos
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from datetime import datetime

# Garante que o diretório raiz do projeto (onde está a pasta "scripts") esteja no sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.utils.inspecao import registrar_ambiente, auditar_df, auditar_csv

registrar_ambiente(etapa="modelagem_feature_based", contexto="inicio")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# -----------------------------------------------------------
# Utilitários
# -----------------------------------------------------------

def garantir_pastas_modelos():
    os.makedirs("./modelos/modelos_salvos", exist_ok=True)
    os.makedirs("./modelos/resultados_modelos", exist_ok=True)


def carregar_dados(caminho="./dados/processado/dados_ingresso_evasao_conclusao.csv"):
    print(f"Lendo dados consolidados de: {caminho}")
    df = pd.read_csv(
        caminho,
        sep=';',              # <- principal ajuste
        encoding='utf-8',
        low_memory=False
    )

    print("\nPré-visualização dos dados carregados:")
    print(df.head())
    print(f"\nColunas disponíveis ({len(df.columns)}): {list(df.columns)}")

    # Garantir que há a coluna 'ano'
    if "ano" not in df.columns:
        raise ValueError("A coluna 'ano' não foi encontrada no dataset consolidado.")

    # Garantir que há a coluna alvo 'taxa_evasao'
    if "taxa_evasao" not in df.columns:
        raise ValueError("A coluna 'taxa_evasao' (alvo) não foi encontrada no dataset.")

    return df


def split_temporal(df, ano_treino_max=2018):
    """Separa treino (<= ano_treino_max) e teste (> ano_treino_max)."""
    df_treino = df[df["ano"] <= ano_treino_max].copy()
    df_teste = df[df["ano"] > ano_treino_max].copy()

    print(f"\nSplit temporal aplicado: treino até {ano_treino_max}, "
          f"teste em anos posteriores.")
    print(f"Anos no treino: {sorted(df_treino['ano'].unique())}")
    print(f"Anos no teste:  {sorted(df_teste['ano'].unique())}")

    return df_treino, df_teste


def definir_features(df):
    """
    Define features numéricas e categóricas dinamicamente com base nas colunas existentes.
    Mantém flexibilidade caso o dataset mude um pouco.
    """
    colunas = list(df.columns)

    # Colunas que NUNCA devem entrar como features
    colunas_excluir = {
        "taxa_evasao",
        "taxa_conclusao",
        "taxa_ingresso",
        "evasao_alta",     # se já existir no CSV
        # IDs podem ser excluídos como features, se você quiser:
        # "id_ies", "id_curso"
    }

    # Garante que só exclui colunas que existem de fato
    colunas_excluir = [c for c in colunas_excluir if c in colunas]

    # Lista de features inicialmente como "todas as colunas menos as excluídas"
    features_candidatas = [c for c in colunas if c not in colunas_excluir]

    # Se quiser excluir explicitamente alguns identificadores:
    for id_col in ["id_ies", "id_curso"]:
        if id_col in features_candidatas:
            features_candidatas.remove(id_col)

    print("\nFeatures candidatas (após remoções explícitas):")
    print(features_candidatas)

    # Separar numéricas e categóricas pela dtype
    numericas = []
    categoricas = []

    for c in features_candidatas:
        if pd.api.types.is_numeric_dtype(df[c]):
            numericas.append(c)
        else:
            categoricas.append(c)

    print(f"\nFeatures numéricas ({len(numericas)}): {numericas}")
    print(f"Features categóricas ({len(categoricas)}): {categoricas}")

    return numericas, categoricas


def montar_preprocessador(colunas_numericas, colunas_categoricas):
    """
    Cria o ColumnTransformer com:
      - StandardScaler para numéricas
      - OneHotEncoder para categóricas
    """
    transformers = []

    if colunas_numericas:
        transformers.append(
            ("num", StandardScaler(), colunas_numericas)
        )

    if colunas_categoricas:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas)
        )

    if not transformers:
        raise ValueError("Nenhuma coluna numérica ou categórica válida foi encontrada.")

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor


# -----------------------------------------------------------
# Modelagem de Regressão — taxa_evasao
# -----------------------------------------------------------

def treinar_regressao_tempo(
    df_treino,
    df_teste,
    colunas_numericas,
    colunas_categoricas,
    salvar_modelo=True,
):
    print("\n=== REGRESSÃO: Predição da taxa_evasao ===")

    preprocessor = montar_preprocessador(colunas_numericas, colunas_categoricas)

    # Alvo
    y_train = df_treino["taxa_evasao"].values
    y_test = df_teste["taxa_evasao"].values

    # Features
    X_train = df_treino[colunas_numericas + colunas_categoricas]
    X_test = df_teste[colunas_numericas + colunas_categoricas]

    # Remover linhas com NaN no treino/teste (simples e direto)
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)

    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    print(f"Tamanho do treino (regressão): {X_train.shape}")
    print(f"Tamanho do teste  (regressão): {X_test.shape}")
    auditar_df(X_train, etapa="modelagem_feature_based", contexto="X_train_regressao", n=5)
    auditar_df(X_test,  etapa="modelagem_feature_based", contexto="X_test_regressao", n=5)

    base_regressor = RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
    )

    pipe_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_regressor),
        ]
    )

    # Pequena busca de hiperparâmetros só no treino
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 5, 10],
    }

    # TimeSeriesSplit usando ordem original do treino
    tscv = TimeSeriesSplit(n_splits=3)

    search = GridSearchCV(
        pipe_reg,
        param_grid,
        cv=tscv,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )

    print("\nIniciando GridSearchCV para regressão (Random Forest)...")
    search.fit(X_train, y_train)

    print(f"\nMelhores hiperparâmetros (regressão): {search.best_params_}")
    best_regressor = search.best_estimator_

    # Avaliar no conjunto de teste (anos 2019–2024)
    y_pred = best_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nResultados no conjunto de teste (regressão):")
    print(f"MSE: {mse:.4f}")
    print(f"R² : {r2:.4f}")

    if salvar_modelo:
        caminho_modelo = "./modelos/modelos_salvos/modelo_evasao_reg_feature_based.pkl"
        joblib.dump(best_regressor, caminho_modelo)
        print(f"Modelo de regressão salvo em: {caminho_modelo}")

    return {
        "mse": mse,
        "r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# -----------------------------------------------------------
# Modelagem de Classificação — evasão alta (> 0.5)
# -----------------------------------------------------------

def treinar_classificacao_tempo(
    df_treino,
    df_teste,
    colunas_numericas,
    colunas_categoricas,
    threshold=0.5,
    salvar_modelo=True,
):
    print("\n=== CLASSIFICAÇÃO: Evasão alta (> {:.2f}) ===".format(threshold))

    preprocessor = montar_preprocessador(colunas_numericas, colunas_categoricas)

    # Rótulos binários
    y_train = (df_treino["taxa_evasao"] > threshold).astype(int).values
    y_test = (df_teste["taxa_evasao"] > threshold).astype(int).values

    X_train = df_treino[colunas_numericas + colunas_categoricas]
    X_test = df_teste[colunas_numericas + colunas_categoricas]

    # Remover linhas com NaN na taxa_evasao, se houver
    mask_train = ~df_treino["taxa_evasao"].isna().values
    mask_test = ~df_teste["taxa_evasao"].isna().values

    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    print(f"Tamanho do treino (classificação): {X_train.shape}")
    print(f"Tamanho do teste  (classificação): {X_test.shape}")
    auditar_df(X_train, etapa="modelagem_feature_based", contexto="X_train_classificacao", n=5)
    auditar_df(X_test,  etapa="modelagem_feature_based", contexto="X_test_classificacao", n=5)

    base_clf = RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_clf),
        ]
    )

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 5, 10],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    search = GridSearchCV(
        pipe_clf,
        param_grid,
        cv=tscv,
        scoring="f1",  # f1 para lidar melhor com eventual desbalanceamento
        n_jobs=-1,
        verbose=1,
    )

    print("\nIniciando GridSearchCV para classificação (Random Forest)...")
    search.fit(X_train, y_train)

    print(f"\nMelhores hiperparâmetros (classificação): {search.best_params_}")
    best_clf = search.best_estimator_

    # Avaliar no teste
    y_pred = best_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Baixa evasão", "Alta evasão"])

    print("\nResultados no conjunto de teste (classificação):")
    print(f"Acurácia : {acc:.4f}")
    print(f"Precisão : {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nMatriz de confusão (linhas = verdadeiro, colunas = predito):")
    print(cm)
    print("\nRelatório de classificação:")
    print(report)

    if salvar_modelo:
        caminho_modelo = "./modelos/modelos_salvos/modelo_evasao_clf_feature_based.pkl"
        joblib.dump(best_clf, caminho_modelo)
        print(f"Modelo de classificação salvo em: {caminho_modelo}")

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
        "report": report,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# -----------------------------------------------------------
# Rotina principal
# -----------------------------------------------------------

def main():
    inicio = datetime.now()
    print(f"\n>>> Início do treinamento (feature_based): {inicio.strftime('%Y-%m-%d %H:%M:%S')}")

    garantir_pastas_modelos()

    df = carregar_dados()
    auditar_df(df, etapa="modelagem_feature_based", contexto="df_carregado", n=5)

    # Split temporal fixo: 2009–2018 treino, 2019–2024 teste
    df_treino, df_teste = split_temporal(df, ano_treino_max=2018)

    # Definir features com base no dataset carregado
    colunas_numericas, colunas_categoricas = definir_features(df)

    # Regressão
    resultados_reg = treinar_regressao_tempo(
        df_treino, df_teste, colunas_numericas, colunas_categoricas
    )

    # Classificação
    resultados_clf = treinar_classificacao_tempo(
        df_treino, df_teste, colunas_numericas, colunas_categoricas
    )

    # Salvar métricas em arquivo texto
    caminho_txt = "./modelos/resultados_modelos/metricas_modelos_feature_based.txt"
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("=== REGRESSÃO — taxa_evasao ===\n")
        f.write(f"MSE: {resultados_reg['mse']:.4f}\n")
        f.write(f"R² : {resultados_reg['r2']:.4f}\n\n")

        f.write("=== CLASSIFICAÇÃO — evasao_alta (> 0.5) ===\n")
        f.write(f"Acurácia : {resultados_clf['acc']:.4f}\n")
        f.write(f"Precisão : {resultados_clf['precision']:.4f}\n")
        f.write(f"Recall   : {resultados_clf['recall']:.4f}\n")
        f.write(f"F1-score : {resultados_clf['f1']:.4f}\n\n")
        f.write("Matriz de confusão:\n")
        f.write(str(resultados_clf["cm"]) + "\n\n")
        f.write("Relatório de classificação:\n")
        f.write(resultados_clf["report"] + "\n")

    print(f"\nMétricas salvas em: {caminho_txt}")
    print("\nTreinamento (feature_based) concluído.")

    fim = datetime.now()
    duracao = fim - inicio
    print(f"\n>>> Fim do treinamento (feature_based): {fim.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f">>> Tempo total de execução: {str(duracao)}")


if __name__ == "__main__":
    main()