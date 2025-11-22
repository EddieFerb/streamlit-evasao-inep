# 1º MODELO ORIGINAL
# # /scripts/modelagem/treinamento_modelo.py
# # Este script treina modelos de aprendizado de máquina para prever a taxa de evasão,
# # bem como explorar as relações entre taxa de ingresso, taxa de evasão e taxa de conclusão,
# # e realizar previsões para ingressantes e concluintes.

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
# import joblib
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

# def main():
#     # Caminho para os dados processados
#     caminho_dados = './dados/processado/dados_ingresso_evasao_conclusao.csv'
#     df = pd.read_csv(caminho_dados, sep=';', encoding='utf-8', low_memory=False)
    
#     # Definir caminho_modelo (com caminho relativo) antes de utilizá-lo em qualquer parte
#     caminho_modelo = './modelos/modelos_salvos'
#     os.makedirs(caminho_modelo, exist_ok=True)
    
#     # Treinamento para prever ingressantes e concluintes
#     if all(col in df.columns for col in ['numero_cursos', 'vagas', 'inscritos', 'docentes', 'ingressantes', 'concluintes']):
#         print("Treinando modelos para ingressantes e concluintes...")

#         # Features e targets para previsões de ingressantes e concluintes
#         X = df[['numero_cursos', 'vagas', 'inscritos', 'docentes']]
#         y_ingressantes = df['ingressantes']
#         y_concluintes = df['concluintes']

#         # Divisão dos dados para ingressantes
#         X_train_ing, X_test_ing, y_train_ing, y_test_ing = train_test_split(X, y_ingressantes, test_size=0.2, random_state=42)

#         # Modelo Random Forest para ingressantes
#         modelo_ingressantes = RandomForestRegressor(random_state=42)
#         modelo_ingressantes.fit(X_train_ing, y_train_ing)
#         score_ingressantes = modelo_ingressantes.score(X_test_ing, y_test_ing)
#         print(f"Score para ingressantes: {score_ingressantes:.4f}")

#         # Divisão dos dados para concluintes
#         X_train_conc, X_test_conc, y_train_conc, y_test_conc = train_test_split(X, y_concluintes, test_size=0.2, random_state=42)

#         # Modelo Random Forest para concluintes
#         modelo_concluintes = RandomForestRegressor(random_state=42)
#         modelo_concluintes.fit(X_train_conc, y_train_conc)
#         score_concluintes = modelo_concluintes.score(X_test_conc, y_test_conc)
#         print(f"Score para concluintes: {score_concluintes:.4f}")

#         # Salvar modelos de ingressantes e concluintes
#         joblib.dump(modelo_ingressantes, os.path.join(caminho_modelo, 'modelo_ingressantes.pkl'))
#         joblib.dump(modelo_concluintes, os.path.join(caminho_modelo, 'modelo_concluintes.pkl'))
    
#     # Treinamento para taxas de evasão
#     print("Treinando modelos para taxa de evasão...")

#     # Filtrar as variáveis relevantes
#     colunas_relevantes = ['taxa_ingresso', 'taxa_evasao', 'taxa_conclusao']
#     df = df[colunas_relevantes].dropna()

#     # Dividir as features e o target
#     X = df[['taxa_ingresso', 'taxa_conclusao']]
#     y = df['taxa_evasao']

#     # Dividir os dados em treino e teste
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Treinar dois modelos: Regressão Linear e Random Forest
#     # Modelo 1: Regressão Linear
#     modelo_linear = LinearRegression()
#     modelo_linear.fit(X_train, y_train)
#     y_pred_linear = modelo_linear.predict(X_test)
#     mse_linear = mean_squared_error(y_test, y_pred_linear)
#     r2_linear = r2_score(y_test, y_pred_linear)

#     # Modelo 2: Random Forest
#     modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
#     modelo_rf.fit(X_train, y_train)
#     y_pred_rf = modelo_rf.predict(X_test)
#     mse_rf = mean_squared_error(y_test, y_pred_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)

#     # Escolher o melhor modelo baseado no R²
#     melhor_modelo = modelo_rf if r2_rf > r2_linear else modelo_linear
#     nome_melhor_modelo = 'Random Forest' if r2_rf > r2_linear else 'Regressão Linear'

#     # Salvar o modelo escolhido
#     joblib.dump(melhor_modelo, os.path.join(caminho_modelo, 'modelo_melhor_evasao.pkl'))

#     # Salvar as métricas dos modelos
#     caminho_metricas = './modelos/resultados_modelos'
#     os.makedirs(caminho_metricas, exist_ok=True)
#     with open(os.path.join(caminho_metricas, 'metricas_modelos.txt'), 'w') as f:
#         f.write(f"Modelo: {nome_melhor_modelo}\n")
#         f.write(f"MSE - Regressão Linear: {mse_linear:.4f}, R²: {r2_linear:.4f}\n")
#         f.write(f"MSE - Random Forest: {mse_rf:.4f}, R²: {r2_rf:.4f}\n")
#         f.write(f"Melhor Modelo Selecionado: {nome_melhor_modelo}\n")
    
#     # Geração da Matriz de Confusão (binarizando a taxa de evasão com threshold de 0.5)
#     threshold = 0.5
#     y_test_class = (y_test >= threshold).astype(int)
#     y_pred_class = (melhor_modelo.predict(X_test) >= threshold).astype(int)
#     cm = confusion_matrix(y_test_class, y_pred_class)
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f"Matriz de Confusão - {nome_melhor_modelo}")
#     plt.xlabel("Predito")
#     plt.ylabel("Verdadeiro")
#     caminho_cm = os.path.join(caminho_metricas, 'matriz_confusao.png')
#     plt.savefig(caminho_cm)
#     plt.close()
    
#     print(f"Treinamento concluído. Melhor modelo para evasão: {nome_melhor_modelo}")
#     print(f"Métricas salvas em: {caminho_metricas}")
#     print(f"Modelos salvos em: {caminho_modelo}")

# if __name__ == '__main__':
#     main()



# NOVO SCRIPT PARA DISCIPLINA DE 2COP507 - Reconhecimento de Padrões
# Prof. Bruno B. Zarpelão / Prof. Sylvio Barbon Jr.
# Projeto Prático de Aprendizado de Máquina com Streamlit
# Este script treina modelos de regressão para prever a taxa de evasão acadêmica
# utilizando as variáveis 'taxa_ingresso' e 'vagas_totais' como preditoras.
# Avalia o desempenho de Regressão Linear e Random Forest, seleciona o melhor
# modelo com base no R², e salva o modelo e as métricas em arquivos apropriados.


# 2º MODELO ADAPTADO DISCIPLINA 2COP507 - RECONHECIMENTO DE PADRÕES 
# Professores Bruno B. Zarpelão e Sylvio Barbon Jr.
# Projeto Prático de Aprendizado de Máquina com Streamlit
# Este script treina modelos de regressão para prever a taxa de evasão acadêmica
# utilizando as variáveis 'taxa_ingresso' e 'taxa_conclusao' como preditoras.
# Avalia o desempenho de Regressão Linear e Random Forest, seleciona o melhor
# modelo com base no R², e salva o modelo e as métricas em arquivos apropriados.

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import (
#     mean_squared_error,
#     r2_score,
#     confusion_matrix,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
# )
# import joblib
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns


# def main():
#     """
#     Treina modelos para:
#     1) Prever número de ingressantes e concluintes (Random Forest).
#     2) Prever taxa de evasão (Regressão Linear x Random Forest),
#        selecionando automaticamente o melhor modelo.

#     O script:
#     - Salva os modelos em ./modelos/modelos_salvos
#     - Salva métricas em ./modelos/resultados_modelos/metricas_modelos.txt
#     - Gera e salva a matriz de confusão binária em ./modelos/resultados_modelos/matriz_confusao.png
#     - Imprime um resumo dos resultados no terminal em uma seção
#       "Exibição de resultados", inspirada no exemplo KNN do Prof. Zarpelão.
#     """
#     # Caminho para os dados processados
#     caminho_dados = "./dados/processado/dados_ingresso_evasao_conclusao.csv"
#     df = pd.read_csv(caminho_dados, sep=";", encoding="utf-8", low_memory=False)

#     # Definir caminho_modelo (com caminho relativo) antes de utilizá-lo em qualquer parte
#     caminho_modelo = "./modelos/modelos_salvos"
#     os.makedirs(caminho_modelo, exist_ok=True)

#     resultados = {
#         "modelos_ingressantes_concluintes": {},
#         "modelo_evasao_regressao": {},
#         "classificacao_evasao_binaria": {},
#         "caminhos": {},
#     }

#     # ------------------------------------------------------------------
#     # 1) Treinamento para prever ingressantes e concluintes
#     # ------------------------------------------------------------------
#     if all(
#         col in df.columns
#         for col in [
#             "numero_cursos",
#             "vagas_totais",
#             "inscritos_totais",
#             "ingressantes",
#             "concluintes",
#         ]
#     ):
#         print("Treinando modelos para ingressantes e concluintes...")

#         # Features e targets para previsões de ingressantes e concluintes
#         X = df[["numero_cursos", "vagas_totais", "inscritos_totais"]]
#         y_ingressantes = df["ingressantes"]
#         y_concluintes = df["concluintes"]

#         # Divisão dos dados para ingressantes
#         X_train_ing, X_test_ing, y_train_ing, y_test_ing = train_test_split(
#             X, y_ingressantes, test_size=0.2, random_state=42
#         )

#         # Modelo Random Forest para ingressantes
#         modelo_ingressantes = RandomForestRegressor(random_state=42)
#         modelo_ingressantes.fit(X_train_ing, y_train_ing)
#         score_ingressantes = modelo_ingressantes.score(X_test_ing, y_test_ing)
#         print(f"Score R² para ingressantes (Random Forest): {score_ingressantes:.4f}")

#         # Divisão dos dados para concluintes
#         X_train_conc, X_test_conc, y_train_conc, y_test_conc = train_test_split(
#             X, y_concluintes, test_size=0.2, random_state=42
#         )

#         # Modelo Random Forest para concluintes
#         modelo_concluintes = RandomForestRegressor(random_state=42)
#         modelo_concluintes.fit(X_train_conc, y_train_conc)
#         score_concluintes = modelo_concluintes.score(X_test_conc, y_test_conc)
#         print(f"Score R² para concluintes (Random Forest): {score_concluintes:.4f}")

#         # Salvar modelos de ingressantes e concluintes
#         caminho_modelo_ing = os.path.join(caminho_modelo, "modelo_ingressantes.pkl")
#         caminho_modelo_conc = os.path.join(caminho_modelo, "modelo_concluintes.pkl")
#         joblib.dump(modelo_ingressantes, caminho_modelo_ing)
#         joblib.dump(modelo_concluintes, caminho_modelo_conc)

#         resultados["modelos_ingressantes_concluintes"] = {
#             "score_r2_ingressantes": score_ingressantes,
#             "score_r2_concluintes": score_concluintes,
#             "caminho_modelo_ingressantes": caminho_modelo_ing,
#             "caminho_modelo_concluintes": caminho_modelo_conc,
#         }
#     else:
#         print(
#             "Aviso: colunas para ingressantes/concluintes não encontradas. "
#             "Pulando etapa de modelos de ingresso/conclusão."
#         )

#     # ------------------------------------------------------------------
#     # 2) Treinamento para taxas de evasão (Regressão)
#     # ------------------------------------------------------------------
#     print("\nTreinando modelos para taxa de evasão...")

#     # Filtrar as variáveis relevantes
#     colunas_relevantes = ["taxa_ingresso", "taxa_evasao", "taxa_conclusao"]
#     df_reg = df[colunas_relevantes].dropna()

#     # Dividir as features e o target
#     X = df_reg[["taxa_ingresso", "taxa_conclusao"]]
#     y = df_reg["taxa_evasao"]

#     # Dividir os dados em treino e teste
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Modelo 1: Regressão Linear
#     modelo_linear = LinearRegression()
#     modelo_linear.fit(X_train, y_train)
#     y_pred_linear = modelo_linear.predict(X_test)
#     mse_linear = mean_squared_error(y_test, y_pred_linear)
#     r2_linear = r2_score(y_test, y_pred_linear)

#     # Modelo 2: Random Forest
#     modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
#     modelo_rf.fit(X_train, y_train)
#     y_pred_rf = modelo_rf.predict(X_test)
#     mse_rf = mean_squared_error(y_test, y_pred_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)

#     # Escolher o melhor modelo baseado no R²
#     if r2_rf > r2_linear:
#         melhor_modelo = modelo_rf
#         nome_melhor_modelo = "Random Forest"
#     else:
#         melhor_modelo = modelo_linear
#         nome_melhor_modelo = "Regressão Linear"

#     # Salvar o modelo escolhido
#     caminho_modelo_evasao = os.path.join(caminho_modelo, "modelo_melhor_evasao.pkl")
#     joblib.dump(melhor_modelo, caminho_modelo_evasao)

#     # ------------------------------------------------------------------
#     # Exibição de resultados no terminal (estilo Prof. Zarpelão)
#     # ------------------------------------------------------------------
#     print("\n" + "=" * 70)
#     print("Exibição de resultados: Modelos de regressão para taxa de evasão")
#     print("=" * 70)
#     print(f"Regressão Linear  -> MSE: {mse_linear:.4f} | R²: {r2_linear:.4f}")
#     print(f"Random Forest     -> MSE: {mse_rf:.4f} | R²: {r2_rf:.4f}")
#     print(f"\nMelhor modelo (critério R²): {nome_melhor_modelo}")
#     print(f"Modelo salvo em: {caminho_modelo_evasao}")

#     resultados["modelo_evasao_regressao"] = {
#         "mse_linear": mse_linear,
#         "r2_linear": r2_linear,
#         "mse_random_forest": mse_rf,
#         "r2_random_forest": r2_rf,
#         "melhor_modelo": nome_melhor_modelo,
#         "caminho_modelo": caminho_modelo_evasao,
#     }

#     # ------------------------------------------------------------------
#     # 3) Análise binária da evasão (matriz de confusão + métricas)
#     # ------------------------------------------------------------------
#     caminho_metricas = "./modelos/resultados_modelos"
#     os.makedirs(caminho_metricas, exist_ok=True)

#     # Geração da Matriz de Confusão (binarizando a taxa de evasão com threshold de 0.5)
#     threshold = 0.5
#     y_test_class = (y_test >= threshold).astype(int)
#     y_pred_class = (melhor_modelo.predict(X_test) >= threshold).astype(int)

#     cm = confusion_matrix(y_test_class, y_pred_class)
#     acc = accuracy_score(y_test_class, y_pred_class)
#     prec = precision_score(y_test_class, y_pred_class, zero_division=0)
#     rec = recall_score(y_test_class, y_pred_class, zero_division=0)
#     f1 = f1_score(y_test_class, y_pred_class, zero_division=0)

#     print("\n" + "=" * 70)
#     print("Exibição de resultados: Classificação binária da evasão (threshold 0.5)")
#     print("=" * 70)
#     print(f"Acurácia:  {acc:.4f}")
#     print(f"Precisão:  {prec:.4f}")
#     print(f"Recall:    {rec:.4f}")
#     print(f"F1-score:  {f1:.4f}")

#     print("\nMatriz de Confusão (linhas = verdadeiro, colunas = predito):")
#     print(cm)

#     print("\nRelatório de classificação:")
#     print(
#         classification_report(
#             y_test_class,
#             y_pred_class,
#             target_names=["Baixa evasão", "Alta evasão"],
#             zero_division=0,
#         )
#     )

#     # Salvar as métricas dos modelos em arquivo texto (regressão + classificação)
#     caminho_metricas_txt = os.path.join(caminho_metricas, "metricas_modelos.txt")
#     with open(caminho_metricas_txt, "w") as f:
#         f.write(f"Modelo de evasão escolhido: {nome_melhor_modelo}\n\n")
#         f.write("=== Regressão: Taxa de evasão ===\n")
#         f.write(f"MSE - Regressão Linear: {mse_linear:.4f}, R²: {r2_linear:.4f}\n")
#         f.write(f"MSE - Random Forest:   {mse_rf:.4f}, R²: {r2_rf:.4f}\n\n")

#         f.write("=== Classificação binária da evasão (threshold 0.5) ===\n")
#         f.write(f"Acurácia: {acc:.4f}\n")
#         f.write(f"Precisão: {prec:.4f}\n")
#         f.write(f"Recall:   {rec:.4f}\n")
#         f.write(f"F1-score: {f1:.4f}\n\n")

#         f.write("Matriz de confusão:\n")
#         f.write(str(cm) + "\n\n")

#         f.write("Relatório de classificação:\n")
#         f.write(
#             classification_report(
#                 y_test_class,
#                 y_pred_class,
#                 target_names=["Baixa evasão", "Alta evasão"],
#                 zero_division=0,
#             )
#         )

#     # Salvar figura da matriz de confusão
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#     plt.title(f"Matriz de Confusão - {nome_melhor_modelo}")
#     plt.xlabel("Predito")
#     plt.ylabel("Verdadeiro")
#     caminho_cm = os.path.join(caminho_metricas, "matriz_confusao.png")
#     plt.tight_layout()
#     plt.savefig(caminho_cm)
#     plt.close()

#     print(f"\nArquivos de métricas salvos em: {caminho_metricas_txt}")
#     print(f"Figura da matriz de confusão salva em: {caminho_cm}")

#     resultados["classificacao_evasao_binaria"] = {
#         "threshold": threshold,
#         "accuracy": acc,
#         "precision": prec,
#         "recall": rec,
#         "f1": f1,
#         "matriz_confusao": cm.tolist(),
#         "caminho_metricas_txt": caminho_metricas_txt,
#         "caminho_matriz_confusao_png": caminho_cm,
#     }

#     resultados["caminhos"]["metricas"] = caminho_metricas
#     resultados["caminhos"]["modelos"] = caminho_modelo

#     print("\nTreinamento concluído.")
#     return resultados


# if __name__ == "__main__":
#     main()

# # 3º MODELO ADAPTADO DISCIPLINA 2COP507 - RECONHECIMENTO DE PADRÕES
# # Professores Bruno B. Zarpelão e Sylvio Barbon Jr.
# # Projeto Prático de Aprendizado de Máquina com Streamlit
# # Este script treina modelos de regressão para prever a taxa de evasão acadêmica
# # utilizando as variáveis 'taxa_ingresso' e 'taxa_conclusao' como preditoras.
# # Avalia o desempenho de Regressão Linear e Random Forest, seleciona o melhor
# # modelo com base no R², e salva o modelo e as métricas em arquivos apropriados.

# import os
# import sys

# # Garante que o diretório raiz do projeto (onde está a pasta "scripts") esteja no sys.path
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# if BASE_DIR not in sys.path:
#     sys.path.insert(0, BASE_DIR)

# from scripts.utils.inspecao import registrar_ambiente, auditar_df, auditar_csv

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import (
#     mean_squared_error,
#     r2_score,
#     confusion_matrix,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
# )
# import joblib
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns


# def main():
#     """
#     Treina modelos para:
#     1) Prever número de ingressantes e concluintes (Random Forest).
#     2) Prever taxa de evasão (Regressão Linear x Random Forest),
#        selecionando automaticamente o melhor modelo.

#     O script agora:
#     - evita leakage removendo variáveis derivadas diretamente da taxa de evasão das features;
#     - utiliza split temporal (anos de ingresso) quando possível, para avaliar capacidade de predição futura.

#     Além disso:
#     - Salva os modelos em ./modelos/modelos_salvos
#     - Salva métricas em ./modelos/resultados_modelos/metricas_modelos.txt
#     - Gera e salva a matriz de confusão binária em ./modelos/resultados_modelos/matriz_confusao.png
#     - Imprime um resumo dos resultados no terminal em uma seção
#       "Exibição de resultados", inspirada no exemplo KNN do Prof. Zarpelão.
#     """
#     registrar_ambiente(etapa="modelagem", contexto="inicio_modelagem_base")

#     # Caminho para os dados processados
#     caminho_dados = "./dados/processado/dados_ingresso_evasao_conclusao.csv"
#     df = pd.read_csv(caminho_dados, sep=";", encoding="utf-8", low_memory=False)

#     # Auditoria do dataframe carregado e split temporal 2009–2018 / 2019–2024
#     auditar_df(df, etapa="modelagem_base", contexto="df_carregado", n=5)

#     if "ano" in df.columns:
#         df_treino = df[df["ano"] <= 2018]
#         df_teste = df[df["ano"] > 2018]

#         auditar_df(df_treino, etapa="modelagem_base", contexto="treino_ate_2018", n=5)
#         auditar_df(df_teste, etapa="modelagem_base", contexto="teste_2019_2024", n=5)

#     # Definir caminho_modelo (com caminho relativo) antes de utilizá-lo em qualquer parte
#     caminho_modelo = "./modelos/modelos_salvos"
#     os.makedirs(caminho_modelo, exist_ok=True)

#     resultados = {
#         "modelos_ingressantes_concluintes": {},
#         "modelo_evasao_regressao": {},
#         "classificacao_evasao_binaria": {},
#         "caminhos": {},
#     }

#     # ------------------------------------------------------------------
#     # 1) Treinamento para prever ingressantes e concluintes
#     # ------------------------------------------------------------------
#     if all(
#         col in df.columns
#         for col in [
#             "numero_cursos",
#             "vagas_totais",
#             "inscritos_totais",
#             "ingressantes",
#             "concluintes",
#         ]
#     ):
#         print("Treinando modelos para ingressantes e concluintes...")

#         # Features e targets para previsões de ingressantes e concluintes
#         X = df[["numero_cursos", "vagas_totais", "inscritos_totais"]]
#         y_ingressantes = df["ingressantes"]
#         y_concluintes = df["concluintes"]

#         # Divisão dos dados para ingressantes
#         X_train_ing, X_test_ing, y_train_ing, y_test_ing = train_test_split(
#             X, y_ingressantes, test_size=0.2, random_state=42
#         )

#         # Modelo Random Forest para ingressantes
#         modelo_ingressantes = RandomForestRegressor(random_state=42)
#         modelo_ingressantes.fit(X_train_ing, y_train_ing)
#         score_ingressantes = modelo_ingressantes.score(X_test_ing, y_test_ing)
#         print(f"Score R² para ingressantes (Random Forest): {score_ingressantes:.4f}")

#         # Divisão dos dados para concluintes
#         X_train_conc, X_test_conc, y_train_conc, y_test_conc = train_test_split(
#             X, y_concluintes, test_size=0.2, random_state=42
#         )

#         # Modelo Random Forest para concluintes
#         modelo_concluintes = RandomForestRegressor(random_state=42)
#         modelo_concluintes.fit(X_train_conc, y_train_conc)
#         score_concluintes = modelo_concluintes.score(X_test_conc, y_test_conc)
#         print(f"Score R² para concluintes (Random Forest): {score_concluintes:.4f}")

#         # Salvar modelos de ingressantes e concluintes
#         caminho_modelo_ing = os.path.join(caminho_modelo, "modelo_ingressantes.pkl")
#         caminho_modelo_conc = os.path.join(caminho_modelo, "modelo_concluintes.pkl")
#         joblib.dump(modelo_ingressantes, caminho_modelo_ing)
#         joblib.dump(modelo_concluintes, caminho_modelo_conc)

#         resultados["modelos_ingressantes_concluintes"] = {
#             "score_r2_ingressantes": score_ingressantes,
#             "score_r2_concluintes": score_concluintes,
#             "caminho_modelo_ingressantes": caminho_modelo_ing,
#             "caminho_modelo_concluintes": caminho_modelo_conc,
#         }
#     else:
#         print(
#             "Aviso: colunas para ingressantes/concluintes não encontradas. "
#             "Pulando etapa de modelos de ingresso/conclusão."
#         )

#     # ------------------------------------------------------------------
#     # 2) Treinamento para taxas de evasão (Regressão)
#     # ------------------------------------------------------------------
#     print("\nTreinando modelos para taxa de evasão...")

#     # Definir features "brutas" para evitar leakage com taxa_evasao
#     feature_cols = [
#         "numero_cursos",
#         "vagas_totais",
#         "inscritos_totais",
#         "ingressantes",
#         "matriculados",
#         "concluintes",
#     ]

#     # Definir colunas a serem carregadas, incluindo ano e taxa_evasao
#     colunas_relevantes = feature_cols + ["taxa_evasao", "ano"]
#     df_reg = df[colunas_relevantes].dropna()

#     # Verifica se coluna de ano está disponível para split temporal
#     if "ano" in df_reg.columns:
#         ano_limite_treino = 2018  # treina em 2009–2018, testa em 2019–2024
#         df_treino = df_reg[df_reg["ano"] <= ano_limite_treino]
#         df_teste = df_reg[df_reg["ano"] > ano_limite_treino]

#         if df_treino.empty or df_teste.empty:
#             print(
#                 "Aviso: split temporal não pôde ser aplicado (sem dados suficientes em alguma faixa de anos). "
#                 "Usando hold-out aleatório (train_test_split)."
#             )
#             X = df_reg[feature_cols]
#             y = df_reg["taxa_evasao"]
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42
#             )
#         else:
#             print(
#                 f"Split temporal aplicado para evasão: treino até {ano_limite_treino}, "
#                 "teste em anos posteriores."
#             )
#             print(
#                 "Anos no treino:", sorted(df_treino["ano"].unique()),
#                 "| Anos no teste:", sorted(df_teste["ano"].unique()),
#             )
#             X_train = df_treino[feature_cols]
#             y_train = df_treino["taxa_evasao"]
#             X_test = df_teste[feature_cols]
#             y_test = df_teste["taxa_evasao"]
#     else:
#         print(
#             "Aviso: coluna 'ano' não encontrada em df_reg. "
#             "Usando hold-out aleatório (train_test_split)."
#         )
#         X = df_reg[feature_cols]
#         y = df_reg["taxa_evasao"]
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#     # Modelo 1: Regressão Linear
#     modelo_linear = LinearRegression()
#     modelo_linear.fit(X_train, y_train)
#     y_pred_linear = modelo_linear.predict(X_test)
#     mse_linear = mean_squared_error(y_test, y_pred_linear)
#     r2_linear = r2_score(y_test, y_pred_linear)

#     # Modelo 2: Random Forest
#     modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
#     modelo_rf.fit(X_train, y_train)
#     y_pred_rf = modelo_rf.predict(X_test)
#     mse_rf = mean_squared_error(y_test, y_pred_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)

#     # Escolher o melhor modelo baseado no R²
#     if r2_rf > r2_linear:
#         melhor_modelo = modelo_rf
#         nome_melhor_modelo = "Random Forest"
#     else:
#         melhor_modelo = modelo_linear
#         nome_melhor_modelo = "Regressão Linear"

#     # Salvar o modelo escolhido
#     caminho_modelo_evasao = os.path.join(caminho_modelo, "modelo_melhor_evasao.pkl")
#     joblib.dump(melhor_modelo, caminho_modelo_evasao)

#     # ------------------------------------------------------------------
#     # Exibição de resultados no terminal (estilo Prof. Zarpelão)
#     # ------------------------------------------------------------------
#     print("\n" + "=" * 70)
#     print("Exibição de resultados: Modelos de regressão para taxa de evasão")
#     print("=" * 70)
#     print(f"Regressão Linear  -> MSE: {mse_linear:.4f} | R²: {r2_linear:.4f}")
#     print(f"Random Forest     -> MSE: {mse_rf:.4f} | R²: {r2_rf:.4f}")
#     print(f"\nMelhor modelo (critério R²): {nome_melhor_modelo}")
#     print(f"Modelo salvo em: {caminho_modelo_evasao}")

#     resultados["modelo_evasao_regressao"] = {
#         "mse_linear": mse_linear,
#         "r2_linear": r2_linear,
#         "mse_random_forest": mse_rf,
#         "r2_random_forest": r2_rf,
#         "melhor_modelo": nome_melhor_modelo,
#         "caminho_modelo": caminho_modelo_evasao,
#     }

#     # ------------------------------------------------------------------
#     # 3) Análise binária da evasão (matriz de confusão + métricas)
#     # ------------------------------------------------------------------
#     caminho_metricas = "./modelos/resultados_modelos"
#     os.makedirs(caminho_metricas, exist_ok=True)

#     # Geração da Matriz de Confusão (binarizando a taxa de evasão com threshold de 0.5)
#     threshold = 0.5
#     y_test_class = (y_test >= threshold).astype(int)
#     y_pred_class = (melhor_modelo.predict(X_test) >= threshold).astype(int)

#     cm = confusion_matrix(y_test_class, y_pred_class)
#     acc = accuracy_score(y_test_class, y_pred_class)
#     prec = precision_score(y_test_class, y_pred_class, zero_division=0)
#     rec = recall_score(y_test_class, y_pred_class, zero_division=0)
#     f1 = f1_score(y_test_class, y_pred_class, zero_division=0)

#     print("\n" + "=" * 70)
#     print("Exibição de resultados: Classificação binária da evasão (threshold 0.5)")
#     print("=" * 70)
#     print(f"Acurácia:  {acc:.4f}")
#     print(f"Precisão:  {prec:.4f}")
#     print(f"Recall:    {rec:.4f}")
#     print(f"F1-score:  {f1:.4f}")

#     print("\nMatriz de Confusão (linhas = verdadeiro, colunas = predito):")
#     print(cm)

#     print("\nRelatório de classificação:")
#     print(
#         classification_report(
#             y_test_class,
#             y_pred_class,
#             target_names=["Baixa evasão", "Alta evasão"],
#             zero_division=0,
#         )
#     )

#     # Salvar as métricas dos modelos em arquivo texto (regressão + classificação)
#     caminho_metricas_txt = os.path.join(caminho_metricas, "metricas_modelos.txt")
#     with open(caminho_metricas_txt, "w") as f:
#         f.write(f"Modelo de evasão escolhido: {nome_melhor_modelo}\n\n")
#         f.write("=== Regressão: Taxa de evasão ===\n")
#         f.write(f"MSE - Regressão Linear: {mse_linear:.4f}, R²: {r2_linear:.4f}\n")
#         f.write(f"MSE - Random Forest:   {mse_rf:.4f}, R²: {r2_rf:.4f}\n\n")

#         f.write("=== Classificação binária da evasão (threshold 0.5) ===\n")
#         f.write(f"Acurácia: {acc:.4f}\n")
#         f.write(f"Precisão: {prec:.4f}\n")
#         f.write(f"Recall:   {rec:.4f}\n")
#         f.write(f"F1-score: {f1:.4f}\n\n")

#         f.write("Matriz de confusão:\n")
#         f.write(str(cm) + "\n\n")

#         f.write("Relatório de classificação:\n")
#         f.write(
#             classification_report(
#                 y_test_class,
#                 y_pred_class,
#                 target_names=["Baixa evasão", "Alta evasão"],
#                 zero_division=0,
#             )
#         )

#     # Salvar figura da matriz de confusão
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#     plt.title(f"Matriz de Confusão - {nome_melhor_modelo}")
#     plt.xlabel("Predito")
#     plt.ylabel("Verdadeiro")
#     caminho_cm = os.path.join(caminho_metricas, "matriz_confusao.png")
#     plt.tight_layout()
#     plt.savefig(caminho_cm)
#     plt.close()

#     print(f"\nArquivos de métricas salvos em: {caminho_metricas_txt}")
#     print(f"Figura da matriz de confusão salva em: {caminho_cm}")

#     resultados["classificacao_evasao_binaria"] = {
#         "threshold": threshold,
#         "accuracy": acc,
#         "precision": prec,
#         "recall": rec,
#         "f1": f1,
#         "matriz_confusao": cm.tolist(),
#         "caminho_metricas_txt": caminho_metricas_txt,
#         "caminho_matriz_confusao_png": caminho_cm,
#     }

#     resultados["caminhos"]["metricas"] = caminho_metricas
#     resultados["caminhos"]["modelos"] = caminho_modelo

#     print("\nTreinamento concluído.")
#     return resultados


# if __name__ == "__main__":
#     main()

# 4º MODELO ADAPTADO DISCIPLINA 2COP507 - RECONHECIMENTO DE PADRÕES
# Professores Bruno B. Zarpelão e Sylvio Barbon Jr.
# Projeto Prático de Aprendizado de Máquina com Streamlit
# Este script treina modelos de regressão para prever a taxa de evasão acadêmica
# utilizando as variáveis 'taxa_ingresso' e 'taxa_conclusao' como preditoras.
# Avalia o desempenho de Regressão Linear e Random Forest, seleciona o melhor
# modelo com base no R², e salva o modelo e as métricas em arquivos apropriados.


import os
import sys

# Garante que o diretório raiz do projeto (onde está a pasta "scripts") esteja no sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.utils.inspecao import registrar_ambiente, auditar_df, auditar_csv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """
    Treina modelos para:
    1) Prever número de ingressantes e concluintes (Random Forest).
    2) Prever taxa de evasão (Regressão Linear x Random Forest),
       selecionando automaticamente o melhor modelo.

    O script agora:
    - evita leakage removendo variáveis derivadas diretamente da taxa de evasão das features;
    - utiliza split temporal (anos de ingresso) quando possível, para avaliar capacidade de predição futura.

    Além disso:
    - Salva os modelos em ./modelos/modelos_salvos
    - Salva métricas em ./modelos/resultados_modelos/metricas_modelos.txt
    - Gera e salva a matriz de confusão binária em ./modelos/resultados_modelos/matriz_confusao.png
    - Imprime um resumo dos resultados no terminal em uma seção
      "Exibição de resultados", inspirada no exemplo KNN do Prof. Zarpelão.
    """
    registrar_ambiente(etapa="modelagem", contexto="inicio_modelagem_base")

    # Caminho para os dados processados
    caminho_dados = "./dados/processado/dados_ingresso_evasao_conclusao.csv"
    df = pd.read_csv(caminho_dados, sep=";", encoding="utf-8", low_memory=False)

    # Auditoria do dataframe carregado e split temporal 2009–2018 / 2019–2024
    auditar_df(df, etapa="modelagem_base", contexto="df_carregado", n=5)

    if "ano" in df.columns:
        df_treino = df[df["ano"] <= 2018]
        df_teste = df[df["ano"] > 2018]

        auditar_df(df_treino, etapa="modelagem_base", contexto="treino_ate_2018", n=5)
        auditar_df(df_teste, etapa="modelagem_base", contexto="teste_2019_2024", n=5)

    # Definir caminho_modelo (com caminho relativo) antes de utilizá-lo em qualquer parte
    caminho_modelo = "./modelos/modelos_salvos"
    os.makedirs(caminho_modelo, exist_ok=True)

    resultados = {
        "modelos_ingressantes_concluintes": {},
        "modelo_evasao_regressao": {},
        "classificacao_evasao_binaria": {},
        "caminhos": {},
    }

    # ------------------------------------------------------------------
    # 1) Treinamento para prever ingressantes e concluintes
    # ------------------------------------------------------------------
    if all(
        col in df.columns
        for col in [
            "numero_cursos",
            "vagas_totais",
            "inscritos_totais",
            "ingressantes",
            "concluintes",
        ]
    ):
        print("Treinando modelos para ingressantes e concluintes...")

        # Features e targets para previsões de ingressantes e concluintes
        X = df[["numero_cursos", "vagas_totais", "inscritos_totais"]]
        y_ingressantes = df["ingressantes"]
        y_concluintes = df["concluintes"]

        # Divisão dos dados para ingressantes
        X_train_ing, X_test_ing, y_train_ing, y_test_ing = train_test_split(
            X, y_ingressantes, test_size=0.2, random_state=42
        )

        # Modelo Random Forest para ingressantes
        modelo_ingressantes = RandomForestRegressor(random_state=42)
        modelo_ingressantes.fit(X_train_ing, y_train_ing)
        score_ingressantes = modelo_ingressantes.score(X_test_ing, y_test_ing)
        print(f"Score R² para ingressantes (Random Forest): {score_ingressantes:.4f}")

        # Divisão dos dados para concluintes
        X_train_conc, X_test_conc, y_train_conc, y_test_conc = train_test_split(
            X, y_concluintes, test_size=0.2, random_state=42
        )

        # Modelo Random Forest para concluintes
        modelo_concluintes = RandomForestRegressor(random_state=42)
        modelo_concluintes.fit(X_train_conc, y_train_conc)
        score_concluintes = modelo_concluintes.score(X_test_conc, y_test_conc)
        print(f"Score R² para concluintes (Random Forest): {score_concluintes:.4f}")

        # Salvar modelos de ingressantes e concluintes
        caminho_modelo_ing = os.path.join(caminho_modelo, "modelo_ingressantes.pkl")
        caminho_modelo_conc = os.path.join(caminho_modelo, "modelo_concluintes.pkl")
        joblib.dump(modelo_ingressantes, caminho_modelo_ing)
        joblib.dump(modelo_concluintes, caminho_modelo_conc)

        resultados["modelos_ingressantes_concluintes"] = {
            "score_r2_ingressantes": score_ingressantes,
            "score_r2_concluintes": score_concluintes,
            "caminho_modelo_ingressantes": caminho_modelo_ing,
            "caminho_modelo_concluintes": caminho_modelo_conc,
        }
    else:
        print(
            "Aviso: colunas para ingressantes/concluintes não encontradas. "
            "Pulando etapa de modelos de ingresso/conclusão."
        )

    # ------------------------------------------------------------------
    # 2) Treinamento para taxas de evasão (Regressão)
    # ------------------------------------------------------------------
    print("\nTreinando modelos para taxa de evasão...")

    # Definir features "brutas" para evitar leakage com taxa_evasao
    feature_cols = [
        "numero_cursos",
        "vagas_totais",
        "inscritos_totais",
        "ingressantes",
        "matriculados",
        "concluintes",
    ]

    # Definir colunas a serem carregadas, incluindo ano e taxa_evasao
    colunas_relevantes = feature_cols + ["taxa_evasao", "ano"]
    df_reg = df[colunas_relevantes].dropna()

    # Verifica se coluna de ano está disponível para split temporal
    if "ano" in df_reg.columns:
        ano_limite_treino = 2018  # treina em 2009–2018, testa em 2019–2024
        df_treino = df_reg[df_reg["ano"] <= ano_limite_treino]
        df_teste = df_reg[df_reg["ano"] > ano_limite_treino]

        if df_treino.empty or df_teste.empty:
            print(
                "Aviso: split temporal não pôde ser aplicado (sem dados suficientes em alguma faixa de anos). "
                "Usando hold-out aleatório (train_test_split)."
            )
            X = df_reg[feature_cols]
            y = df_reg["taxa_evasao"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            print(
                f"Split temporal aplicado para evasão: treino até {ano_limite_treino}, "
                "teste em anos posteriores."
            )
            print(
                "Anos no treino:", sorted(df_treino["ano"].unique()),
                "| Anos no teste:", sorted(df_teste["ano"].unique()),
            )
            X_train = df_treino[feature_cols]
            y_train = df_treino["taxa_evasao"]
            X_test = df_teste[feature_cols]
            y_test = df_teste["taxa_evasao"]
    else:
        print(
            "Aviso: coluna 'ano' não encontrada em df_reg. "
            "Usando hold-out aleatório (train_test_split)."
        )
        X = df_reg[feature_cols]
        y = df_reg["taxa_evasao"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Modelo 1: Regressão Linear
    modelo_linear = LinearRegression()
    modelo_linear.fit(X_train, y_train)
    y_pred_linear = modelo_linear.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    # Modelo 2: Random Forest
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    # Escolher o melhor modelo baseado no R²
    if r2_rf > r2_linear:
        melhor_modelo = modelo_rf
        nome_melhor_modelo = "Random Forest"
    else:
        melhor_modelo = modelo_linear
        nome_melhor_modelo = "Regressão Linear"

    # Salvar o modelo escolhido
    caminho_modelo_evasao = os.path.join(caminho_modelo, "modelo_melhor_evasao.pkl")
    joblib.dump(melhor_modelo, caminho_modelo_evasao)

    # ------------------------------------------------------------------
    # Exibição de resultados no terminal (estilo Prof. Zarpelão)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Exibição de resultados: Modelos de regressão para taxa de evasão")
    print("=" * 70)
    print(f"Regressão Linear  -> MSE: {mse_linear:.4f} | R²: {r2_linear:.4f}")
    print(f"Random Forest     -> MSE: {mse_rf:.4f} | R²: {r2_rf:.4f}")
    print(f"\nMelhor modelo (critério R²): {nome_melhor_modelo}")
    print(f"Modelo salvo em: {caminho_modelo_evasao}")

    resultados["modelo_evasao_regressao"] = {
        "mse_linear": mse_linear,
        "r2_linear": r2_linear,
        "mse_random_forest": mse_rf,
        "r2_random_forest": r2_rf,
        "melhor_modelo": nome_melhor_modelo,
        "caminho_modelo": caminho_modelo_evasao,
    }

    # ------------------------------------------------------------------
    # 3) Análise binária da evasão (matriz de confusão + métricas)
    # ------------------------------------------------------------------
    caminho_metricas = "./modelos/resultados_modelos"
    os.makedirs(caminho_metricas, exist_ok=True)

    # Geração da Matriz de Confusão (binarizando a taxa de evasão com threshold de 0.5)
    threshold = 0.5
    y_test_class = (y_test >= threshold).astype(int)
    y_pred_class = (melhor_modelo.predict(X_test) >= threshold).astype(int)

    cm = confusion_matrix(y_test_class, y_pred_class)
    acc = accuracy_score(y_test_class, y_pred_class)
    prec = precision_score(y_test_class, y_pred_class, zero_division=0)
    rec = recall_score(y_test_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_test_class, y_pred_class, zero_division=0)

    print("\n" + "=" * 70)
    print("Exibição de resultados: Classificação binária da evasão (threshold 0.5)")
    print("=" * 70)
    print(f"Acurácia:  {acc:.4f}")
    print(f"Precisão:  {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nMatriz de Confusão (linhas = verdadeiro, colunas = predito):")
    print(cm)

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_test_class,
            y_pred_class,
            target_names=["Baixa evasão", "Alta evasão"],
            zero_division=0,
        )
    )

    # Salvar as métricas dos modelos em arquivo texto (regressão + classificação)
    caminho_metricas_txt = os.path.join(caminho_metricas, "metricas_modelos.txt")
    with open(caminho_metricas_txt, "w") as f:
        f.write(f"Modelo de evasão escolhido: {nome_melhor_modelo}\n\n")
        f.write("=== Regressão: Taxa de evasão ===\n")
        f.write(f"MSE - Regressão Linear: {mse_linear:.4f}, R²: {r2_linear:.4f}\n")
        f.write(f"MSE - Random Forest:   {mse_rf:.4f}, R²: {r2_rf:.4f}\n\n")

        f.write("=== Classificação binária da evasão (threshold 0.5) ===\n")
        f.write(f"Acurácia: {acc:.4f}\n")
        f.write(f"Precisão: {prec:.4f}\n")
        f.write(f"Recall:   {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")

        f.write("Matriz de confusão:\n")
        f.write(str(cm) + "\n\n")

        f.write("Relatório de classificação:\n")
        f.write(
            classification_report(
                y_test_class,
                y_pred_class,
                target_names=["Baixa evasão", "Alta evasão"],
                zero_division=0,
            )
        )

    # Salvar figura da matriz de confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão - {nome_melhor_modelo}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    caminho_cm = os.path.join(caminho_metricas, "matriz_confusao.png")
    plt.tight_layout()
    plt.savefig(caminho_cm)
    plt.close()

    print(f"\nArquivos de métricas salvos em: {caminho_metricas_txt}")
    print(f"Figura da matriz de confusão salva em: {caminho_cm}")
    resultados["classificacao_evasao_binaria"] = {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "matriz_confusao": cm.tolist(),
        "caminho_metricas_txt": caminho_metricas_txt,
        "caminho_matriz_confusao_png": caminho_cm,
    }

    resultados["caminhos"]["metricas"] = caminho_metricas
    resultados["caminhos"]["modelos"] = caminho_modelo

    # ------------------------------------------------------------------
    # 4) Treinamento de modelos de evasão por subconjuntos de variáveis
    #    (baseado nos subconjuntos definidos em pre_processamento.py)
    # ------------------------------------------------------------------
    subconjuntos_features = {
        # Oferta e demanda (curso_oferta_demanda)
        "oferta_demanda": [
            "numero_cursos",
            "vagas_totais",
            "inscritos_totais",
            "ingressantes",
        ],
        # Fluxo acadêmico (curso_fluxo_academico)
        "fluxo_academico": [
            "ingressantes",
            "matriculados",
            "concluintes",
        ],
        # Visão geral compacta (curso_geral_compacto)
        "geral_compacto": [
            "numero_cursos",
            "vagas_totais",
            "inscritos_totais",
            "ingressantes",
            "matriculados",
            "concluintes",
        ],
    }

    # Arquivo adicional com métricas por subconjunto
    caminho_metricas_sub = os.path.join(
        caminho_metricas, "metricas_modelos_subconjuntos.txt"
    )
    with open(caminho_metricas_sub, "w") as f_sub:
        f_sub.write("Resultados por subconjunto de variáveis (taxa_evasao)\n")
        f_sub.write("=" * 70 + "\n\n")

    resultados["subconjuntos_evasao"] = {}
    resultados["caminhos"]["metricas_subconjuntos"] = caminho_metricas_sub

    for nome_sub, cols in subconjuntos_features.items():
        # Garante que as colunas existem no dataframe consolidado
        cols_validas = [c for c in cols if c in df.columns]
        if not cols_validas:
            print(
                f"Aviso: subconjunto '{nome_sub}' sem colunas válidas em df. Pulando."
            )
            continue

        print(
            f"\nTreinando modelo de evasão para subconjunto: {nome_sub} "
            f"com features: {cols_validas}"
        )

        colunas_relevantes_sub = cols_validas + ["taxa_evasao"]
        if "ano" in df.columns:
            colunas_relevantes_sub.append("ano")

        df_sub = df[colunas_relevantes_sub].dropna()

        if df_sub.empty:
            print(
                f"Aviso: subconjunto '{nome_sub}' ficou vazio após dropna(). Pulando."
            )
            continue

        # Split temporal quando possível (2009–2018 treino, 2019–2024 teste)
        if "ano" in df_sub.columns:
            ano_limite_treino_sub = 2018
            df_treino_sub = df_sub[df_sub["ano"] <= ano_limite_treino_sub]
            df_teste_sub = df_sub[df_sub["ano"] > ano_limite_treino_sub]

            if df_treino_sub.empty or df_teste_sub.empty:
                print(
                    f"Aviso: split temporal não pôde ser aplicado para '{nome_sub}'. "
                    "Usando hold-out aleatório."
                )
                X_sub = df_sub[cols_validas]
                y_sub = df_sub["taxa_evasao"]
                X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                    X_sub, y_sub, test_size=0.2, random_state=42
                )
            else:
                print(
                    f"Split temporal para '{nome_sub}': treino até {ano_limite_treino_sub}, "
                    "teste em anos posteriores."
                )
                X_train_sub = df_treino_sub[cols_validas]
                y_train_sub = df_treino_sub["taxa_evasao"]
                X_test_sub = df_teste_sub[cols_validas]
                y_test_sub = df_teste_sub["taxa_evasao"]
        else:
            print(
                f"Aviso: coluna 'ano' não encontrada em df_sub ({nome_sub}). "
                "Usando hold-out aleatório."
            )
            X_sub = df_sub[cols_validas]
            y_sub = df_sub["taxa_evasao"]
            X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                X_sub, y_sub, test_size=0.2, random_state=42
            )

        # Modelo 1: Regressão Linear
        modelo_linear_sub = LinearRegression()
        modelo_linear_sub.fit(X_train_sub, y_train_sub)
        y_pred_linear_sub = modelo_linear_sub.predict(X_test_sub)
        mse_linear_sub = mean_squared_error(y_test_sub, y_pred_linear_sub)
        r2_linear_sub = r2_score(y_test_sub, y_pred_linear_sub)

        # Modelo 2: Random Forest
        modelo_rf_sub = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo_rf_sub.fit(X_train_sub, y_train_sub)
        y_pred_rf_sub = modelo_rf_sub.predict(X_test_sub)
        mse_rf_sub = mean_squared_error(y_test_sub, y_pred_rf_sub)
        r2_rf_sub = r2_score(y_test_sub, y_pred_rf_sub)

        if r2_rf_sub > r2_linear_sub:
            melhor_sub = modelo_rf_sub
            nome_melhor_sub = "Random Forest"
        else:
            melhor_sub = modelo_linear_sub
            nome_melhor_sub = "Regressão Linear"

        caminho_modelo_sub = os.path.join(
            caminho_modelo, f"modelo_evasao_{nome_sub}.pkl"
        )
        joblib.dump(melhor_sub, caminho_modelo_sub)

        # Classificação binária para o subconjunto (mesmo limiar da parte geral)
        y_test_class_sub = (y_test_sub >= threshold).astype(int)
        y_pred_class_sub = (melhor_sub.predict(X_test_sub) >= threshold).astype(int)
        cm_sub = confusion_matrix(y_test_class_sub, y_pred_class_sub)
        acc_sub = accuracy_score(y_test_class_sub, y_pred_class_sub)
        prec_sub = precision_score(
            y_test_class_sub, y_pred_class_sub, zero_division=0
        )
        rec_sub = recall_score(y_test_class_sub, y_pred_class_sub, zero_division=0)
        f1_sub = f1_score(y_test_class_sub, y_pred_class_sub, zero_division=0)

        # Persistência de métricas por subconjunto
        with open(caminho_metricas_sub, "a") as f_sub:
            f_sub.write(f"Subconjunto: {nome_sub}\n")
            f_sub.write(f"Features: {cols_validas}\n")
            f_sub.write(
                f"Regressão Linear  -> MSE: {mse_linear_sub:.4f}, "
                f"R²: {r2_linear_sub:.4f}\n"
            )
            f_sub.write(
                f"Random Forest     -> MSE: {mse_rf_sub:.4f}, "
                f"R²: {r2_rf_sub:.4f}\n"
            )
            f_sub.write(f"Melhor modelo: {nome_melhor_sub}\n")
            f_sub.write(
                "Classificação binária (threshold 0.5) -> "
                f"Acurácia: {acc_sub:.4f}, Precisão: {prec_sub:.4f}, "
                f"Recall: {rec_sub:.4f}, F1: {f1_sub:.4f}\n"
            )
            f_sub.write(f"Matriz de confusão:\n{cm_sub}\n")
            f_sub.write("-" * 70 + "\n\n")

        resultados["subconjuntos_evasao"][nome_sub] = {
            "features": cols_validas,
            "mse_linear": mse_linear_sub,
            "r2_linear": r2_linear_sub,
            "mse_random_forest": mse_rf_sub,
            "r2_random_forest": r2_rf_sub,
            "melhor_modelo": nome_melhor_sub,
            "caminho_modelo": caminho_modelo_sub,
            "accuracy": acc_sub,
            "precision": prec_sub,
            "recall": rec_sub,
            "f1": f1_sub,
            "matriz_confusao": cm_sub.tolist(),
        }

    print("\nTreinamento concluído.")
    return resultados

if __name__ == "__main__":
    main()