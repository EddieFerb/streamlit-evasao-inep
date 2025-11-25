# scripts/rodar_pipeline.py
# Objetivo: Script para rodar todo o pipeline de coleta, processamento, análise, modelagem e visualização de dados.
# Obejective: Script to run the entire pipeline of data collection, processing, analysis, modeling, and visualization.

import subprocess

def run(cmd):
    print(f"\n>>> Rodando: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    run("python scripts/coleta_dados/coletar_links_inep.py")
    run("python scripts/coleta_dados/coleta_dados_oficiais.py")
    run("python scripts/processamento_dados/pre_processamento.py")
    run("python scripts/processamento_dados/tratar_dados.py")
    run("python scripts/analises/analises.py")
    # Escolha do modelo / Model selection
    print("\nSelecione o modelo para rodar / Select which model to run:")
    print("1 - randomforest.py (rápido, 3min) / (fast, 3min)")
    print("2 - fine_tuning.py (muito rápido, 1min) / (fastest, 1min)")
    print("3 - feature_based.py (demora 1–2 horas) / (takes 1–2 hours)")
    escolha = input("Digite 1, 2 ou 3 / Enter 1, 2 or 3: ").strip()

    if escolha == "1":
        run("python scripts/modelagem/randomforest.py")

    elif escolha == "2":
        run("python scripts/modelagem/fine_tuning.py")

    elif escolha == "3":
        print("\n⚠️ Atenção / Warning:")
        print("O script feature_based.py pode levar entre 1 a 2 horas para rodar. / The script feature_based.py may take between 1 and 2 hours to run.")
        print("Este tempo é baseado em referência real  (MacBook Pro M1 Max, 32GB). / This time is based on real reference (MacBook Pro M1 Max, 32GB).")
        confirm = input("Tem certeza que deseja continuar? (s/n) / Are you sure? (y/n): ").strip().lower()

        if confirm in ["s", "y"]:
            run("python scripts/modelagem/feature_based.py")
        else:
            print("\nExecução cancelada pelo usuário. / Execution canceled by user.")
    else:
        print("\nOpção inválida. Pulando etapa de modelagem. / Invalid option. Skipping modeling step.")

    run("python scripts/visualizacao/gerar_graficos.py")
    # dashboard rode manualmente: streamlit run scripts/dashboard/app_evasao.py
    # dashboard run manually: streamlit run scripts/dashboard/app_evasao.py