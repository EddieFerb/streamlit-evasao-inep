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
    print("1 - modelagem.py (rápido) / (fast)")
    print("2 - feature_based.py (demora 1–2 horas) / (takes 1–2 hours)")
    escolha = input("Digite 1 ou 2 / Enter 1 or 2: ").strip()

    if escolha == "1":
        run("python scripts/modelagem/modelagem.py")

    elif escolha == "2":
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