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
    run("python scripts/modelagem/modelagem.py")
    run("python scripts/visualizacao/gerar_graficos.py")
    # dashboard é manual: streamlit run scripts/dashboard/app_evasao.py