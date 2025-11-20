import pandapower.networks as pn
import simulation_engine as sim
import analysis_tools as tools
import os
import shutil
import sys
import time

# ==============================================================================
# ARQUIVO PRINCIPAL (MAIN)
# ==============================================================================
# Orquestra a execução do simulador, configurando o sistema, os parâmetros
# e chamando as funções de cálculo e geração de relatórios.
# ==============================================================================

TOTAL_STEPS = 7

def log_step(step_num, message):
    """Função auxiliar para imprimir etapas do processo de forma organizada."""
    print(f"\n[{step_num}/{TOTAL_STEPS}] {message}")
    print("-" * 60)

def print_intro():
    """
    Exibe o cabeçalho inicial com explicações teóricas sobre o código.
    Isso garante que o usuário entenda o que está sendo simulado.
    """
    intro = """
    ============================================================
       SIMULADOR DE ESTABILIDADE DE TENSÃO (CPF - PYTHON)
    ============================================================
    
    OBJETIVO:
    Determinar a Margem de Estabilidade de Tensão e identificar
    elementos críticos (linhas/barras) através do método de 
    Fluxo de Potência Continuado (CPF).
    
    DEFINIÇÕES FUNDAMENTAIS:
    1. Fator de Escala (Lambda): 
       Multiplicador aplicado à carga base e geração ativa.
       Lambda = 1.0 representa o carregamento original do caso.
       
    2. Ponto de Colapso (Máximo Carregamento):
       Ponto onde a matriz Jacobiana se torna singular e o fluxo
       de potência não tem mais solução (nariz da curva PV).
       
    METODOLOGIA (REPLICAÇÃO ANAREDE/TCC):
    - Passo Adaptativo: O passo diminui ao detectar divergência.
    - Alta Resolução: Incrementos pequenos (0.5%) para traçar a curva suave.
    - Despacho Distribuído: Geradores ativos assumem o aumento de carga.
    - Limites Q: Configurável (Ligado = Realista / Desligado = Teórico).
      
    ============================================================
    """
    print(intro)

def select_system():
    """
    Menu interativo para seleção do sistema de teste IEEE.
    Retorna o nome do sistema e a função geradora do pandapower.
    """
    systems = {
        "1": ("IEEE 14 Barras", pn.case14),
        "2": ("IEEE 30 Barras", pn.case30),
        "3": ("IEEE 39 Barras (New England)", pn.case39),
        "4": ("IEEE 57 Barras", pn.case57),
        "5": ("IEEE 118 Barras", pn.case118)
    }
    
    print("\nSELEÇÃO DO SISTEMA ELÉTRICO:")
    for key, (name, _) in systems.items():
        print(f"  [{key}] {name}")
        
    choice = input("\nDigite o número do sistema desejado (1-5): ").strip()
    
    if choice in systems:
        return systems[choice]
    else:
        print("Opção inválida. Usando IEEE 30 por padrão.")
        return systems["2"]

def main():
    # Inicia a contagem de tempo
    start_time = time.time()
    
    # Exibe a explicação inicial
    print_intro()
    
    # --- ETAPA 1: INICIALIZAÇÃO ---
    log_step(1, "Configuração e Inicialização")
    
    # Seleciona e carrega a rede
    system_name, case_func = select_system()
    net = case_func()
    
    # CONFIGURAÇÃO DE SIMULAÇÃO (ALTA FIDELIDADE)
    # Ajustada para replicar o TCC do Madureira e o relatório do ANAREDE
    CONFIG = {
        'load_scaling_bus_id': None,  # None = Escala todo o sistema (Global)
        'enforce_q_lims': False,      # False = Q Infinito (Curva Teórica/TCC) | True = Realista
        'distributed_slack': True,    # True = Geradores ajudam a Slack | False = Slack isolada
        'max_scale': 5.0,             # Lambda máximo
        'steps': 0.005,               # Tamanho do passo (0.005 = 0.5%)
        'min_step': 0.001             # Precisão mínima para detecção do colapso
    }
    
    print(f"Sistema Selecionado: {system_name}")
    print(f"Parâmetros de Configuração: {CONFIG}")
    
    # Criação/Limpeza da pasta de resultados
    output_dir = 'tabelas_resultados'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Limpando arquivos antigos em /{output_dir}...")
    for f in os.listdir(output_dir):
        try: os.remove(os.path.join(output_dir, f))
        except: pass

    # --- ETAPA 2: MATRIZES ESTÁTICAS ---
    log_step(2, "Pré-Cálculo de Matrizes da Rede")
    # Calcula Ybus e F-matrix uma única vez para acelerar os índices depois
    try:
        static_matrices = tools.pre_calculate_matrices(net)
    except Exception as e:
        print(f"Erro crítico na análise topológica: {e}")
        return

    # --- ETAPA 3: EXECUÇÃO DO CPF ---
    log_step(3, "Executando Fluxo de Potência Continuado (Alta Resolução)")
    print("Iniciando varredura da Curva PV. Isso pode levar alguns instantes...")
    
    # Executa a simulação.
    # history: Lista de snapshots convergentes (dados limpos para plot).
    # full_log: Lista completa com erros e tentativas (para relatório ANAREDE).
    history, full_log = sim.run_continuation_process(
        net, 
        load_scaling_bus_id=CONFIG['load_scaling_bus_id'],
        max_scale=CONFIG['max_scale'],
        initial_step=CONFIG['steps'],
        min_step=CONFIG['min_step'],
        enforce_q_lims=CONFIG['enforce_q_lims'],
        distributed_slack=CONFIG['distributed_slack']
    )
    
    if not history:
        print("ERRO: A simulação não convergiu. Verifique o caso base.")
        return

    # --- ETAPA 4: CÁLCULO DE ÍNDICES ---
    log_step(4, "Extração dos Cenários de Interesse")
    # Seleciona os pontos da curva correspondentes a 0%, 25%... 100% da margem
    scenarios = sim.extract_scenarios(history, [0, 25, 50, 75, 95, 99, 100])
    all_results = {}
    
    print("Calculando 17 Índices de Estabilidade para cada cenário...")
    for pct, snapshot in scenarios.items():
        # Calcula índices para o snapshot atual usando as matrizes pré-calculadas
        df = tools.calculate_indices_for_scenario(snapshot, static_matrices)
        all_results[pct] = df
        
        try:
            # Salva tabela CSV individual
            csv_name = f"resultados_indices_cenario_{pct}.csv"
            df.to_csv(os.path.join(output_dir, csv_name), index=False)
        except: pass

    # --- ETAPA 5: GRÁFICOS ---
    log_step(5, "Geração de Gráficos Comparativos")
    try:
        # Plota a Curva PV completa
        tools.plot_pv_curves(history, title=f"Curva PV - {system_name}")
        # Plota a evolução de cada índice (FVSI, Lmn, etc.)
        tools.plot_comparative_indices(all_results)
    except Exception as e:
        print(f"Aviso: Erro ao gerar gráficos ({e})")

    # --- ETAPA 6: RELATÓRIO DE COLAPSO ---
    log_step(6, "Gerando Relatório de Colapso (Detalhado)")
    # Gera o "Raio-X" do sistema no momento da falha
    report_name_col = f"relatorio_colapso_{system_name.replace(' ', '_').lower()}.txt"
    tools.generate_anarede_report(history, system_name, report_name_col)

    # --- ETAPA 7: RELATÓRIO DE CONVERGÊNCIA ---
    log_step(7, "Gerando Relatório de Convergência (Passo-a-Passo)")
    # Gera o log sequencial idêntico ao do ANAREDE
    report_name_conv = f"relatorio_convergencia_{system_name.replace(' ', '_').lower()}.txt"
    tools.generate_convergence_report(full_log, system_name, report_name_conv)

    # --- FIM E ESTATÍSTICAS ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print(f"\n{'='*60}")
    print(f"PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*60}")
    print(f"Estatísticas da Simulação:")
    print(f"  - Total de Tentativas (Iterações): {len(full_log)}")
    print(f"  - Passos Convergentes (Gráfico):   {len(history)}")
    print(f"  - Tempo Total de Execução:         {minutes}m {seconds:.2f}s")
    print(f"\nArquivos Gerados:")
    print(f"  - Gráficos (.png):    Na pasta raiz")
    print(f"  - Tabelas (.csv):     Na pasta /{output_dir}")
    print(f"  - Relatórios (.txt):  {report_name_col}")
    print(f"                        {report_name_conv}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()