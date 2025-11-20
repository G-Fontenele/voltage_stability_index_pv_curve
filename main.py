import pandapower.networks as pn
import pandapower as pp

import simulation_engine as sim
import analysis_tools as tools
import os
import shutil
import sys
import time

# ==============================================================================
# ARQUIVO PRINCIPAL (MAIN)
# ==============================================================================

TOTAL_STEPS = 7

def log_step(step_num, message):
    print(f"\n[{step_num}/{TOTAL_STEPS}] {message}")
    print("-" * 60)

def print_intro():
    intro = """
    ============================================================
       SIMULADOR DE ESTABILIDADE DE TENSÃO (CPF - PYTHON)
    ============================================================
    OBJETIVO: Identificar Margem de Estabilidade via CPF.
    ============================================================
    """
    print(intro)

def select_system():
    """
    Permite selecionar um único sistema ou todos para execução em lote.
    Retorna uma lista de tuplas: [(name, func), (name, func), ...]
    """
    systems = {
        "1": ("IEEE 14 Barras", pn.case14),
        "2": ("IEEE 30 Barras", pn.case30),
        "3": ("IEEE 39 Barras (New England)", pn.case39),
        "4": ("IEEE 57 Barras", pn.case57),
        "5": ("IEEE 118 Barras", pn.case118)
    }
    
    print("\nSELEÇÃO DO SISTEMA ELÉTRICO:")
    print(f"  [0] TODAS AS REDES (Bateria de Testes)")
    for key, (name, _) in systems.items(): 
        print(f"  [{key}] {name}")
        
    choice = input("\nDigite a opção desejada (0-5): ").strip()
    
    if choice == "0":
        print("\n>> MODO BATERIA: Executando todos os sistemas sequencialmente.")
        return list(systems.values())
    elif choice in systems:
        return [systems[choice]]
    else:
        print("Opção inválida. Usando IEEE 30 por padrão.")
        return [systems["2"]]

def adjust_generator_participation(net):
    """Ajusta o despacho inicial do Gerador 2 para 13.3% (Apenas IEEE 30)."""
    print("\n--- AJUSTE FINO DE DESPACHO (TCC Madureira) ---")
    try: pp.runpp(net)
    except: pass
        
    total_load = net.res_load.p_mw.sum()
    total_loss = net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
    total_gen = total_load + total_loss
    target_mw_g2 = total_gen * 0.133 
    
    gen2_candidates = net.gen[net.gen.bus == 1]
    if gen2_candidates.empty: gen2_candidates = net.gen[net.gen.bus == 2]
        
    if gen2_candidates.empty: return

    gen2_idx = gen2_candidates.index[0]
    net.gen.at[gen2_idx, 'p_mw'] = target_mw_g2
    print(f"Gerador 2 ajustado para: {target_mw_g2:.2f} MW")
    try: pp.runpp(net); print("Ajuste aplicado.")
    except: pass
    print("-" * 50)

def main():
    global_start_time = time.time()
    print_intro()
    
    # 1. Seleção (Retorna uma lista de sistemas)
    systems_to_run = select_system()
    
    # --- CONFIGURAÇÃO GERAL DA SIMULAÇÃO ---
    CONFIG = {
        'load_scaling_bus_id': None, 
        'enforce_q_lims': False,      
        'distributed_slack': True,    
        'max_scale': 5.0,             
        'steps': 0.002,                
        'min_step': 0.00001             
    }
    
    print(f"Parâmetros Globais: {CONFIG}")

    # --- LOOP DE EXECUÇÃO (BATERIA DE TESTES) ---
    for system_index, (system_name, case_func) in enumerate(systems_to_run, 1):
        
        print(f"\n{'#'*80}")
        print(f" INICIANDO SIMULAÇÃO {system_index}/{len(systems_to_run)}: {system_name.upper()}")
        print(f"{'#'*80}")
        
        case_start_time = time.time()
        
        # Inicializa a rede
        log_step(1, f"Inicialização: {system_name}")
        net = case_func()
        
        # Ajuste específico para IEEE 30
        if "IEEE 30" in system_name:
            adjust_generator_participation(net)
        
        # Preparação de Pastas para ESTE caso
        case_folder_name = system_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        base_output_dir = os.path.join("outputs", case_folder_name)
        sheets_dir = os.path.join(base_output_dir, "index_sheets")
        figures_dir = os.path.join(base_output_dir, "index_figures")
        pv_dir = os.path.join(base_output_dir, "pv_figures")
        reports_dir = os.path.join(base_output_dir, "reports")
        
        if os.path.exists(base_output_dir):
            try: shutil.rmtree(base_output_dir)
            except: pass
            
        os.makedirs(sheets_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(pv_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        # 2. Pré-Cálculo
        log_step(2, "Pré-Cálculo de Matrizes")
        try: static_matrices = tools.pre_calculate_matrices(net)
        except Exception as e: 
            print(f"Erro ({system_name}): {e}")
            continue # Pula para o próximo sistema

        # 3. Execução CPF
        log_step(3, "Executando Fluxo de Potência Continuado")
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
            print(f"Aviso: {system_name} não convergiu. Pulando...")
            continue

        # 4. Extração
        log_step(4, "Extração e Tabelas")
        scenarios = sim.extract_scenarios(history, [0, 25, 50, 75, 95, 99, 100])
        all_results = {}
        for pct, snapshot in scenarios.items():
            df = tools.calculate_indices_for_scenario(snapshot, static_matrices)
            all_results[pct] = df
            try:
                csv_name = f"resultados_indices_cenario_{pct}.csv"
                df.to_csv(os.path.join(sheets_dir, csv_name), index=False)
            except: pass

        # 5. Gráficos
        log_step(5, "Geração de Gráficos")
        try:
            tools.plot_pv_curves(history, title=f"Curva PV - {system_name}", save_dir=pv_dir)
            tools.plot_comparative_indices(all_results, save_dir=figures_dir)
        except Exception as e: print(f"Erro gráfico: {e}")

        # 6. Relatórios
        log_step(6, "Gerando Relatórios")
        rep_col = os.path.join(reports_dir, "relatorio_colapso.txt")
        rep_conv = os.path.join(reports_dir, "relatorio_convergencia.txt")
        
        tools.generate_anarede_report(history, system_name, rep_col)
        
        log_step(7, "Finalizando Caso")
        tools.generate_convergence_report(full_log, system_name, rep_conv)

        case_elapsed = time.time() - case_start_time
        print(f"--> {system_name} finalizado em {case_elapsed:.2f}s")

    # --- FIM GERAL ---
    total_elapsed = time.time() - global_start_time
    mins, secs = int(total_elapsed // 60), total_elapsed % 60

    print(f"\n{'='*60}")
    print(f"BATERIA DE TESTES CONCLUÍDA!")
    print(f"Sistemas Processados: {len(systems_to_run)}")
    print(f"Tempo Total: {mins}m {secs:.2f}s")
    print(f"Todos os resultados estão em: /outputs/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()