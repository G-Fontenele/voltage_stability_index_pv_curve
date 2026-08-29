import os
import tempfile
os.environ["NUMBA_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "numba_cache")

import pandapower.networks as pn
import pandapower as pp
import simulation_engine as sim
import analysis_tools as tools
import shutil
import sys
import time
import custom_systems as cs

# ==============================================================================
# ARQUIVO PRINCIPAL (MAIN)
# ==============================================================================

TOTAL_STEPS = 9 # Aumentado para 9 (Incluindo exportação PWF)

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
    systems = {
        "1": ("IEEE 14 Barras", pn.case14),
        "2": ("IEEE 30 Barras (Padrão)", pn.case30),
        "3": ("IEEE 39 Barras (New England)", pn.case39),
        "4": ("IEEE 57 Barras", pn.case57),
        "5": ("IEEE 118 Barras", pn.case118),
        "6": ("IEEE 30 ANAREDE (PWF TCC)", cs.create_ieee30_anarede)
    }
    print("\nSELEÇÃO DO SISTEMA ELÉTRICO:")
    print(f"  [0] TODAS AS REDES (Bateria de Testes)")
    for key, (name, _) in systems.items(): print(f"  [{key}] {name}")
    choice = input("\nDigite a opção desejada (0-6): ").strip()
    if choice == "0": return list(systems.values())
    elif choice in systems: return [systems[choice]]
    else: return [systems["2"]]

def select_mode():
    print("\n" + "="*50)
    print(" MODO DE OPERACAO:")
    print("="*50)
    print("  [1] N-0 (SPF - Successive Power Flow, sem QLIM)")
    print("  [2] N-0 (CPF Verdadeiro - Preditor-Corretor, PADRAO ARTIGO)")
    print("  [3] N-0 com QLIM (Pandapower - enforce_q_lims)")
    print("  [4] N-0 com QLIM (Agregacao de PV para PQ)")
    print("  [5] N-1 (Todas as Contingencias de Linha - CPF)")
    choice = input("\nEscolha a opcao (1-5) [Padrao: 2 - CPF]: ").strip()
    
    if choice == "5": return True, "none", "cpf"
    if choice == "4": return False, "pv_to_pq", "spf"
    if choice == "3": return False, "pandapower", "spf"
    if choice == "1": return False, "none", "spf"
    
    return False, "none", "cpf"  # default: CPF verdadeiro

def adjust_generator_participation(net):
    """Ajusta o despacho inicial do Gerador 2 para 13.3% (Apenas IEEE 30 Padrão)."""
    print("\n--- AJUSTE FINO DE DESPACHO (TCC Madureira) ---")
    try: pp.runpp(net)
    except: pass
        
    total_load = net.res_load.p_mw.sum()
    total_loss = net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
    total_gen = total_load + total_loss
    target_mw_g2 = total_gen * 0.133 
    
    # Procura gerador na barra 1 ou 2
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
    start_time = time.time()
    print_intro()
    
    systems_to_run = select_system()
    run_n1, qlim_mode, cpf_mode = select_mode()
    
    CONFIG = {
        'load_scaling_bus_id': None, 
        'qlim_mode': qlim_mode,      
        'distributed_slack': True,    
        'max_scale': 5.0,             
        'steps': 0.002,                
        'min_step': 0.0005,
        'max_iters': 2000,
        'max_failures': 15,
        'solver_max_iter': 20,
        'solver_tol': 0.1,
        'cpf_mode': cpf_mode,    # 'cpf' = CPF verdadeiro (artigo), 'spf' = retrocompativel
    }
    print(f"Parâmetros Globais: {CONFIG}")

    for system_index, (system_name, case_func) in enumerate(systems_to_run, 1):
        print(f"\n{'#'*80}\n INICIANDO SIMULAÇÃO {system_index}/{len(systems_to_run)}: {system_name.upper()}\n{'#'*80}")
        
        log_step(1, f"Inicialização: {system_name}")
        net = case_func()
        bus_count = len(net.bus)
        
        # Ajuste apenas se for o caso padrão
        if "IEEE 30" in system_name and "ANAREDE" not in system_name:
            adjust_generator_participation(net)
        
        case_folder_name = system_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        base_output_dir = os.path.join("outputs", case_folder_name)
        
        if os.path.exists(base_output_dir):
            try: shutil.rmtree(base_output_dir)
            except: pass
            
        print(f"\n=== EXECUTANDO CASO BASE (N-0) ===")
        max_scale_base = run_scenario(net, system_name, base_output_dir, bus_count, CONFIG, scenario_name="base")
        
        if run_n1:
            print(f"\n{'='*80}")
            print(f" INICIANDO ANÁLISE N-1 PARA {system_name}")
            print(f"{'='*80}")
            
            contingency_dir = os.path.join(base_output_dir, "contingencies")
            os.makedirs(contingency_dir, exist_ok=True)
            
            ranking = []
            import copy
            import pandas as pd
            
            for branch_idx in net.line.index:
                print(f"\n---> Testando Contingência: Linha {branch_idx}...")
                net_cont = copy.deepcopy(net)
                net_cont.line.at[branch_idx, 'in_service'] = False
                
                cont_output_dir = os.path.join(contingency_dir, f"line_{branch_idx}")
                try:
                    max_scale_cont = run_scenario(net_cont, f"{system_name} (S/ L{branch_idx})", cont_output_dir, bus_count, CONFIG, scenario_name=f"cont_L{branch_idx}")
                    if max_scale_cont is not None:
                        ranking.append({'Linha': branch_idx, 'Lambda_Max': max_scale_cont})
                except Exception as e:
                    print(f"Erro na contingência da linha {branch_idx}: {e}")
            
            if ranking:
                df_ranking = pd.DataFrame(ranking)
                df_ranking = df_ranking.sort_values(by='Lambda_Max').reset_index(drop=True)
                ranking_path = os.path.join(contingency_dir, "ranking_contingencias.csv")
                df_ranking.to_csv(ranking_path, index=False)
                
                pior_caso = df_ranking.iloc[0]
                print(f"\n*** PIOR CASO N-1 ENCONTRADO: Linha {int(pior_caso['Linha'])} com Lambda Max = {pior_caso['Lambda_Max']:.4f} ***")
                
                # Copiar os gráficos do pior caso para evidenciar o colapso precoce
                worst_case_dir = os.path.join(base_output_dir, "worst_case_analysis")
                os.makedirs(worst_case_dir, exist_ok=True)
                
                src_worst_cont = os.path.join(contingency_dir, f"line_{int(pior_caso['Linha'])}")
                if os.path.exists(src_worst_cont):
                    for folder in ['index_figures', 'pv_figures', 'reports']:
                        src_folder = os.path.join(src_worst_cont, folder)
                        dst_folder = os.path.join(worst_case_dir, folder)
                        if os.path.exists(src_folder):
                            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
                            
                print(f"Resultados do pior caso copiados para: {worst_case_dir}")

    total_elapsed = time.time() - start_time
    mins, secs = int(total_elapsed // 60), total_elapsed % 60

    print(f"\n{'='*60}")
    print(f"BATERIA DE TESTES CONCLUÍDA!")
    print(f"Tempo Total: {mins}m {secs:.2f}s")
    print(f"Resultados em: /outputs/")
    print(f"{'='*60}")

def run_scenario(net, system_name, output_dir, bus_count, CONFIG, scenario_name="base"):
    sheets_dir = os.path.join(output_dir, "index_sheets")
    figures_dir = os.path.join(output_dir, "index_figures")
    pv_dir = os.path.join(output_dir, "pv_figures")
    reports_dir = os.path.join(output_dir, "reports")
    network_dir = os.path.join(output_dir, "network")
    
    os.makedirs(sheets_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(pv_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)

    pwf_name = f"{scenario_name}_{bus_count}.pwf"
    pwf_path = os.path.join(network_dir, pwf_name)
    try: cs.export_pwf_anarede(net, pwf_path)
    except: pass

    rep_initial = os.path.join(reports_dir, f"relatorio_inicial_{scenario_name}_{bus_count}.txt")
    tools.generate_initial_report(net, system_name, rep_initial, bus_count=bus_count)

    try: static_matrices = tools.pre_calculate_matrices(net)
    except Exception as e: 
        print(f"Erro fatal na topologia: {e}")
        return None

    history, full_log = sim.run_continuation_process(
        net, 
        initial_static_matrices=static_matrices,
        load_scaling_bus_id=CONFIG['load_scaling_bus_id'],
        max_scale=CONFIG['max_scale'],
        initial_step=CONFIG['steps'],
        min_step=CONFIG['min_step'],
        max_iters=CONFIG['max_iters'],
        max_failures=CONFIG['max_failures'],
        qlim_mode=CONFIG['qlim_mode'],
        distributed_slack=CONFIG['distributed_slack'],
        solver_max_iter=CONFIG['solver_max_iter'],
        solver_tol=CONFIG['solver_tol'],
        cpf_mode=CONFIG.get('cpf_mode', 'spf')  # default spf para retrocompatibilidade
    )
    
    if not history: 
        print(f"Aviso: Não convergiu. Pulando...")
        return None

    scenarios = sim.extract_scenarios(history, [0, 25, 50, 75, 95, 99, 100])
    branch_results_scenarios = {} 
    bus_results_scenarios = {}
    
    for pct, snapshot in scenarios.items():
        branch_df, bus_df = tools.calculate_indices_for_scenario(snapshot, static_matrices)
        branch_results_scenarios[pct] = branch_df
        bus_results_scenarios[pct] = bus_df
        try:
            branch_df.to_csv(os.path.join(sheets_dir, f"resultados_indices_ramos_{pct}_{bus_count}.csv"), index=False)
            bus_df.to_csv(os.path.join(sheets_dir, f"resultados_indices_barras_{pct}_{bus_count}.csv"), index=False)
        except: pass

    # Extrair Correlação Estatística (Spearman/Kendall) no ponto de colapso
    if 100 in branch_results_scenarios and 100 in bus_results_scenarios:
        tools.generate_correlation_reports(branch_results_scenarios[100], bus_results_scenarios[100], reports_dir, bus_count)

    try:
        tools.plot_pv_curves(history, title=f"Curva PV - {system_name}", save_dir=pv_dir, bus_count=bus_count)
        tools.plot_comparative_indices(branch_results_scenarios, bus_results_scenarios, save_dir=figures_dir, bus_count=bus_count)
    except Exception as e: print(f"Erro gráfico: {e}")

    rep_col = os.path.join(reports_dir, f"relatorio_colapso_{scenario_name}_{bus_count}.txt")
    rep_conv = os.path.join(reports_dir, f"relatorio_convergencia_{scenario_name}_{bus_count}.txt")
    
    tools.generate_anarede_report(history, system_name, rep_col, bus_count=bus_count)
    tools.generate_convergence_report(full_log, system_name, rep_conv, bus_count=bus_count)

    max_scale_found = history[-1]['scale'] if history else None
    return max_scale_found

if __name__ == "__main__":
    main()