import pandapower as pp
import pandapower.networks as pn
import simulation_engine as sim
import analysis_tools as tools
import os
import shutil
import sys
import time
import custom_systems as cs

# ==============================================================================
# ARQUIVO PRINCIPAL (MAIN)
# ==============================================================================

TOTAL_STEPS = 8

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
        "3": ("IEEE 39 Barras", pn.case39),
        "4": ("IEEE 57 Barras", pn.case57),
        "5": ("IEEE 118 Barras", pn.case118),
        "6": ("IEEE 30 ANAREDE (PWF TCC)", cs.create_ieee30_anarede)
    }
    print("\nSELEÇÃO DO SISTEMA ELÉTRICO:")
    print(f"  [0] TODAS AS REDES (Bateria de Testes)")
    for key, (name, _) in systems.items(): print(f"  [{key}] {name}")
    
    choice = input("\nDigite a opção desejada (0-6): ").strip()
    
    if choice == "0":
        return list(systems.values())
    elif choice in systems:
        return [systems[choice]]
    else:
        print("Opção inválida. Usando IEEE 30 Padrão.")
        return [systems["2"]]

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
    
    # --- CONFIGURAÇÃO DE ALTA PRECISÃO ---
    CONFIG = {
        'load_scaling_bus_id': None, 
        'enforce_q_lims': False,      
        'distributed_slack': True,    
        'max_scale': 5.0,             
        'steps': 0.002,                
        'min_step': 0.0005,
        'max_iters': 2000,
        'max_failures': 15,
        
        # NOVOS PARÂMETROS SOLVER
        'solver_max_iter': 20,  # Dá mais tempo para convergir no nariz
        'solver_tol': 1e-4      # Tolerância padrão (pode relaxar para 1e-5 se estiver muito difícil)
    }
    print(f"Parâmetros Globais: {CONFIG}")

    for system_index, (system_name, case_func) in enumerate(systems_to_run, 1):
        print(f"\n{'#'*80}\n INICIANDO SIMULAÇÃO {system_index}/{len(systems_to_run)}: {system_name.upper()}\n{'#'*80}")
        
        log_step(1, f"Inicialização: {system_name}")
        net = case_func()
        
        # Ajuste apenas se for o caso padrão (o ANAREDE já vem ajustado)
        if "IEEE 30" in system_name and "ANAREDE" not in system_name:
            adjust_generator_participation(net)
        
        # Preparação de Pastas
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

        # 2. Relatório Inicial
        log_step(2, "Gerando Relatório do Caso Base")
        rep_initial = os.path.join(reports_dir, "relatorio_inicial_base.txt")
        tools.generate_initial_report(net, system_name, rep_initial)

        # 3. Matrizes
        log_step(3, "Pré-Cálculo de Matrizes")
        try: static_matrices = tools.pre_calculate_matrices(net)
        except Exception as e: 
            print(f"Erro fatal na topologia: {e}")
            continue

        # 4. Execução CPF
        log_step(4, "Executando Fluxo de Potência Continuado")
        history, full_log = sim.run_continuation_process(
                net, 
                load_scaling_bus_id=CONFIG['load_scaling_bus_id'],
                max_scale=CONFIG['max_scale'],
                initial_step=CONFIG['steps'],
                min_step=CONFIG['min_step'],
                max_iters=CONFIG['max_iters'],
                max_failures=CONFIG['max_failures'],
                enforce_q_lims=CONFIG['enforce_q_lims'],
                distributed_slack=CONFIG['distributed_slack'],
                
                # Passa os parâmetros do solver
                solver_max_iter=CONFIG['solver_max_iter'],
                solver_tol=CONFIG['solver_tol']
            )
        
        if not history: 
            print(f"Aviso: {system_name} não convergiu. Pulando...")
            continue

        # 5. Extração e Tabelas
        log_step(5, "Extração e Tabelas")
        scenarios = sim.extract_scenarios(history, [0, 25, 50, 75, 95, 99, 100])
        all_results = {} # Dicionário para plotagem
        
        for pct, snapshot in scenarios.items():
            df = tools.calculate_indices_for_scenario(snapshot, static_matrices)
            all_results[pct] = df
            try:
                csv_name = f"resultados_indices_cenario_{pct}.csv"
                df.to_csv(os.path.join(sheets_dir, csv_name), index=False)
            except: pass

        # 6. Gráficos
        log_step(6, "Geração de Gráficos")
        try:
            tools.plot_pv_curves(history, title=f"Curva PV - {system_name}", save_dir=pv_dir)
            tools.plot_comparative_indices(all_results, save_dir=figures_dir)
        except Exception as e: print(f"Erro gráfico: {e}")

        # 7. Relatórios Finais
        log_step(7, "Gerando Relatórios Finais")
        rep_col = os.path.join(reports_dir, "relatorio_colapso.txt")
        rep_conv = os.path.join(reports_dir, "relatorio_convergencia.txt")
        
        tools.generate_anarede_report(history, system_name, rep_col)
        log_step(8, "Finalizando Caso")
        tools.generate_convergence_report(full_log, system_name, rep_conv)

    # --- FIM GERAL ---
    total_elapsed = time.time() - start_time
    mins, secs = int(total_elapsed // 60), total_elapsed % 60

    print(f"\n{'='*60}")
    print(f"BATERIA DE TESTES CONCLUÍDA!")
    print(f"Tempo Total: {mins}m {secs:.2f}s")
    print(f"Resultados em: /outputs/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()