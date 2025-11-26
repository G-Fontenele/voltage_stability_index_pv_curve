import pandapower as pp
import numpy as np
import copy

# ==============================================================================
# MOTOR DE SIMULAÇÃO (CPF) - CORRIGIDO COM SOLVER PARAMS
# ==============================================================================

def run_continuation_process(net, load_scaling_bus_id=None, max_scale=5.0, 
                             initial_step=0.1, min_step=0.001, 
                             max_iters=2000, max_failures=10,
                             enforce_q_lims=True, distributed_slack=True,
                             # NOVOS PARÂMETROS DO SOLVER (NEWTON-RAPHSON)
                             solver_max_iter=50,  # Aumentado de 10 para 50 (Crucial perto do colapso)
                             solver_tol=1e-6):    # Tolerância de Mismatch (MVA)
    
    print(f"\n{'='*80}")
    print(f" MOTOR DE SIMULAÇÃO: FLUXO DE POTÊNCIA CONTINUADO (CPF)")
    print(f"{'='*80}")
    print(f" CONFIGURAÇÃO DE PARADA:")
    print(f"  > DINC (Limite):    {max_scale}")
    print(f"  > ICMN (Passo Min): {min_step}")
    print(f"  > ICIT (Max Iter):  {max_iters}")
    print(f"  > DMAX (Max Fail):  {max_failures}")
    print(f" CONFIGURAÇÃO DO SOLVER (NEWTON-RAPHSON):")
    print(f"  > Max Iterações:    {solver_max_iter} (Padrão PP=10)")
    print(f"  > Tolerância:       {solver_tol} MVA")
    print(f"{'='*80}\n")

    net_sim = copy.deepcopy(net)
    
    # --- 1. Identificação de Cargas ---
    if load_scaling_bus_id is None:
        load_idx = net_sim.load.index
    else:
        load_idx = net_sim.load[net_sim.load.bus == load_scaling_bus_id].index
        if load_idx.empty: return [], []
    
    base_p_load = net_sim.load.loc[load_idx, 'p_mw'].copy()
    base_q_load = net_sim.load.loc[load_idx, 'q_mvar'].copy()

    # --- 2. Identificação de Geradores ---
    active_gen_idx = []
    base_p_gen = []
    if distributed_slack:
        mask_gen = net_sim.gen.p_mw > 1.0
        active_gen_idx = net_sim.gen[mask_gen].index
        base_p_gen = net_sim.gen.loc[active_gen_idx, 'p_mw'].copy()

    # --- 3. Inicialização ---
    current_scale = 1.0
    current_step = initial_step
    history = []   
    full_log = []  
    consecutive_failures = 0
    total_iterations = 0

    # --- 4. Caso Base ---
    print(f" [...] Rodando Caso Base...")
    try:
        # Na primeira vez, usamos init="auto" (padrão) pois não temos histórico
        pp.runpp(net_sim, enforce_q_lims=enforce_q_lims, 
                 max_iteration=solver_max_iter, tolerance_mva=solver_tol)
        
        _save_snapshot(net_sim, 1.0, history)
        _log_attempt(full_log, 0, 1.0, 0.0, "Convergente", net_sim)
        print(f"   -> Convergiu Base OK")
    except:
        _log_attempt(full_log, 0, 1.0, 0.0, "Divergente", None)
        print("ERRO CRÍTICO: Caso base não converge.")
        return [], full_log

    # --- 5. Loop Principal ---
    while current_scale < max_scale:
        if total_iterations >= max_iters:
            print(f"[PARADA] Max iterações ({max_iters}).")
            break

        next_scale = current_scale + current_step
        if next_scale > max_scale: next_scale = max_scale
        total_iterations += 1

        # Aplica Escala
        net_sim.load.loc[load_idx, 'p_mw'] = base_p_load * next_scale
        net_sim.load.loc[load_idx, 'q_mvar'] = base_q_load * next_scale
        if distributed_slack and not active_gen_idx.empty:
            net_sim.gen.loc[active_gen_idx, 'p_mw'] = base_p_gen * next_scale

        try:
            # --- SOLVER ROBUSTO ---
            # init="results": Usa a tensão do passo anterior como chute inicial.
            # Isso é essencial para CPF, pois evita "flat start" (1.0 pu) quando a tensão real já é 0.8 pu.
            # max_iteration: Aumentado para dar chance de convergir no "nariz".
            
            pp.runpp(net_sim, enforce_q_lims=enforce_q_lims, 
                     init="results", 
                     max_iteration=solver_max_iter, 
                     tolerance_mva=solver_tol)
            
            # Sucesso
            consecutive_failures = 0
            _save_snapshot(net_sim, next_scale, history)
            _log_attempt(full_log, total_iterations, next_scale, current_step, "Convergente", net_sim)
            
            p_tot = net_sim.res_load.p_mw.sum()
            print(f"   Iter {total_iterations}: Scale {next_scale:.5f} OK | Carga: {p_tot:.1f} MW")
            
            current_scale = next_scale

        except pp.LoadflowNotConverged:
            # Falha
            consecutive_failures += 1
            _log_attempt(full_log, total_iterations, next_scale, current_step, "Divergente", None)
            print(f"   Iter {total_iterations}: Falha em {next_scale:.5f}. Reduzindo passo...")

            if consecutive_failures >= max_failures: break
            if current_step < min_step:
                print(f"--> COLAPSO EM: {current_scale:.5f}")
                break
            
            current_step /= 2.0
            # Importante: Ao falhar, o net_sim pode estar com tensões "sujas" da iteração falha.
            # Se tivéssemos um deepcopy do 'last_good_net', restauraríamos aqui.
            # Como o pandapower mantém o res_bus anterior se falhar (geralmente), init="results"
            # na próxima tentativa (com passo menor) ainda deve pegar o último bom.

    return history, full_log

def _save_snapshot(net, scale, history_list):
    snapshot = {
        'scale': scale,
        'total_load_mw': net.res_load.p_mw.sum(),
        'total_load_mvar': net.res_load.q_mvar.sum(),
        'res_bus': net.res_bus.copy(),
        'res_line': net.res_line.copy(),
        'res_trafo': net.res_trafo.copy(),
        'line_data': net.line.copy(),
        'trafo_data': net.trafo.copy(),
        'bus_data': net.bus.copy()
    }
    history_list.append(snapshot)

def _log_attempt(log_list, iter_num, scale, step_used, status, net=None):
    row = {'iter': iter_num, 'scale': scale, 'step': step_used, 'status': status, 
           'mw': 0.0, 'mvar': 0.0, 'p_gen': 0.0, 'p_slack': 0.0, 'vmin': 0.0}
    if net is not None:
        row['mw'] = net.res_load.p_mw.sum()
        row['mvar'] = net.res_load.q_mvar.sum()
        row['vmin'] = net.res_bus['vm_pu'].min()
        gen_p = net.res_gen.p_mw.sum() if not net.res_gen.empty else 0.0
        ext_p = net.res_ext_grid.p_mw.sum() if not net.res_ext_grid.empty else 0.0
        row['p_gen'] = gen_p + ext_p
        row['p_slack'] = ext_p
    log_list.append(row)

def extract_scenarios(history, percentages):
    if not history: return {}
    base_load = history[0]['total_load_mw']
    max_load = history[-1]['total_load_mw']
    load_margin = max_load - base_load
    scenarios = {}
    for pct in percentages:
        target = base_load + (pct/100.0)*load_margin
        best_snap = min(history, key=lambda x: abs(x['total_load_mw'] - target))
        scenarios[pct] = best_snap
    return scenarios