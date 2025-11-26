import pandapower as pp
import numpy as np
import copy

# ==============================================================================
# MOTOR DE SIMULAÇÃO (CPF) - PADRÃO ANAREDE
# ==============================================================================
# Implementação rigorosa baseada no Manual do Usuário ANAREDE.
# Critérios de Parada:
# 1. DINC (Limite de Carregamento): max_scale
# 2. ICMN (Passo Mínimo): min_step
# 3. ICIT (Máximo de Iterações Totais): max_iters
# 4. DMAX (Máximo de Divergências Consecutivas): max_failures
# ==============================================================================

def run_continuation_process(net, load_scaling_bus_id=None, 
                             max_scale=5.0,       # DINC (Limite)
                             initial_step=0.1, 
                             min_step=0.001,      # ICMN
                             max_iters=2000,      # ICIT (Novo)
                             max_failures=10,     # DMAX (Novo)
                             enforce_q_lims=True, 
                             distributed_slack=True):
    
    # --- LOG INICIAL ---
    print(f"\n{'='*80}")
    print(f" CPF ENGINE - PADRÃO ANAREDE")
    print(f"{'='*80}")
    print(f" PARÂMETROS DE PARADA:")
    print(f"  > DINC (Max Lambda): {max_scale}")
    print(f"  > ICMN (Min Passo):  {min_step}")
    print(f"  > ICIT (Max Iter):   {max_iters}")
    print(f"  > DMAX (Max Falhas): {max_failures}")
    print(f"{'-'*80}")

    net_sim = copy.deepcopy(net)
    
    # --- 1. Identificação de Cargas e Geradores ---
    if load_scaling_bus_id is None:
        load_idx = net_sim.load.index
    else:
        load_idx = net_sim.load[net_sim.load.bus == load_scaling_bus_id].index
        if load_idx.empty: return [], []
    
    base_p_load = net_sim.load.loc[load_idx, 'p_mw'].copy()
    base_q_load = net_sim.load.loc[load_idx, 'q_mvar'].copy()

    active_gen_idx = []
    base_p_gen = []
    if distributed_slack:
        mask_gen = net_sim.gen.p_mw > 1.0
        active_gen_idx = net_sim.gen[mask_gen].index
        base_p_gen = net_sim.gen.loc[active_gen_idx, 'p_mw'].copy()

    # --- 2. Inicialização de Variáveis de Estado ---
    current_scale = 1.0      # Lambda atual (confirmado)
    current_step = initial_step 
    
    history = []   
    full_log = []  
    
    consecutive_failures = 0 # Contador para DMAX
    total_iterations = 0     # Contador para ICIT

    # --- 3. Caso Base (Iteração 0) ---
    print(f" [...] Rodando Caso Base (Scale=1.0)...")
    try:
        pp.runpp(net_sim, enforce_q_lims=enforce_q_lims)
        _save_snapshot(net_sim, 1.0, history)
        _log_attempt(full_log, 0, 1.0, "Convergente", net_sim)
    except:
        print("ERRO CRÍTICO: Caso base não converge.")
        return [], []

    # --- 4. Loop Principal ---
    while current_scale < max_scale:
        # Critério ICIT (Iterações Totais)
        if total_iterations >= max_iters:
            print(f"\n[PARADA] ICIT atingido: {total_iterations} iterações realizadas.")
            break

        # Previsão do Próximo Ponto
        next_scale = current_scale + current_step
        
        # Respeita o teto DINC
        if next_scale > max_scale: 
            next_scale = max_scale

        total_iterations += 1

        # A. Aplica Incrementos
        net_sim.load.loc[load_idx, 'p_mw'] = base_p_load * next_scale
        net_sim.load.loc[load_idx, 'q_mvar'] = base_q_load * next_scale # P e Q crescem juntos
        
        if distributed_slack and not active_gen_idx.empty:
            net_sim.gen.loc[active_gen_idx, 'p_mw'] = base_p_gen * next_scale

        # B. Tenta Resolver (Newton-Raphson)
        try:
            pp.runpp(net_sim, enforce_q_lims=enforce_q_lims)
            
            # --- SUCESSO ---
            consecutive_failures = 0 # Reseta DMAX
            
            # Salva
            _save_snapshot(net_sim, next_scale, history)
            _log_attempt(full_log, total_iterations, next_scale, "Convergente", net_sim)
            
            # Feedback Visual
            p_tot = net_sim.res_load.p_mw.sum()
            print(f"   Iter {total_iterations}: Scale {next_scale:.4f} OK | Carga: {p_tot:.1f} MW")
            
            # Consolida o passo e avança
            current_scale = next_scale
            
            # Opcional: Se o passo estava muito pequeno, pode tentar recuperar (aceleração)
            # Mas o manual diz "novo incremento menor", não fala em aumentar de volta.
            # Mantemos current_step constante até falhar.

        except pp.LoadflowNotConverged:
            # --- FALHA (Divergência) ---
            consecutive_failures += 1
            
            _log_attempt(full_log, total_iterations, next_scale, "Divergente", None)
            print(f"   Iter {total_iterations}: Falha em {next_scale:.4f}. (Falhas seguidas: {consecutive_failures})")

            # Critério DMAX (Falhas Consecutivas)
            if consecutive_failures >= max_failures:
                print(f"\n[PARADA] DMAX atingido: {consecutive_failures} divergências consecutivas.")
                break

            # Critério ICMN (Passo Mínimo)
            if current_step < min_step:
                print(f"\n[PARADA] ICMN atingido: Passo {current_step:.6f} < {min_step}")
                print(f"--> PONTO DE COLAPSO: {current_scale:.4f}")
                break
            
            # Lógica de Corte de Passo (Backtracking)
            # "O último caso convergido é restabelecido... e um novo incremento menor é aplicado"
            current_step /= 2.0
            print(f"   -> Reduzindo passo para {current_step:.5f} e tentando novamente...")
            
            # Nota: Não atualizamos 'current_scale', então o próximo loop tentará
            # current_scale (último bom) + novo current_step (menor).

    return history, full_log

def _save_snapshot(net, scale, history_list):
    snapshot = {
        'scale': scale,
        'total_load_mw': net.res_load.p_mw.sum(),
        'res_bus': net.res_bus.copy(),
        'res_line': net.res_line.copy(),
        'line_data': net.line.copy(),
        'bus_data': net.bus.copy()
    }
    history_list.append(snapshot)

def _log_attempt(log_list, iter_num, scale, status, net=None):
    row = {'iter': iter_num, 'scale': scale, 'status': status, 'mw': 0.0, 'mvar': 0.0, 'vmin': 0.0}
    if net is not None:
        row['mw'] = net.res_load.p_mw.sum()
        row['mvar'] = net.res_load.q_mvar.sum()
        row['vmin'] = net.res_bus['vm_pu'].min()
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