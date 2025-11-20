import pandapower as pp
import numpy as np
import copy

# ==============================================================================
# MOTOR DE SIMULAÇÃO (CONTINUOUS POWER FLOW)
# ==============================================================================
# Este módulo executa o fluxo de potência continuado utilizando o método
# Preditor (Incremento de Carga) - Corretor (Newton-Raphson padrão).
# Implementa a lógica adaptativa do ANAREDE para refinar o passo perto do colapso.
# ==============================================================================

def run_continuation_process(net, load_scaling_bus_id=None, max_scale=5.0, 
                             initial_step=0.1, min_step=0.001, 
                             enforce_q_lims=True, distributed_slack=True):
    """
    Executa a varredura da curva PV até a divergência (colapso).
    
    Retorna:
        history: Lista de snapshots convergentes (para gráficos).
        full_log: Lista de todas as tentativas (para relatório de convergência).
    """
    # --- PRINT INICIAL EXPLICATIVO ---
    print(f"\n{'='*80}")
    print(f" MOTOR DE SIMULAÇÃO: FLUXO DE POTÊNCIA CONTINUADO (CPF)")
    print(f"{'='*80}")
    print(f" OBJETIVO:    Incrementar carga até o ponto de colapso (Nariz da Curva PV).")
    print(f" MÉTODO:      Passo Adaptativo (Reduz o passo em caso de divergência).")
    print(f"{'-'*80}")
    print(f" PARÂMETROS:")
    print(f"  > Escopo Carga:     {'Global (Todo Sistema)' if load_scaling_bus_id is None else f'Local (Barra {load_scaling_bus_id})'}")
    print(f"  > Limites Q (QLIM): {'ATIVADO (Realista)' if enforce_q_lims else 'DESATIVADO (TCC/Teórico)'}")
    print(f"  > Despacho (BPSI):  {'Distribuído (Geradores ajudam)' if distributed_slack else 'Concentrado (Apenas Slack)'}")
    print(f"  > Passo Inicial:    {initial_step} pu")
    print(f"  > Passo Mínimo:     {min_step} pu")
    print(f"{'='*80}\n")

    # Cópia de segurança
    net_sim = copy.deepcopy(net)
    
    # --- 1. Identificação de Cargas ---
    if load_scaling_bus_id is None:
        load_idx = net_sim.load.index
    else:
        load_idx = net_sim.load[net_sim.load.bus == load_scaling_bus_id].index
        if load_idx.empty: return [], []
    
    base_p_load = net_sim.load.loc[load_idx, 'p_mw'].copy()
    base_q_load = net_sim.load.loc[load_idx, 'q_mvar'].copy()

    # --- 2. Identificação de Geradores (Despacho Distribuído) ---
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
    
    iteration_counter = 0 

    # --- 4. Rodar Caso Base (Lambda = 1.0) ---
    iteration_counter += 1
    print(f" [...] Rodando Caso Base (Scale=1.000)...")
    try:
        pp.runpp(net_sim, enforce_q_lims=enforce_q_lims)
        
        total_load = net_sim.res_load.p_mw.sum()
        print(f"   -> Convergiu: Scale=1.0000 | Carga Total={total_load:.2f} MW")
        
        _save_snapshot(net_sim, 1.0, history)
        _log_attempt(full_log, iteration_counter, 1.0, "Convergente", net_sim)
        
    except:
        _log_attempt(full_log, iteration_counter, 1.0, "Divergente (Base)", None)
        print("ERRO CRÍTICO: Caso base não converge. Verifique os dados de entrada.")
        return [], full_log

    # --- 5. Loop de Continuação ---
    while current_scale < max_scale:
        next_scale = current_scale + current_step
        if next_scale > max_scale: next_scale = max_scale
        
        iteration_counter += 1

        # A. Aplica Escala
        net_sim.load.loc[load_idx, 'p_mw'] = base_p_load * next_scale
        net_sim.load.loc[load_idx, 'q_mvar'] = base_q_load * next_scale
        if distributed_slack and not active_gen_idx.empty:
            net_sim.gen.loc[active_gen_idx, 'p_mw'] = base_p_gen * next_scale

        try:
            # B. Tenta resolver
            pp.runpp(net_sim, enforce_q_lims=enforce_q_lims)
            
            # SUCESSO
            total_load = net_sim.res_load.p_mw.sum()
            # -- PRINT RESTAURADO (Formato que você gostava) --
            print(f"   -> Convergiu: Scale={next_scale:.4f} | Carga Total={total_load:.2f} MW")
            
            _save_snapshot(net_sim, next_scale, history)
            _log_attempt(full_log, iteration_counter, next_scale, "Convergente", net_sim)
            
            current_scale = next_scale

        except pp.LoadflowNotConverged:
            # FALHA
            _log_attempt(full_log, iteration_counter, next_scale, "Divergente", None)
            
            # -- PRINT RESTAURADO (Mais informativo sobre o erro) --
            print(f"   ! DIVERGÊNCIA em Scale={next_scale:.4f}. Refinando passo: {current_step:.5f} -> {current_step/2.0:.5f}")

            # Critério de Parada
            if current_step < min_step:
                print(f"\n{'!'*60}")
                print(f" PONTO DE COLAPSO ENCONTRADO!")
                print(f" Limite de precisão ({min_step}) atingido em Scale={current_scale:.5f}")
                print(f"{'!'*60}\n")
                break
            
            # Backtracking
            current_step /= 2.0

    return history, full_log

def _save_snapshot(net, scale, history_list):
    """Salva dados da rede."""
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
    """Registra log para relatório."""
    row = {
        'iter': iter_num,
        'scale': scale,
        'status': status,
        'mw': 0.0, 'mvar': 0.0, 'vmin': 0.0
    }
    if net is not None:
        row['mw'] = net.res_bus['p_mw'].clip(upper=0).abs().sum()
        row['mvar'] = net.res_bus['q_mvar'].clip(upper=0).abs().sum()
        row['vmin'] = net.res_bus['vm_pu'].min()
    log_list.append(row)

def extract_scenarios(history, percentages):
    """Extrai snapshots."""
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