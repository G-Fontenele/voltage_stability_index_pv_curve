import pandapower as pp
import numpy as np
import copy

def run_continuation_process(net, load_scaling_bus_id=None, max_scale=5.0, 
                             initial_step=0.1, min_step=0.001, 
                             enforce_q_lims=True, distributed_slack=True):
    """
    Roda um Fluxo de Potência Continuado (CPF) genérico com passo adaptativo.
    
    Parâmetros:
        net: Rede pandapower.
        load_scaling_bus_id: ID do barramento para aumentar carga (None = Sistema Todo).
        max_scale: Fator máximo de multiplicação da carga.
        initial_step: Tamanho do passo inicial.
        min_step: Precisão mínima (critério de parada do nariz da curva).
        enforce_q_lims: (Bool) Se True, respeita limites de Mvar dos geradores (Curva PV cai mais cedo).
                        Se False, assume Q infinito (Curva PV vai até o limite de transmissão).
        distributed_slack: (Bool) Se True, aumenta a geração de todos os geradores ativos proporcionalmente.
                           Se False, apenas a barra Slack (Ref) assume o aumento de carga.
    """
    net_sim = copy.deepcopy(net)
    
    # --- 1. Configuração de Carga ---
    if load_scaling_bus_id is None:
        print(f"Modo Carga: Global (Todo o sistema)")
        load_idx = net_sim.load.index
    else:
        print(f"Modo Carga: Local (Barra {load_scaling_bus_id})")
        load_idx = net_sim.load[net_sim.load.bus == load_scaling_bus_id].index
        if load_idx.empty:
            print(f"AVISO CRÍTICO: Não há carga conectada na barra {load_scaling_bus_id} para escalonar!")
            return []
    
    # Armazena valores base para aplicar o fator de escala
    base_p_load = net_sim.load.loc[load_idx, 'p_mw'].copy()
    base_q_load = net_sim.load.loc[load_idx, 'q_mvar'].copy()

    # --- 2. Configuração de Geração (Distributed Slack Genérico) ---
    # Identifica automaticamente quem são os geradores de potência ativa (P > 0)
    # para não forçar compensadores síncronos (P=0) a gerar MW.
    active_gen_idx = []
    base_p_gen = []
    
    if distributed_slack:
        # Filtra geradores com P nominal > 1 MW (ignora compensadores e microrredes desligadas)
        mask_gen = net_sim.gen.p_mw > 1.0
        active_gen_idx = net_sim.gen[mask_gen].index
        base_p_gen = net_sim.gen.loc[active_gen_idx, 'p_mw'].copy()
        
        print(f"Despacho Distribuído: ATIVADO")
        print(f" -> Geradores participando: {len(active_gen_idx)} (Barras: {list(net_sim.gen.loc[active_gen_idx, 'bus'].values)})")
    else:
        print(f"Despacho Distribuído: DESATIVADO (Apenas Slack assume)")

    # --- 3. Loop de Simulação ---
    current_scale = 1.0
    current_step = initial_step
    history = []
    
    print(f"Configuração: QLIM={enforce_q_lims} | Passo Ini={initial_step} | Min={min_step}")

    # Roda caso base (1.0)
    try:
        pp.runpp(net_sim, enforce_q_lims=enforce_q_lims)
        _save_snapshot(net_sim, 1.0, history)
    except pp.LoadflowNotConverged:
        # Tenta fallback sem Qlims se o caso base for muito pesado
        if enforce_q_lims:
            print("Aviso: Caso base não convergiu com QLIM. Tentando sem limites...")
            try:
                pp.runpp(net_sim, enforce_q_lims=False)
                _save_snapshot(net_sim, 1.0, history)
            except:
                print("Erro: Caso base não converge.")
                return []
        else:
            return []

    while current_scale < max_scale:
        next_scale = current_scale + current_step
        if next_scale > max_scale: next_scale = max_scale

        # A. Aplica Escala na Carga
        net_sim.load.loc[load_idx, 'p_mw'] = base_p_load * next_scale
        net_sim.load.loc[load_idx, 'q_mvar'] = base_q_load * next_scale
        
        # B. Aplica Escala na Geração (Se ativado)
        if distributed_slack and not active_gen_idx.empty:
            net_sim.gen.loc[active_gen_idx, 'p_mw'] = base_p_gen * next_scale

        try:
            # Tenta resolver o fluxo de potência
            pp.runpp(net_sim, enforce_q_lims=enforce_q_lims)
            
            # Sucesso: Salva e avança
            current_scale = next_scale
            total_load = net_sim.res_load.p_mw.sum()
            print(f"   -> Convergiu: Scale={current_scale:.4f} | Carga Total={total_load:.2f} MW")
            
            _save_snapshot(net_sim, current_scale, history)

        except pp.LoadflowNotConverged:
            # Falha: Refinamento de Passo (Backtracking)
            if current_step < min_step:
                print(f"--> Ponto de Colapso encontrado em Scale={current_scale:.5f}")
                break
            
            # Reduz passo pela metade e tenta de novo (sem avançar current_scale)
            print(f"   ! Divergiu em {next_scale:.4f}. Refinando passo: {current_step} -> {current_step/2:.5f}")
            current_step /= 2.0

    return history

def _save_snapshot(net, scale, history_list):
    # Salva apenas os dados essenciais para economizar memória
    snapshot = {
        'scale': scale,
        'total_load_mw': net.res_load.p_mw.sum(),
        'res_bus': net.res_bus.copy(),
        'res_line': net.res_line.copy(),
        'line_data': net.line.copy(), # Necessário para dados de R, X
        'bus_data': net.bus.copy()    # Necessário para dados de VN_kv
    }
    history_list.append(snapshot)

def extract_scenarios(history, percentages=[0, 25, 50, 75, 95, 99, 100]):
    """Identifica os snapshots mais próximos das percentagens da margem de carregamento."""
    if not history: return {}
    
    base_load = history[0]['total_load_mw']
    max_load = history[-1]['total_load_mw']
    load_margin = max_load - base_load
    
    scenarios = {}
    print(f"\n--- Definição dos Cenários ---")
    print(f"Base: {base_load:.2f} MW | Máximo: {max_load:.2f} MW | Margem: {load_margin:.2f} MW")
    
    for pct in percentages:
        target = base_load + (pct/100.0)*load_margin
        # Busca o snapshot com menor erro absoluto em relação ao alvo
        best_snap = min(history, key=lambda x: abs(x['total_load_mw'] - target))
        scenarios[pct] = best_snap
        print(f"Cenário {pct:3d}%: Alvo {target:.2f} MW -> Obtido {best_snap['total_load_mw']:.2f} MW")
        
    return scenarios