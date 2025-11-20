import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vsi_lib as vsi
import pandapower as pp
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

def pre_calculate_matrices(net):
    """
    Calcula as matrizes estáticas (Ybus e F-matrix) UMA ÚNICA VEZ baseada na topologia.
    Retorna um dicionário com os objetos prontos para uso rápido.
    """
    print("  -> Pré-calculando matrizes de admitância (Ybus) e participação (F)...")
    
    # 1. Garante que a rede está inicializada internamente (cria estrutura _ppc)
    pp.runpp(net)
    
    # 2. Extrai Ybus (Matriz Esparsa Complexa)
    # O pandapower armazena Ybus em net._ppc['internal']['Ybus']
    Ybus = net._ppc['internal']['Ybus']
    
    # 3. Mapeamento de índices (Pandapower ID -> Índice da Matriz 0..N)
    # O Ybus segue a ordem dos barramentos "in-service" no ppc
    bus_lookup = net._ppc['bus'][:, 0].astype(int) # IDs reais das barras na ordem da matriz
    bus_to_idx = {bus_id: i for i, bus_id in enumerate(bus_lookup)}
    
    # 4. Matriz F para L-index (Separação Carga vs Geração)
    # Consideramos Geradores: Slack + PV buses (mesmo que virem PQ depois, a topologia 'F' mantém)
    gen_buses = net.gen.bus.values.tolist() + net.ext_grid.bus.values.tolist()
    gen_buses = list(set(gen_buses)) # Remove duplicatas
    load_buses = [b for b in net.bus.index if b not in gen_buses and net.bus.at[b, 'in_service']]
    
    # Índices matriciais
    idx_gen = [bus_to_idx[b] for b in gen_buses if b in bus_to_idx]
    idx_load = [bus_to_idx[b] for b in load_buses if b in bus_to_idx]
    
    # Submatrizes
    Y_LL = Ybus[idx_load, :][:, idx_load]
    Y_LG = Ybus[idx_load, :][:, idx_gen]
    
    # Cálculo da Matriz F = -inv(Y_LL) * Y_LG
    # Usamos esparsas para velocidade
    try:
        Y_LL_inv = inv(Y_LL)
        F_matrix = -Y_LL_inv.dot(Y_LG)
    except:
        print("AVISO: Matriz Y_LL singular. L-index pode falhar.")
        F_matrix = None

    return {
        'Ybus': Ybus,
        'F_matrix': F_matrix,
        'bus_to_idx': bus_to_idx,
        'idx_gen': idx_gen,
        'idx_load': idx_load,
        'load_buses_ids': load_buses # IDs reais para mapear volta
    }

def calculate_indices_for_scenario(snapshot, static_matrices):
    """
    Calcula VSIs de Linha e Barra usando as matrizes pré-calculadas.
    """
    res_bus = snapshot['res_bus']
    res_line = snapshot['res_line']
    line_data = snapshot['line_data']
    bus_data = snapshot['bus_data']
    
    # --- 1. Preparação dos Vetores de Tensão (Complexos) ---
    # Reconstrói o vetor V complexo na ordem correta da matriz Ybus
    bus_map = static_matrices['bus_to_idx']
    num_buses = len(bus_map)
    V_complex = np.zeros(num_buses, dtype=complex)
    
    for bus_id, matrix_idx in bus_map.items():
        vm = res_bus.at[bus_id, 'vm_pu']
        va_rad = np.radians(res_bus.at[bus_id, 'va_degree'])
        V_complex[matrix_idx] = vm * np.exp(1j * va_rad)
        
    # --- 2. Cálculo em Lote dos Índices de Barra ---
    
    # A. L-index (Apenas para barras de carga)
    l_index_map = {}
    if static_matrices['F_matrix'] is not None:
        L_vals = vsi.calculate_l_index_vectorized(
            V_complex, 
            static_matrices['F_matrix'], 
            static_matrices['idx_gen'], 
            static_matrices['idx_load']
        )
        # Mapeia de volta para ID da barra
        for i, bus_id in enumerate(static_matrices['load_buses_ids']):
            l_index_map[bus_id] = L_vals[i]
            
    # B. VCPI_bus (Para todas as barras)
    vcpi_vals = vsi.calculate_vcpi_bus_vectorized(V_complex, static_matrices['Ybus'])
    # Mapeia (bus_id -> valor)
    # Invertemos o dicionário bus_to_idx para idx_to_bus
    idx_to_bus = {v: k for k, v in bus_map.items()}
    vcpi_map = {idx_to_bus[i]: val for i, val in enumerate(vcpi_vals)}

    # --- 3. Cálculo dos Índices de Linha (Iterativo) ---
    results = []
    s_base = 100.0
    
    for idx, line in line_data.iterrows():
        from_bus = line.from_bus
        to_bus = line.to_bus
        
        # Dados Elétricos e PU
        vn_kv = bus_data.at[from_bus, 'vn_kv']
        z_base = (vn_kv ** 2) / s_base
        R_pu = (line.r_ohm_per_km * line.length_km) / z_base
        X_pu = (line.x_ohm_per_km * line.length_km) / z_base
        Z_pu, theta = vsi.get_line_params(R_pu, X_pu)
        
        # Fluxos e Direção
        p_from = res_line.at[idx, 'p_from_mw']
        q_to = res_line.at[idx, 'q_to_mvar'] # Q na ponta receptora (aprox)
        
        # Normaliza
        p_from_pu = p_from / s_base
        q_to_pu = q_to / s_base
        
        # Direção
        if p_from >= 0:
            idx_s, idx_r = from_bus, to_bus
            Q_r = abs(q_to_pu)
            P_s = abs(p_from_pu)
            P_r = abs(res_line.at[idx, 'p_to_mw'] / s_base)
        else:
            idx_s, idx_r = to_bus, from_bus
            Q_r = abs(res_line.at[idx, 'q_from_mvar'] / s_base)
            P_s = abs(res_line.at[idx, 'p_to_mw'] / s_base)
            P_r = abs(p_from_pu)
            
        # Tensões
        V_s = res_bus.at[idx_s, 'vm_pu']
        V_r = res_bus.at[idx_r, 'vm_pu']
        delta = np.radians(res_bus.at[idx_s, 'va_degree'] - res_bus.at[idx_r, 'va_degree'])
        
        # Pega os índices de barra pré-calculados para a barra RECEPTORA
        # (A barra receptora é a que sofre o colapso)
        l_index_val = l_index_map.get(idx_r, 0.0) # Se for gerador, L=0
        vcpi_bus_val = vcpi_map.get(idx_r, 0.0)
        
        row = {
            'Line_ID': idx, 
            # Índices de Linha
            'FVSI': vsi.calculate_fvsi(V_s, X_pu, Q_r, Z_pu),
            'Lmn': vsi.calculate_lmn(V_s, X_pu, Q_r, theta, delta),
            'LQP': vsi.calculate_lqp(V_s, X_pu, Q_r, P_s),
            'NLSI': vsi.calculate_nlsi(V_s, P_r, R_pu, Q_r, X_pu),
            'NVSI': vsi.calculate_nvsi(V_s, X_pu, P_r, Q_r),
            'VSLI': vsi.calculate_vsli(V_s, V_r, delta),
            'VSI_2': vsi.calculate_vsi2(V_s, Q_r, R_pu, X_pu),
            # Índices de Barra (Atribuídos à linha)
            'L_index': l_index_val,
            'VCPI_bus': vcpi_bus_val
        }
        results.append(row)
        
    return pd.DataFrame(results)

def plot_comparative_indices(all_scenarios_results):
    """Plota gráficos para todos os índices encontrados no DataFrame."""
    first_key = list(all_scenarios_results.keys())[0]
    indices_cols = [c for c in all_scenarios_results[first_key].columns if c not in ['Line_ID', 'From', 'To']]
    
    scenario_keys = sorted(list(all_scenarios_results.keys()))
    cmap = plt.cm.get_cmap('RdYlGn_r')
    colors = [cmap(i) for i in np.linspace(0, 1, len(scenario_keys))]
    
    print(f"Gerando gráficos para {len(indices_cols)} índices...")
    
    for ind_name in indices_cols:
        plt.figure(figsize=(10, 6))
        limit = 1.0 # A maioria colapsa em 1.0
        
        for i, pct in enumerate(scenario_keys):
            df = all_scenarios_results[pct]
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ind_name])
            # Filtro visual para evitar picos de divisão por zero
            df_clean = df_clean[df_clean[ind_name] < 5.0]
            
            plt.scatter(df_clean['Line_ID'], df_clean[ind_name], label=f'{pct}%', color=colors[i], alpha=0.7)
            
        plt.title(f'Evolução do Índice {ind_name}')
        plt.xlabel('ID da Linha')
        plt.ylabel('Valor')
        plt.axhline(y=limit, color='r', linestyle='--', label='Limite (1.0)')
        plt.legend(title="Carga (%)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'analise_{ind_name}.png')
        plt.close()
    
    print("Gráficos salvos.")