import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vsi_lib as vsi
import pandapower as pp
from scipy.sparse.linalg import inv
import os

# --- MANTIDO: Pré-Cálculo de Matrizes (Sem alterações na lógica) ---
def pre_calculate_matrices(net):
    print("  -> Pré-calculando matrizes de admitância (Ybus) e participação (F)...")
    pp.runpp(net)
    Ybus = net._ppc['internal']['Ybus']
    bus_lookup = net._ppc['bus'][:, 0].astype(int)
    bus_to_idx = {bus_id: i for i, bus_id in enumerate(bus_lookup)}
    
    gen_buses = net.gen.bus.values.tolist() + net.ext_grid.bus.values.tolist()
    gen_buses = list(set(gen_buses)) 
    load_buses = [b for b in net.bus.index if b not in gen_buses and net.bus.at[b, 'in_service']]
    
    idx_gen = [bus_to_idx[b] for b in gen_buses if b in bus_to_idx]
    idx_load = [bus_to_idx[b] for b in load_buses if b in bus_to_idx]
    
    Y_LL = Ybus[idx_load, :][:, idx_load]
    Y_LG = Ybus[idx_load, :][:, idx_gen]
    
    try:
        Y_LL_inv = inv(Y_LL)
        F_matrix = -Y_LL_inv.dot(Y_LG)
    except:
        print("AVISO: Matriz Y_LL singular. L-index pode falhar.")
        F_matrix = None

    return {
        'Ybus': Ybus, 'F_matrix': F_matrix, 'bus_to_idx': bus_to_idx,
        'idx_gen': idx_gen, 'idx_load': idx_load, 'load_buses_ids': load_buses
    }

# --- MANTIDO: Cálculo dos Índices (Lógica mantida, apenas organização) ---
def calculate_indices_for_scenario(snapshot, static_matrices):
    res_bus = snapshot['res_bus']
    res_line = snapshot['res_line']
    line_data = snapshot['line_data']
    bus_data = snapshot['bus_data']
    
    # 1. Vetores de Tensão
    bus_map = static_matrices['bus_to_idx']
    num_buses = len(bus_map)
    V_complex = np.zeros(num_buses, dtype=complex)
    for bus_id, matrix_idx in bus_map.items():
        vm = res_bus.at[bus_id, 'vm_pu']
        va_rad = np.radians(res_bus.at[bus_id, 'va_degree'])
        V_complex[matrix_idx] = vm * np.exp(1j * va_rad)
        
    # 2. Índices de Barra (Batch)
    l_index_map = {}
    if static_matrices['F_matrix'] is not None:
        L_vals = vsi.calculate_l_index_vectorized(V_complex, static_matrices['F_matrix'], static_matrices['idx_gen'], static_matrices['idx_load'])
        for i, bus_id in enumerate(static_matrices['load_buses_ids']):
            l_index_map[bus_id] = L_vals[i]
            
    vcpi_vals = vsi.calculate_vcpi_bus_vectorized(V_complex, static_matrices['Ybus'])
    idx_to_bus = {v: k for k, v in bus_map.items()}
    vcpi_map = {idx_to_bus[i]: val for i, val in enumerate(vcpi_vals)}

    # 3. Índices de Linha e Mapeamento
    results = []
    s_base = 100.0
    
    for idx, line in line_data.iterrows():
        from_bus, to_bus = line.from_bus, line.to_bus
        vn_kv = bus_data.at[from_bus, 'vn_kv']
        z_base = (vn_kv ** 2) / s_base
        R_pu = (line.r_ohm_per_km * line.length_km) / z_base
        X_pu = (line.x_ohm_per_km * line.length_km) / z_base
        Z_pu, theta = vsi.get_line_params(R_pu, X_pu)
        
        p_from, q_to = res_line.at[idx, 'p_from_mw'], res_line.at[idx, 'q_to_mvar']
        p_from_pu, q_to_pu = p_from / s_base, q_to / s_base
        
        # Direção do Fluxo
        if p_from >= 0:
            idx_s, idx_r = from_bus, to_bus
            Q_r, P_s, P_r = abs(q_to_pu), abs(p_from_pu), abs(res_line.at[idx, 'p_to_mw'] / s_base)
        else:
            idx_s, idx_r = to_bus, from_bus
            Q_r, P_s, P_r = abs(res_line.at[idx, 'q_from_mvar'] / s_base), abs(res_line.at[idx, 'p_to_mw'] / s_base), abs(p_from_pu)
            
        V_s, V_r = res_bus.at[idx_s, 'vm_pu'], res_bus.at[idx_r, 'vm_pu']
        delta = np.radians(res_bus.at[idx_s, 'va_degree'] - res_bus.at[idx_r, 'va_degree'])
        S_r, phi = vsi.get_load_params(P_r, Q_r)

        row = {
            'Line_ID': idx, 'From': idx_s, 'To': idx_r,
            'FVSI': vsi.calculate_fvsi(V_s, X_pu, Q_r, Z_pu),
            'Lmn': vsi.calculate_lmn(V_s, X_pu, Q_r, theta, delta),
            'LQP': vsi.calculate_lqp(V_s, X_pu, Q_r, P_s),
            'Lp': vsi.calculate_lp(V_s, R_pu, P_r, theta, delta),
            'NLSI': vsi.calculate_nlsi(V_s, P_r, R_pu, Q_r, X_pu),
            'NVSI': vsi.calculate_nvsi(V_s, X_pu, P_r, Q_r),
            'VQI': vsi.calculate_vqi(V_s, Q_r, X_pu, R_pu),
            'PTSI': vsi.calculate_ptsi(V_s, P_r, Q_r, Z_pu, theta, phi),
            'VSI_2': vsi.calculate_vsi2(V_s, Q_r, R_pu, X_pu),
            'L_simple': vsi.calculate_l_simple(V_s, V_r),
            'VSLI': vsi.calculate_vsli(V_s, V_r, delta),
            'VCPI_P': vsi.calculate_vcpi_power(V_s, Z_pu, theta, phi, P_r, Q_r, 'P'),
            'VCPI_Q': vsi.calculate_vcpi_power(V_s, Z_pu, theta, phi, P_r, Q_r, 'Q'),
            'Lsr': vsi.calculate_vcpi_power(V_s, Z_pu, theta, phi, P_r, Q_r, 'S'),
            'SI': vsi.calculate_si(V_s, V_r, P_r, Q_r, R_pu, X_pu, Z_pu),
            'VCPI_1': vsi.calculate_vcpi_1_voltage(V_s, V_r, delta),
            'VSMI': vsi.calculate_vsmi(delta, theta, phi),
            'VSLBI': vsi.calculate_vslbi(V_s, V_r, delta),
            'VSI_1': vsi.calculate_vsi1(V_s, P_r, Q_r, X_pu),
            # Mapeamento correto para a barra receptora
            'L_index': l_index_map.get(idx_r, 0.0),
            'VCPI_bus': vcpi_map.get(idx_r, 0.0)
        }
        results.append(row)
    return pd.DataFrame(results)

# --- NOVO: Plotagem Profissional e Corrigida ---

def plot_comparative_indices(all_scenarios_results):
    """Gera gráficos melhorados, separando lógica de Linha e Barra."""
    first_key = list(all_scenarios_results.keys())[0]
    # Identifica colunas de índices
    all_cols = all_scenarios_results[first_key].columns
    indices_cols = [c for c in all_cols if c not in ['Line_ID', 'From', 'To']]
    
    # Define quais são índices de BARRA para tratamento especial
    bus_indices_names = ['L_index', 'VCPI_bus'] 
    
    scenario_keys = sorted(list(all_scenarios_results.keys()))
    # Paleta de cores profissional
    cmap = plt.cm.get_cmap('turbo') # Turbo é excelente para gradientes visuais
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(scenario_keys))]
    
    print(f"Gerando gráficos para {len(indices_cols)} índices...")
    
    for ind_name in indices_cols:
        plt.figure(figsize=(10, 6))
        is_bus_index = ind_name in bus_indices_names
        
        # Configuração de Limites
        limit = 1.0
        limit_label = "Instável > 1.0"
        if ind_name in ['SI', 'VCPI_1', 'VSMI', 'VSI_1']:
            limit = 0.0
            limit_label = "Instável < 0.0"
        
        max_val_scenario = 0.0
        
        for i, pct in enumerate(scenario_keys):
            df = all_scenarios_results[pct]
            
            # Limpeza básica
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ind_name])
            
            # LÓGICA DE FILTRAGEM
            if is_bus_index:
                # Se é índice de Barra, o CSV tem linhas repetidas (várias linhas chegam na mesma barra).
                # Precisamos pegar valores únicos por barra RECEPTORA ('To')
                df_plot = df_clean[['To', ind_name]].drop_duplicates(subset=['To'])
                
                # FILTRO DE ZEROS: Remove barras onde o índice é 0 (geradores ou sem carga relevante)
                # Isso limpa o gráfico de barras irrelevantes.
                df_plot = df_plot[df_plot[ind_name] > 0.001]
                
                x_data = df_plot['To'] # Eixo X = ID da Barra
                y_data = df_plot[ind_name]
                marker = 's' # Quadrado para barras
                label_prefix = "Barras"
            else:
                # Índice de Linha normal
                df_plot = df_clean[df_clean[ind_name] < 10.0] # Remove outliers visuais
                x_data = df_plot['Line_ID']
                y_data = df_plot[ind_name]
                marker = 'o' # Círculo para linhas
                label_prefix = "Linhas"

            # Captura máximo para legenda
            if not y_data.empty:
                current_max = y_data.max()
                max_val_scenario = max(max_val_scenario, current_max)
                
                # Plota
                plt.scatter(x_data, y_data, label=f'{pct}% (Max: {current_max:.3f})', 
                            color=colors[i], alpha=0.75, marker=marker, s=40, edgecolors='w', linewidth=0.5)

        # Cosmética do Gráfico
        title_type = "Índice de BARRA" if is_bus_index else "Índice de LINHA"
        plt.title(f'{ind_name} - {title_type}\n(Máximo Global: {max_val_scenario:.3f})', fontsize=12)
        plt.xlabel('ID da Barra' if is_bus_index else 'ID da Linha', fontsize=10)
        plt.ylabel(f'Valor do Índice', fontsize=10)
        
        # Linha de Limite
        plt.axhline(y=limit, color='red', linestyle='--', linewidth=1.5, label=f'Limite Crítico ({limit})')
        
        # Legenda Externa Inteligente
        plt.legend(title=f"Cenário Carga", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Salva em minúsculo
        filename = f'analise_{ind_name.lower()}.png'
        plt.savefig(filename, dpi=150)
        plt.close()
    
    print("Gráficos de índices salvos.")

def plot_pv_curves(history, title="Curvas PV"):
    """
    Plota Curvas PV destacando a barra crítica e limpando a legenda.
    """
    p_total = [snap['total_load_mw'] for snap in history]
    # Extrai tensões
    vm_data = [snap['res_bus']['vm_pu'].values for snap in history]
    df_vm = pd.DataFrame(vm_data, index=p_total)
    
    # Identifica a Barra Crítica (Menor tensão no último passo)
    last_step_voltages = df_vm.iloc[-1]
    critical_bus_idx = last_step_voltages.idxmin()
    critical_val = last_step_voltages.min()
    max_load = p_total[-1]
    
    plt.figure(figsize=(10, 7))
    
    # Plota todas as barras em cinza claro (background)
    for col in df_vm.columns:
        if col != critical_bus_idx:
            plt.plot(df_vm.index, df_vm[col], color='grey', linewidth=0.8, alpha=0.3)
    
    # Plota a Barra Crítica com destaque
    plt.plot(df_vm.index, df_vm[critical_bus_idx], color='red', linewidth=2.5, 
             label=f'Barra Crítica: {critical_bus_idx}\n(Vmin: {critical_val:.3f} pu)')
    
    # Plota ponto de colapso
    plt.plot(max_load, critical_val, 'X', color='black', markersize=10, label=f'Colapso: {max_load:.1f} MW')
    
    # Dummy plot para legenda das outras barras
    plt.plot([], [], color='grey', linewidth=1, alpha=0.5, label='Outras Barras')
    
    plt.title(f'{title}', fontsize=14, fontweight='bold')
    plt.xlabel('Potência Ativa Total (MW)', fontsize=12)
    plt.ylabel('Tensão (pu)', fontsize=12)
    plt.axvline(x=max_load, color='black', linestyle='--', alpha=0.5)
    
    plt.legend(loc='lower left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plt.savefig("curva_pv_sistema.png", dpi=150)
    plt.close()
    print(f"Gráfico da Curva PV salvo (Barra Crítica: {critical_bus_idx}).")