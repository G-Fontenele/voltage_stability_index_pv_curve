import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vsi_lib as vsi
import pandapower as pp
from scipy.sparse.linalg import inv
import os
from datetime import datetime

# ==============================================================================
# FERRAMENTAS DE ANÁLISE E RELATÓRIOS
# ==============================================================================

# --- 1. PRÉ-CÁLCULO DE MATRIZES (OTIMIZAÇÃO) ---
def pre_calculate_matrices(net):
    """
    Calcula as matrizes estáticas (Ybus e F-matrix) UMA ÚNICA VEZ.
    Como a topologia da rede não muda durante o CPF (apenas cargas mudam),
    podemos reutilizar essas matrizes para acelerar o cálculo dos índices.
    """
    print("  -> Pré-calculando matrizes de admitância (Ybus) e participação (F)...")
    
    # Inicializa estruturas internas do pandapower
    pp.runpp(net)
    
    # Extrai Ybus (Matriz Esparsa Complexa)
    Ybus = net._ppc['internal']['Ybus']
    
    # Mapeamento de índices internos para índices do DataFrame
    bus_lookup = net._ppc['bus'][:, 0].astype(int)
    bus_to_idx = {bus_id: i for i, bus_id in enumerate(bus_lookup)}
    
    # Separação de Barras para Matriz F (L-Index)
    gen_buses = net.gen.bus.values.tolist() + net.ext_grid.bus.values.tolist()
    gen_buses = list(set(gen_buses)) 
    load_buses = [b for b in net.bus.index if b not in gen_buses and net.bus.at[b, 'in_service']]
    
    idx_gen = [bus_to_idx[b] for b in gen_buses if b in bus_to_idx]
    idx_load = [bus_to_idx[b] for b in load_buses if b in bus_to_idx]
    
    # Particionamento da Ybus
    Y_LL = Ybus[idx_load, :][:, idx_load] # Carga-Carga
    Y_LG = Ybus[idx_load, :][:, idx_gen]  # Carga-Geração
    
    try:
        # Cálculo da Matriz F = -inv(Y_LL) * Y_LG
        Y_LL_inv = inv(Y_LL)
        F_matrix = -Y_LL_inv.dot(Y_LG)
    except:
        print("AVISO: Matriz Y_LL singular. L-index pode falhar.")
        F_matrix = None

    return {
        'Ybus': Ybus, 'F_matrix': F_matrix, 'bus_to_idx': bus_to_idx,
        'idx_gen': idx_gen, 'idx_load': idx_load, 'load_buses_ids': load_buses
    }

# --- 2. CÁLCULO DOS ÍNDICES PARA UM CENÁRIO ---
def calculate_indices_for_scenario(snapshot, static_matrices):
    """
    Calcula todos os índices (Linha e Barra) para um ponto específico da curva PV.
    """
    res_bus = snapshot['res_bus']
    res_line = snapshot['res_line']
    line_data = snapshot['line_data']
    bus_data = snapshot['bus_data']
    
    # A. Monta vetor de tensões complexas (V = |V| < theta)
    bus_map = static_matrices['bus_to_idx']
    num_buses = len(bus_map)
    V_complex = np.zeros(num_buses, dtype=complex)
    for bus_id, matrix_idx in bus_map.items():
        vm = res_bus.at[bus_id, 'vm_pu']
        va_rad = np.radians(res_bus.at[bus_id, 'va_degree'])
        V_complex[matrix_idx] = vm * np.exp(1j * va_rad)
        
    # B. Calcula Índices de Barra (L-Index, VCPI) em lote (Vectorized)
    l_index_map = {}
    if static_matrices['F_matrix'] is not None:
        L_vals = vsi.calculate_l_index_vectorized(V_complex, static_matrices['F_matrix'], static_matrices['idx_gen'], static_matrices['idx_load'])
        for i, bus_id in enumerate(static_matrices['load_buses_ids']):
            l_index_map[bus_id] = L_vals[i]
            
    vcpi_vals = vsi.calculate_vcpi_bus_vectorized(V_complex, static_matrices['Ybus'])
    idx_to_bus = {v: k for k, v in bus_map.items()}
    vcpi_map = {idx_to_bus[i]: val for i, val in enumerate(vcpi_vals)}

    # C. Calcula Índices de Linha (Iterando sobre cada linha)
    results = []
    s_base = 100.0 # Base MVA padrão do pandapower
    
    for idx, line in line_data.iterrows():
        from_bus, to_bus = line.from_bus, line.to_bus
        
        # Parâmetros de Linha em PU
        vn_kv = bus_data.at[from_bus, 'vn_kv']
        z_base = (vn_kv ** 2) / s_base
        R_pu = (line.r_ohm_per_km * line.length_km) / z_base
        X_pu = (line.x_ohm_per_km * line.length_km) / z_base
        Z_pu, theta = vsi.get_line_params(R_pu, X_pu)
        
        # Fluxos de Potência
        p_from, q_to = res_line.at[idx, 'p_from_mw'], res_line.at[idx, 'q_to_mvar']
        p_from_pu, q_to_pu = p_from / s_base, q_to / s_base
        
        # Determina direção do fluxo para saber quem é Vs (Sending) e Vr (Receiving)
        if p_from >= 0:
            idx_s, idx_r = from_bus, to_bus
            Q_r, P_s, P_r = abs(q_to_pu), abs(p_from_pu), abs(res_line.at[idx, 'p_to_mw'] / s_base)
        else:
            idx_s, idx_r = to_bus, from_bus
            Q_r, P_s, P_r = abs(res_line.at[idx, 'q_from_mvar'] / s_base), abs(res_line.at[idx, 'p_to_mw'] / s_base), abs(p_from_pu)
            
        # Dados elétricos das barras
        V_s, V_r = res_bus.at[idx_s, 'vm_pu'], res_bus.at[idx_r, 'vm_pu']
        delta = np.radians(res_bus.at[idx_s, 'va_degree'] - res_bus.at[idx_r, 'va_degree'])
        S_r, phi = vsi.get_load_params(P_r, Q_r)

        # D. Chama a biblioteca VSI para cada índice
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
            # Atribui os índices de barra calculados no passo B à linha correspondente
            'L_index': l_index_map.get(idx_r, 0.0),
            'VCPI_bus': vcpi_map.get(idx_r, 0.0)
        }
        results.append(row)
    return pd.DataFrame(results)

# --- 3. PLOTAGEM ---
def plot_comparative_indices(all_scenarios_results):
    """Gera gráficos comparativos para cada índice, salvando em minúsculo."""
    first_key = list(all_scenarios_results.keys())[0]
    all_cols = all_scenarios_results[first_key].columns
    indices_cols = [c for c in all_cols if c not in ['Line_ID', 'From', 'To']]
    
    bus_indices_names = ['L_index', 'VCPI_bus'] 
    
    scenario_keys = sorted(list(all_scenarios_results.keys()))
    cmap = plt.cm.get_cmap('turbo')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(scenario_keys))]
    
    print(f"Gerando gráficos para {len(indices_cols)} índices...")
    
    for ind_name in indices_cols:
        plt.figure(figsize=(10, 6))
        is_bus_index = ind_name in bus_indices_names
        
        # Define limites visuais (1.0 ou 0.0 dependendo do índice)
        limit = 1.0
        if ind_name in ['SI', 'VCPI_1', 'VSMI', 'VSI_1']: limit = 0.0
        
        max_val_scenario = 0.0
        
        for i, pct in enumerate(scenario_keys):
            df = all_scenarios_results[pct]
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ind_name])
            
            if is_bus_index:
                # Filtro para gráficos de Barra (remove repetidos e zeros)
                df_plot = df_clean[['To', ind_name]].drop_duplicates(subset=['To'])
                df_plot = df_plot[df_plot[ind_name] > 0.001]
                x_data, y_data, marker = df_plot['To'], df_plot[ind_name], 's'
            else:
                # Filtro para gráficos de Linha (remove outliers visuais > 10)
                df_plot = df_clean[df_clean[ind_name] < 10.0]
                x_data, y_data, marker = df_plot['Line_ID'], df_plot[ind_name], 'o'

            if not y_data.empty:
                current_max = y_data.max()
                max_val_scenario = max(max_val_scenario, current_max)
                plt.scatter(x_data, y_data, label=f'{pct}% (Max: {current_max:.3f})', 
                            color=colors[i], alpha=0.75, marker=marker, s=40)

        title_type = "BARRA" if is_bus_index else "LINHA"
        plt.title(f'{ind_name} - {title_type}\n(Máximo Global: {max_val_scenario:.3f})', fontsize=12)
        plt.xlabel('ID', fontsize=10)
        plt.ylabel('Valor', fontsize=10)
        plt.axhline(y=limit, color='red', linestyle='--', label=f'Limite ({limit})')
        plt.legend(title="Carga (%)", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'analise_{ind_name.lower()}.png' # Nome minúsculo
        plt.savefig(filename, dpi=150)
        plt.close()
    
    print("Gráficos de índices salvos.")

def plot_pv_curves(history, title="Curvas PV"):
    """Plota a Curva PV com destaque para a barra crítica."""
    p_total = [snap['total_load_mw'] for snap in history]
    vm_data = [snap['res_bus']['vm_pu'].values for snap in history]
    df_vm = pd.DataFrame(vm_data, index=p_total)
    
    last_step_voltages = df_vm.iloc[-1]
    critical_bus_idx = last_step_voltages.idxmin()
    critical_val = last_step_voltages.min()
    max_load = p_total[-1]
    
    plt.figure(figsize=(10, 7))
    # Background
    for col in df_vm.columns:
        if col != critical_bus_idx:
            plt.plot(df_vm.index, df_vm[col], color='grey', linewidth=0.8, alpha=0.3)
    # Crítica
    plt.plot(df_vm.index, df_vm[critical_bus_idx], color='red', linewidth=2.5, 
             label=f'Crítica: {critical_bus_idx}\n(Vmin: {critical_val:.3f} pu)')
    # Colapso
    plt.plot(max_load, critical_val, 'X', color='black', markersize=10, label=f'Colapso: {max_load:.1f} MW')
    # Legenda
    plt.plot([], [], color='grey', linewidth=1, alpha=0.5, label='Outras Barras')
    
    plt.title(f'{title}', fontsize=14, fontweight='bold')
    plt.xlabel('Potência Ativa Total (MW)', fontsize=12)
    plt.ylabel('Tensão (pu)', fontsize=12)
    plt.axvline(x=max_load, color='black', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=10, frameon=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plt.savefig("curva_pv_sistema.png", dpi=150)
    plt.close()
    print(f"Gráfico da Curva PV salvo (Barra Crítica: {critical_bus_idx}).")

# --- 4. RELATÓRIOS TXT ---

def generate_anarede_report(history, system_name, filename="relatorio_colapso.txt"):
    """Gera relatório detalhado do último snapshot (Colapso)."""
    snap = history[-1]
    res_bus, res_line = snap['res_bus'], snap['res_line']
    max_load, scale = snap['total_load_mw'], snap['scale']
    
    header = f"""
{'='*80}
RELATORIO DE ANALISE DE ESTABILIDADE DE TENSAO (SIMULADOR PYTHON)
SISTEMA: {system_name.upper()}
DATA: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
{'='*80}

RESUMO DO PONTO DE COLAPSO
--------------------------
Fator de Escala Final (Lambda): {scale:.4f}
Carregamento Total do Sistema : {max_load:.2f} MW
Tensao Minima do Sistema      : {res_bus['vm_pu'].min():.4f} pu (Barra {res_bus['vm_pu'].idxmin()})

ESTADO DAS BARRAS
{'='*80}
BARRA   | V (pu)  | ANG (deg) | P_INJ (MW) | Q_INJ (Mvar) | TIPO
{'-'*80}
"""
    content = header
    sorted_buses = res_bus.sort_values(by='vm_pu')
    for bus_id, row in sorted_buses.iterrows():
        b_type = "PQ"
        if abs(row['p_mw']) > 0 and row['vm_pu'] > 0.99: b_type = "PV/REF"
        content += f"{bus_id:<7} | {row['vm_pu']:<7.4f} | {row['va_degree']:<9.2f} | {row['p_mw']:<10.2f} | {row['q_mvar']:<12.2f} | {b_type}\n"
        
    content += f"\n{'='*80}\nFLUXO NAS LINHAS\n{'='*80}\n"
    content += f"LINHA   | DE      | PARA    | P_DE (MW) | Q_DE (Mvar) | CARREG (%)\n{'-'*80}\n"
    
    col_load = 'loading_percent' if 'loading_percent' in res_line.columns else 'p_from_mw'
    sorted_lines = res_line.sort_values(by=col_load, ascending=False)

    for line_id, row in sorted_lines.iterrows():
        from_bus = snap['line_data'].at[line_id, 'from_bus']
        to_bus = snap['line_data'].at[line_id, 'to_bus']
        load_val = row[col_load] if col_load == 'loading_percent' else 0.0
        content += f"{line_id:<7} | {from_bus:<7} | {to_bus:<7} | {row['p_from_mw']:<9.2f} | {row['q_from_mvar']:<11.2f} | {load_val:.1f}\n"
    
    content += f"\n{'='*80}\nFIM DO RELATORIO\n{'='*80}\n"
    with open(filename, "w") as f: f.write(content)
    print(f"  -> Relatório de Colapso salvo: {filename}")

def generate_convergence_report(full_log, system_name, filename="relatorio_convergencia.txt"):
    """Gera relatório passo-a-passo (Convergência) igual ao do ANAREDE."""
    header = f"""
{'='*100}
RELATORIO DE EXECUCAO DO FLUXO DE POTENCIA CONTINUADO (CONVERGENCIA)
SISTEMA: {system_name.upper()}
DATA: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
{'='*100}

NUM.    STATUS          PASSO (LAMBDA)    CARGA TOTAL (MW)    CARGA TOTAL (Mvar)    TENSAO MIN (PU)
{'-'*100}
"""
    content = header
    
    for row in full_log:
        iter_num = row['iter']
        status = row['status']
        scale = row['scale']
        
        if row['mw'] > 0:
            mw_str = f"{row['mw']:.2f}"
            mvar_str = f"{row['mvar']:.2f}"
            vmin_str = f"{row['vmin']:.4f}"
        else:
            mw_str = "---"
            mvar_str = "---"
            vmin_str = "---"
            
        line_str = f"{iter_num:<6}  {status:<14}  {scale:<16.4f}  {mw_str:<18}  {mvar_str:<18}    {vmin_str}\n"
        content += line_str
    
    content += f"{'='*100}\n"
    with open(filename, "w") as f: f.write(content)
    print(f"  -> Relatório de Convergência salvo: {filename}")