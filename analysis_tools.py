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

# --- 1. PRÉ-CÁLCULO DE MATRIZES ---
def pre_calculate_matrices(net):
    """Calcula matrizes estáticas (Ybus e F) para acelerar índices."""
    print("  -> Pré-calculando matrizes de admitância (Ybus) e participação (F)...")
    
    try:
        pp.runpp(net)
    except:
        print("Aviso: runpp falhou no pré-cálculo. Matrizes podem estar incompletas.")
        
    # Ybus Interna
    try:
        Ybus = net._ppc['internal']['Ybus']
        bus_lookup = net._pd2ppc_lookups['bus']
    except:
        return {'Ybus': None, 'F_matrix': None, 'bus_to_idx': {}, 'idx_gen': [], 'idx_load': [], 'load_buses_ids': []}
    
    bus_to_idx = {}
    
    # Mapeia apenas barras que o pandapower considerou no cálculo
    for ext_id in net.bus.index:
        if net.bus.at[ext_id, 'in_service'] and ext_id in bus_lookup:
            int_idx = bus_lookup[ext_id]
            bus_to_idx[ext_id] = int(int_idx)
            
    gen_buses_ext = list(set(net.gen.bus.values.tolist() + net.ext_grid.bus.values.tolist()))
    all_load_buses = [b for b in net.bus.index if b not in gen_buses_ext and net.bus.at[b, 'in_service']]
    
    valid_load_buses = [b for b in all_load_buses if b in bus_to_idx]
    valid_gen_buses = [b for b in gen_buses_ext if b in bus_to_idx]
    
    idx_gen_int = [bus_to_idx[b] for b in valid_gen_buses]
    idx_load_int = [bus_to_idx[b] for b in valid_load_buses]
    
    if idx_load_int:
        Y_LL = Ybus[idx_load_int, :][:, idx_load_int]
        Y_LG = Ybus[idx_load_int, :][:, idx_gen_int]
        try:
            F_matrix = -inv(Y_LL).dot(Y_LG)
        except:
            print("AVISO: Matriz Y_LL singular. L-index pode falhar.")
            F_matrix = None
    else:
        F_matrix = None

    return {
        'Ybus': Ybus, 'F_matrix': F_matrix, 'bus_to_idx': bus_to_idx, 
        'idx_gen': idx_gen_int, 'idx_load': idx_load_int, 
        'load_buses_ids': valid_load_buses
    }

# --- 2. CÁLCULO DOS ÍNDICES ---
def calculate_indices_for_scenario(snapshot, static_matrices):
    res_bus = snapshot['res_bus']
    res_line = snapshot['res_line']
    line_data = snapshot['line_data']
    
    bus_map = static_matrices['bus_to_idx']
    ybus = static_matrices['Ybus']
    
    if not bus_map or ybus is None: return pd.DataFrame()
    
    V_complex = np.zeros(ybus.shape[0], dtype=complex)
    
    for bus_id_ext, matrix_idx in bus_map.items():
        if bus_id_ext in res_bus.index:
            vm = res_bus.at[bus_id_ext, 'vm_pu']
            va_rad = np.radians(res_bus.at[bus_id_ext, 'va_degree'])
            if matrix_idx < len(V_complex):
                V_complex[matrix_idx] = vm * np.exp(1j * va_rad)
        
    l_index_map = {}
    if static_matrices['F_matrix'] is not None:
        L_vals = vsi.calculate_l_index_vectorized(V_complex, static_matrices['F_matrix'], static_matrices['idx_gen'], static_matrices['idx_load'])
        for i, bus_id_ext in enumerate(static_matrices['load_buses_ids']):
            if i < len(L_vals):
                l_index_map[bus_id_ext] = L_vals[i]
            
    vcpi_vals = vsi.calculate_vcpi_bus_vectorized(V_complex, static_matrices['Ybus'])
    idx_int_to_ext = {v: k for k, v in bus_map.items()}
    vcpi_map = {}
    for int_idx, val in enumerate(vcpi_vals):
        if int_idx in idx_int_to_ext:
            vcpi_map[idx_int_to_ext[int_idx]] = val

    results = []
    s_base = 100.0
    for idx, line in line_data.iterrows():
        if idx not in res_line.index: continue
        from_bus, to_bus = line.from_bus, line.to_bus
        if from_bus not in res_bus.index or to_bus not in res_bus.index: continue
        
        bus_data = snapshot['bus_data']
        vn_kv = bus_data.at[from_bus, 'vn_kv']
        z_base = (vn_kv ** 2) / s_base
        R_pu = (line.r_ohm_per_km * line.length_km) / z_base
        X_pu = (line.x_ohm_per_km * line.length_km) / z_base
        Z_pu, theta = vsi.get_line_params(R_pu, X_pu)
        
        p_from = res_line.at[idx, 'p_from_mw']
        q_to = res_line.at[idx, 'q_to_mvar']
        p_from_pu, q_to_pu = p_from / s_base, q_to / s_base
        
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
            'L_index': l_index_map.get(idx_r, 0.0),
            'VCPI_bus': vcpi_map.get(idx_r, 0.0)
        }
        results.append(row)
    return pd.DataFrame(results)

# --- 3. PLOTAGEM (COM LINHA DE CRITICIDADE) ---

def plot_pv_curves(history, title="Curvas PV", save_dir="."):
    p_total = [snap['total_load_mw'] for snap in history]
    vm_data = [snap['res_bus']['vm_pu'].values for snap in history]
    df_vm = pd.DataFrame(vm_data, index=p_total)
    bus_ids = history[0]['res_bus'].index
    df_vm.columns = bus_ids
    last_step_voltages = df_vm.iloc[-1]
    critical_bus_idx = last_step_voltages.idxmin()
    critical_val = last_step_voltages.min()
    max_load = p_total[-1]
    
    plt.figure(figsize=(14, 8))
    other_buses = [b for b in df_vm.columns if b != critical_bus_idx]
    cmap = plt.cm.get_cmap('viridis', len(other_buses))
    for i, bus_id in enumerate(other_buses):
        plt.plot(df_vm.index, df_vm[bus_id], color=cmap(i), linewidth=1.2, alpha=0.6, label=f'Barra {bus_id}')
    plt.plot(df_vm.index, df_vm[critical_bus_idx], color='red', linewidth=3.5, zorder=10, label=f'CRÍTICA: Barra {critical_bus_idx} ({critical_val:.3f} pu)')
    plt.plot(max_load, critical_val, 'X', color='black', markersize=12, zorder=11, label=f'Colapso: {max_load:.1f} MW')
    plt.title(f'{title}', fontsize=16, fontweight='bold')
    plt.xlabel('Potência Ativa Total (MW)', fontsize=13)
    plt.ylabel('Tensão (pu)', fontsize=13)
    plt.axvline(x=max_load, color='black', linestyle='--', alpha=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small', ncol=2, title="Barras do Sistema", frameon=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    filename = os.path.join(save_dir, "curva_pv_sistema.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico da Curva PV salvo em: {filename}")

def plot_comparative_indices(all_scenarios_results, save_dir="."):
    first_key = list(all_scenarios_results.keys())[0]
    all_cols = all_scenarios_results[first_key].columns
    indices_cols = [c for c in all_cols if c not in ['Line_ID', 'From', 'To']]
    bus_indices_names = ['L_index', 'VCPI_bus'] 
    scenario_keys = sorted(list(all_scenarios_results.keys()))
    cmap = plt.cm.get_cmap('turbo')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(scenario_keys))]
    
    print(f"Gerando gráficos para {len(indices_cols)} índices em {save_dir}...")
    
    for ind_name in indices_cols:
        plt.figure(figsize=(10, 6))
        is_bus_index = ind_name in bus_indices_names
        
        # Define limite de referência (Restaurado!)
        limit = 1.0
        if ind_name in ['SI', 'VCPI_1', 'VSMI', 'VSI_1']: 
            limit = 0.0
        
        max_val_scenario = 0.0
        
        for i, pct in enumerate(scenario_keys):
            df = all_scenarios_results[pct]
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ind_name])
            
            if is_bus_index:
                df_plot = df_clean[['To', ind_name]].drop_duplicates(subset=['To'])
                df_plot = df_plot[df_plot[ind_name] > 0.001]
                x_data, y_data, marker = df_plot['To'], df_plot[ind_name], 's'
            else:
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
        
        # LINHA VERMELHA RESTAURADA
        plt.axhline(y=limit, color='red', linestyle='--', linewidth=1.5, label=f'Limite ({limit})')
        
        plt.legend(title="Carga (%)", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = os.path.join(save_dir, f'analise_{ind_name.lower()}.png')
        plt.savefig(filename, dpi=150)
        plt.close()
    print("Gráficos de índices salvos.")

# --- 4. RELATÓRIOS TXT ---

def generate_initial_report(net, system_name, filepath):
    try: pp.runpp(net)
    except: pass
    header = f"""
{'='*100}
RELATORIO DO CASO BASE (PONTO DE PARTIDA)
SISTEMA: {system_name.upper()}
DATA: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
{'='*100}
RESUMO GERAL:
-------------
Carga Total:   {net.res_load.p_mw.sum():.2f} MW  |  {net.res_load.q_mvar.sum():.2f} Mvar
Geração Total: {net.res_gen.p_mw.sum() + net.res_ext_grid.p_mw.sum():.2f} MW
DETALHE DAS BARRAS:
{'='*80}
BARRA   | V (pu)  | ANG (deg) | P_INJ (MW) | Q_INJ (Mvar) | TIPO
{'-'*80}
"""
    content = header
    sorted_buses = net.res_bus.sort_values(by='vm_pu')
    for bus_id, row in sorted_buses.iterrows():
        b_type = "PQ"
        if bus_id in net.gen.bus.values or bus_id in net.ext_grid.bus.values: b_type = "PV/REF"
        content += f"{bus_id:<7} | {row['vm_pu']:<7.4f} | {row['va_degree']:<9.2f} | {row['p_mw']:<10.2f} | {row['q_mvar']:<12.2f} | {b_type}\n"
    content += f"\n{'='*80}\n"
    with open(filepath, "w") as f: f.write(content)
    print(f"  -> Relatório Inicial salvo: {filepath}")

def generate_anarede_report(history, system_name, filepath):
    snap = history[-1]
    res_bus = snap['res_bus']
    res_line = snap.get('res_line', pd.DataFrame())
    line_data = snap.get('line_data', pd.DataFrame())
    res_trafo = snap.get('res_trafo', pd.DataFrame())
    trafo_data = snap.get('trafo_data', pd.DataFrame())
    max_load, scale = snap['total_load_mw'], snap['scale']
    header = f"""
{'='*80}
RELATORIO DE ANALISE DE ESTABILIDADE DE TENSAO (COLAPSO)
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
BARRA   | V (pu)  | ANG (deg) | P_INJ (MW) | Q_INJ (Mvar)
{'-'*80}
"""
    content = header
    sorted_buses = res_bus.sort_values(by='vm_pu')
    for bus_id, row in sorted_buses.iterrows():
        content += f"{bus_id:<7} | {row['vm_pu']:<7.4f} | {row['va_degree']:<9.2f} | {row['p_mw']:<10.2f} | {row['q_mvar']:<12.2f}\n"
    content += f"\n{'='*80}\nFLUXO NOS RAMOS (LINHAS E TRANSFORMADORES)\n{'='*80}\n"
    content += f"TIPO  | ID    | DE      | PARA    | P_DE (MW) | Q_DE (Mvar) | CARREG (%)\n{'-'*80}\n"
    branches = []
    for idx, row in res_line.iterrows():
        if idx in line_data.index:
            branches.append({'type': 'LIN', 'id': idx, 'from': line_data.at[idx, 'from_bus'], 'to': line_data.at[idx, 'to_bus'], 'p': row['p_from_mw'], 'q': row['q_from_mvar'], 'load': row.get('loading_percent', 0.0)})
    for idx, row in res_trafo.iterrows():
        if idx in trafo_data.index:
            branches.append({'type': 'TRF', 'id': idx, 'from': trafo_data.at[idx, 'hv_bus'], 'to': trafo_data.at[idx, 'lv_bus'], 'p': row['p_hv_mw'], 'q': row['q_hv_mvar'], 'load': row.get('loading_percent', 0.0)})
    branches.sort(key=lambda x: x['load'], reverse=True)
    for b in branches:
        content += f"{b['type']:<5} | {b['id']:<5} | {b['from']:<7} | {b['to']:<7} | {b['p']:<9.2f} | {b['q']:<11.2f} | {b['load']:.1f}\n"
    content += f"\n{'='*80}\nFIM DO RELATORIO\n{'='*80}\n"
    with open(filepath, "w") as f: f.write(content)
    print(f"  -> Relatório de Colapso salvo em: {filepath}")

def generate_convergence_report(full_log, system_name, filepath):
    header = f"""
X----X----------------X--------------------------X-------------------------X---------X
                               AUMENTO DA CARGA                           
  NUM.   CONVERGENCIA       ATIVA E REATIVA (%)        CARGA TOTAL         PASSO MAX 
          STATUS          MAXIMO (LAMBDA-1)          MW    /   Mvar           (%)    
X----X----------------X--------------------------X-------------------------X---------X
"""
    content = header
    for row in full_log:
        iter_num = row['iter']
        status = row['status']
        scale = row['scale']
        step_used = row.get('step', 0.0)
        increase_pct = (scale - 1.0) * 100.0
        if increase_pct < 0: increase_pct = 0.0
        step_pct = step_used * 100.0
        if row['mw'] > 0:
            mw_val = row['mw']
            mvar_val = row['mvar']
            line_str = (
                f"  {iter_num:<4} {status:<13}   {increase_pct:8.3f} {increase_pct:8.3f} {increase_pct:8.3f}   "
                f"{mw_val:8.2f} MW   {step_pct:8.4f}\n"
                f"                                                         {mvar_val:8.2f} Mvar {step_pct:8.4f}\n"
            )
        else:
            line_str = (
                f"  {iter_num:<4} {status:<13}   {increase_pct:8.3f} {increase_pct:8.3f} {increase_pct:8.3f}   "
                f"   ---      MW   {step_pct:8.4f}\n"
                f"                                                            ---      Mvar {step_pct:8.4f}\n"
            )
        content += line_str
    content += f"X----X----------------X--------------------------X-------------------------X---------X\n"
    with open(filepath, "w") as f: f.write(content)
    print(f"  -> Relatório de Convergência salvo em: {filepath}")