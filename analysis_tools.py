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

def set_ieee_style():
    """Configura o Matplotlib para o padrão IEEE (Times New Roman, 10pt)."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': (3.5, 2.6), # Tamanho para coluna simples IEEE
        'savefig.dpi': 600,
        'savefig.format': 'eps',
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

# --- 1. PRÉ-CÁLCULO DE MATRIZES ---
def pre_calculate_matrices(net):
    """Calcula matrizes estáticas (Ybus e F) e parâmetros de linha p.u."""
    print("  -> Pré-calculando matrizes e parâmetros estáticos de rede...")
    
    try:
        pp.runpp(net)
    except:
        print("Aviso: runpp falhou no pré-cálculo.")
        
    # Ybus e F-Matrix
    try:
        Ybus = net._ppc['internal']['Ybus']
        bus_lookup = net._pd2ppc_lookups['bus']
    except:
        return {'Ybus': None, 'F_matrix': None, 'bus_to_idx': {}, 'idx_gen': [], 'idx_load': [], 'load_buses_ids': [], 'line_params': {}}
    
    bus_to_idx = {ext_id: int(bus_lookup[ext_id]) for ext_id in net.bus.index if ext_id in bus_lookup}
            
    gen_buses_ext = list(set(net.gen.bus.values.tolist() + net.ext_grid.bus.values.tolist()))
    valid_load_buses = [b for b in net.bus.index if b not in gen_buses_ext and b in bus_to_idx]
    valid_gen_buses = [b for b in gen_buses_ext if b in bus_to_idx]
    
    idx_gen_int = [bus_to_idx[b] for b in valid_gen_buses]
    idx_load_int = [bus_to_idx[b] for b in valid_load_buses]
    
    F_matrix = None
    if idx_load_int:
        Y_LL = Ybus[idx_load_int, :][:, idx_load_int]
        Y_LG = Ybus[idx_load_int, :][:, idx_gen_int]
        try: F_matrix = -inv(Y_LL).dot(Y_LG)
        except: print("AVISO: Matriz Y_LL singular.")

    # Parâmetros de Linha p.u.
    s_base = 100.0
    line_data = net.line[net.line.in_service]
    from_buses = line_data.from_bus.values
    vn_kv = net.bus.loc[from_buses, 'vn_kv'].values
    z_base = (vn_kv ** 2) / s_base
    R_pu = (line_data.r_ohm_per_km.values * line_data.length_km.values) / z_base
    X_pu = (line_data.x_ohm_per_km.values * line_data.length_km.values) / z_base
    Z_pu, theta = vsi.get_line_params(R_pu, X_pu)
    
    line_params = {
        'indices': line_data.index.values,
        'from_bus': from_buses,
        'to_bus': line_data.to_bus.values,
        'R_pu': R_pu, 'X_pu': X_pu, 'Z_pu': Z_pu, 'theta': theta
    }

    return {
        'Ybus': Ybus, 'F_matrix': F_matrix, 'bus_to_idx': bus_to_idx, 
        'idx_gen': idx_gen_int, 'idx_load': idx_load_int, 
        'load_buses_ids': valid_load_buses, 'line_params': line_params
    }

# --- 2. CÁLCULO DOS ÍNDICES (VETORIZADO) ---
def calculate_indices_for_scenario(snapshot, static_matrices):
    res_bus = snapshot['res_bus']
    res_line = snapshot['res_line']
    lp = static_matrices['line_params']
    
    if not static_matrices['bus_to_idx'] or not lp: return pd.DataFrame()
    
    # 2.1 Tensões Complexas e Mapas Globais
    ybus_size = static_matrices['Ybus'].shape[0]
    V_complex = np.zeros(ybus_size, dtype=complex)
    for ext_id, int_idx in static_matrices['bus_to_idx'].items():
        if ext_id in res_bus.index:
            vm = res_bus.at[ext_id, 'vm_pu']
            va = np.radians(res_bus.at[ext_id, 'va_degree'])
            V_complex[int_idx] = vm * np.exp(1j * va)
        
    # 2.2 Índices de Barra (L-index, VCPI_bus)
    l_index_map = {}
    if static_matrices['F_matrix'] is not None:
        L_vals = vsi.calculate_l_index_vectorized(V_complex, static_matrices['F_matrix'], static_matrices['idx_gen'], static_matrices['idx_load'])
        l_index_map = dict(zip(static_matrices['load_buses_ids'], L_vals))
            
    vcpi_bus_vals = vsi.calculate_vcpi_bus_vectorized(V_complex, static_matrices['Ybus'])
    idx_int_to_ext = {v: k for k, v in static_matrices['bus_to_idx'].items()}
    vcpi_map = {idx_int_to_ext[i]: val for i, val in enumerate(vcpi_bus_vals) if i in idx_int_to_ext}

    # 2.3 Extração Vetorizada de Dados das Linhas
    line_idx = lp['indices']
    mask = np.isin(line_idx, res_line.index)
    line_idx = line_idx[mask]
    
    from_b = lp['from_bus'][mask]
    to_b = lp['to_bus'][mask]
    
    # Tensões e Ângulos
    V_from = res_bus.loc[from_b, 'vm_pu'].values
    V_to = res_bus.loc[to_b, 'vm_pu'].values
    Va_from = np.radians(res_bus.loc[from_b, 'va_degree'].values)
    Va_to = np.radians(res_bus.loc[to_b, 'va_degree'].values)
    
    # Fluxos (p.u.)
    s_base = 100.0
    p_from = res_line.loc[line_idx, 'p_from_mw'].values / s_base
    q_from = res_line.loc[line_idx, 'q_from_mvar'].values / s_base
    p_to = res_line.loc[line_idx, 'p_to_mw'].values / s_base
    q_to = res_line.loc[line_idx, 'q_to_mvar'].values / s_base
    
    # Sentido do Fluxo: Se p_from >= 0, From=Source, To=Receiver
    is_fwd = p_from >= 0
    V_s = np.where(is_fwd, V_from, V_to)
    V_r = np.where(is_fwd, V_to, V_from)
    delta = np.where(is_fwd, Va_from - Va_to, Va_to - Va_from)
    
    P_r = np.where(is_fwd, np.abs(p_to), np.abs(p_from))
    Q_r = np.where(is_fwd, np.abs(q_to), np.abs(q_from))
    P_s = np.where(is_fwd, np.abs(p_from), np.abs(p_to))
    
    R_pu, X_pu, Z_pu, theta = lp['R_pu'][mask], lp['X_pu'][mask], lp['Z_pu'][mask], lp['theta'][mask]
    _, phi = vsi.get_load_params(P_r, Q_r)

    # 2.4 Cálculos Vetorizados em Massa
    results_df = pd.DataFrame({
        'Line_ID': line_idx, 'From': np.where(is_fwd, from_b, to_b), 'To': np.where(is_fwd, to_b, from_b),
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
        'VSI_1': vsi.calculate_vsi1(V_s, P_r, Q_r, X_pu)
    })

    # Mapeamento de Índices de Barra para o DataFrame de Linhas (usando o Receiver 'To')
    results_df['L_index'] = results_df['To'].map(l_index_map).fillna(0.0)
    results_df['VCPI_bus'] = results_df['To'].map(vcpi_map).fillna(0.0)

    return results_df


# --- 3. PLOTAGEM (PADRÃO IEEE - SVG) ---

def plot_pv_curves(history, title="Curvas PV", save_dir="."):
    set_ieee_style()
    p_total = [snap['total_load_mw'] for snap in history]
    vm_data = [snap['res_bus']['vm_pu'].values for snap in history]
    df_vm = pd.DataFrame(vm_data, index=p_total)
    bus_ids = history[0]['res_bus'].index
    df_vm.columns = bus_ids
    last_step_voltages = df_vm.iloc[-1]
    critical_bus_idx = last_step_voltages.idxmin()
    critical_val = last_step_voltages.min()
    max_load = p_total[-1]
    
    plt.figure(figsize=(5.5, 3.5)) # Dimensão aumentada para acomodar legenda externa
    other_buses = [b for b in df_vm.columns if b != critical_bus_idx]
    
    for i, bus_id in enumerate(other_buses):
        plt.plot(df_vm.index, df_vm[bus_id], color='gray', linewidth=0.5, alpha=0.3)
    
    plt.plot(df_vm.index, df_vm[critical_bus_idx], color='black', linewidth=1.5, label=f'Critical Bus {critical_bus_idx}')
    plt.plot(max_load, critical_val, 'o', color='red', markersize=4, label='Collapse Point')
    
    plt.xlabel('Total Active Power (MW)')
    plt.ylabel('Voltage (pu)')
    
    # Legenda fora do gráfico, à direita
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    filename = os.path.join(save_dir, "curva_pv_sistema.svg")
    plt.savefig(filename)
    plt.close()
    print(f"  -> Gráfico PV (SVG) salvo com legenda externa.")

def plot_comparative_indices(all_scenarios_results, save_dir="."):
    set_ieee_style()
    first_key = list(all_scenarios_results.keys())[0]
    all_cols = all_scenarios_results[first_key].columns
    indices_cols = [c for c in all_cols if c not in ['Line_ID', 'From', 'To']]
    bus_indices_names = ['L_index', 'VCPI_bus'] 
    scenario_keys = sorted(list(all_scenarios_results.keys()))
    
    # Paleta de cores por percentual
    cmap = plt.cm.get_cmap('turbo')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(scenario_keys))]
    
    for ind_name in indices_cols:
        plt.figure(figsize=(5.5, 3.5)) # Dimensão aumentada para acomodar legenda externa
        is_bus_index = ind_name in bus_indices_names
        marker = 's' if is_bus_index else 'o' # Quadrados para Barras, Círculos para Linhas
        
        limit = 1.0
        if ind_name in ['SI', 'VCPI_1', 'VSMI', 'VSI_1']: limit = 0.0
        
        for i, pct in enumerate(scenario_keys):
            df = all_scenarios_results[pct]
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ind_name])
            
            if is_bus_index:
                df_plot = df_clean[['To', ind_name]].drop_duplicates(subset=['To'])
                x_data, y_data = df_plot['To'], df_plot[ind_name]
            else:
                df_plot = df_clean[df_clean[ind_name] < 5.0]
                x_data, y_data = df_plot['Line_ID'], df_plot[ind_name]

            if not y_data.empty:
                plt.scatter(x_data, y_data, label=f'{pct}%', 
                            marker=marker, color=colors[i], s=20, alpha=0.8)

        plt.xlabel('Bus ID' if is_bus_index else 'Line ID')
        plt.ylabel(f'{ind_name} Value')
        plt.axhline(y=limit, color='black', linestyle=':', linewidth=1.0)
        
        # Legenda fora do gráfico, à direita
        plt.legend(title="Load (%)", loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=1)
            
        filename = os.path.join(save_dir, f'analise_{ind_name.lower()}.svg')
        plt.savefig(filename)
        plt.close()
    print(f"  -> Gráficos de Índices (SVG) salvos com legendas externas.")


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