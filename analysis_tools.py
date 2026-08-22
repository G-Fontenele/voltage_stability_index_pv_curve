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
    """Calcula matrizes estáticas (Ybus e F) e parâmetros de ramos p.u. (Linhas e Trafos)"""
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
        return {'Ybus': None, 'F_matrix': None, 'bus_to_idx': {}, 'idx_gen': [], 'idx_load': [], 'load_buses_ids': [], 'branch_params': {}}
    
    bus_to_idx = {}
    for ext_id in net.bus.index:
        try:
            internal_idx = int(bus_lookup[ext_id])
            if internal_idx >= 0:
                bus_to_idx[ext_id] = internal_idx
        except (IndexError, KeyError):
            pass
    active_gens = net.gen[net.gen.in_service]
    active_ext = net.ext_grid[net.ext_grid.in_service]
    gen_buses_ext = list(set(active_gens.bus.values.tolist() + active_ext.bus.values.tolist()))
    valid_load_buses = [b for b in net.bus.index if b not in gen_buses_ext and b in bus_to_idx]
    valid_gen_buses = [b for b in gen_buses_ext if b in bus_to_idx]
    
    idx_gen_int = [bus_to_idx[b] for b in valid_gen_buses]
    idx_load_int = [bus_to_idx[b] for b in valid_load_buses]
    
    F_matrix = None
    if idx_load_int:
        Y_LL = Ybus[idx_load_int, :][:, idx_load_int]
        Y_LG = Ybus[idx_load_int, :][:, idx_gen_int]
        try: 
            Y_LL_dense = Y_LL.toarray() if hasattr(Y_LL, 'toarray') else Y_LL
            Y_LG_dense = Y_LG.toarray() if hasattr(Y_LG, 'toarray') else Y_LG
            F_matrix = -np.linalg.inv(Y_LL_dense).dot(Y_LG_dense)
        except: print("AVISO: Matriz Y_LL singular.")

    s_base = 100.0
    
    # 1. Parâmetros de Linha
    line_data = net.line[net.line.in_service]
    from_buses_l = line_data.from_bus.values
    to_buses_l = line_data.to_bus.values
    vn_kv_l = net.bus.loc[from_buses_l, 'vn_kv'].values
    z_base_l = (vn_kv_l ** 2) / s_base
    R_pu_l = (line_data.r_ohm_per_km.values * line_data.length_km.values) / z_base_l
    X_pu_l = (line_data.x_ohm_per_km.values * line_data.length_km.values) / z_base_l
    branch_type_l = np.array(['LINE'] * len(line_data))
    branch_id_l = line_data.index.values

    # 2. Parâmetros de Transformador
    trafo_data = net.trafo[net.trafo.in_service]
    from_buses_t = trafo_data.hv_bus.values
    to_buses_t = trafo_data.lv_bus.values
    R_pu_t = (trafo_data.vkr_percent.values / 100.0) * (s_base / trafo_data.sn_mva.values)
    Z_pu_t = (trafo_data.vk_percent.values / 100.0) * (s_base / trafo_data.sn_mva.values)
    X_pu_t = np.sqrt(np.maximum(0, Z_pu_t**2 - R_pu_t**2))
    branch_type_t = np.array(['TRAFO'] * len(trafo_data))
    branch_id_t = trafo_data.index.values

    # 3. Unificação
    all_from = np.concatenate([from_buses_l, from_buses_t])
    all_to = np.concatenate([to_buses_l, to_buses_t])
    all_R = np.concatenate([R_pu_l, R_pu_t])
    all_X = np.concatenate([X_pu_l, X_pu_t])
    all_types = np.concatenate([branch_type_l, branch_type_t])
    all_ids = np.concatenate([branch_id_l, branch_id_t])
    
    Z_pu, theta = vsi.get_line_params(all_R, all_X)
    
    branch_params = {
        'indices': all_ids,
        'from_bus': all_from,
        'to_bus': all_to,
        'R_pu': all_R, 'X_pu': all_X, 'Z_pu': Z_pu, 'theta': theta,
        'type': all_types
    }

    return {
        'Ybus': Ybus, 'F_matrix': F_matrix, 'bus_to_idx': bus_to_idx, 
        'idx_gen': idx_gen_int, 'idx_load': idx_load_int, 
        'load_buses_ids': valid_load_buses, 'branch_params': branch_params
    }

# --- 2. CÁLCULO DOS ÍNDICES (VETORIZADO) ---
def calculate_indices_for_scenario(snapshot, default_matrices):
    static_matrices = snapshot.get('static_matrices', default_matrices)
    
    res_bus = snapshot['res_bus']
    res_line = snapshot['res_line']
    res_trafo = snapshot['res_trafo']
    bp = static_matrices['branch_params']
    
    if not static_matrices['bus_to_idx'] or not bp: return pd.DataFrame(), pd.DataFrame()
    
    # --- 2.1 RELATÓRIO DE BARRAS (COMPLETO) ---
    ybus_size = static_matrices['Ybus'].shape[0]
    V_complex = np.zeros(ybus_size, dtype=complex)
    idx_int_to_ext = {v: k for k, v in static_matrices['bus_to_idx'].items()}
    
    bus_ids = sorted(list(static_matrices['bus_to_idx'].keys()))
    vm_vals = res_bus.loc[bus_ids, 'vm_pu'].values
    va_vals = res_bus.loc[bus_ids, 'va_degree'].values
    
    for ext_id in bus_ids:
        int_idx = static_matrices['bus_to_idx'][ext_id]
        vm = res_bus.at[ext_id, 'vm_pu']
        va = np.radians(res_bus.at[ext_id, 'va_degree'])
        V_complex[int_idx] = vm * np.exp(1j * va)

    # L-index
    l_index_vals = np.zeros(len(bus_ids))
    if static_matrices['F_matrix'] is not None:
        L_vector = vsi.calculate_l_index_vectorized(V_complex, static_matrices['F_matrix'], static_matrices['idx_gen'], static_matrices['idx_load'])
        l_map = dict(zip(static_matrices['load_buses_ids'], L_vector))
        l_index_vals = np.array([l_map.get(b, 0.0) for b in bus_ids])

    # VCPI_bus
    vcpi_bus_full = vsi.calculate_vcpi_bus_vectorized(V_complex, static_matrices['Ybus'])
    vcpi_bus_vals = np.array([vcpi_bus_full[static_matrices['bus_to_idx'][b]] for b in bus_ids])

    bus_df = pd.DataFrame({
        'Bus_ID': bus_ids,
        'V_pu': vm_vals,
        'Angle_deg': va_vals,
        'L_index': l_index_vals,
        'VCPI_bus': vcpi_bus_vals
    })

    # --- 2.2 RELATÓRIO DE RAMOS (LINHAS + TRAFOS) ---
    # Extração de Fluxos Unificada
    p_from = []
    q_from = []
    p_to = []
    q_to = []
    
    for i, b_type in enumerate(bp['type']):
        b_idx = bp['indices'][i]
        if b_type == 'LINE':
            if b_idx in res_line.index:
                p_from.append(res_line.at[b_idx, 'p_from_mw'])
                q_from.append(res_line.at[b_idx, 'q_from_mvar'])
                p_to.append(res_line.at[b_idx, 'p_to_mw'])
                q_to.append(res_line.at[b_idx, 'q_to_mvar'])
            else:
                p_from.append(0.0); q_from.append(0.0); p_to.append(0.0); q_to.append(0.0)
        else: # TRAFO
            if b_idx in res_trafo.index:
                p_from.append(res_trafo.at[b_idx, 'p_hv_mw'])
                q_from.append(res_trafo.at[b_idx, 'q_hv_mvar'])
                p_to.append(res_trafo.at[b_idx, 'p_lv_mw'])
                q_to.append(res_trafo.at[b_idx, 'q_lv_mvar'])
            else:
                p_from.append(0.0); q_from.append(0.0); p_to.append(0.0); q_to.append(0.0)

    s_base = 100.0
    p_from = np.array(p_from) / s_base
    q_from = np.array(q_from) / s_base
    p_to = np.array(p_to) / s_base
    q_to = np.array(q_to) / s_base
    
    from_b = bp['from_bus']
    to_b = bp['to_bus']
    
    # Tensões e Ângulos
    V_from = res_bus.loc[from_b, 'vm_pu'].values
    V_to = res_bus.loc[to_b, 'vm_pu'].values
    Va_from = np.radians(res_bus.loc[from_b, 'va_degree'].values)
    Va_to = np.radians(res_bus.loc[to_b, 'va_degree'].values)
    
    # Sentido do Fluxo
    is_fwd = p_from >= 0
    V_s = np.where(is_fwd, V_from, V_to)
    V_r = np.where(is_fwd, V_to, V_from)
    delta = np.where(is_fwd, Va_from - Va_to, Va_to - Va_from)
    
    # Convencao Pandapower: p_from/p_to e fluxo injetado no ramo.
    # A potencia entregue ao barramento receptor e o negativo do fluxo injetado pelo terminal.
    P_r = np.where(is_fwd, -p_to, -p_from)
    Q_r = np.where(is_fwd, -q_to, -q_from)
    P_s = np.where(is_fwd, p_from, p_to)
    
    R_pu, X_pu, Z_pu, theta = bp['R_pu'], bp['X_pu'], bp['Z_pu'], bp['theta']
    _, phi = vsi.get_load_params(P_r, Q_r)

    branch_df = pd.DataFrame({
        'Branch_ID': bp['indices'], 'Type': bp['type'], 
        'From': np.where(is_fwd, from_b, to_b), 'To': np.where(is_fwd, to_b, from_b),
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

    # Mapeamento para o DataFrame de Ramos
    l_map = dict(zip(bus_df['Bus_ID'], bus_df['L_index']))
    vcpi_map = dict(zip(bus_df['Bus_ID'], bus_df['VCPI_bus']))
    branch_df['L_index'] = branch_df['To'].map(l_map).fillna(0.0)
    branch_df['VCPI_bus'] = branch_df['To'].map(vcpi_map).fillna(0.0)

    return branch_df, bus_df


# --- 3. PLOTAGEM (PADRÃO IEEE - SVG) ---

def plot_pv_curves(history, title="Curvas PV", save_dir=".", bus_count=0):
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
    
    suffix = f"_{bus_count}" if bus_count > 0 else ""
    filename = os.path.join(save_dir, f"curva_pv_sistema{suffix}.svg")
    plt.savefig(filename)
    plt.close()
    print(f"  -> Gráfico PV (SVG) salvo com sufixo {suffix}.")

def plot_comparative_indices(branch_scenarios, bus_scenarios, save_dir=".", bus_count=0):
    set_ieee_style()
    first_key = list(branch_scenarios.keys())[0]
    
    branch_cols = branch_scenarios[first_key].columns
    bus_cols = bus_scenarios[first_key].columns
    
    branch_indices = [c for c in branch_cols if c not in ['Branch_ID', 'Type', 'From', 'To', 'L_index', 'VCPI_bus']]
    bus_indices = [c for c in bus_cols if c not in ['Bus_ID', 'V_pu', 'Angle_deg']]
    
    scenario_keys = sorted(list(branch_scenarios.keys()))
    
    # Paleta de cores por percentual
    cmap = plt.cm.get_cmap('turbo')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(scenario_keys))]
    
    suffix = f"_{bus_count}" if bus_count > 0 else ""

    # 1. Plotar Índices de Ramo
    for ind_name in branch_indices:
        plt.figure(figsize=(5.5, 3.5))
        limit = 1.0
        if ind_name in ['SI', 'VCPI_1', 'VSMI', 'VSI_1']: limit = 0.0
        
        for i, pct in enumerate(scenario_keys):
            df = branch_scenarios[pct]
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ind_name])
            df_plot = df_clean[df_clean[ind_name] < 5.0]
            
            x_data, y_data = df_plot['Branch_ID'], df_plot[ind_name]
            if not y_data.empty:
                plt.scatter(x_data, y_data, label=f'{pct}%', marker='o', color=colors[i], s=20, alpha=0.8)

        plt.xlabel('Branch ID')
        plt.ylabel(f'{ind_name} Value')
        plt.axhline(y=limit, color='black', linestyle=':', linewidth=1.0)
        plt.legend(title="Load (%)", loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=1)
            
        filename = os.path.join(save_dir, f'analise_{ind_name.lower()}{suffix}.svg')
        plt.savefig(filename)
        plt.close()

    # 2. Plotar Índices Nodais
    for ind_name in bus_indices:
        plt.figure(figsize=(5.5, 3.5))
        limit = 1.0
        
        for i, pct in enumerate(scenario_keys):
            df = bus_scenarios[pct]
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ind_name])
            
            x_data, y_data = df_clean['Bus_ID'], df_clean[ind_name]
            if not y_data.empty:
                plt.scatter(x_data, y_data, label=f'{pct}%', marker='s', color=colors[i], s=20, alpha=0.8)

        plt.xlabel('Bus ID')
        plt.ylabel(f'{ind_name} Value')
        plt.axhline(y=limit, color='black', linestyle=':', linewidth=1.0)
        plt.legend(title="Load (%)", loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=1)
            
        filename = os.path.join(save_dir, f'analise_{ind_name.lower()}{suffix}.svg')
        plt.savefig(filename)
        plt.close()

    print(f"  -> Gráficos de Índices (SVG) salvos com sufixo {suffix}.")


# --- 4. RELATÓRIOS TXT ---

def generate_initial_report(net, system_name, filepath, bus_count=0):
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
BARRA   | V (pu)  | ANG (deg) | P_GEN (MW) | Q_GEN (Mvar) | P_INJ (MW) | Q_INJ (Mvar) | TIPO
{'-'*95}
"""
    content = header
    sorted_buses = net.res_bus.sort_values(by='vm_pu')
    for bus_id, row in sorted_buses.iterrows():
        b_type = "PQ"
        p_gen = 0.0
        q_gen = 0.0
        
        # Check if it's a generator bus
        if bus_id in net.gen.bus.values:
            b_type = "PV/REF"
            gen_idx = net.gen[net.gen.bus == bus_id].index
            if not net.res_gen.empty and len(gen_idx) > 0:
                p_gen = net.res_gen.loc[gen_idx, 'p_mw'].sum()
                q_gen = net.res_gen.loc[gen_idx, 'q_mvar'].sum()
        
        # Check if it's an external grid (slack) bus
        if bus_id in net.ext_grid.bus.values:
            b_type = "PV/REF"
            ext_idx = net.ext_grid[net.ext_grid.bus == bus_id].index
            if not net.res_ext_grid.empty and len(ext_idx) > 0:
                p_gen += net.res_ext_grid.loc[ext_idx, 'p_mw'].sum()
                q_gen += net.res_ext_grid.loc[ext_idx, 'q_mvar'].sum()
                
        content += f"{bus_id:<7} | {row['vm_pu']:<7.4f} | {row['va_degree']:<9.2f} | {p_gen:<10.2f} | {q_gen:<12.2f} | {row['p_mw']:<10.2f} | {row['q_mvar']:<12.2f} | {b_type}\n"
    content += f"\n{'='*95}\n"

    # Ajusta o nome do arquivo se necessário (main já deve passar o caminho com sufixo)
    with open(filepath, "w") as f: f.write(content)
    print(f"  -> Relatório Inicial salvo: {filepath}")

def generate_anarede_report(history, system_name, filepath, bus_count=0):
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

def generate_convergence_report(full_log, system_name, filepath, bus_count=0):
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

def generate_correlation_reports(branch_df, bus_df, save_dir, bus_count=0):
    """Calcula matrizes de correlação de Spearman e Kendall Tau para os índices no ponto de colapso."""
    suffix = f"_{bus_count}" if bus_count > 0 else ""
    
    # --- BARRAS ---
    bus_cols = ['V_pu', 'Angle_deg', 'L_index', 'VCPI_bus']
    # Mantem apenas colunas numéricas disponíveis
    bus_cols = [c for c in bus_cols if c in bus_df.columns]
    bus_df_clean = bus_df[bus_cols].copy()
    bus_df_clean = bus_df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    if not bus_df_clean.empty and len(bus_df_clean) > 1:
        corr_spearman_bus = bus_df_clean.corr(method='spearman')
        corr_kendall_bus = bus_df_clean.corr(method='kendall')
        
        corr_spearman_bus.to_csv(os.path.join(save_dir, f"correlacao_spearman_barras{suffix}.csv"))
        corr_kendall_bus.to_csv(os.path.join(save_dir, f"correlacao_kendall_barras{suffix}.csv"))
    
    # --- RAMOS ---
    branch_cols_to_drop = ['Branch_ID', 'Type', 'From', 'To']
    branch_df_clean = branch_df.drop(columns=[c for c in branch_cols_to_drop if c in branch_df.columns])
    branch_df_clean = branch_df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    if not branch_df_clean.empty and len(branch_df_clean) > 1:
        corr_spearman_branch = branch_df_clean.corr(method='spearman')
        corr_kendall_branch = branch_df_clean.corr(method='kendall')
        
        corr_spearman_branch.to_csv(os.path.join(save_dir, f"correlacao_spearman_ramos{suffix}.csv"))
        corr_kendall_branch.to_csv(os.path.join(save_dir, f"correlacao_kendall_ramos{suffix}.csv"))
        
        try:
            set_ieee_style()
            plt.figure(figsize=(7, 6))
            plt.imshow(corr_spearman_branch.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(label='Spearman Correlation')
            plt.xticks(range(len(corr_spearman_branch.columns)), corr_spearman_branch.columns, rotation=90)
            plt.yticks(range(len(corr_spearman_branch.index)), corr_spearman_branch.index)
            plt.title('Spearman Rank Correlation - Line VSIs (Collapse Point)')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"heatmap_spearman_ramos{suffix}.svg"))
            plt.close()
        except Exception as e:
            print(f"Erro ao gerar heatmap de correlação: {e}")
            
    print(f"  -> Matrizes de Correlação (Spearman/Kendall) geradas em {save_dir}.")