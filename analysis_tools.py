import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vsi_lib as vsi
import pandapower as pp
from scipy.sparse.linalg import inv
import os
from datetime import datetime

# ==============================================================================
# FERRAMENTAS DE ANÁLISE
# ==============================================================================

def pre_calculate_matrices(net):
    print("  -> Pré-calculando matrizes de admitância (Ybus) e participação (F)...")
    try: pp.runpp(net)
    except: return {'Ybus': None, 'F_matrix': None, 'bus_to_idx': {}}
        
    Ybus = net._ppc['internal']['Ybus']
    bus_lookup = net._pd2ppc_lookups['bus']
    
    bus_to_idx = {}
    for ext_id in net.bus.index:
        if net.bus.at[ext_id, 'in_service'] and ext_id in bus_lookup:
            bus_to_idx[ext_id] = int(bus_lookup[ext_id])
            
    gen_buses_ext = list(set(net.gen.bus.values.tolist() + net.ext_grid.bus.values.tolist()))
    all_load = [b for b in net.bus.index if b not in gen_buses_ext and net.bus.at[b, 'in_service']]
    
    valid_load = [b for b in all_load if b in bus_to_idx]
    idx_gen_int = [bus_to_idx[b] for b in gen_buses_ext if b in bus_to_idx]
    idx_load_int = [bus_to_idx[b] for b in valid_load]
    
    if idx_load_int:
        Y_LL = Ybus[idx_load_int, :][:, idx_load_int]
        Y_LG = Ybus[idx_load_int, :][:, idx_gen_int]
        try: F_matrix = -inv(Y_LL).dot(Y_LG)
        except: F_matrix = None
    else: F_matrix = None

    return {'Ybus': Ybus, 'F_matrix': F_matrix, 'bus_to_idx': bus_to_idx, 
            'idx_gen': idx_gen_int, 'idx_load': idx_load_int, 'load_buses_ids': valid_load}

def calculate_indices_for_scenario(snapshot, static_matrices):
    res_bus = snapshot['res_bus']
    res_line = snapshot['res_line']
    line_data = snapshot['line_data']
    
    bus_map = static_matrices.get('bus_to_idx', {})
    ybus = static_matrices.get('Ybus')
    
    if not bus_map or ybus is None: return pd.DataFrame()
    
    V_complex = np.zeros(ybus.shape[0], dtype=complex)
    for bus_id_ext, matrix_idx in bus_map.items():
        if bus_id_ext in res_bus.index:
            vm = res_bus.at[bus_id_ext, 'vm_pu']
            va = np.radians(res_bus.at[bus_id_ext, 'va_degree'])
            if matrix_idx < len(V_complex):
                V_complex[matrix_idx] = vm * np.exp(1j * va)
        
    l_index_map = {}
    if static_matrices['F_matrix'] is not None:
        L_vals = vsi.calculate_l_index_vectorized(V_complex, static_matrices['F_matrix'], 
                                                  static_matrices['idx_gen'], static_matrices['idx_load'])
        for i, bus_id_ext in enumerate(static_matrices['load_buses_ids']):
            if i < len(L_vals): l_index_map[bus_id_ext] = L_vals[i]
            
    vcpi_vals = vsi.calculate_vcpi_bus_vectorized(V_complex, static_matrices['Ybus'])
    idx_int_to_ext = {v: k for k, v in bus_map.items()}
    vcpi_map = {idx_int_to_ext[i]: val for i, val in enumerate(vcpi_vals) if i in idx_int_to_ext}

    results = []
    s_base = 100.0
    for idx, line in line_data.iterrows():
        if idx not in res_line.index: continue
        f, t = line.from_bus, line.to_bus
        if f not in res_bus.index or t not in res_bus.index: continue
        
        vn_kv = snapshot['bus_data'].at[f, 'vn_kv']
        z_base = (vn_kv**2)/s_base
        R_pu = (line.r_ohm_per_km * line.length_km)/z_base
        X_pu = (line.x_ohm_per_km * line.length_km)/z_base
        Z_pu, theta = vsi.get_line_params(R_pu, X_pu)
        
        p_from = res_line.at[idx, 'p_from_mw']
        if p_from >= 0:
            idx_s, idx_r = f, t
            Q_r = abs(res_line.at[idx, 'q_to_mvar'])/s_base
            P_s = abs(p_from)/s_base
            P_r = abs(res_line.at[idx, 'p_to_mw'])/s_base
        else:
            idx_s, idx_r = t, f
            Q_r = abs(res_line.at[idx, 'q_from_mvar'])/s_base
            P_s = abs(res_line.at[idx, 'p_to_mw'])/s_base
            P_r = abs(p_from)/s_base
            
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

def plot_pv_curves(history, title="Curvas PV", save_dir="."):
    p_total = [snap['total_load_mw'] for snap in history]
    vm_data = [snap['res_bus']['vm_pu'].values for snap in history]
    df_vm = pd.DataFrame(vm_data, index=p_total)
    bus_ids = history[0]['res_bus'].index
    df_vm.columns = bus_ids
    crit_idx = df_vm.iloc[-1].idxmin()
    
    plt.figure(figsize=(14, 8))
    other = [b for b in df_vm.columns if b != crit_idx]
    cmap = plt.cm.get_cmap('viridis', len(other))
    for i, b in enumerate(other):
        plt.plot(df_vm.index, df_vm[b], color=cmap(i), linewidth=1.2, alpha=0.6, label=f'Barra {b}')
    plt.plot(df_vm.index, df_vm[crit_idx], color='red', linewidth=3.5, zorder=10, label=f'CRÍTICA: {crit_idx}')
    plt.axvline(p_total[-1], color='black', linestyle='--')
    plt.title(title, fontsize=16); plt.xlabel('MW'); plt.ylabel('V (pu)')
    plt.legend(bbox_to_anchor=(1.01, 0.5), fontsize='small', ncol=2)
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curva_pv_sistema.png"), dpi=150, bbox_inches='tight'); plt.close()

def plot_comparative_indices(all_res, save_dir="."):
    first = list(all_res.keys())[0]
    cols = [c for c in all_res[first].columns if c not in ['Line_ID', 'From', 'To']]
    scenarios = sorted(list(all_res.keys()))
    cmap = plt.cm.get_cmap('turbo')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(scenarios))]
    
    for ind in cols:
        plt.figure(figsize=(10, 6))
        is_bus = ind in ['L_index', 'VCPI_bus']
        for i, pct in enumerate(scenarios):
            df = all_res[pct].replace([np.inf, -np.inf], np.nan).dropna(subset=[ind])
            if is_bus:
                df = df[['To', ind]].drop_duplicates(subset=['To'])
                df = df[df[ind] > 0.001]
                x, y, m = df['To'], df[ind], 's'
            else:
                df = df[df[ind] < 10]
                x, y, m = df['Line_ID'], df[ind], 'o'
            if not y.empty:
                plt.scatter(x, y, label=f'{pct}%', color=colors[i], alpha=0.75, marker=m)
        plt.title(f'{ind}'); plt.xlabel('ID'); plt.ylabel('Valor')
        plt.legend(bbox_to_anchor=(1.02, 1)); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'analise_{ind.lower()}.png'), dpi=150); plt.close()

def generate_initial_report(net, sys, path):
    try: pp.runpp(net)
    except: pass
    with open(path, "w") as f:
        f.write(f"RELATORIO INICIAL {sys}\n{'-'*40}\n")
        f.write(f"Carga Total: {net.res_load.p_mw.sum():.2f} MW\n")
        f.write(f"Geração Total: {net.res_gen.p_mw.sum() + net.res_ext_grid.p_mw.sum():.2f} MW\n")

def generate_anarede_report(history, sys, path):
    snap = history[-1]; bus = snap['res_bus']; load = snap['total_load_mw']
    line = snap.get('res_line', pd.DataFrame())
    trafo = snap.get('res_trafo', pd.DataFrame())
    with open(path, "w") as f:
        f.write(f"RELATORIO COLAPSO {sys}\nCarga Max: {load:.2f} MW\n")
        f.write("--- BARRAS ---\nID      V(pu)   Ang     P       Q\n")
        for i, r in bus.sort_values('vm_pu').iterrows():
            f.write(f"{i:<7} {r.vm_pu:.4f} {r.va_degree:<7.2f} {r.p_mw:<7.2f} {r.q_mvar:<7.2f}\n")
        f.write("\n--- RAMOS (LIN/TRAFO) ---\nID    De   Para  Carreg(%)\n")
        branches = []
        for i, r in line.iterrows(): branches.append((i, r.get('loading_percent',0)))
        for i, r in trafo.iterrows(): branches.append((i, r.get('loading_percent',0)))
        for i, l in sorted(branches, key=lambda x: x[1], reverse=True):
            f.write(f"{i:<5} ... {l:.2f}\n")

def generate_convergence_report(log, sys, path):
    with open(path, "w") as f:
        f.write(f"RELATORIO CONVERGENCIA {sys}\n")
        f.write("NUM    STATUS         PASSO            LAMBDA           MW\n")
        for r in log:
            mw = f"{r['mw']:.2f}" if r['mw'] > 0 else "---"
            f.write(f"{r['iter']:<6} {r['status']:<14} {r.get('step',0):<16.5f} {r['scale']:<16.4f} {mw}\n")