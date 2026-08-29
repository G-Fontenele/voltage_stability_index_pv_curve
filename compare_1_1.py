import custom_systems as cs
import copy
import pandapower as pp
import numpy as np

import analysis_tools as tools
import cpf_solver as cpf

def run_compare():
    net = cs.create_ieee30_anarede(use_taps=True)
    pp.runpp(net)
    
    static_matrices = tools.pre_calculate_matrices(net)
    
    # Run CPF up to 1.65
    snapshots, logs = cpf.run_cpf(
        net,
        initial_static_matrices=static_matrices,
        distributed_slack=True,
        qlim_mode='none',
        solver_max_iter=20,
        solver_tol=0.1,
        initial_step=0.1,
        min_step=0.001,
        max_failures=10,
        max_scale=2.0,
        max_iters=50
    )
    
    V_cpf = None
    lam_cpf = 0
    for snap in snapshots:
        if abs(snap['scale'] - 1.65) < 0.05:
            V_cpf = snap['res_bus']['vm_pu'].values * np.exp(1j * np.deg2rad(snap['res_bus']['va_degree'].values))
            lam_cpf = snap['scale']
            break
            
    if V_cpf is not None:
        lam = lam_cpf
        net_spf = copy.deepcopy(net)
        load_idx = net_spf.load.index
        net_spf.load.loc[load_idx, 'p_mw'] *= lam
        net_spf.load.loc[load_idx, 'q_mvar'] *= lam
        active_gen_idx = net_spf.gen[net_spf.gen.p_mw > 1.0].index
        net_spf.gen.loc[active_gen_idx, 'p_mw'] *= lam
        
        pp.runpp(net_spf, init="results")
        V_spf = net_spf._ppc['internal']['V']

        print(f"Comparing at lam_cpf={lam_cpf}, lam_spf={lam}")
        print("Max diff in Vm:", np.max(np.abs(np.abs(V_spf) - np.abs(V_cpf))))
        print("Max diff in Va:", np.max(np.abs(np.angle(V_spf) - np.angle(V_cpf))))
        
run_compare()
