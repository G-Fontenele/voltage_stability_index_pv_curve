import custom_systems as cs
import copy
import pandapower as pp
import numpy as np

def run_compare():
    net = cs.create_ieee30_anarede(use_taps=True)
    pp.runpp(net)
    
    # SPF at lam=1.7
    lam = 1.7
    net_spf = copy.deepcopy(net)
    load_idx = net_spf.load.index
    net_spf.load.loc[load_idx, 'p_mw'] *= lam
    net_spf.load.loc[load_idx, 'q_mvar'] *= lam
    active_gen_idx = net_spf.gen[net_spf.gen.p_mw > 1.0].index
    net_spf.gen.loc[active_gen_idx, 'p_mw'] *= lam
    
    pp.runpp(net_spf, init="results")
    V_spf = net_spf._ppc['internal']['V']
    
    import sys
    sys.path.append('.')
    import cpf_solver as cpf
    
    # Run CPF up to 1.7
    snapshots, logs = cpf.run_cpf(
        net,
        distributed_slack=True,
        qlim_mode='none',
        solver_max_iter=20,
        solver_tol=0.1,
        initial_step=0.05,
        min_step=0.001,
        max_failures=10,
        max_scale=3.0,
        max_iters=100
    )
    
    V_cpf = None
    lam_cpf = 0
    for snap in snapshots:
        if abs(snap['lambda'] - 1.7) < 0.05:
            V_cpf = snap['V']
            lam_cpf = snap['lambda']
            break
            
    if V_cpf is not None:
        print(f"Comparing at lam_cpf={lam_cpf}, lam_spf=1.7")
        print("Max diff in Vm:", np.max(np.abs(np.abs(V_spf) - np.abs(V_cpf))))
        print("Max diff in Va:", np.max(np.abs(np.angle(V_spf) - np.angle(V_cpf))))
        print("SPF Vm[29]:", np.abs(V_spf[29]))
        print("CPF Vm[29]:", np.abs(V_cpf[29]))
        
run_compare()
