import custom_systems as cs
import copy
import pandapower as pp
import numpy as np

def run_compare():
    net = cs.create_ieee30_anarede(use_taps=True)
    pp.runpp(net)
    
    lam = 2.0
    net_spf = copy.deepcopy(net)
    load_idx = net_spf.load.index
    net_spf.load.loc[load_idx, 'p_mw'] *= lam
    net_spf.load.loc[load_idx, 'q_mvar'] *= lam
    
    active_gen_idx = net_spf.gen[net_spf.gen.p_mw > 1.0].index
    net_spf.gen.loc[active_gen_idx, 'p_mw'] *= lam
    
    pp.runpp(net_spf, init="results")
    ppc_spf = net_spf._ppc
    
    # Calculate mismatch using CPF functions
    import sys
    sys.path.append('.')
    import cpf_solver as cpf
    
    baseMVA = ppc_spf['baseMVA']
    pvpq = np.r_[ppc_spf['internal']['pv'], ppc_spf['internal']['pq']]
    pq = ppc_spf['internal']['pq']
    
    # Scale base ppc using lambda=2.0
    ppc_base = net._ppc
    S_inj = cpf._compute_Sbus_scaled(ppc_base, baseMVA, lam, None, None, None, [1], [], distributed_slack=True)
    
    V_spf = ppc_spf['internal']['V']
    Ybus = ppc_spf['internal']['Ybus']
    S_calc = V_spf * np.conj(Ybus.dot(V_spf))
    
    mismatch = S_inj - S_calc
    F = np.r_[mismatch[pvpq].real, mismatch[pq].imag]
    
    Vm = np.abs(V_spf)
    Va = np.angle(V_spf, deg=True)
    print("Vm:", Vm)
    print("Va:", Va)
    
run_compare()
