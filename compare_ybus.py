import custom_systems as cs
import copy
import pandapower as pp
import numpy as np

def run_compare():
    net = cs.create_ieee30_anarede(use_taps=True)
    pp.runpp(net)
    ppc_base = net._ppc
    
    lam = 2.0
    net_spf = copy.deepcopy(net)
    load_idx = net_spf.load.index
    net_spf.load.loc[load_idx, 'p_mw'] *= lam
    net_spf.load.loc[load_idx, 'q_mvar'] *= lam
    
    active_gen_idx = net_spf.gen[net_spf.gen.p_mw > 1.0].index
    net_spf.gen.loc[active_gen_idx, 'p_mw'] *= lam
    
    pp.runpp(net_spf, init="results")
    ppc_spf = net_spf._ppc
    
    Ybus_base = ppc_base['internal']['Ybus']
    Ybus_spf = ppc_spf['internal']['Ybus']
    
    diff = np.max(np.abs(Ybus_base - Ybus_spf))
    print("Max diff in Ybus:", diff)
    
run_compare()
