import custom_systems as cs
import analysis_tools as tools
import copy
import pandapower as pp
import numpy as np

def run_compare():
    net = cs.create_ieee30_anarede(use_taps=True)
    
    # Base
    pp.runpp(net)
    ppc_base = copy.deepcopy(net._ppc)
    
    # SPF at lam = 2.0
    lam = 2.0
    net_spf = copy.deepcopy(net)
    load_idx = net_spf.load.index
    base_p_load = net_spf.load.loc[load_idx, 'p_mw'].copy()
    base_q_load = net_spf.load.loc[load_idx, 'q_mvar'].copy()
    
    active_gen_idx = net_spf.gen[net_spf.gen.p_mw > 1.0].index
    base_p_gen = net_spf.gen.loc[active_gen_idx, 'p_mw'].copy()
    
    net_spf.load.loc[load_idx, 'p_mw'] = base_p_load * lam
    net_spf.load.loc[load_idx, 'q_mvar'] = base_q_load * lam
    net_spf.gen.loc[active_gen_idx, 'p_mw'] = base_p_gen * lam
    
    print("Running SPF at lam=2.0...")
    try:
        pp.runpp(net_spf)
        print("SPF converged!")
        ppc_spf = net_spf._ppc
        print("Slack power SPF:", ppc_spf['gen'][0, 1])
        print("PV Gen power SPF:", ppc_spf['gen'][1:, 1])
    except Exception as e:
        print("SPF diverged!", e)
        return

run_compare()
