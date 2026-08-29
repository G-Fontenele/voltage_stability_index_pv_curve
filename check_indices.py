import custom_systems as cs
import analysis_tools as tools
import pandapower as pp
import numpy as np

def run_check():
    net = cs.create_ieee30_anarede(use_taps=True)
    pp.runpp(net)
    ppc = net._ppc
    
    pv = np.where(ppc['bus'][:, 1] == 2)[0]
    pq = np.where(ppc['bus'][:, 1] == 1)[0]
    ref = np.where(ppc['bus'][:, 1] == 3)[0]
    
    print(f"Total buses: {len(ppc['bus'])}")
    print(f"len(pv) = {len(pv)}")
    print(f"len(pq) = {len(pq)}")
    print(f"len(ref) = {len(ref)}")
    
    pvpq = np.sort(np.concatenate((pv, pq)))
    print(f"len(pvpq) = {len(pvpq)}")
    
    x = np.r_[np.deg2rad(ppc['bus'][pvpq, 8]), ppc['bus'][pq, 7]]
    print(f"len(x) = {len(x)}")
    
    # What is x[44]?
    if len(x) > 44:
        print(f"x[44] is: {x[44]}")
        if 44 < len(pvpq):
            print(f"Index 44 is ANGLE of bus {pvpq[44]}")
        else:
            print(f"Index 44 is MAGNITUDE of bus {pq[44 - len(pvpq)]}")
            
run_check()
