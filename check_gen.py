import custom_systems as cs
import pandapower as pp

def run():
    net = cs.create_ieee30_anarede(use_taps=True)
    pp.runpp(net)
    ppc = net._ppc
    
    print("ext_grid:", net.ext_grid.bus.values)
    print("gen:", net.gen.bus.values)
    print(net.gen[['bus', 'p_mw', 'vm_pu']])
    
    print("\nppc['gen'] (bus, Pg, Qg):")
    for i in range(ppc['gen'].shape[0]):
        print(f"Gen {i}: bus={ppc['gen'][i, 0]}, Pg={ppc['gen'][i, 1]}, Qg={ppc['gen'][i, 2]}")

run()
