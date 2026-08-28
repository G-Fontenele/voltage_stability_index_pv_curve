import pandapower as pp
import pandapower.networks as pn
import copy

net = pn.case14()
pp.runpp(net)
print("Converged res_bus vm_pu (first 3):", net.res_bus.vm_pu.head(3).values)
print("Iterations:", net["_ppc"]["iterations"])

net_good = copy.deepcopy(net)

# Force divergence by absurd load
net.load.p_mw *= 1000
try:
    pp.runpp(net, init="results", max_iteration=10)
    print("Converged?")
except pp.LoadflowNotConverged:
    print("Failed as expected.")
    print("Diverged res_bus vm_pu (first 3):", net.res_bus.vm_pu.head(3).values)
    
