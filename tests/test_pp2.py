import pandapower as pp
import pandapower.networks as pn
import copy
import numpy as np

net = pn.case14()
pp.runpp(net)
res_bus_good = copy.deepcopy(net.res_bus)

net.load.p_mw *= 1000
try:
    pp.runpp(net, init="results", max_iteration=20)
except pp.LoadflowNotConverged:
    pass

is_same_vm = np.allclose(res_bus_good.vm_pu.values, net.res_bus.vm_pu.values, equal_nan=True)
is_same_va = np.allclose(res_bus_good.va_degree.values, net.res_bus.va_degree.values, equal_nan=True)

print("vm_pu is exactly same:", is_same_vm)
print("va_degree is exactly same:", is_same_va)
print("Good va:", res_bus_good.va_degree.values[:3])
print("Bad va:", net.res_bus.va_degree.values[:3])
