import custom_systems as cs
import pandapower as pp
from pandapower.powerflow import _ppc2pd

net = cs.create_ieee30_anarede(use_taps=True)
pp.runpp(net)

ppc = net._ppc
# Modify V slightly
ppc['internal']['V'] *= 0.95

# Calculate results
_ppc2pd(net, ppc)

print(net.res_bus.vm_pu.head())
print(net.res_line.p_from_mw.head())
