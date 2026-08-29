import custom_systems as cs
import pandapower as pp
import numpy as np

import analysis_tools as tools
import cpf_solver as cpf

def check_voltages():
    net = cs.create_ieee30_anarede(use_taps=True)
    pp.runpp(net)
    
    static_matrices = tools.pre_calculate_matrices(net)
    
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
        max_iters=15
    )
    
    for snap in snapshots:
        lam = snap['scale']
        vm = snap['res_bus'].at[21, 'vm_pu']
        va = snap['res_bus'].at[21, 'va_degree']
        print(f"lam={lam:.5f} -> Bus 21 Vm = {vm:.5f}, Va = {va:.5f}")

check_voltages()
