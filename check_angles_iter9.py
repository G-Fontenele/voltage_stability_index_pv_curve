import custom_systems as cs
import pandapower as pp
import numpy as np
import analysis_tools as tools
import cpf_solver as cpf

def run_check():
    net = cs.create_ieee30_anarede(use_taps=True)
    static_matrices = tools.pre_calculate_matrices(net)
    snapshots, logs = cpf.run_cpf(
        net, initial_static_matrices=static_matrices,
        distributed_slack=True, qlim_mode='none',
        solver_max_iter=20, solver_tol=0.1,
        initial_step=0.1, min_step=0.001,
        max_failures=10, max_scale=2.0, max_iters=10
    )
    for snap in snapshots:
        if abs(snap['scale'] - 1.69440) < 0.01:
            va = snap['res_bus']['va_degree'].values
            print(f"lam={snap['scale']:.5f}")
            print(f"Max Va: {np.max(va):.2f} deg, Min Va: {np.min(va):.2f} deg")
            print(f"Va of all buses (deg):\n{va}")

run_check()
