import custom_systems as cs
import pandapower as pp
import numpy as np
import analysis_tools as tools
import cpf_solver as cpf

def run_debug():
    net = cs.create_ieee30_anarede(use_taps=True)
    static_matrices = tools.pre_calculate_matrices(net)
    
    # We will instrument the predictor function
    original_predictor = cpf._predictor_tangent
    
    def my_predictor(J_conv, b, k_param, n_x, direction):
        t = original_predictor(J_conv, b, k_param, n_x, direction)
        print(f"\n[INSIDE PREDICTOR]")
        print(f"t max: {np.max(t):.5f}, t min: {np.min(t):.5f}")
        print(f"t_lam = {t[-1]:.5f}")
        return t
        
    cpf._predictor_tangent = my_predictor
    
    # Also instrument select param
    original_select = cpf._select_continuation_param
    def my_select(t_norm, n_x):
        k = original_select(t_norm, n_x)
        print(f"t_norm norm = {np.linalg.norm(t_norm):.5f}")
        print(f"t_norm max = {np.max(t_norm):.5f}, min = {np.min(t_norm):.5f}")
        if k < n_x:
            print(f"Selected k={k}, t_norm[k] = {t_norm[k]:.5f}")
        return k
    cpf._select_continuation_param = my_select

    snapshots, logs = cpf.run_cpf(
        net, initial_static_matrices=static_matrices,
        distributed_slack=True, qlim_mode='none',
        solver_max_iter=20, solver_tol=0.1,
        initial_step=0.1, min_step=0.001,
        max_failures=10, max_scale=2.0, max_iters=10
    )

run_debug()
