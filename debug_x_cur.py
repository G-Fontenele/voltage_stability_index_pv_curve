import custom_systems as cs
import pandapower as pp
import numpy as np
import analysis_tools as tools
import cpf_solver as cpf

def run_debug():
    net = cs.create_ieee30_anarede(use_taps=True)
    static_matrices = tools.pre_calculate_matrices(net)
    
    original_corrector = cpf._corrector_newton
    
    def my_corrector(net, x_pred, lam_pred, ppc_base, baseMVA, pvpq, pq, ref, 
                     base_p_load, base_q_load, active_gen_idx, base_p_gen, 
                     distributed_slack, max_iter, tol, k_param, target_k):
                     
        x, lam, iters, success = original_corrector(net, x_pred, lam_pred, ppc_base, baseMVA, pvpq, pq, ref, 
                     base_p_load, base_q_load, active_gen_idx, base_p_gen, 
                     distributed_slack, max_iter, tol, k_param, target_k)
                     
        print(f"\n[AFTER CORRECTOR] lam={lam:.5f}, success={success}")
        if success:
            print(f"x[28] = {x[28]:.5f} rad ({np.rad2deg(x[28]):.2f} deg)")
            print(f"x[44] = {x[44]:.5f} pu")
        return x, lam, iters, success
        
    cpf._corrector_newton = my_corrector

    cpf.run_cpf(
        net, initial_static_matrices=static_matrices,
        distributed_slack=True, qlim_mode='none',
        solver_max_iter=20, solver_tol=0.1,
        initial_step=0.1, min_step=0.001,
        max_failures=10, max_scale=2.0, max_iters=10
    )

run_debug()
