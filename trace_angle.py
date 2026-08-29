import custom_systems as cs
import copy
import pandapower as pp
import numpy as np
import analysis_tools as tools
import simulation_engine as sim

def run_trace():
    net = cs.create_ieee30_anarede(use_taps=True)
    static_matrices = tools.pre_calculate_matrices(net)
    
    # Run SPF
    hist_spf, _ = sim.run_continuation_process(
        net,
        initial_static_matrices=static_matrices,
        cpf_mode='spf',
        distributed_slack=True,
        qlim_mode='none',
        max_scale=3.0,
        initial_step=0.1
    )
    
    print("SPF Trace for Bus 29 (Index 28):")
    for snap in hist_spf:
        lam = snap['scale']
        v_bus = snap['res_bus']
        va = v_bus.at[29, 'va_degree']
        vm = v_bus.at[29, 'vm_pu']
        print(f"lam={lam:.4f}, Vm={vm:.4f}, Va={va:.4f}")

run_trace()
