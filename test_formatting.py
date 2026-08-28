import sys
import custom_systems as cs
import analysis_tools as tools
import simulation_engine as sim

def run_test():
    CONFIG = {
        'load_scaling_bus_id': None, 
        'qlim_mode': 'none',      
        'distributed_slack': True,    
        'max_scale': 5.0,             
        'initial_step': 0.002,                
        'min_step': 0.00001,
        'max_iters': 2000,
        'max_failures': 15,
        'solver_max_iter': 20,
        'solver_tol': 0.1
    }
    
    net = cs.create_ieee30_anarede(use_taps=True)
    static_matrices = tools.pre_calculate_matrices(net)
    
    history, log = sim.run_continuation_process(
        net,
        initial_static_matrices=static_matrices,
        **CONFIG
    )
    
    tools.generate_convergence_report(log, "IEEE30", "teste_relatorio.txt")
    print("Relatorio gerado em teste_relatorio.txt")

if __name__ == "__main__":
    run_test()
