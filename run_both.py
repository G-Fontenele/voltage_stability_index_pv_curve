import custom_systems as cs
import analysis_tools as tools
import simulation_engine as sim

def run_both():
    net = cs.create_ieee30_anarede(use_taps=True)
    static_matrices = tools.pre_calculate_matrices(net)
    
    # Run CPF
    print("Running CPF...")
    hist_cpf, _ = sim.run_continuation_process(
        net,
        initial_static_matrices=static_matrices,
        cpf_mode='cpf',
        distributed_slack=True,
        qlim_mode='none',
        max_scale=4.0
    )
    print("CPF Max Lambda:", hist_cpf[-1]['scale'])
    
    # Run SPF
    print("Running SPF...")
    hist_spf, _ = sim.run_continuation_process(
        net,
        initial_static_matrices=static_matrices,
        cpf_mode='spf',
        distributed_slack=True,
        qlim_mode='none',
        max_scale=4.0,
        initial_step=0.05
    )
    print("SPF Max Lambda:", hist_spf[-1]['scale'])

run_both()
