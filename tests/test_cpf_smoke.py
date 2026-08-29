import sys
import custom_systems as cs
import analysis_tools as tools
import simulation_engine as sim

def run_cpf_test():
    """Smoke test do CPF verdadeiro no IEEE 30 ANAREDE."""
    print("=" * 60)
    print("SMOKE TEST: CPF Verdadeiro (Preditor-Corretor)")
    print("=" * 60)

    CONFIG = {
        'load_scaling_bus_id': None,
        'qlim_mode': 'none',
        'distributed_slack': True,
        'max_scale': 3.0,          # limite reduzido para smoke test rapido
        'initial_step': 0.05,
        'min_step': 0.001,
        'max_iters': 200,
        'max_failures': 10,
        'solver_max_iter': 20,
        'solver_tol': 0.1,
        'cpf_mode': 'cpf'
    }

    net = cs.create_ieee30_anarede(use_taps=True)
    static_matrices = tools.pre_calculate_matrices(net)

    history, log = sim.run_continuation_process(
        net,
        initial_static_matrices=static_matrices,
        load_scaling_bus_id=CONFIG['load_scaling_bus_id'],
        max_scale=CONFIG['max_scale'],
        initial_step=CONFIG['initial_step'],
        min_step=CONFIG['min_step'],
        max_iters=CONFIG['max_iters'],
        max_failures=CONFIG['max_failures'],
        qlim_mode=CONFIG['qlim_mode'],
        distributed_slack=CONFIG['distributed_slack'],
        solver_max_iter=CONFIG['solver_max_iter'],
        solver_tol=CONFIG['solver_tol'],
        cpf_mode=CONFIG['cpf_mode']
    )

    if not history:
        print("\n[FALHA] CPF nao retornou historico.")
        sys.exit(1)

    last = history[-1]
    print(f"\n{'='*60}")
    print(f"RESULTADO CPF:")
    print(f"  Lambda Max: {last['scale']:.5f}")
    print(f"  Carga Max:  {last['total_load_mw']:.2f} MW")
    print(f"  Pontos:     {len(history)}")
    convergentes = sum(1 for r in log if r['status'] == 'Convergente')
    divergentes = sum(1 for r in log if r['status'] == 'Divergente')
    print(f"  Convergentes: {convergentes} | Divergentes: {divergentes}")
    print(f"{'='*60}")

    if last['total_load_mw'] > 500:
        print("\n[SUCESSO] CPF funcionando - carga acima de 500 MW!")
        sys.exit(0)
    else:
        print(f"\n[FALHA] Carga muito baixa: {last['total_load_mw']:.2f} MW")
        sys.exit(1)

if __name__ == "__main__":
    run_cpf_test()
