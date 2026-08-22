"""
Teste de validação: IEEE 30 ANAREDE (sem QLIM)
Valores de referência confirmados por teste lado-a-lado (old vs new engine):
  - Lambda máximo (scale): 2.85313
  - Carga no colapso: 808.58 MW
  - Steps convergidos: 930
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import custom_systems as cs
import simulation_engine as sim
import analysis_tools as tools

def test_ieee30_anarede_n0_no_qlim():
    net = cs.create_ieee30_anarede(use_taps=False)
    static_matrices = tools.pre_calculate_matrices(net)
    
    history, full_log = sim.run_continuation_process(
        net,
        initial_static_matrices=static_matrices,
        load_scaling_bus_id=None,
        max_scale=5.0,
        initial_step=0.002,
        min_step=0.00001,
        max_iters=2000,
        max_failures=15,
        distributed_slack=True,
        qlim_mode='none',
        solver_max_iter=50,
        solver_tol=1e-6
    )
    
    assert len(history) > 0, "Simulação não convergiu no caso base"
    
    last = history[-1]
    lambda_max = last['scale']
    load_mw = last['total_load_mw']
    
    # Tolerâncias: 0.5% para lambda, 1 MW para carga
    assert abs(lambda_max - 2.85313) < 0.015, \
        f"Lambda máximo {lambda_max:.5f} fora da tolerância (esperado ≈ 2.85313)"
    assert abs(load_mw - 808.58) < 1.0, \
        f"Carga no colapso {load_mw:.2f} MW fora da tolerância (esperado ≈ 808.58 MW)"
    assert len(history) == 930, \
        f"Número de steps convergidos {len(history)} diferente do esperado (930)"
