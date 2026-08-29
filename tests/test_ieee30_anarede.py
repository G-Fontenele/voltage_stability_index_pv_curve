import sys
import os
import custom_systems as cs
import analysis_tools as tools
import simulation_engine as sim

def run_test():
    print("Iniciando Teste Automatizado: IEEE 30 Barras (ANAREDE) - N-0 sem QLIM")
    
    # 1. Configurações Exatas (Conforme Validado)
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
    
    # 2. Criação da Rede
    net = cs.create_ieee30_anarede(use_taps=True)
    
    # 3. Pré-cálculo de Matrizes
    static_matrices = tools.pre_calculate_matrices(net)
    
    # 4. Execução
    history, log = sim.run_continuation_process(
        net,
        initial_static_matrices=static_matrices,
        **CONFIG
    )
    
    if not history:
        print("FALHA: O processo de continuação não retornou nenhum histórico.")
        sys.exit(1)
        
    last_step = history[-1]
    max_load = last_step['total_load_mw']
    max_lambda = last_step['scale']
    
    print(f"\n========================================")
    print(f"RESULTADO DO TESTE:")
    print(f"========================================")
    print(f"Lambda Máximo Atingido: {max_lambda:.5f}")
    print(f"Carga Máxima Atingida: {max_load:.2f} MW")
    
    # Validação contra o valor matemático provado no ambiente (809.68 MW)
    EXPECTED_LOAD = 809.68
    TOLERANCE = 1.0 # 1 MW de tolerância
    
    if abs(max_load - EXPECTED_LOAD) <= TOLERANCE:
        print("\n[SUCESSO] O motor atingiu a carga esperada de ~809.7 MW com sucesso!")
        sys.exit(0)
    else:
        print(f"\n[ERRO] O motor atingiu {max_load:.2f} MW, mas era esperado {EXPECTED_LOAD} MW.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
