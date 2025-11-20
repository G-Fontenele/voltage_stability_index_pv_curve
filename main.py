import pandapower.networks as pn
import simulation_engine as sim
import analysis_tools as tools
import os
import shutil

def main():
    # --- CONFIGURAÇÃO ---
    net = pn.case30()
    system_name = "IEEE 30 Barras"
    
    CONFIG = {
        'load_scaling_bus_id': None,  # Global
        'enforce_q_lims': False,      # True para realismo, False para teste de estresse puro (TCC)
        'distributed_slack': True,    
        'max_scale': 5.0,
        'steps': 0.1,
        'min_step': 0.001
    }
    
    # Pasta de saída
    output_dir = 'tabelas_resultados'
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Pré-Cálculo de Matrizes Estáticas (NOVO) ---
    # Calcula Ybus e F-matrix uma única vez, pois a topologia não muda
    print(f"--- Inicializando Rede {system_name} ---")
    static_matrices = tools.pre_calculate_matrices(net)

    # --- 2. Simulação (Gera a Curva PV) ---
    print(f"\n--- Simulando Fluxo Continuado ---")
    history = sim.run_continuation_process(
        net, 
        load_scaling_bus_id=CONFIG['load_scaling_bus_id'],
        max_scale=CONFIG['max_scale'],
        initial_step=CONFIG['steps'],
        min_step=CONFIG['min_step'],
        enforce_q_lims=CONFIG['enforce_q_lims'],
        distributed_slack=CONFIG['distributed_slack']
    )
    
    if not history: return

    # --- 3. Extração e Cálculo ---
    scenarios = sim.extract_scenarios(history, [0, 25, 50, 75, 95, 99, 100])
    all_results = {}
    
    print("\n--- Calculando Índices (Otimizado) ---")
    for pct, snapshot in scenarios.items():
        # Agora passamos as matrizes pré-calculadas
        df = tools.calculate_indices_for_scenario(snapshot, static_matrices)
        all_results[pct] = df
        
        try:
            df.to_csv(os.path.join(output_dir, f'resultados_{pct}_pct.csv'), index=False)
        except: pass

    # --- 4. Gráficos ---
    print("\n--- Gerando Visualizações ---")
    # Gráfico da Curva PV (Adicionado conforme pedido anterior)
    sim.tools.plot_pv_curves(history, title=f"Curva PV - {system_name}") # Precisa importar tools dentro do sim ou usar local
    # Correção: a função de plotar PV está no analysis_tools, use tools.plot_pv_curves
    # Vou corrigir a chamada abaixo:
    
    tools.plot_pv_curves(history, title=f"Curva PV - {system_name}")
    tools.plot_comparative_indices(all_results)

    print("\nConcluído.")

if __name__ == "__main__":
    main()