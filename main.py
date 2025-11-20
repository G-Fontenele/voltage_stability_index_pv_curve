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
        'load_scaling_bus_id': None,  # Global (Correto: escala todo o sistema)
        
        # MUDANÇA 1: Desligar limites para ter Q infinito (como no TCC)
        'enforce_q_lims': False,      
        
        # MUDANÇA 2: Ativar para que a Barra 2 ajude a Barra 1 (como no TCC)
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
    
    # Correção aqui: chamamos direto do módulo 'tools', não 'sim.tools'
    try:
        tools.plot_pv_curves(history, title=f"Curva PV - {system_name}")
        tools.plot_comparative_indices(all_results)
    except Exception as e:
        print(f"Erro ao gerar gráficos: {e}")

    print("\nConcluído.")

if __name__ == "__main__":
    main()