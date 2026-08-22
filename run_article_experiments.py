import os
import tempfile
os.environ["NUMBA_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "numba_cache")

import time
import shutil
import pandapower.networks as pn
import custom_systems as cs
import copy
import pandas as pd
from main import run_scenario, adjust_generator_participation

# ==============================================================================
# SCRIPT DE EXECUÇÃO AUTOMATIZADA - WCNPS 2026
# ==============================================================================

def main():
    print("="*80)
    print(" INICIANDO BATERIA DE EXPERIMENTOS DO ARTIGO WCNPS 2026")
    print("="*80)
    start_time = time.time()

    # Definir pasta base
    base_results_dir = "outputs"
    os.makedirs(base_results_dir, exist_ok=True)

    # Configuração Padrão
    config_default = {
        'load_scaling_bus_id': None, 
        'distributed_slack': True,    
        'max_scale': 5.0,             
        'steps': 0.002,                
        'min_step': 0.00001,
        'max_iters': 2000,
        'max_failures': 15,
        'solver_max_iter': 20,
        'solver_tol': 0.1
    }

    experiments = [
        {
            "name": "IEEE 30 ANAREDE (Validação)",
            "net_func": lambda: cs.create_ieee30_anarede(use_taps=False),
            "folder": "ieee30_anarede",
            "adjust_gen": False
        },
        {
            "name": "IEEE 30 Padrão",
            "net_func": pn.case30,
            "folder": "ieee30_standard",
            "adjust_gen": True
        },
        {
            "name": "IEEE 39 New England",
            "net_func": pn.case39,
            "folder": "ieee39",
            "adjust_gen": False
        },
        {
            "name": "IEEE 57",
            "net_func": pn.case57,
            "folder": "ieee57",
            "adjust_gen": False
        },
        {
            "name": "IEEE 118",
            "net_func": pn.case118,
            "folder": "ieee118",
            "adjust_gen": False
        }
    ]

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] EXECUTANDO EXPERIMENTO: {exp['name']}")
        print("-" * 80)
        
        system_dir = os.path.join(base_results_dir, exp['folder'])
        if os.path.exists(system_dir):
            try: shutil.rmtree(system_dir)
            except: pass
            
        # --- MODO 1: N-0 sem QLIM ---
        print("\n-> [1/3] Analisando Caso Base N-0 (Sem QLIM)...")
        net_n0 = exp['net_func']()
        bus_count = len(net_n0.bus)
        if exp['adjust_gen']:
            adjust_generator_participation(net_n0)
            
        config_n0 = copy.deepcopy(config_default)
        config_n0['qlim_mode'] = 'none'
        dir_n0 = os.path.join(system_dir, "n0")
        max_scale_n0 = run_scenario(net_n0, exp['name'], dir_n0, bus_count, config_n0, scenario_name="base_n0")
        print(f"   Max Scale N-0 (Sem QLIM): {max_scale_n0}")

        # --- MODO 2: N-0 com QLIM (Pandapower) ---
        print("\n-> [2/3] Analisando Caso Base N-0 (Com QLIM Pandapower)...")
        net_n0_pp = exp['net_func']()
        if exp['adjust_gen']:
            adjust_generator_participation(net_n0_pp)
            
        config_n0_pp = copy.deepcopy(config_default)
        config_n0_pp['qlim_mode'] = 'pandapower'
        dir_n0_pp = os.path.join(system_dir, "n0_qlim_pp")
        max_scale_n0_pp = run_scenario(net_n0_pp, exp['name'], dir_n0_pp, bus_count, config_n0_pp, scenario_name="base_n0_qlim_pp")
        print(f"   Max Scale N-0 (QLIM PP): {max_scale_n0_pp}")

        # --- MODO 3: N-1 com QLIM Avançado ---
        print("\n-> [3/3] Iniciando Análise N-1 (Com Conversão PV->PQ)...")
        net_n1 = exp['net_func']()
        if exp['adjust_gen']:
            adjust_generator_participation(net_n1)
            
        config_n1 = copy.deepcopy(config_default)
        config_n1['qlim_mode'] = 'pv_to_pq'
        
        dir_n1 = os.path.join(system_dir, "n1")
        contingency_dir = os.path.join(dir_n1, "contingencies")
        os.makedirs(contingency_dir, exist_ok=True)
        
        ranking = []
        total_lines = len(net_n1.line)
        for count, branch_idx in enumerate(net_n1.line.index, 1):
            print(f"   Testando Linha {branch_idx} ({count}/{total_lines})...")
            net_cont = copy.deepcopy(net_n1)
            net_cont.line.at[branch_idx, 'in_service'] = False
            
            cont_output_dir = os.path.join(contingency_dir, f"line_{branch_idx}")
            try:
                max_scale_cont = run_scenario(net_cont, f"{exp['name']} (S/ L{branch_idx})", cont_output_dir, bus_count, config_n1, scenario_name=f"cont_L{branch_idx}")
                if max_scale_cont is not None:
                    ranking.append({'Linha': branch_idx, 'Lambda_Max': max_scale_cont})
            except Exception as e:
                print(f"   [ERRO] Contingência linha {branch_idx}: {e}")
        
        if ranking:
            df_ranking = pd.DataFrame(ranking)
            df_ranking = df_ranking.sort_values(by='Lambda_Max').reset_index(drop=True)
            ranking_path = os.path.join(contingency_dir, "ranking_contingencias.csv")
            df_ranking.to_csv(ranking_path, index=False)
            
            pior_caso = df_ranking.iloc[0]
            print(f"\n*** PIOR CASO N-1 ENCONTRADO: Linha {int(pior_caso['Linha'])} com Lambda Max = {pior_caso['Lambda_Max']:.4f} ***")

    total_elapsed = time.time() - start_time
    mins, secs = int(total_elapsed // 60), total_elapsed % 60
    print(f"\n{'='*80}")
    print(f"BATERIA DE EXPERIMENTOS WCNPS CONCLUÍDA EM {mins}m {secs:.2f}s!")
    print(f"Todos os resultados salvos em: {base_results_dir}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
