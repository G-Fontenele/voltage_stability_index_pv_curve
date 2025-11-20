import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# Ignorar o UserWarning sobre o geojson
warnings.filterwarnings("ignore", category=UserWarning)

def calculate_multi_pv_curve(net, load_scaling_bus_id, monitored_bus_ids, load_scale_factor=1.05, steps=50):
    """
    Calcula múltiplas curvas PV para um sistema de potência, escalando a carga em um
    barramento e monitorando a tensão em múltiplos barramentos.

    Args:
        net (dict): A rede pandapower.
        load_scaling_bus_id (int): O ID do barramento de carga a ser escalado.
        monitored_bus_ids (list): Lista de IDs dos barramentos a serem monitorados.
        load_scale_factor (float): O fator máximo de escala da carga.
        steps (int): O número de passos para o aumento da carga.

    Returns:
        tuple: Uma tupla contendo (potências_ativas, tensoes_barramento_df).
               tensoes_barramento_df é um DataFrame com as tensões de cada barramento monitorado.
    """
    net_copy = pp.pandapowerNet(net.copy())

    # Identifica o índice do barramento de carga na tabela de cargas
    load_indices = net_copy['load'][net_copy['load'].bus == load_scaling_bus_id].index
    if load_indices.empty:
        raise ValueError(f"Nenhuma carga encontrada no barramento {load_scaling_bus_id}")

    # Armazena os valores originais da carga
    original_p = net_copy['load'].loc[load_indices, "p_mw"].copy()
    original_q = net_copy['load'].loc[load_indices, "q_mvar"].copy()

    active_powers = []
    voltages_data = {bus_id: [] for bus_id in monitored_bus_ids}

    scale_factors = np.linspace(1.0, load_scale_factor, steps)

    for i, scale in enumerate(scale_factors):
        # Escala a carga ativa e reativa
        net_copy['load'].loc[load_indices, "p_mw"] = original_p * scale
        net_copy['load'].loc[load_indices, "q_mvar"] = original_q * scale

        try:
            # Tenta executar o fluxo de potência
            pp.runpp(net_copy, enforce_q_lims=True)
            
            # Potência ativa total injetada no sistema (slack)
            p_slack = net_copy['res_ext_grid']['p_mw'].sum()
            
            # Tensão no barramento de interesse (pu)
            for bus_id in monitored_bus_ids:
                v_bus = net_copy['res_bus']['vm_pu'].at[bus_id]
                voltages_data[bus_id].append(v_bus)
            
            active_powers.append(p_slack)
            
            # Adiciona um print de progresso para o usuário
            v_min = min(voltages_data[bus_id][-1] for bus_id in monitored_bus_ids)
            print(f"  -> Passo {i+1}/{steps}: Fator de escala = {scale:.3f}, Tensão Mínima = {v_min:.3f} pu")

        except pp.LoadflowNotConverged:
            print(f"Fluxo de potência não convergiu com fator de escala: {scale:.3f}")
            break
            
    # Converte os dados de tensão para um DataFrame
    voltages_df = pd.DataFrame(voltages_data)
    
    return np.array(active_powers), voltages_df

def plot_multi_pv_curve(P, V_df, load_scaling_bus_name, system_name):
    """
    Plota múltiplas curvas PV em um único gráfico.
    """
    plt.figure(figsize=(12, 8))
    
    # Cores para as curvas
    colors = plt.cm.jet(np.linspace(0, 1, len(V_df.columns)))
    
    # Encontra o barramento mais crítico (menor tensão final)
    if not V_df.empty:
        last_voltages = V_df.iloc[-1]
        critical_bus_id = last_voltages.idxmin()
        critical_bus_name = critical_bus_id + 1
    else:
        critical_bus_id = None
        critical_bus_name = "N/A"

    for i, bus_id in enumerate(V_df.columns):
        # O pandapower usa índices baseados em 0. O barramento 14 é o índice 13.
        # Para o usuário, é mais intuitivo o número do barramento (1 a 14).
        bus_name = bus_id + 1
        V = V_df[bus_id].values
        
        # Plota a curva
        plt.plot(P[:len(V)], V, marker='.', linestyle='-', color=colors[i], markersize=4, label=f'V, Barramento {bus_name}')
        
        # Encontra o ponto de colapso (último ponto de convergência)
        if len(V) > 0 and len(P) > len(V):
            collapse_P = P[len(V)-1]
            collapse_V = V[-1]
            # Marca o ponto de colapso
            plt.plot(collapse_P, collapse_V, 'o', color=colors[i], markersize=6)
            
            # Adiciona anotação apenas para o barramento mais crítico
            if bus_id == critical_bus_id:
                 plt.annotate(f'Ponto Crítico: ({collapse_P:.2f}, {collapse_V:.3f})', 
                              (collapse_P, collapse_V), 
                              textcoords="offset points", 
                              xytext=(5,-10), 
                              ha='left',
                              color='red',
                              fontweight='bold')

    plt.title(f'Curvas PV - Escalonamento de Carga no Barramento {load_scaling_bus_name} ({system_name})')
    plt.xlabel('Potência Ativa Total Gerada (MW)')
    plt.ylabel('Tensão (pu)')
    plt.grid(True)
    plt.legend(loc='lower left', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig('all_pv_curve_ieee14.png')
    # plt.show() # Desabilitado para execução em ambiente sem GUI
    
    print(f"\nBarramento mais crítico (menor tensão final): Barramento {critical_bus_name}")

def main():
    # 1. Carregar o sistema IEEE 14 barras
    net = pn.case14()
    system_name = "IEEE 14 Barras"
    
    # 2. Definir o barramento onde a carga será escalada
    # Barramento 14 (índice 13) é um bom barramento de carga para escalonar.
    load_scaling_bus_id = 13
    load_scaling_bus_name = 14
    
    # 3. Definir os barramentos a serem monitorados (TODOS os barramentos)
    # O pandapower usa índices de 0 a 13 para o IEEE 14.
    monitored_bus_ids = list(net.bus.index)
    
    # 4. Calcular as múltiplas curvas PV
    print(f"Calculando Curvas PV para TODOS os Barramentos, escalando a carga no Barramento {load_scaling_bus_name} ({system_name})...")
    # Aumenta a carga no barramento 14 (índice 13) e monitora a tensão em todos os barramentos.
    P, V_df = calculate_multi_pv_curve(net, 
                                       load_scaling_bus_id=load_scaling_bus_id, 
                                       monitored_bus_ids=monitored_bus_ids,
                                       load_scale_factor=4.0, 
                                       steps=100)
    
    # 5. Salvar os dados em um arquivo para análise futura
    # Combina a potência ativa com as tensões em um único DataFrame
    data_df = pd.DataFrame({'Potencia_Ativa_MW': P[:len(V_df)]})
    data_df = pd.concat([data_df, V_df.rename(columns=lambda x: f'Tensao_Barramento_{x+1}_pu')], axis=1)
    data_df.to_csv('all_pv_curve_data_ieee14.csv', index=False)
    print("Dados das Curvas PV salvos em 'all_pv_curve_data_ieee14.csv'")
    
    # 6. Plotar as múltiplas curvas PV
    plot_multi_pv_curve(P, V_df, load_scaling_bus_name, system_name)
    print("Gráfico das Curvas PV salvo como 'all_pv_curve_ieee14.png'")

if __name__ == "__main__":
    main()
