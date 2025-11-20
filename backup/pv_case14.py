import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def calculate_pv_curve(net, load_bus_id, load_scale_factor=1.05, steps=50):
    """
    Calcula a curva PV para um sistema de potência.

    Args:
        net (dict): A rede pandapower.
        load_bus_id (int): O ID do barramento de carga a ser escalado.
        load_scale_factor (float): O fator máximo de escala da carga.
        steps (int): O número de passos para o aumento da carga.

    Returns:
        tuple: Uma tupla contendo (potências_ativas, tensoes_barramento).
    """
    # Garante que a cópia é um objeto pandapowerNet, o que é mais robusto no ambiente sandbox
    net_copy = pp.pandapowerNet(net.copy())

    # Identifica o índice do barramento de carga na tabela de cargas
    load_indices = net_copy['load'][net_copy['load'].bus == load_bus_id].index
    if load_indices.empty:
        raise ValueError(f"Nenhuma carga encontrada no barramento {load_bus_id}")

    # Armazena os valores originais da carga
    original_p = net_copy['load'].loc[load_indices, "p_mw"].copy()
    original_q = net_copy['load'].loc[load_indices, "q_mvar"].copy()

    active_powers = []
    voltages = []

    scale_factors = np.linspace(1.0, load_scale_factor, steps)

    for scale in scale_factors:
        # Escala a carga ativa e reativa
        net_copy['load'].loc[load_indices, "p_mw"] = original_p * scale
        net_copy['load'].loc[load_indices, "q_mvar"] = original_q * scale

        try:
            # Tenta executar o fluxo de potência
            pp.runpp(net_copy, enforce_q_lims=True)
            
            # Potência ativa total injetada no sistema (slack)
            p_slack = net_copy['res_ext_grid']['p_mw'].sum()
            
            # Tensão no barramento de interesse (pu)
            v_bus = net_copy['res_bus']['vm_pu'].at[load_bus_id]
            
            active_powers.append(p_slack)
            voltages.append(v_bus)

        except pp.LoadflowNotConverged:
            print(f"Fluxo de potência não convergiu com fator de escala: {scale:.3f}")
            break
            
    return np.array(active_powers), np.array(voltages)

def plot_pv_curve(P, V, bus_name):
    """
    Plota a curva PV.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(P, V, marker='o', linestyle='-', color='b', markersize=4)
    
    if len(P) > 0:
        collapse_P = P[-1]
        collapse_V = V[-1]
        plt.plot(collapse_P, collapse_V, 'ro', markersize=8, label='Ponto de Colapso (Nariz)')
        plt.annotate(f'({collapse_P:.2f}, {collapse_V:.3f})', 
                     (collapse_P, collapse_V), 
                     textcoords="offset points", 
                     xytext=(5,-10), 
                     ha='left')

    plt.title(f'Curva PV para o Barramento {bus_name} (IEEE 14 Barras)')
    plt.xlabel('Potência Ativa Total Gerada (MW)')
    plt.ylabel(f'Tensão no Barramento {bus_name} (pu)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pv_curve_ieee14.png')
    # plt.show() # Desabilitado para execução em ambiente sem GUI

def main():
    # 1. Carregar o sistema IEEE 14 barras
    net = pn.case30()
    
    # 2. Definir o barramento de interesse para a curva PV
    # O barramento 14 (índice 13) é um bom candidato para análise.
    # O pandapower usa índices baseados em 0. O barramento 14 é o índice 13.
    bus_to_analyze_id = 13
    bus_to_analyze_name = 14 # Nome do barramento para o título do gráfico
    
    # 3. Calcular a curva PV
    print(f"Calculando a Curva PV para o Barramento {bus_to_analyze_name}...")
    # Aumenta a carga no barramento 14 (índice 13) e monitora a tensão no mesmo barramento.
    P, V = calculate_pv_curve(net, load_bus_id=bus_to_analyze_id, load_scale_factor=4.0, steps=100)
    
    # 4. Salvar os dados em um arquivo para análise futura
    data = np.column_stack((P, V))
    np.savetxt('pv_curve_data_ieee14.csv', data, delimiter=',', header='Potencia_Ativa_MW,Tensao_pu', comments='')
    print("Dados da Curva PV salvos em 'pv_curve_data_ieee14.csv'")
    
    # 5. Plotar a curva PV
    plot_pv_curve(P, V, bus_to_analyze_name)
    print("Gráfico da Curva PV salvo como 'pv_curve_ieee14.png'")

if __name__ == "__main__":
    main()