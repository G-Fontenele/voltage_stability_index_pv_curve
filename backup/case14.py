# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:24:36 2025

@author: gonca_pc
"""

import pandapower as pp
import pandas as pd

def run_pandapower_ieee14():
    # 1. Criar a rede IEEE 14 barras padrão do pandapower
    net = pp.networks.case14()

    # 2. Rodar o fluxo de potência com as configurações padrão
    # O pandapower usa o método Newton-Raphson por padrão (runpp)
    pp.runpp(net)

    # 3. Preparar a saída no formato Anarede (tabela de barras)
    # Selecionar as colunas relevantes para a saída de barras:
    # - bus: Índice da barra
    # - vm_pu: Tensão em pu
    # - va_degree: Ângulo da tensão em graus
    # - p_mw: Potência ativa injetada (geração - carga)
    # - q_mvar: Potência reativa injetada (geração - carga)
    
    # O pandapower armazena os resultados em net.res_bus
    res_bus = net.res_bus.copy()
    
    # Adicionar o índice da barra como uma coluna
    res_bus['bus'] = res_bus.index
    
    # Calcular as potências injetadas (Geração - Carga)
    # net.res_bus.p_mw e net.res_bus.q_mvar são as potências injetadas (P_inj = P_G - P_L)
    
    # Selecionar e renomear as colunas para um formato mais limpo e semelhante ao Anarede
    output_df = res_bus[['bus', 'vm_pu', 'va_degree', 'p_mw', 'q_mvar']]
    
    # Renomear colunas para clareza
    output_df.columns = ['Barra', 'V (pu)', 'Angulo (graus)', 'P Injetada (MW)', 'Q Injetada (Mvar)']
    
    # Formatar a saída como uma string para impressão
    # Usar to_string para garantir que todas as linhas e colunas sejam exibidas
    # e formatar os números para melhor legibilidade (ex: 4 casas decimais)
    output_string = output_df.to_string(
        index=False, 
        float_format=lambda x: f"{x:.4f}"
    )

    print("--- Resultado do Fluxo de Potência (IEEE 14 Barras) ---")
    print(output_string)
    print("------------------------------------------------------")

if __name__ == "__main__":
    run_pandapower_ieee14()