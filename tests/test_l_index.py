import numpy as np
import pytest
import pandapower as pp
from analysis_tools import pre_calculate_matrices

def test_bus_to_idx_missing_buses():
    """
    Testa a heurística bus_to_idx em um sistema 1-based não-contíguo.
    Objetivo: Mostrar que a barra final (que não bate com o índice do ppc interno) é ignorada.
    """
    net = pp.create_empty_network()
    # IDs externos = 1, 3, 5
    pp.create_bus(net, index=1, vn_kv=132, type="b")
    pp.create_bus(net, index=3, vn_kv=132, type="b")
    pp.create_bus(net, index=5, vn_kv=132, type="b")
    
    pp.create_ext_grid(net, bus=1)
    pp.create_load(net, bus=3, p_mw=10)
    pp.create_load(net, bus=5, p_mw=10)
    
    pp.create_line(net, from_bus=1, to_bus=3, length_km=1, std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=3, to_bus=5, length_km=1, std_type="NAYY 4x50 SE")
    
    # Ao calcular matrizes estáticas, 'bus_to_idx' tentará mapear as barras
    matrices = pre_calculate_matrices(net)
    bus_to_idx = matrices['bus_to_idx']
    
    assert 5 in bus_to_idx, "Barra 5 desapareceu do mapeamento!"
    assert len(bus_to_idx) == 3, "Quantidade errada de barras mapeadas."
    assert matrices['idx_load'] is not None
    assert len(matrices['idx_load']) == 2 # Barras 3 e 5
