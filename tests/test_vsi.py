import numpy as np
import pytest
from vsi_lib import calculate_nvsi, calculate_fvsi, calculate_lmn, calculate_nlsi

def test_nvsi_formula():
    """
    Verifica se a fórmula do NVSI permite denominadores negativos (indicando colapso)
    e não dobra o gráfico artificialmente com np.abs().
    """
    V_s = np.array([1.0])
    X = np.array([0.1])
    P_r = np.array([0.0])
    
    # Q_r = 5.0 (2*Q_r*X = 1.0, V_s^2 = 1.0). Denominador = 0
    Q_r_crit = np.array([5.0])
    nvsi_crit = calculate_nvsi(V_s, X, P_r, Q_r_crit)
    
    # Q_r = 6.0 (2*Q_r*X = 1.2, V_s^2 = 1.0). Denominador = 1.2 - 1.0 = 0.2 (positivo na fórmula corrigida?)
    # A fórmula corrigida de NVSI na ref é denom = (V_s**2 - 2*X*Q_r) ou (2*X*Q_r - V_s**2)?
    # Original (NVSI = 2X * S / (V_s^2 - 2X*Q)).
    # Se usarmos denom = V_s**2 - 2X*Q_r: 
    # Q_r=6 -> 1.0 - 1.2 = -0.2 (negativo)
    
    Q_r_post = np.array([6.0])
    nvsi_post = calculate_nvsi(V_s, X, P_r, Q_r_post)
    
    # Após a correção do np.abs, se Q_r = 6.0, o denominador é 1 - 1.2 = -0.2 (negativo)
    # val = (0.2 * 6.0) / -0.2 = 1.2 / -0.2 = -6.0
    # nvsi_post = -6.0
    assert nvsi_post[0] < 0, "NVSI deveria sinalizar limite excedido (negativo) ou explodir, mas retornou valor positivo normal!"
    
def test_fvsi_direction():
    """
    Testa FVSI com injeção reativa (Q_r < 0).
    Se não for usado abs(), FVSI deve refletir a estabilidade extra (valor negativo).
    """
    V_s = np.array([1.0])
    X = np.array([0.1])
    Z = np.array([0.1]) # R=0
    
    Q_r_load = np.array([1.0])
    fvsi_load = calculate_fvsi(V_s, X, Q_r_load, Z)
    assert fvsi_load > 0
    
    Q_r_gen = np.array([-1.0]) # Compensação reativa superior à carga
    fvsi_gen = calculate_fvsi(V_s, X, Q_r_gen, Z)
    # A fórmula puramente matemática de FVSI manterá o sinal de Q_r
    # FVSI < 0 significa que a linha está suportando a tensão.
    assert fvsi_gen[0] < 0
