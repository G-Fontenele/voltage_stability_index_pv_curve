# Salve este arquivo como: vsi_library.py

import math
import numpy as np
from numpy.linalg import inv

# --- Funções Auxiliares ---

def get_line_params(R, X):
    """Calcula Z (magnitude) e theta (ângulo) da impedância da linha."""
    Z = math.sqrt(R**2 + X**2)
    theta = math.atan2(X, R)
    return Z, theta

def get_load_params(P_r, Q_r):
    """Calcula S_r (potência aparente) e phi (ângulo) da carga."""
    S_r = math.sqrt(P_r**2 + Q_r**2)
    phi = math.atan2(Q_r, P_r) if P_r != 0 else (math.pi / 2 * np.sign(Q_r))
    return S_r, phi

def get_angle_diff(delta_s, delta_r):
    """Calcula a diferença de ângulo delta."""
    return delta_s - delta_r

# --- 1. Índices de Estabilidade de Tensão de Linha (Line VSIs) ---
# [cite: 254-512]

def calculate_fvsi(V_s, X, Q_r, Z):
    """
    Calcula o Fast Voltage Stability Index (FVSI).
    [cite: 263] Estável se FVSI < 1.
    """
    if V_s == 0 or X == 0:
        return np.nan
    # Garante que Q_r seja positivo para o cálculo (índice foca em carga)
    Q_r_load = max(Q_r, 0)
    return (4 * Z**2 * Q_r_load) / (V_s**2 * X)

def calculate_Lmn(V_s, X, Q_r, theta, delta):
    """
    Calcula o Line Stability Index (L_mn).
    [cite: 305] Estável se L_mn < 1.
    """
    # Garante que Q_r seja positivo para o cálculo
    Q_r_load = max(Q_r, 0)
    denominator = (V_s * math.sin(theta - delta))**2
    if denominator == 0:
        return np.nan
    return (4 * X * Q_r_load) / denominator

def calculate_NLSI(V_s, P_r, R, Q_r, X):
    """
    Calcula o Novel Line Stability Index (NLSI).
    [cite: 287] Estável se NLSI < 1.
    """
    if V_s == 0:
        return np.nan
    # O índice considera o fluxo P e Q na barra receptora
    return (P_r * R + Q_r * X) / (0.25 * V_s**2)

def calculate_NVSI(V_s, X, P_r, Q_r):
    """
    Calcula o New Voltage Stability Index (NVSI).
    [cite: 334] Estável se NVSI < 1.
    """
    S_r, _ = get_load_params(P_r, Q_r)
    denominator = (2 * Q_r * X - V_s**2)
    if denominator == 0:
        return np.nan
    return abs((2 * X * S_r) / denominator) # abs() para lidar com numeradores/denominadores negativos

def calculate_VSLI(V_s, V_r, delta):
    """
    Calcula o Voltage Stability Load Index (VSLI).
    [cite: 344, 404] Estável se VSLI < 1.
    """
    if V_s == 0:
        return np.nan
    cos_delta = math.cos(delta)
    if cos_delta < 0: # Cenário fisicamente improvável, mas evita sqrt(negativo)
        return np.nan
    return (4 * (V_s * V_r * cos_delta - V_r**2 * cos_delta**2)) / V_s**2

def calculate_VSI_2(V_s, Q_r, R, X):
    """
    Calcula o Voltage Stability Indicator (VSI_2).
    [cite: 415] Estável se VSI_2 < 1.
    """
    # Garante que Q_r seja positivo para o cálculo
    Q_r_load = max(Q_r, 0)
    denominator = X * (V_s**2 + 8 * R * Q_r_load)
    if denominator == 0:
        return np.nan
    return (4 * Q_r_load * (R + X)**2) / denominator

def calculate_SI(V_s, V_r, P_r, Q_r, R, X, Z):
    """
    Calcula o Stability Index (SI).
    [cite: 501] Estável se SI > 0.
    """
    term1 = 2 * V_s**2 * V_r**2
    term2 = V_r**4
    term3 = 2 * V_r**2 * (P_r * R + Q_r * X)
    S_r_sq = P_r**2 + Q_r**2
    term4 = (Z**2) * S_r_sq
    return term1 - term2 - term3 - term4

# --- 2. Índices de Estabilidade de Tensão de Barra (Bus VSIs) ---
# [cite: 513-724]

def calculate_VCPI_bus_per_bus(V_phasors, Y_bus):
    """
    Calcula o Voltage Collapse Prediction Index (VCPI_bus) para CADA barra.
    Baseado no Artigo Eq. (48-50) [cite: 522, 595-598, 608].
    Retorna um dicionário {bus_index: vcpi_value}.
    """
    V = np.asarray(V_phasors, dtype=complex)
    Y = np.asarray(Y_bus, dtype=complex)
    num_buses = len(V)
    vcpi_results = {}

    for i in range(num_buses):
        sum_Yij_Vj = 0
        
        for j in range(num_buses):
            if i != j:
                sum_Yij_Vj += Y[i, j] * V[j]
        
        # A formulação do artigo [cite: 595-598, 608] pode ser simplificada
        # V_tilde_m = (Y_im / sum(Y_ij)) * V_m
        # sum(V_tilde_m) = (1 / sum(Y_ij)) * sum(Y_im * V_m)
        # O termo sum(Y_im * V_m) para m != i é sum_Yij_Vj
        
        # A formulação do TCC [cite: 2244] é |1 - (sum(V_m_prime) / V_k)|
        # Onde V_m_prime = (Y_km / sum(Y_kj)) * V_m [cite: 2222]
        # Vamos seguir a formulação do TCC, que é mais clara.

        sum_Ykj = 0
        sum_V_prime = 0
        
        for j in range(num_buses):
             if i != j:
                sum_Ykj += Y[i,j] # Y_kj
        
        if sum_Ykj == 0 or V[i] == 0:
            vcpi_results[i] = np.nan
            continue
            
        for m in range(num_buses):
            if m != i:
                V_prime_m = (Y[i, m] * V[m]) / sum_Ykj
                sum_V_prime += V_prime_m
        
        # Eq. (3.53) do TCC [cite: 2244]
        vcpi_index = abs(1 - (sum_V_prime / V[i]))
        vcpi_results[i] = vcpi_index

    return vcpi_results