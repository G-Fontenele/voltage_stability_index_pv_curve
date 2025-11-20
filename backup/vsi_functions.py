import math
import numpy as np
from numpy.linalg import inv

# --- Funções Auxiliares ---
# Estas funções calculam valores comuns usados em múltiplos índices
# a partir dos parâmetros básicos de um modelo de 2 barras.

def get_line_params(R, X):
    """Calcula Z (magnitude) e theta (ângulo) da impedância da linha."""
    Z = math.sqrt(R**2 + X**2)
    theta = math.atan2(X, R)
    return Z, theta

def get_load_params(P_r, Q_r):
    """Calcula S_r (potência aparente) e phi (ângulo) da carga."""
    S_r = math.sqrt(P_r**2 + Q_r**2)
    phi = math.atan2(Q_r, P_r) if P_r != 0 else math.pi / 2
    return S_r, phi

def get_angle_diff(delta_s, delta_r):
    """Calcula a diferença de ângulo delta."""
    return delta_s - delta_r

# --- 1. Índices de Estabilidade de Tensão de Linha (Line VSIs) ---
# Baseados no artigo, Seção 4.1 [cite: 254-512] e Aula 04 [cite: 1101-1140]

def calculate_fvsi(V_s, X, Q_r, Z):
    """
    Calcula o Fast Voltage Stability Index (FVSI).
    Aula 04[cite: 1104], Artigo Eq. (7) [cite: 263].
    Estável se FVSI < 1.
    """
    if V_s == 0 or X == 0:
        return float('inf')
    return (4 * Z**2 * Q_r) / (V_s**2 * X)

def calculate_Lmn(V_s, X, Q_r, theta, delta):
    """
    Calcula o Line Stability Index (L_mn).
    Aula 04[cite: 1109], Artigo Eq. (10) [cite: 305].
    Estável se L_mn < 1.
    """
    denominator = (V_s * math.sin(theta - delta))**2
    if denominator == 0:
        return float('inf')
    return (4 * X * Q_r) / denominator

def calculate_LQP(V_s, X, Q_r, P_s):
    """
    Calcula o Line Stability Factor (LQP).
    Aula 04[cite: 1114], Artigo Eq. (11) [cite: 271].
    Estável se LQP < 1.
    """
    if V_s == 0:
        return float('inf')
    return 4 * (X / V_s**2) * (Q_r + (P_s**2 * X) / V_s**2)

def calculate_Lp(V_s, R, P_r, theta, delta):
    """
    Calcula o Line Stability Index (L_p).
    Aula 04[cite: 1119], Artigo Eq. (12) [cite: 278-281].
    Estável se L_p < 1.
    """
    denominator = (V_s * math.cos(theta - delta))**2
    if denominator == 0:
        return float('inf')
    return (4 * R * P_r) / denominator

def calculate_NLSI(V_s, P_r, R, Q_r, X):
    """
    Calcula o Novel Line Stability Index (NLSI).
    Artigo Eq. (13) [cite: 287].
    Estável se NLSI < 1.
    """
    if V_s == 0:
        return float('inf')
    return (P_r * R + Q_r * X) / (0.25 * V_s**2)

def calculate_NVSI(V_s, X, P_r, Q_r):
    """
    Calcula o New Voltage Stability Index (NVSI).
    Aula 04[cite: 1124], Artigo Eq. (23) [cite: 334].
    Estável se NVSI < 1.
    """
    S_r, _ = get_load_params(P_r, Q_r)
    denominator = (2 * Q_r * X - V_s**2)
    if denominator == 0:
        return float('inf')
    return (2 * X * S_r) / denominator

def calculate_VQI_line(V_s, Q_r, Z, X):
    """
    Calcula o Voltage Reactive Power Index (VQI_Line).
    Aula 04[cite: 1129], Artigo Eq. (24) [cite: 355].
    Nota: O artigo define B = Im(1/(R+jX)) = -X/Z^2. |B| = X/Z^2[cite: 357].
    A fórmula se torna idêntica ao FVSI.
    Estável se VQI_Line < 1.
    """
    if V_s == 0 or X == 0:
        return float('inf')
    B_mag = X / Z**2
    return (4 * Q_r) / (B_mag * V_s**2)

def calculate_PTSI(V_s, P_r, Q_r, Z, theta, phi):
    """
    Calcula o Power Transfer Stability Index (PTSI).
    Aula 04[cite: 1134], Artigo Eq. (25) [cite: 361].
    Estável se PTSI < 1.
    """
    S_r, _ = get_load_params(P_r, Q_r)
    if V_s == 0:
        return float('inf')
    return (2 * S_r * Z * (1 + math.cos(theta - phi))) / V_s**2

def calculate_VSLI(V_s, V_r, delta):
    """
    Calcula o Voltage Stability Load Index (VSLI).
    Aula 04, Artigo Eq. (33) [cite: 344] e Tabela 1[cite: 404].
    Usa a fórmula da Tabela 1: 4[VsVr*cos(delta) - Vr^2*cos^2(delta)] / Vs^2
    Estável se VSLI < 1.
    """
    if V_s == 0:
        return float('inf')
    cos_delta = math.cos(delta)
    return (4 * (V_s * V_r * cos_delta - V_r**2 * cos_delta**2)) / V_s**2

def calculate_L_index_simple(V_s, V_r):
    """
    Calcula o Índice L (simplificação do VSLI com delta=0).
    Artigo Eq. (34) [cite: 379].
    Estável se L < 1.
    """
    if V_s == 0:
        return float('inf')
    return (4 * (V_s * V_r - V_r**2)) / V_s**2

def calculate_VCPI_1(V_s, V_r, delta):
    """
    Calcula o Voltage Collapse Proximity Index (VCPI_1).
    Artigo Eq. (37) [cite: 394].
    Estável se VCPI_1 >= 0.
    """
    return V_r * math.cos(delta) - 0.5 * V_s

def calculate_VSI_2(V_s, Q_r, R, X):
    """
    Calcula o Voltage Stability Indicator (VSI_2).
    Artigo Eq. (39) [cite: 415].
    Estável se VSI_2 < 1.
    """
    denominator = X * (V_s**2 + 8 * R * Q_r)
    if denominator == 0:
        return float('inf')
    return (4 * Q_r * (R + X)**2) / denominator

def calculate_SI(V_s, V_r, P_r, Q_r, R, X, Z):
    """
    Calcula o Stability Index (SI).
    Artigo Eq. (44) [cite: 501].
    Estável se SI > 0.
    """
    term1 = 2 * V_s**2 * V_r**2
    term2 = V_r**4
    term3 = 2 * V_r**2 * (P_r * R + Q_r * X)
    term4 = Z**2 * (P_r**2 + Q_r**2)
    return term1 - term2 - term3 - term4


# --- 2. Índices de Estabilidade de Tensão de Barra (Bus VSIs) ---
# Baseados no artigo, Seção 4.2 [cite: 513-724] e Aula 04 [cite: 1141-1152]
# NOTA: Muitos destes índices requerem a matriz de admitância (Y_bus)
# ou medições em dois instantes de tempo (PMU).

def calculate_VCPI_bus(V_phasors, Y_bus):
    """
    Calcula o Voltage Collapse Prediction Index (VCPI_bus).
    Aula 04[cite: 1144, 1145], Artigo Eq. (48-50) [cite: 522, 595-598, 608].
    Requer vetores de fasores de tensão (complexo) e a matriz Y_bus (complexo).
    """
    V = np.asarray(V_phasors, dtype=complex)
    Y = np.asarray(Y_bus, dtype=complex)
    num_buses = len(V)
    VCPI_i_list = []

    for i in range(num_buses):
        sum_Yij = 0
        sum_V_tilde_m = 0
        
        for j in range(num_buses):
            if i != j:
                sum_Yij += Y[i, j]
        
        if sum_Yij == 0 or V[i] == 0:
            continue

        for m in range(num_buses):
            if m != i:
                V_tilde_m = (Y[i, m] * V[m]) / sum_Yij
                sum_V_tilde_m += V_tilde_m
        
        VCPI_i = abs(1 - (sum_V_tilde_m / V[i]))
        VCPI_i_list.append(VCPI_i)

    return min(VCPI_i_list) if VCPI_i_list else float('nan')

def calculate_L_index(V_phasors, Y_bus, gen_buses, load_buses):
    """
    Calcula o L-index.
    Artigo Eq. (51-53) [cite: 553, 554, 617].
    Requer particionamento da rede em barras de carga (LL) e geração (LG).
    """
    # Particionando a Y_bus
    Y_LL = Y_bus[np.ix_(load_buses, load_buses)]
    Y_LG = Y_bus[np.ix_(load_buses, gen_buses)]
    
    V_L = V_phasors[load_buses]
    V_G = V_phasors[gen_buses]
    
    # F = -inv(Y_LL) @ Y_LG
    F = -np.dot(inv(Y_LL), Y_LG)
    
    L_j_list = []
    for j_idx, j in enumerate(load_buses):
        sum_F_V = 0
        for i_idx, i in enumerate(gen_buses):
            sum_F_V += F[j_idx, i_idx] * V_G[i_idx]
            
        if V_L[j_idx] == 0:
            continue
            
        L_j = abs(1 - (sum_F_V / V_L[j_idx]))
        L_j_list.append(L_j)
        
    return max(L_j_list) if L_j_list else float('nan')


def calculate_SDC(V_r1_phasor, I_r1_phasor, V_r2_phasor, I_r2_phasor):
    """
    Calcula o S Difference Criterion (SDC).
    Artigo Eq. (54) [cite: 622].
    Requer medições de fasores em dois instantes (1 e 2).
    """
    delta_V_r = V_r2_phasor - V_r1_phasor
    delta_I_r = I_r2_phasor - I_r1_phasor
    
    if V_r1_phasor == 0 or delta_I_r == 0:
        return float('inf')
        
    term = (delta_V_r * np.conjugate(I_r1_phasor)) / (V_r1_phasor * np.conjugate(delta_I_r))
    return abs(1 + term)

def calculate_SVSI(V_r_mag, V_s_mag):
    """
    Calcula o Simplified Voltage Stability Index (SVSI) para um sistema de 2 barras.
    Aula 04 [cite: 1150-1151], Artigo Eq. (60-62) [cite: 716-722].
    Simplificação: Vg = Vs, beta calculado para 2 barras.
    """
    if V_r_mag == 0:
        return float('inf')
        
    delta_V_r = abs(V_s_mag - V_r_mag)
    beta = 1 - (abs(V_s_mag - V_r_mag))**2
    
    if beta * V_r_mag == 0:
        return float('inf')
        
    return delta_V_r / (beta * V_r_mag)

# --- 3. Índices Gerais de Estabilidade de Tensão (Overall VSIs) ---
# Baseados no artigo, Seção 4.3 [cite: 725-731]

def calculate_SG(P_gt, P_dt, Q_dt):
    """
    Calcula o Network Sensitivity Approach (SG).
    Artigo Eq. (63-64) [cite: 695-697].
    Retorna SG_p e SG_q.
    """
    SG_p = P_gt / P_dt if P_dt != 0 else float('inf')
    SG_q = P_gt / Q_dt if Q_dt != 0 else float('inf')
    return SG_p, SG_q


# # --- Exemplo de Uso ---
# if __name__ == "__main__":
#     # --- Parâmetros de Exemplo para o Modelo de 2 Barras ---
#     V_s = 1.0  # pu (magnitude tensão envio)
#     V_r = 0.95 # pu (magnitude tensão recebimento)
#     delta_s_deg = 10.0 # graus (ângulo tensão envio)
#     delta_r_deg = 0.0  # graus (ângulo tensão recebimento)
    
#     R = 0.05 # pu (resistência linha)
#     X = 0.15 # pu (reatância linha)
    
#     P_s = 0.8  # pu (potência ativa envio)
#     P_r = 0.75 # pu (potência ativa recebimento)
#     Q_r = 0.4  # pu (potência reativa recebimento)

#     # --- Cálculos Auxiliares ---
#     Z, theta = get_line_params(R, X)
#     S_r, phi = get_load_params(P_r, Q_r)
#     delta_s_rad = math.radians(delta_s_deg)
#     delta_r_rad = math.radians(delta_r_deg)
#     delta = get_angle_diff(delta_s_rad, delta_r_rad)

#     print(f"--- Parâmetros de Linha e Carga ---")
#     print(f"Z: {Z:.4f} pu, theta: {math.degrees(theta):.2f}°")
#     print(f"S_r: {S_r:.4f} pu, phi: {math.degrees(phi):.2f}°")
#     print(f"delta: {math.degrees(delta):.2f}°")
#     print("-" * 30)
    
#     print("--- 1. Testando Índices de Linha ---")
    
#     # FVSI
#     fvsi = calculate_fvsi(V_s, X, Q_r, Z)
#     print(f"FVSI: {fvsi:.4f} (Estável < 1)")
    
#     # Lmn
#     lmn = calculate_Lmn(V_s, X, Q_r, theta, delta)
#     print(f"L_mn: {lmn:.4f} (Estável < 1)")

#     # LQP
#     lqp = calculate_LQP(V_s, X, Q_r, P_s)
#     print(f"LQP: {lqp:.4f} (Estável < 1)")
    
#     # Lp
#     lp = calculate_Lp(V_s, R, P_r, theta, delta)
#     print(f"L_p: {lp:.4f} (Estável < 1)")

#     # VSLI
#     vsli = calculate_VSLI(V_s, V_r, delta)
#     print(f"VSLI: {vsli:.4f} (Estável < 1)")
    
#     # VCPI_1
#     vcpi_1 = calculate_VCPI_1(V_s, V_r, delta)
#     print(f"VCPI_1: {vcpi_1:.4f} (Estável >= 0)")

#     # SI
#     si = calculate_SI(V_s, V_r, P_r, Q_r, R, X, Z)
#     print(f"SI: {si:.4f} (Estável > 0)")
    
#     print("\n--- 2. Testando Índices de Barra (SVSI) ---")
#     svsi = calculate_SVSI(V_r, V_s)
#     print(f"SVSI (2-barras): {svsi:.4f} (Colapso em 1)")

#     print("\n--- 3. Testando Índices Gerais (SG) ---")
#     P_gt_total = 1000 # MW (Geração total)
#     P_dt_total = 950  # MW (Demanda ativa total)
#     Q_dt_total = 400  # MVAR (Demanda reativa total)
#     sg_p, sg_q = calculate_SG(P_gt_total, P_dt_total, Q_dt_total)
#     print(f"SG_p: {sg_p:.4f}, SG_q: {sg_q:.4f} (Colapso com aumento rápido)")