import numpy as np

# --- FUNÇÕES AUXILIARES ---
def get_line_params(R, X):
    Z = np.sqrt(R**2 + X**2)
    theta = np.arctan2(X, R)
    return Z, theta

def get_load_params(P, Q):
    S = np.sqrt(P**2 + Q**2)
    phi = np.arctan2(Q, P)
    return S, phi

# --- ÍNDICES DE LINHA (Line VSIs) ---
# (Mantidos iguais, focados no fluxo entre duas barras)

def calculate_fvsi(V_s, X, Q_r, Z):
    if V_s == 0 or X == 0: return np.nan
    return (4 * Z**2 * Q_r) / (V_s**2 * X)

def calculate_lmn(V_s, X, Q_r, theta, delta):
    denom = (V_s * np.sin(theta - delta))**2
    if denom == 0: return np.nan
    return (4 * X * Q_r) / denom

def calculate_lqp(V_s, X, Q_r, P_s):
    if V_s == 0: return np.nan
    return 4 * (X / V_s**2) * (Q_r + (P_s**2 * X) / V_s**2)

def calculate_lp(V_s, R, P_r, theta, delta):
    denom = (V_s * np.cos(theta - delta))**2
    if denom == 0: return np.nan
    return (4 * R * P_r) / denom

def calculate_nlsi(V_s, P_r, R, Q_r, X):
    if V_s == 0: return np.nan
    return (P_r * R + Q_r * X) / (0.25 * V_s**2)

def calculate_nvsi(V_s, X, P_r, Q_r):
    S_r = np.sqrt(P_r**2 + Q_r**2)
    denom = (2 * Q_r * X - V_s**2)
    if denom == 0: return np.nan
    return abs((2 * X * S_r) / denom)

def calculate_vsli(V_s, V_r, delta):
    if V_s == 0: return np.nan
    term = V_r * np.cos(delta)
    return (4 * (V_s * term - term**2)) / V_s**2

def calculate_vsi2(V_s, Q_r, R, X):
    denom = X * (V_s**2 + 8 * R * Q_r)
    if denom == 0: return np.nan
    return (4 * Q_r * (R + X)**2) / denom

# --- ÍNDICES DE BARRA (Bus VSIs) ---
# Calculados usando matrizes pré-processadas para velocidade

def calculate_l_index_vectorized(V_complex, F_matrix, gen_indices, load_indices):
    """
    Calcula o L-index para todas as barras de carga de uma vez.
    L_j = |1 - sum(F_ji * V_i / V_j)|
    """
    # V_load (vetor das tensões nas barras de carga)
    V_L = V_complex[load_indices]
    # V_gen (vetor das tensões nas barras de geração)
    V_G = V_complex[gen_indices]
    
    # Numerador: sum(F_ji * V_i) -> Multiplicação Matricial F * V_G
    # F_matrix deve ser shape (n_load, n_gen)
    if V_L.size == 0 or V_G.size == 0:
        return np.array([])
        
    numerator = F_matrix.dot(V_G)
    
    # Evita divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        L_values = np.abs(1 - (numerator / V_L))
    
    # Trata NaNs ou Infinitos resultantes de V_L ~ 0
    L_values[~np.isfinite(L_values)] = 1.0 
    
    return L_values

def calculate_vcpi_bus_vectorized(V_complex, Y_bus_matrix):
    """
    Calcula VCPI_bus para todas as barras.
    VCPI_k = |1 - sum(V_m') / V_k|
    Onde V_m' = (Y_km / sum(Y_kj)) * V_m
    """
    # Soma das admitâncias de cada linha (diagonal da Ybus não serve diretamente aqui, 
    # precisamos da soma da linha da matriz completa)
    # Y_sum_rows = soma de elementos da linha k (excluindo a própria barra se a fórmula pedir, 
    # mas a eq. usual é sobre Y_km V_m)
    
    # Implementação otimizada da Eq. 50 e 49:
    # V_node = soma(Y_km * V_m) / soma(Y_km)
    # Isso é basicamente (Y_bus * V) ./ (Y_bus * 1) se Ybus incluir shunts corretamente
    
    num_buses = V_complex.shape[0]
    results = np.zeros(num_buses)
    
    # Soma das admitâncias conectadas a cada barra k (soma da linha k da Ybus)
    Y_sum = np.array(Y_bus_matrix.sum(axis=1)).flatten()
    
    # Corrente injetada "equivalente" I = Y * V
    I_inj = Y_bus_matrix.dot(V_complex)
    
    # O termo sum(V_m') da equação pode ser simplificado matricialmente
    # V_m' = Y_km * V_m / Y_sum_k
    # Sum(V_m') = (1/Y_sum_k) * Sum(Y_km * V_m) = (1/Y_sum_k) * I_inj_k
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Numerador total da fração interna
        term = I_inj / Y_sum
        # Índice final
        VCPI = np.abs(1 - (term / V_complex))
        
    results = VCPI
    return results