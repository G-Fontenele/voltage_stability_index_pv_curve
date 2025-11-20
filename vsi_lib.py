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

def calculate_vqi(V_s, Q_r, X, R):
    Z2 = R**2 + X**2
    if Z2 == 0 or V_s == 0: return np.nan
    B_mag = abs(X / Z2)
    if B_mag == 0: return np.nan
    return (4 * Q_r) / (B_mag * V_s**2)

def calculate_ptsi(V_s, P_r, Q_r, Z, theta, phi):
    if V_s == 0: return np.nan
    S_r = np.sqrt(P_r**2 + Q_r**2)
    return (2 * S_r * Z * (1 + np.cos(theta - phi))) / V_s**2

def calculate_l_simple(V_s, V_r):
    if V_s == 0: return np.nan
    return (4 * (V_s * V_r - V_r**2)) / V_s**2

def calculate_vcpi_power(V_s, Z, theta, phi, P_r, Q_r, kind='P'):
    denom_max = 4 * Z * (np.cos((theta - phi)/2))**2
    if denom_max == 0: return np.nan
    if kind == 'P':
        P_max = (V_s**2 / denom_max) * np.cos(phi)
        return P_r / P_max if P_max != 0 else np.nan
    elif kind == 'Q':
        Q_max = (V_s**2 / denom_max) * np.sin(phi)
        return Q_r / Q_max if Q_max != 0 else np.nan
    elif kind == 'S':
        S_max = (V_s**2 / denom_max)
        S_r = np.sqrt(P_r**2 + Q_r**2)
        return S_r / S_max if S_max != 0 else np.nan
    return np.nan

def calculate_si(V_s, V_r, P_r, Q_r, R, X, Z):
    term1 = 2 * V_s**2 * V_r**2
    term2 = V_r**4
    term3 = 2 * V_r**2 * (P_r * R + Q_r * X)
    term4 = Z**2 * (P_r**2 + Q_r**2)
    return term1 - term2 - term3 - term4

def calculate_vcpi_1_voltage(V_s, V_r, delta):
    return V_r * np.cos(delta) - 0.5 * V_s

def calculate_vsmi(delta, theta, phi):
    delta_max = (theta - phi) / 2
    if delta_max == 0: return np.nan
    return (delta_max - abs(delta)) / delta_max

def calculate_vslbi(V_s, V_r, delta):
    v_drop_sq = V_s**2 + V_r**2 - 2*V_s*V_r*np.cos(delta)
    if v_drop_sq <= 0: return 99.0
    return V_r / np.sqrt(v_drop_sq)

def calculate_vsi1(V_s, P_r, Q_r, X):
    if X == 0 or V_s == 0: return np.nan
    try:
        P_max = V_s**2 / (4 * X)
        Q_max = V_s**2 / (4 * X)
        margin_p = 1 - (P_r / P_max)
        margin_q = 1 - (Q_r / Q_max)
        return min(margin_p, margin_q)
    except:
        return np.nan

# --- 2. ÍNDICES DE BARRA VETORIZADOS ---

def calculate_l_index_vectorized(V_complex, F_matrix, gen_indices, load_indices):
    V_L = V_complex[load_indices]
    V_G = V_complex[gen_indices]
    if V_L.size == 0 or V_G.size == 0: return np.array([])
    numerator = F_matrix.dot(V_G)
    with np.errstate(divide='ignore', invalid='ignore'):
        L_values = np.abs(1 - (numerator / V_L))
    L_values[~np.isfinite(L_values)] = 0.0 # Se erro, assume 0 para não poluir gráfico
    return L_values

def calculate_vcpi_bus_vectorized(V_complex, Y_bus_matrix):
    Y_sum = np.array(Y_bus_matrix.sum(axis=1)).flatten()
    I_inj = Y_bus_matrix.dot(V_complex)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = I_inj / Y_sum
        VCPI = np.abs(1 - (term / V_complex))
    VCPI[~np.isfinite(VCPI)] = 0.0 # Limpeza de NaNs
    return VCPI