import numpy as np

# ==============================================================================
# BIBLIOTECA DE ÍNDICES DE ESTABILIDADE DE TENSÃO (VSI) - VETORIZADA
# ==============================================================================

def get_line_params(R, X):
    """Retorna a impedância Z e o ângulo theta (rad)."""
    Z = np.sqrt(R**2 + X**2)
    theta = np.arctan2(X, R)
    return Z, theta

def get_load_params(P, Q):
    """Retorna a potência aparente S e o ângulo phi (rad)."""
    S = np.sqrt(P**2 + Q**2)
    phi = np.arctan2(Q, P)
    return S, phi

def calculate_fvsi(V_s, X, Q_r, Z):
    with np.errstate(divide='ignore', invalid='ignore'):
        val = (4 * Z**2 * Q_r) / (V_s**2 * X)
    return np.where(np.isfinite(val), val, np.nan)

def calculate_lmn(V_s, X, Q_r, theta, delta):
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = (V_s * np.sin(theta - delta))**2
        val = (4 * X * Q_r) / denom
    return np.where(np.isfinite(val), val, np.nan)

def calculate_lqp(V_s, X, Q_r, P_s):
    with np.errstate(divide='ignore', invalid='ignore'):
        val = 4 * (X / V_s**2) * (Q_r + (P_s**2 * X) / V_s**2)
    return np.where(np.isfinite(val), val, np.nan)

def calculate_lp(V_s, R, P_r, theta, delta):
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = (V_s * np.cos(theta - delta))**2
        val = (4 * R * P_r) / denom
    return np.where(np.isfinite(val), val, np.nan)

def calculate_nlsi(V_s, P_r, R, Q_r, X):
    with np.errstate(divide='ignore', invalid='ignore'):
        val = (P_r * R + Q_r * X) / (0.25 * V_s**2)
    return np.where(np.isfinite(val), val, np.nan)

def calculate_nvsi(V_s, X, P_r, Q_r):
    S_r = np.sqrt(P_r**2 + Q_r**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = (2 * Q_r * X - V_s**2)
        val = np.abs((2 * X * S_r) / denom)
    return np.where(np.isfinite(val), val, np.nan)

def calculate_vsli(V_s, V_r, delta):
    with np.errstate(divide='ignore', invalid='ignore'):
        term = V_r * np.cos(delta)
        val = (4 * (V_s * term - term**2)) / V_s**2
    return np.where(np.isfinite(val), val, np.nan)

def calculate_vsi2(V_s, Q_r, R, X):
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = X * (V_s**2 + 8 * R * Q_r)
        val = (4 * Q_r * (R + X)**2) / denom
    return np.where(np.isfinite(val), val, np.nan)

def calculate_vqi(V_s, Q_r, X, R):
    Z2 = R**2 + X**2
    with np.errstate(divide='ignore', invalid='ignore'):
        B_mag = np.abs(X / Z2)
        val = (4 * Q_r) / (B_mag * V_s**2)
    return np.where(np.isfinite(val), val, np.nan)

def calculate_ptsi(V_s, P_r, Q_r, Z, theta, phi):
    S_r = np.sqrt(P_r**2 + Q_r**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = (2 * S_r * Z * (1 + np.cos(theta - phi))) / V_s**2
    return np.where(np.isfinite(val), val, np.nan)

def calculate_l_simple(V_s, V_r):
    with np.errstate(divide='ignore', invalid='ignore'):
        val = (4 * (V_s * V_r - V_r**2)) / V_s**2
    return np.where(np.isfinite(val), val, np.nan)

def calculate_vcpi_power(V_s, Z, theta, phi, P_r, Q_r, kind='P'):
    with np.errstate(divide='ignore', invalid='ignore'):
        denom_max = 4 * Z * (np.cos((theta - phi)/2))**2
        if kind == 'P':
            P_max = (V_s**2 / denom_max) * np.cos(phi)
            val = P_r / P_max
        elif kind == 'Q':
            Q_max = (V_s**2 / denom_max) * np.sin(phi)
            val = Q_r / Q_max
        elif kind == 'S':
            S_max = (V_s**2 / denom_max)
            S_r = np.sqrt(P_r**2 + Q_r**2)
            val = S_r / S_max
        else:
            return np.nan
    return np.where(np.isfinite(val), val, np.nan)

def calculate_si(V_s, V_r, P_r, Q_r, R, X, Z):
    term1 = 2 * V_s**2 * V_r**2
    term2 = V_r**4
    term3 = 2 * V_r**2 * (P_r * R + Q_r * X)
    term4 = Z**2 * (P_r**2 + Q_r**2)
    return term1 - term2 - term3 - term4

def calculate_vcpi_1_voltage(V_s, V_r, delta):
    return V_r * np.cos(delta) - 0.5 * V_s

def calculate_vsmi(delta, theta, phi):
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_max = (theta - phi) / 2
        val = (delta_max - np.abs(delta)) / delta_max
    return np.where(np.isfinite(val), val, np.nan)

def calculate_vslbi(V_s, V_r, delta):
    v_drop_sq = V_s**2 + V_r**2 - 2*V_s*V_r*np.cos(delta)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = V_r / np.sqrt(v_drop_sq)
    return np.where((v_drop_sq > 0) & np.isfinite(val), val, 99.0)

def calculate_vsi1(V_s, P_r, Q_r, X):
    with np.errstate(divide='ignore', invalid='ignore'):
        P_max = V_s**2 / (4 * X)
        Q_max = V_s**2 / (4 * X)
        val = np.minimum(1 - (P_r / P_max), 1 - (Q_r / Q_max))
    return np.where(np.isfinite(val), val, np.nan)

def calculate_l_index_vectorized(V_complex, F_matrix, gen_indices, load_indices):
    if F_matrix is None: return np.zeros(len(load_indices))
    V_L = V_complex[load_indices]
    V_G = V_complex[gen_indices]
    if V_L.size == 0 or V_G.size == 0: return np.array([])
    numerator = F_matrix.dot(V_G)
    with np.errstate(divide='ignore', invalid='ignore'):
        L_values = np.abs(1 - (numerator / V_L))
    return np.where(np.isfinite(L_values), L_values, 0.0)

def calculate_vcpi_bus_vectorized(V_complex, Y_bus_matrix):
    if Y_bus_matrix is None: return np.zeros(len(V_complex))
    Y_sum = np.array(Y_bus_matrix.sum(axis=1)).flatten()
    I_inj = Y_bus_matrix.dot(V_complex)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = I_inj / Y_sum
        VCPI = np.abs(1 - (term / V_complex))
    return np.where(np.isfinite(VCPI), VCPI, 0.0)
