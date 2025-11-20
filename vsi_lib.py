import numpy as np

# ==============================================================================
# BIBLIOTECA DE ÍNDICES DE ESTABILIDADE DE TENSÃO (VSI LIB)
# ==============================================================================
# Este arquivo contém as implementações matemáticas puras dos índices.
# Baseado na revisão bibliográfica de Modarresi et al. (2016).
# ==============================================================================

# --- FUNÇÕES AUXILIARES ---
def get_line_params(R, X):
    """
    Calcula a impedância magnitude (Z) e o ângulo da impedância (theta).
    Necessário pois o Pandapower fornece R e X separados.
    """
    Z = np.sqrt(R**2 + X**2)
    theta = np.arctan2(X, R)
    return Z, theta

def get_load_params(P, Q):
    """
    Calcula a potência aparente (S) e o ângulo da potência (phi).
    """
    S = np.sqrt(P**2 + Q**2)
    phi = np.arctan2(Q, P)
    return S, phi

# --- 1. ÍNDICES DE LINHA (Line VSIs) ---
# Focam no colapso de tensão devido à transferência de potência entre duas barras.
# A maioria destes índices deriva do modelo de linha de transmissão de 2 barras.

def calculate_fvsi(V_s, X, Q_r, Z):
    """
    Fast Voltage Stability Index (Musirin, 2002).
    Baseado na discrimante da equação quadrática do fluxo de potência.
    Estável se FVSI < 1.0.
    """
    if V_s == 0 or X == 0: return np.nan
    return (4 * Z**2 * Q_r) / (V_s**2 * X)

def calculate_lmn(V_s, X, Q_r, theta, delta):
    """
    Line Stability Index Lmn (Moghavvemi, 1998).
    Considera o ângulo da linha e a defasagem angular (delta).
    Estável se Lmn < 1.0.
    """
    denom = (V_s * np.sin(theta - delta))**2
    if denom == 0: return np.nan
    return (4 * X * Q_r) / denom

def calculate_lqp(V_s, X, Q_r, P_s):
    """
    Line Stability Factor LQP (Mohamed, 1989).
    Derivado simplificando a resistência da linha, mas aqui usa formula completa.
    Estável se LQP < 1.0.
    """
    if V_s == 0: return np.nan
    return 4 * (X / V_s**2) * (Q_r + (P_s**2 * X) / V_s**2)

def calculate_lp(V_s, R, P_r, theta, delta):
    """
    Line Stability Index Lp (Moghavvemi, 2001).
    Similar ao Lmn, mas focado na potência Ativa (P) e Resistência (R).
    Estável se Lp < 1.0.
    """
    denom = (V_s * np.cos(theta - delta))**2
    if denom == 0: return np.nan
    return (4 * R * P_r) / denom

def calculate_nlsi(V_s, P_r, R, Q_r, X):
    """
    Novel Line Stability Index (Yazdanpanah-Goharrizi, 2007).
    Baseado na máxima transferência de potência considerando R e X.
    Estável se NLSI < 1.0.
    """
    if V_s == 0: return np.nan
    return (P_r * R + Q_r * X) / (0.25 * V_s**2)

def calculate_nvsi(V_s, X, P_r, Q_r):
    """
    New Voltage Stability Index (Kanimozhi, 2013).
    Foca na potência aparente da carga.
    Estável se NVSI < 1.0.
    """
    S_r = np.sqrt(P_r**2 + Q_r**2)
    denom = (2 * Q_r * X - V_s**2)
    if denom == 0: return np.nan
    return abs((2 * X * S_r) / denom)

def calculate_vsli(V_s, V_r, delta):
    """
    Voltage Stability Load Index (Abdul Rahman, 1995).
    Baseado apenas nas tensões e ângulos.
    Estável se VSLI < 1.0.
    """
    if V_s == 0: return np.nan
    term = V_r * np.cos(delta)
    return (4 * (V_s * term - term**2)) / V_s**2

def calculate_vsi2(V_s, Q_r, R, X):
    """
    Voltage Stability Indicator 2 (Chattopadhyay, 2014).
    Estável se VSI_2 < 1.0.
    """
    denom = X * (V_s**2 + 8 * R * Q_r)
    if denom == 0: return np.nan
    return (4 * Q_r * (R + X)**2) / denom

# --- ÍNDICES EXTRAS DO ARTIGO DE REVISÃO ---

def calculate_vqi(V_s, Q_r, X, R):
    """Voltage Reactive Power Index (Althowibi, 2010)."""
    Z2 = R**2 + X**2
    if Z2 == 0 or V_s == 0: return np.nan
    B_mag = abs(X / Z2) # Susceptância magnitude
    if B_mag == 0: return np.nan
    return (4 * Q_r) / (B_mag * V_s**2)

def calculate_ptsi(V_s, P_r, Q_r, Z, theta, phi):
    """Power Transfer Stability Index (Nizam, 2006)."""
    if V_s == 0: return np.nan
    S_r = np.sqrt(P_r**2 + Q_r**2)
    return (2 * S_r * Z * (1 + np.cos(theta - phi))) / V_s**2

def calculate_l_simple(V_s, V_r):
    """Simplificação do Índice L assumindo ângulo zero (Sahari, 2003)."""
    if V_s == 0: return np.nan
    return (4 * (V_s * V_r - V_r**2)) / V_s**2

def calculate_vcpi_power(V_s, Z, theta, phi, P_r, Q_r, kind='P'):
    """
    Voltage Collapse Proximity Index (Power).
    Calcula a razão entre Potência Atual / Potência Máxima Teórica.
    kind='P': VCPI(1), kind='Q': VCPI(2), kind='S': Lsr.
    """
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
    """Stability Index (Eminoglu, 2007). Colapso ocorre quando SI = 0."""
    term1 = 2 * V_s**2 * V_r**2
    term2 = V_r**4
    term3 = 2 * V_r**2 * (P_r * R + Q_r * X)
    term4 = Z**2 * (P_r**2 + Q_r**2)
    return term1 - term2 - term3 - term4

def calculate_vcpi_1_voltage(V_s, V_r, delta):
    """VCPI baseado em tensão (Wang, 2005). Colapso em 0."""
    return V_r * np.cos(delta) - 0.5 * V_s

def calculate_vsmi(delta, theta, phi):
    """Voltage Stability Margin Index (He, 2004). Baseado em margem angular."""
    delta_max = (theta - phi) / 2
    if delta_max == 0: return np.nan
    return (delta_max - abs(delta)) / delta_max

def calculate_vslbi(V_s, V_r, delta):
    """Voltage Stability Load Bus Index (Milosevic, 2003)."""
    v_drop_sq = V_s**2 + V_r**2 - 2*V_s*V_r*np.cos(delta)
    if v_drop_sq <= 0: return 99.0
    return V_r / np.sqrt(v_drop_sq)

def calculate_vsi1(V_s, P_r, Q_r, X):
    """VSI baseado em margem de potência restante (Gong, 2006)."""
    if X == 0 or V_s == 0: return np.nan
    try:
        P_max = V_s**2 / (4 * X)
        Q_max = V_s**2 / (4 * X)
        margin_p = 1 - (P_r / P_max)
        margin_q = 1 - (Q_r / Q_max)
        return min(margin_p, margin_q)
    except:
        return np.nan

# --- 2. ÍNDICES DE BARRA VETORIZADOS (Otimização) ---
# Utilizam álgebra linear para calcular o índice de todas as barras simultaneamente.

def calculate_l_index_vectorized(V_complex, F_matrix, gen_indices, load_indices):
    """
    L-Index (Kessel & Glavitsch).
    Mede a distância da instabilidade baseada na matriz híbrida F.
    Otimizado para numpy (sem loops Python).
    """
    V_L = V_complex[load_indices]
    V_G = V_complex[gen_indices]
    
    if V_L.size == 0 or V_G.size == 0:
        return np.array([])
        
    numerator = F_matrix.dot(V_G)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        L_values = np.abs(1 - (numerator / V_L))
    
    L_values[~np.isfinite(L_values)] = 0.0 # Trata erros numéricos
    return L_values

def calculate_vcpi_bus_vectorized(V_complex, Y_bus_matrix):
    """
    VCPI Bus Vectorizado.
    Baseado na relação entre injeção de corrente e admitância da barra.
    """
    # Soma da linha da matriz Ybus (Admitância total conectada ao nó)
    Y_sum = np.array(Y_bus_matrix.sum(axis=1)).flatten()
    # Corrente injetada calculada
    I_inj = Y_bus_matrix.dot(V_complex)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        term = I_inj / Y_sum
        VCPI = np.abs(1 - (term / V_complex))
    
    VCPI[~np.isfinite(VCPI)] = 0.0 
    return VCPI