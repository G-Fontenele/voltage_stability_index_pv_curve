import copy
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pandapower as pp

# ==============================================================================
# MOTOR DE CPF VERDADEIRO (PREDITOR-CORRETOR COM PARAMETRIZAÇÃO LOCAL)
# Equivalente ao comando /EXCF do ANAREDE (CEPEL)
#
# Baseado em:
#   Ajjarapu & Christy (1992) - "The continuation power flow: a tool for
#   steady state voltage stability analysis"
#   IEEE Trans. Power Systems, vol.7, no.1, pp.416-423
# ==============================================================================

def run_cpf(net, initial_static_matrices, load_scaling_bus_id=None, max_scale=5.0,
             initial_step=0.1, min_step=0.001,
             max_iters=2000, max_failures=15,
             distributed_slack=True, qlim_mode='none',
             solver_max_iter=20, solver_tol=0.1):
    """
    Executa o Fluxo de Potencia Continuado (CPF) verdadeiro via metodo
    Preditor-Corretor com Parametrizacao Local.

    Reproduz o comportamento do ANAREDE /EXCF:
    - Preditor: vetor tangente normalizado
    - Corretor: Newton-Raphson com Jacobiana Aumentada
    - Parametrizacao Local: evita singularidade no nariz da curva PV
    - Para no ponto de maximo carregamento (nariz)
    """
    import analysis_tools as tools

    print(f"\n{'='*80}")
    print(f" CPF VERDADEIRO: PREDITOR-CORRETOR (Estilo ANAREDE /EXCF)")
    print(f"{'='*80}")
    print(f"  > Passo Inicial (sigma):  {initial_step*100:.3f}%")
    print(f"  > Passo Minimo:           {min_step*100:.5f}%")
    print(f"  > Max Iteracoes CPF:      {max_iters}")
    print(f"  > Max Falhas:             {max_failures}")
    print(f"  > Tolerancia NR:          {solver_tol} MVA")
    print(f"{'='*80}\n")

    net_sim = copy.deepcopy(net)
    qlim_mode = str(qlim_mode).lower()
    enforce_q_lims = (qlim_mode in ['pandapower', 'pv_to_pq'])

    # --- 1. Identificacao de cargas ---
    if load_scaling_bus_id is None:
        load_idx = net_sim.load.index
    else:
        load_idx = net_sim.load[net_sim.load.bus == load_scaling_bus_id].index
        if load_idx.empty:
            return [], []

    base_p_load = net_sim.load.loc[load_idx, 'p_mw'].copy()
    base_q_load = net_sim.load.loc[load_idx, 'q_mvar'].copy()

    # --- 2. Identificacao de geradores para slack distribuido ---
    active_gen_idx = []
    base_p_gen = []
    if distributed_slack:
        mask_gen = net_sim.gen.p_mw > 1.0
        active_gen_idx = net_sim.gen[mask_gen].index
        base_p_gen = net_sim.gen.loc[active_gen_idx, 'p_mw'].copy()

    # --- 3. Caso base ---
    print(" [...] Rodando Caso Base (NR convencional)...")
    history = []
    full_log = []
    current_matrices = copy.deepcopy(initial_static_matrices)

    try:
        pp.runpp(net_sim, enforce_q_lims=enforce_q_lims,
                 max_iteration=solver_max_iter, tolerance_mva=solver_tol)
        iters_base = net_sim._ppc.get('iterations', 0) if isinstance(net_sim._ppc, dict) else 0
        _save_snapshot(net_sim, 1.0, history, current_matrices)
        _log_attempt(full_log, 0, 1.0, 0.0, "Convergente", net_sim, iters_base)
        print(f"   -> Caso Base OK (Iter NR: {iters_base})")
    except Exception as e:
        _log_attempt(full_log, 0, 1.0, 0.0, "Divergente", None, 0)
        print(f"ERRO CRITICO: Caso base nao converge: {e}")
        return [], full_log

    # --- 4. Extracao das estruturas internas do pandapower ---
    ppc = net_sim._ppc
    baseMVA = ppc['baseMVA']
    Ybus = ppc['internal']['Ybus']
    V_cur = ppc['internal']['V'].copy()
    pv  = ppc['internal']['pv'].copy()
    pq  = ppc['internal']['pq'].copy()
    ref = ppc['internal']['ref'].copy()
    pvpq = np.r_[pv, pq]
    n_pv = len(pv)
    n_pq = len(pq)
    n_x = n_pv + 2 * n_pq  # dimensao do espaco de estados

    # --- 5. Vetor de direcao b ---
    b = _build_direction_vector(ppc, baseMVA, pvpq, pq)

    # --- 6. Estado inicial do CPF ---
    lam_cur = 1.0
    sigma = initial_step
    x_cur = _V_to_x(V_cur, pvpq, pq)
    total_iters = 0
    consecutive_failures = 0
    k_param = n_x  # comeca controlando lambda

    # Ultimo ponto bom para restauracao em falha
    x_last_good = x_cur.copy()
    lam_last_good = lam_cur
    V_last_good = V_cur.copy()

    print(" [...] Iniciando loop CPF Preditor-Corretor...")

    # Variavel para rastrear o vetor tangente anterior
    t_prev = None

    # --- 7. Loop principal CPF ---
    while lam_cur < max_scale and total_iters < max_iters:
        total_iters += 1

        # == PASSO 1: PREDITOR (Vetor Tangente) ==
        # Recalcula J_conv usando a V_cur atual (evita problemas com pandapower)
        J_conv = _compute_jacobian(Ybus, V_cur, pvpq, pq)
        
        if t_prev is not None:
            direction = np.sign(t_prev[k_param])
            if direction == 0: direction = 1.0
        else:
            direction = 1.0

        t = _predictor_tangent(J_conv, b, k_param, n_x, direction=direction)
        t_norm = t / (np.linalg.norm(t) + 1e-15)
        t_prev = t_norm

        # Define qual o parametro de continuacao (0 a n_x-1 sao variaveis de estado, n_x e lambda)
        k_param_name = "LAMBDA"
        if k_param < len(pvpq):
            k_param_name = f"ANGULO (bus_int={pvpq[k_param]})"
        elif k_param < n_x:
            k_param_name = f"TENSAO (bus_int={pq[k_param - len(pvpq)]})"

        print(f"   [PREDITOR] lam={lam_cur:.5f} -> k_param={k_param_name} (idx={k_param}), dir={direction}")

        x_pred = x_cur + sigma * t_norm[:n_x]
        lam_pred = lam_cur + sigma * t_norm[n_x]
        if lam_pred > max_scale:
            lam_pred = max_scale

        # == PASSO 2: SELECAO DO PARAMETRO DE CONTINUACAO ==
        k_param_new = _select_continuation_param(t_norm, n_x)
        print(f"      [DEBUG] max|t_x|={np.max(np.abs(t_norm[:n_x])):.5f}, |t_lam|={np.abs(t_norm[n_x]):.5f}")
        k_param = k_param_new
        
        target_k = x_pred[k_param] if k_param < n_x else lam_pred
        
        print(f"      [DEBUG] Corrector starts: k_param={k_param}, target_k={target_k:.5f}, lam_pred={lam_pred:.5f}")

        # == PASSO 3: CORRETOR (NR com Jacobiana Aumentada) ==
        x_new, lam_new, nr_iters, success = _corrector_newton(
            Ybus, ppc, baseMVA, pvpq, pq, ref,
            x_pred, lam_pred,
            b, k_param, target_k,
            solver_tol, solver_max_iter,
            base_p_load, base_q_load, load_idx,
            active_gen_idx, base_p_gen, distributed_slack
        )

        if success:
            consecutive_failures = 0
            V_new = _x_to_V_from_ppc(x_new, ppc, pvpq, pq, ref)

            # Manual update instead of runpp
            bus_to_idx = current_matrices['bus_to_idx']
            bus_ids = list(bus_to_idx.keys())
            int_idxs = [bus_to_idx[b] for b in bus_ids]
            
            if 'res_bus' not in net_sim or net_sim.res_bus.empty:
                net_sim.res_bus = pd.DataFrame(index=net_sim.bus.index)
            
            net_sim.res_bus.loc[bus_ids, 'vm_pu'] = np.abs(V_new[int_idxs])
            net_sim.res_bus.loc[bus_ids, 'va_degree'] = np.degrees(np.angle(V_new[int_idxs]))

            net_sim.load.loc[load_idx, 'p_mw'] = base_p_load * lam_new
            net_sim.load.loc[load_idx, 'q_mvar'] = base_q_load * lam_new
            if 'res_load' not in net_sim or net_sim.res_load.empty:
                net_sim.res_load = pd.DataFrame(index=net_sim.load.index)
            net_sim.res_load.loc[load_idx, 'p_mw'] = base_p_load * lam_new
            net_sim.res_load.loc[load_idx, 'q_mvar'] = base_q_load * lam_new

            if distributed_slack and len(active_gen_idx) > 0:
                net_sim.gen.loc[active_gen_idx, 'p_mw'] = base_p_gen.loc[active_gen_idx] * lam_new
                if 'res_gen' not in net_sim or net_sim.res_gen.empty:
                    net_sim.res_gen = pd.DataFrame(index=net_sim.gen.index)
                net_sim.res_gen.loc[active_gen_idx, 'p_mw'] = base_p_gen.loc[active_gen_idx] * lam_new


            # Passo adaptativo
            if nr_iters <= 3:
                sigma = min(sigma * 1.2, initial_step * 5)
            elif nr_iters > 7:
                sigma = max(sigma * 0.5, min_step)

            # Detecta nariz comparando com o lambda do passo anterior
            if lam_new < lam_last_good - 1e-6 and total_iters > 2:
                print(f"--> NARIZ DA CURVA DETECTADO em lambda={lam_new:.5f}")
                # Atualiza resultados para salvar o ponto do nariz
                x_cur = x_new.copy()
                lam_cur = lam_new
                V_cur = V_new.copy()
                _save_snapshot(net_sim, lam_new, history, current_matrices)
                _log_attempt(full_log, total_iters, lam_new, sigma, "Convergente (Nariz)", net_sim, nr_iters)
                break

            x_last_good = x_new.copy()
            lam_last_good = lam_new
            V_last_good = V_new.copy()


            # (Ybus doesn't change, no need to update ppc)


            _save_snapshot(net_sim, lam_new, history, current_matrices)
            _log_attempt(full_log, total_iters, lam_new, sigma, "Convergente", net_sim, nr_iters)

            p_tot = net_sim.res_load.p_mw.sum()
            print(f"   Iter {total_iters}: lambda={lam_new:.5f} OK (Iter NR: {nr_iters}) | Carga: {p_tot:.1f} MW")

            x_cur = x_new.copy()
            lam_cur = lam_new
            V_cur = V_new.copy()

        else:
            consecutive_failures += 1
            _log_attempt(full_log, total_iters, lam_pred, sigma, "Divergente", None, 0)
            print(f"   Iter {total_iters}: Falha corretor em lambda={lam_pred:.5f}. Reduzindo sigma...")

            if consecutive_failures >= max_failures:
                print(f"--> COLAPSO EM: lambda={lam_cur:.5f}")
                break
            if sigma < min_step:
                print(f"--> COLAPSO (passo minimo) em lambda={lam_cur:.5f}")
                break

            sigma /= 2.0
            x_cur = x_last_good.copy()
            lam_cur = lam_last_good
            V_cur = V_last_good.copy()
            k_param = n_x  # volta para parametro lambda apos falha

    return history, full_log


# ==============================================================================
# FUNCOES INTERNAS DO CPF
# ==============================================================================

def _predictor_tangent(J_conv, b, k_param, n_x, direction=+1.0):
    """
    Calcula o vetor tangente [dx; dlambda] resolvendo:
        J_aug * t = e_k * direction
    onde J_aug = [[J, -b], [e_k^T, 0]]
    """
    n = n_x + 1

    b_col = sp.csr_matrix((-b).reshape(-1, 1))

    top = sp.hstack([J_conv, b_col], format='csr')

    if k_param < n_x:
        e_k = np.zeros(n_x)
        e_k[k_param] = 1.0
        bottom = sp.hstack([sp.csr_matrix(e_k.reshape(1, -1)),
                             sp.csr_matrix([[0.0]])], format='csr')
    else:
        bottom = sp.hstack([sp.csr_matrix(np.zeros((1, n_x))),
                             sp.csr_matrix([[1.0]])], format='csr')

    J_aug = sp.vstack([top, bottom], format='csr')

    rhs = np.zeros(n)
    rhs[-1] = direction

    try:
        t = spsolve(J_aug, rhs)
    except Exception:
        t = np.zeros(n)
        t[n_x] = direction

    return t


def _select_continuation_param(t_norm, n_x):
    """
    Parametrizacao local: seleciona indice k tal que |t_k| = max.
    - k < n_x: fixa componente de estado (tensao ou angulo)
    - k == n_x: fixa lambda (longe do nariz)
    """
    return int(np.argmax(np.abs(t_norm)))


def _corrector_newton(Ybus, ppc, baseMVA, pvpq, pq, ref,
                       x_init, lam_init, b, k_param, target_k,
                       tol, max_it,
                       base_p_load, base_q_load, load_idx,
                       active_gen_idx, base_p_gen, distributed_slack):
    """
    Corretor Newton-Raphson com Jacobiana Aumentada.

    Resolve:
        f(x, lambda) = 0         (balanco de potencia)
        g_k(x, lambda) = target_k  (equacao de continuacao)
    """
    x = x_init.copy()
    lam = lam_init
    n_x = len(x)

    for i in range(max_it):
        V_cur = _x_to_V_from_ppc(x, ppc, pvpq, pq, ref)

        S_inj = _compute_Sbus_scaled(ppc, baseMVA, lam,
                                      base_p_load, base_q_load, load_idx,
                                      active_gen_idx, base_p_gen, distributed_slack)
        S_calc = V_cur * np.conj(Ybus.dot(V_cur))
        mismatch = S_inj - S_calc
        F = np.r_[mismatch[pvpq].real, mismatch[pq].imag]

        if k_param < n_x:
            g_k = x[k_param] - target_k
        else:
            g_k = lam - target_k

        # O mismatch real (F) e S_inj - S_calc. Para o Newton, resolvemos J*dx = F.
        # A equacao de continuacao e g_k = 0. Para o Newton, e_k*dx = -g_k.
        res = np.r_[F, -g_k]
        norm_res = np.max(np.abs(F)) * baseMVA if len(F) > 0 else 0.0

        if norm_res < tol and abs(g_k) < 1e-6:
            return x, lam, i + 1, True

        try:
            J_conv = _compute_jacobian(Ybus, V_cur, pvpq, pq)
        except Exception:
            return x, lam, i + 1, False

        b_col = sp.csr_matrix((-b).reshape(-1, 1))
        top = sp.hstack([J_conv, b_col], format='csr')

        if k_param < n_x:
            e_k = np.zeros(n_x)
            e_k[k_param] = 1.0
            bottom = sp.hstack([sp.csr_matrix(e_k.reshape(1, -1)),
                                 sp.csr_matrix([[0.0]])], format='csr')
        else:
            bottom = sp.hstack([sp.csr_matrix(np.zeros((1, n_x))),
                                 sp.csr_matrix([[1.0]])], format='csr')

        J_aug = sp.vstack([top, bottom], format='csr')

        try:
            dx_aug = spsolve(J_aug, res)
        except Exception:
            return x, lam, i + 1, False

        x = x + dx_aug[:n_x]
        lam = max(lam + dx_aug[n_x], 1.0)

    # Check final
    V_cur = _x_to_V_from_ppc(x, ppc, pvpq, pq, ref)
    S_inj = _compute_Sbus_scaled(ppc, baseMVA, lam, base_p_load, base_q_load,
                                  load_idx, active_gen_idx, base_p_gen, distributed_slack)
    S_calc = V_cur * np.conj(Ybus.dot(V_cur))
    mismatch = S_inj - S_calc
    F = np.r_[mismatch[pvpq].real, mismatch[pq].imag]
    norm_final = np.max(np.abs(F)) * baseMVA if len(F) > 0 else 0.0
    return x, lam, max_it, norm_final < tol


def _compute_jacobian(Ybus, V, pvpq, pq):
    """
    Calcula a Jacobiana convencional do fluxo de potencia AC.
    Formulacao polar padrao (dP/dtheta, dP/dV, dQ/dtheta, dQ/dV).
    """
    Vm = np.abs(V)
    Ibus = Ybus.dot(V)
    diagV = sp.diags(V)
    diagIbus = sp.diags(Ibus)
    diagVnorm = sp.diags(V / (Vm + 1e-15))

    dSbus_dVm = diagV * np.conj(Ybus * diagVnorm) + np.conj(diagIbus) * diagVnorm
    dSbus_dVa = 1j * diagV * np.conj(diagIbus - Ybus * diagV)

    J11 = dSbus_dVa[np.ix_(pvpq, pvpq)].real
    J12 = dSbus_dVm[np.ix_(pvpq, pq)].real
    J21 = dSbus_dVa[np.ix_(pq, pvpq)].imag
    J22 = dSbus_dVm[np.ix_(pq, pq)].imag

    J = sp.bmat([[J11, J12], [J21, J22]], format='csr')
    return J


def _build_direction_vector(ppc, baseMVA, pvpq, pq):
    """
    Constroi o vetor de direcao b em pu, representando dS/dlambda.
    Proporcional as cargas base de cada barra.
    """
    n_bus = ppc['bus'].shape[0]
    b_P = np.zeros(n_bus)
    b_Q = np.zeros(n_bus)

    bus_arr = ppc['bus']
    for bus_idx in range(n_bus):
        pd = bus_arr[bus_idx, 2] / baseMVA  # PD em pu
        qd = bus_arr[bus_idx, 3] / baseMVA  # QD em pu
        b_P[bus_idx] -= pd  # carga aumenta = injecao de P diminui
        b_Q[bus_idx] -= qd

    # Geradores PV aumentam com lambda (slack distribuido)
    gen_arr = ppc['gen']
    for g in range(gen_arr.shape[0]):
        bus_g = int(gen_arr[g, 0])
        pg_pu = gen_arr[g, 1] / baseMVA
        if pg_pu > 1e-3:
            b_P[bus_g] += pg_pu  # geracao aumenta = injecao de P aumenta

    # A normalizacao de b foi REMOVIDA.
    # O vetor b DEVE representar a variacao real dS/dlam usada no corretor.
    # Como o corretor usa `load * lam`, dS/dlam = load_base.
    # Se normalizarmos b, o preditor preve para um dS pequeno, mas o corretor 
    # resolve para um dS grande, causando um erro massivo de predicao.

    return np.r_[b_P[pvpq], b_Q[pq]]


def _compute_Sbus_scaled(ppc, baseMVA, lam, base_p_load, base_q_load,
                          load_idx, active_gen_idx, base_p_gen, distributed_slack):
    """
    Calcula o vetor de injecao de potencia S_bus para o fator de carga lambda.
    """
    from pandapower.pypower.makeSbus import makeSbus

    bus_scaled = ppc['bus'].copy()
    gen_scaled = ppc['gen'].copy()

    # Escala cargas proporcionalmente a lambda
    bus_scaled[:, 2] = ppc['bus'][:, 2] * lam  # PD
    bus_scaled[:, 3] = ppc['bus'][:, 3] * lam  # QD

    # Escala geracao (exceto slack)
    if distributed_slack and len(active_gen_idx) > 0:
        for g in range(gen_scaled.shape[0]):
            pg_base = ppc['gen'][g, 1]
            if pg_base > 1.0:
                gen_scaled[g, 1] = pg_base * lam

    return makeSbus(baseMVA, bus_scaled, gen_scaled)


def _V_to_x(V, pvpq, pq):
    """Converte V complexo para vetor de estado x = [theta_pvpq, Vm_pq]."""
    return np.r_[np.angle(V)[pvpq], np.abs(V)[pq]]


def _x_to_V(x, V_prev, pvpq, pq, ref):
    """Reconstroi V complexo a partir de x = [theta_pvpq, Vm_pq]."""
    n_pvpq = len(pvpq)
    theta = np.angle(V_prev).copy()
    vm = np.abs(V_prev).copy()
    theta[pvpq] = x[:n_pvpq]
    vm[pq] = x[n_pvpq:]
    return vm * np.exp(1j * theta)


def _x_to_V_from_ppc(x, ppc, pvpq, pq, ref):
    """Reconstroi V complexo usando o V atual em ppc como base para Vm das barras PV/Ref."""
    V_prev = ppc['internal']['V']
    return _x_to_V(x, V_prev, pvpq, pq, ref)


# ==============================================================================
# FUNCOES AUXILIARES (MESMA ASSINATURA QUE simulation_engine.py)
# ==============================================================================

def _save_snapshot(net, scale, history_list, static_matrices):
    snapshot = {
        'scale': scale,
        'total_load_mw': net.res_load.p_mw.sum(),
        'total_load_mvar': net.res_load.q_mvar.sum(),
        'res_bus': net.res_bus.copy(),
        'res_line': net.res_line.copy(),
        'res_trafo': net.res_trafo.copy(),
        'line_data': net.line.copy(),
        'trafo_data': net.trafo.copy(),
        'bus_data': net.bus.copy(),
        'static_matrices': copy.deepcopy(static_matrices)
    }
    history_list.append(snapshot)


def _log_attempt(log_list, iter_num, scale, step_used, status, net=None, nr_iters=0):
    row = {'iter': iter_num, 'scale': scale, 'step': step_used, 'status': status,
           'nr_iters': nr_iters, 'mw': 0.0, 'mvar': 0.0,
           'p_gen': 0.0, 'p_slack': 0.0, 'vmin': 0.0}
    if net is not None:
        try:
            row['mw'] = net.res_load.p_mw.sum()
            row['mvar'] = net.res_load.q_mvar.sum()
            row['vmin'] = net.res_bus['vm_pu'].min()
            gen_p = net.res_gen.p_mw.sum() if not net.res_gen.empty else 0.0
            ext_p = net.res_ext_grid.p_mw.sum() if not net.res_ext_grid.empty else 0.0
            row['p_gen'] = gen_p + ext_p
            row['p_slack'] = ext_p
        except Exception:
            pass
    log_list.append(row)
