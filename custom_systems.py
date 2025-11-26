import pandapower as pp
import numpy as np

def create_ieee30_anarede():
    """
    Cria o sistema IEEE 30 Barras fiel ao PWF do ANAREDE.
    Inclui limites de reativo (Qmin/Qmax) e parametrização da Slack.
    """
    net = pp.create_empty_network()
    
    # --- 1. BARRAMENTOS (DBAR) ---
    # Estrutura: (ID, Vn_kV, Nome, Tipo, Vm_pu, Va_degree, Shunt_Mvar_Capacitivo)
    # Tipos: 0=PQ, 1=PV, 2=Ref
    buses_data = [
        (1, 132, "Glen-Lyn", 2, 1.060, 0.0, 0.0),
        (2, 132, "Claytor", 1, 1.043, -5.0, 0.0),
        (3, 132, "Kumis", 0, 1.021, -7.0, 0.0),
        (4, 132, "Hancock", 0, 1.012, -9.0, 0.0),
        (5, 132, "Fieldale", 1, 1.010, -14.0, 0.0),
        (6, 132, "Roanoke", 0, 1.010, -11.0, 0.0),
        (7, 132, "Blaine", 0, 1.002, -13.0, 0.0),
        (8, 132, "Reusens", 1, 1.010, -12.0, 0.0),
        (9, 132, "ZRoanoke", 0, 1.051, -14.0, 0.0),
        (10, 33, "TRoanoke", 0, 1.045, -15.0, 19.0), # Capacitor 19 Mvar
        (11, 11, "SRoanoke", 1, 1.082, -14.0, 0.0),
        (12, 33, "THancock", 0, 1.057, -15.0, 0.0),
        (13, 11, "SHancock", 1, 1.071, -15.0, 0.0),
        (14, 33, "Barra14", 0, 1.042, -16.0, 0.0),
        (15, 33, "Barra15", 0, 1.038, -16.0, 0.0),
        (16, 33, "Barra16", 0, 1.045, -15.0, 0.0),
        (17, 33, "Barra17", 0, 1.040, -16.0, 0.0),
        (18, 33, "Barra18", 0, 1.028, -16.0, 0.0),
        (19, 33, "Barra19", 0, 1.026, -17.0, 0.0),
        (20, 33, "Barra20", 0, 1.030, -16.0, 0.0),
        (21, 33, "Barra21", 0, 1.033, -16.0, 0.0),
        (22, 33, "Barra22", 0, 1.033, -16.0, 0.0),
        (23, 33, "Barra23", 0, 1.027, -16.0, 0.0),
        (24, 33, "Barra24", 0, 1.021, -16.0, 4.3), # Capacitor 4.3 Mvar
        (25, 33, "Barra25", 0, 1.017, -16.0, 0.0),
        (26, 33, "Barra26", 0, 1.000, -16.0, 0.0),
        (27, 33, "TCloverdle", 0, 1.023, -15.0, 0.0),
        (28, 132, "ACloverdle", 0, 1.007, -11.0, 0.0),
        (29, 33, "Barra29", 0, 1.003, -17.0, 0.0),
        (30, 33, "Barra30", 0, 0.992, -17.0, 0.0)
    ]

    for b in buses_data:
        pp.create_bus(net, index=b[0], vn_kv=b[1], name=b[2], vm_pu=b[4], type="b")
        # Shunts: No ANAREDE Bc é positivo para capacitor. 
        # No Pandapower, shunt q_mvar positivo é consumo (indutor), negativo é injeção (capacitor).
        if b[6] > 0:
            pp.create_shunt(net, bus=b[0], q_mvar=-b[6], p_mw=0, name=f"Cap-{b[0]}")

    # --- 2. CARGAS (Pl, Ql) ---
    loads_data = [
        (2, 21.7, 12.7), (3, 2.4, 1.2), (4, 7.6, 1.6), (5, 94.2, 19.0),
        (7, 22.8, 10.9), (8, 30.0, 30.0), (10, 5.8, 2.0), (12, 11.2, 7.5),
        (14, 6.2, 1.6), (15, 8.2, 2.5), (16, 3.5, 1.8), (17, 9.0, 5.8),
        (18, 3.2, 0.9), (19, 9.5, 3.4), (20, 2.2, 0.7), (21, 17.5, 11.2),
        (23, 3.2, 1.6), (24, 8.7, 6.7), (26, 3.5, 2.3), (29, 2.4, 0.9),
        (30, 10.6, 1.9)
    ]
    for l in loads_data:
        pp.create_load(net, bus=l[0], p_mw=l[1], q_mvar=l[2])

    # --- 3. GERADORES E SLACK (Incluindo Limites Q) ---
    
    # BARRA 1 (Slack/Ref)
    # Dados PWF: Pg=260.2, Qg=-16.1, Qmin=-9999, Qmax=9999
    # Nota: Pg e Qg na Slack são resultados do fluxo, mas definimos os limites.
    pp.create_ext_grid(
        net, bus=1, 
        vm_pu=1.060, va_degree=0.0, 
        min_p_mw=0, max_p_mw=9999, # Slack assume o que precisar
        min_q_mvar=-9999, max_q_mvar=9999,
        name="Glen-Lyn-Ref"
    )
    
    # OUTROS GERADORES (PV)
    # Estrutura: (Bus, P_mw, V_set, Q_min, Q_max)
    # Dados extraídos do PWF ANAREDE fornecido
    gens_data = [
        (1,  260.2, 1.060, -9999, 99999),  # Claytor
        (2,  40.0, 1.043, -40.0, 50.0),  # Claytor
        (5,   0.0, 1.010, -40.0, 40.0),  # Fieldale (Compensador)
        (8,   0.0, 1.010, -10.0, 40.0),  # Reusens (Compensador)
        (11,  0.0, 1.082,  -6.0, 24.0),  # SRoanoke (Compensador)
        (13,  0.0, 1.071,  -6.0, 24.0)   # SHancock (Compensador)
    ]
    
    for g in gens_data:
        pp.create_gen(
            net, bus=g[0], 
            p_mw=g[1], 
            vm_pu=g[2], 
            min_q_mvar=g[3], 
            max_q_mvar=g[4],
            name=f"Gen-{g[0]}"
        )

    # --- 4. LINHAS E TRANSFORMADORES ---
    # Dados: (From, To, R%, X%, Mvar_ch, Tap)
    branches_data = [
        (1, 2, 1.92, 5.75, 5.28, 0),
        (1, 3, 4.52, 16.52, 4.08, 0),
        (2, 4, 5.70, 17.37, 3.68, 0),
        (2, 5, 4.72, 19.83, 4.18, 0),
        (2, 6, 5.81, 17.63, 3.74, 0),
        (3, 4, 1.32, 3.79, 0.84, 0),
        (4, 6, 1.19, 4.14, 0.90, 0),
        (4, 12, 0.0, 25.6, 0.0, 0.932), # Trafo
        (5, 7, 4.60, 11.6, 2.04, 0),
        (6, 7, 2.67, 8.20, 1.70, 0),
        (6, 8, 1.20, 4.20, 0.90, 0),
        (6, 9, 0.0, 20.8, 0.0, 0.978), # Trafo
        (6, 10, 0.0, 55.6, 0.0, 0.969), # Trafo
        (6, 28, 1.69, 5.99, 1.30, 0),
        (8, 28, 6.36, 20.0, 4.28, 0),
        (9, 10, 0.0, 11.0, 0.0, 0),
        (9, 11, 0.0, 20.8, 0.0, 0),
        (10, 17, 3.24, 8.45, 0.0, 0),
        (10, 20, 9.36, 20.9, 0.0, 0),
        (10, 21, 3.48, 7.49, 0.0, 0),
        (10, 22, 7.27, 14.99, 0.0, 0),
        (12, 13, 0.0, 14.0, 0.0, 0),
        (12, 14, 12.31, 25.59, 0.0, 0),
        (12, 15, 6.62, 13.04, 0.0, 0),
        (12, 16, 9.45, 19.87, 0.0, 0),
        (14, 15, 22.1, 19.97, 0.0, 0),
        (15, 18, 10.73, 21.85, 0.0, 0),
        (15, 23, 10.0, 20.2, 0.0, 0),
        (16, 17, 5.24, 19.23, 0.0, 0),
        (18, 19, 6.39, 12.92, 0.0, 0),
        (19, 20, 3.40, 6.80, 0.0, 0),
        (21, 22, 1.16, 2.36, 0.0, 0),
        (22, 24, 11.5, 17.9, 0.0, 0),
        (23, 24, 13.2, 27.0, 0.0, 0),
        (24, 25, 18.85, 32.92, 0.0, 0),
        (25, 26, 25.44, 38.0, 0.0, 0),
        (25, 27, 10.93, 20.87, 0.0, 0),
        (27, 29, 21.98, 41.53, 0.0, 0),
        (27, 30, 32.02, 60.27, 0.0, 0),
        (28, 27, 0.0, 39.6, 0.0, 0.968), # Trafo
        (29, 30, 23.99, 45.33, 0.0, 0)
    ]

    s_base = 100.0

    for branch in branches_data:
        f, t, r_pct, x_pct, mvar_ch, tap = branch
        
        vn_kv = net.bus.at[f, 'vn_kv']
        z_base = (vn_kv**2) / s_base
        
        r_ohm = (r_pct / 100.0) * z_base
        x_ohm = (x_pct / 100.0) * z_base
        
        # Susceptância (Charging)
        if mvar_ch > 0:
            # B (Siemens) = Mvar / V^2
            # c_nf = B / (2*pi*f) * 1e9
            b_siemens = (mvar_ch) / (vn_kv**2) 
            w = 2 * np.pi * 60
            c_nf = (b_siemens / w) * 1e9
        else:
            c_nf = 0.0

        if tap > 0:
            # Transformador
            vk_pct = np.sqrt(r_pct**2 + x_pct**2)
            pp.create_transformer_from_parameters(
                net, hv_bus=f, lv_bus=t, sn_mva=100, 
                vn_hv_kv=net.bus.at[f, 'vn_kv'], 
                vn_lv_kv=net.bus.at[t, 'vn_kv'],
                vkr_percent=r_pct, vk_percent=vk_pct,
                pfe_kw=0, i0_percent=0,
                tap_pos=0, tap_neutral=0, tap_step_percent=0,
                name=f"Trafo-{f}-{t}"
            )
        else:
            # Linha
            pp.create_line_from_parameters(
                net, from_bus=f, to_bus=t, length_km=1.0,
                r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm,
                c_nf_per_km=c_nf, max_i_ka=1.0,
                name=f"Line-{f}-{t}"
            )

    return net