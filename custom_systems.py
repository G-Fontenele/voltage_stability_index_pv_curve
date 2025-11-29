import pandapower as pp
import numpy as np
import os

def export_pwf_anarede(net, filename="validacao_ieee30.pwf"):
    """
    Gera um arquivo .pwf formatado como ANAREDE.
    ATUALIZADO: Agora lê corretamente Taps modelados via Tap Changer.
    """
    try: pp.runpp(net); 
    except: pass

    s_base = 100.0
    
    with open(filename, 'w') as f:
        f.write("TITU\n")
        f.write("VALIDACAO IEEE 30 - MODELAGEM VIA TAP CHANGER\n")
        f.write("DBAR\n")
        f.write("(Num)OETGb(    nome    )Gl( V)( A)( Pg)( Qg)( Qn)( Qm)(Bc  )( Pl)( Ql)\n")
        
        for idx in sorted(net.bus.index):
            row = net.bus.loc[idx]
            num, nome = idx, str(row['name'])[:12]
            
            if not net.res_bus.empty and idx in net.res_bus.index:
                v_pu, ang = net.res_bus.at[idx, 'vm_pu'], net.res_bus.at[idx, 'va_degree']
            else:
                v_pu, ang = row['vm_pu'], 0.0
            
            v_int = int(v_pu * 1000)
            pl = ql = pg = qg = qmin = qmax = bc = 0.0
            tipo = "L "
            
            loads = net.load[net.load.bus == idx]
            if not loads.empty: pl = loads.p_mw.sum(); ql = loads.q_mvar.sum()
            shunts = net.shunt[net.shunt.bus == idx]
            if not shunts.empty: bc = -shunts.q_mvar.sum()

            slacks = net.ext_grid[net.ext_grid.bus == idx]
            if not slacks.empty:
                tipo = "L2" 
                qmin, qmax = slacks.iloc[0].min_q_mvar, slacks.iloc[0].max_q_mvar
                if not net.res_ext_grid.empty:
                    res_idx = slacks.index[0]
                    pg = net.res_ext_grid.at[res_idx, 'p_mw']
                    qg = net.res_ext_grid.at[res_idx, 'q_mvar']
                
            gens = net.gen[net.gen.bus == idx]
            if not gens.empty:
                if tipo == "L ": tipo = "L1" 
                qmin += gens.min_q_mvar.sum(); qmax += gens.max_q_mvar.sum()
                if not net.res_gen.empty:
                    for gen_i in gens.index:
                        pg += net.res_gen.at[gen_i, 'p_mw']; qg += net.res_gen.at[gen_i, 'q_mvar']
                else:
                    pg += gens.p_mw.sum(); qg += gens.q_mvar.sum()

            qmin_str = f"{int(qmin)}" if abs(qmin) > 9000 else f"{qmin:.1f}"
            qmax_str = f"{int(qmax)}" if abs(qmax) > 9000 else f"{qmax:.1f}"
            
            line = (f"{num:>5} {tipo:<2} A{nome:<12} {v_int:>4} {ang:>5.1f} "
                    f"{pg:>6.1f} {qg:>6.1f} {qmin_str:>6} {qmax_str:>6} "
                    f"{bc:>6.1f} {pl:>6.1f} {ql:>6.1f}\n")
            f.write(line)
            
        f.write("DLIN\n")
        f.write("(De )d O d(Pa )NcEP ( R% )( X% )(Mvar)(Tap)(Tmn)(Tmx)(Phs)(Bc  )(Cn)(Ce)Ns\n")
        
        for i, row in net.line.iterrows():
            fr, to = row.from_bus, row.to_bus
            vn = net.bus.at[fr, 'vn_kv']
            z_base = (vn**2) / 100.0
            r_pct = (row.r_ohm_per_km * row.length_km / z_base) * 100
            x_pct = (row.x_ohm_per_km * row.length_km / z_base) * 100
            w = 2 * np.pi * 60
            b_siemens = (row.c_nf_per_km * row.length_km * 1e-9) * w
            mvar_ch = b_siemens * (vn**2)
            f.write(f"{fr:>5} {to:>5}   1      {r_pct:>6.2f} {x_pct:>6.2f} {mvar_ch:>6.2f}\n")
            
        for i, row in net.trafo.iterrows():
            hv, lv = row.hv_bus, row.lv_bus
            r_pct = row.vkr_percent
            xp = np.sqrt(max(0, row.vk_percent**2 - r_pct**2))
            
            # CÁLCULO DO TAP REAL (Considerando Tap Changer)
            # Fórmula: Ratio = 1 + (pos - neutral) * step/100
            if not np.isnan(row.tap_pos):
                step = row.tap_step_percent
                pos = row.tap_pos
                neutral = row.tap_neutral
                tap_final = 1 + (pos - neutral) * (step / 100.0)
            else:
                # Fallback para método antigo (Vn/Vn)
                vn_base_hv = net.bus.at[hv, 'vn_kv']
                tap_final = row.vn_hv_kv / vn_base_hv

            f.write(f"{hv:>5} {lv:>5}   1      {r_pct:>6.2f} {xp:>6.2f}   0.00  {tap_final:.3f}\n")
            
        f.write("99999\nFIM\n")
    print(f"  -> Arquivo PWF (Tap Changer) gerado: {filename}")


def create_ieee30_anarede():
    """
    Cria o sistema IEEE 30 Barras usando Tap Changers para os transformadores.
    """
    net = pp.create_empty_network()
    
    # 1. BARRAS
    buses_data = [
        (1, 132, "Glen-Lyn", 2, 1.060, 0.0, 0.0), (2, 132, "Claytor", 1, 1.043, -5.0, 0.0),
        (3, 132, "Kumis", 0, 1.021, -7.0, 0.0), (4, 132, "Hancock", 0, 1.012, -9.0, 0.0),
        (5, 132, "Fieldale", 1, 1.010, -14.0, 0.0), (6, 132, "Roanoke", 0, 1.010, -11.0, 0.0),
        (7, 132, "Blaine", 0, 1.002, -13.0, 0.0), (8, 132, "Reusens", 1, 1.010, -12.0, 0.0),
        (9, 132, "ZRoanoke", 0, 1.051, -14.0, 0.0), (10, 33, "TRoanoke", 0, 1.045, -15.0, 19.0),
        (11, 11, "SRoanoke", 1, 1.082, -14.0, 0.0), (12, 33, "THancock", 0, 1.057, -15.0, 0.0),
        (13, 11, "SHancock", 1, 1.071, -15.0, 0.0), (14, 33, "Barra14", 0, 1.042, -16.0, 0.0),
        (15, 33, "Barra15", 0, 1.038, -16.0, 0.0), (16, 33, "Barra16", 0, 1.045, -15.0, 0.0),
        (17, 33, "Barra17", 0, 1.040, -16.0, 0.0), (18, 33, "Barra18", 0, 1.028, -16.0, 0.0),
        (19, 33, "Barra19", 0, 1.026, -17.0, 0.0), (20, 33, "Barra20", 0, 1.030, -16.0, 0.0),
        (21, 33, "Barra21", 0, 1.033, -16.0, 0.0), (22, 33, "Barra22", 0, 1.033, -16.0, 0.0),
        (23, 33, "Barra23", 0, 1.027, -16.0, 0.0), (24, 33, "Barra24", 0, 1.021, -16.0, 4.3),
        (25, 33, "Barra25", 0, 1.017, -16.0, 0.0), (26, 33, "Barra26", 0, 1.000, -16.0, 0.0),
        (27, 33, "TCloverdle", 0, 1.023, -15.0, 0.0), (28, 132, "ACloverdle", 0, 1.007, -11.0, 0.0),
        (29, 33, "Barra29", 0, 1.003, -17.0, 0.0), (30, 33, "Barra30", 0, 0.992, -17.0, 0.0)
    ]
    for b in buses_data:
        pp.create_bus(net, index=b[0], vn_kv=b[1], name=b[2], vm_pu=b[4], type="b")
        if b[6] > 0: pp.create_shunt(net, bus=b[0], q_mvar=-b[6], p_mw=0, name=f"Cap-{b[0]}")

    # 2. CARGAS
    loads_data = [
        (2, 21.7, 12.7), (3, 2.4, 1.2), (4, 7.6, 1.6), (5, 94.2, 19.0),
        (7, 22.8, 10.9), (8, 30.0, 30.0), (10, 5.8, 2.0), (12, 11.2, 7.5),
        (14, 6.2, 1.6), (15, 8.2, 2.5), (16, 3.5, 1.8), (17, 9.0, 5.8),
        (18, 3.2, 0.9), (19, 9.5, 3.4), (20, 2.2, 0.7), (21, 17.5, 11.2),
        (23, 3.2, 1.6), (24, 8.7, 6.7), (26, 3.5, 2.3), (29, 2.4, 0.9),
        (30, 10.6, 1.9)
    ]
    for l in loads_data: pp.create_load(net, bus=l[0], p_mw=l[1], q_mvar=l[2])

    # 3. GERADORES
    pp.create_ext_grid(net, bus=1, vm_pu=1.060, min_q_mvar=-9999, max_q_mvar=9999, name="Glen-Lyn-Ref")
    gens_data = [
        (2, 40.0, 1.043, -40.0, 50.0, 50.0), (5, 0.0, 1.010, -40.0, 40.0, 37.0),
        (8, 0.0, 1.010, -10.0, 40.0, 37.3), (11, 0.0, 1.082, -6.0, 24.0, 16.2),
        (13, 0.0, 1.071, -6.0, 24.0, 10.6)
    ]
    for g in gens_data:
        pp.create_gen(net, bus=g[0], p_mw=g[1], vm_pu=g[2], min_q_mvar=g[3], max_q_mvar=g[4], q_mvar=g[5], name=f"Gen-{g[0]}")

    # 4. RAMOS
    branches_data = [
        (1, 2, 1.92, 5.75, 5.28, 0), (1, 3, 4.52, 16.52, 4.08, 0),
        (2, 4, 5.70, 17.37, 3.68, 0), (2, 5, 4.72, 19.83, 4.18, 0),
        (2, 6, 5.81, 17.63, 3.74, 0), (3, 4, 1.32, 3.79, 0.84, 0),
        (4, 6, 1.19, 4.14, 0.90, 0), (4, 12, 0.0, 25.6, 0.0, 0.932),
        (5, 7, 4.60, 11.6, 2.04, 0), (6, 7, 2.67, 8.20, 1.70, 0),
        (6, 8, 1.20, 4.20, 0.90, 0), (6, 9, 0.0, 20.8, 0.0, 0.978),
        (6, 10, 0.0, 55.6, 0.0, 0.969), (6, 28, 1.69, 5.99, 1.30, 0),
        (8, 28, 6.36, 20.0, 4.28, 0), (9, 10, 0.0, 11.0, 0.0, 0),
        (9, 11, 0.0, 20.8, 0.0, 0), (10, 17, 3.24, 8.45, 0.0, 0),
        (10, 20, 9.36, 20.9, 0.0, 0), (10, 21, 3.48, 7.49, 0.0, 0),
        (10, 22, 7.27, 14.99, 0.0, 0), (12, 13, 0.0, 14.0, 0.0, 0),
        (12, 14, 12.31, 25.59, 0.0, 0), (12, 15, 6.62, 13.04, 0.0, 0),
        (12, 16, 9.45, 19.87, 0.0, 0), (14, 15, 22.1, 19.97, 0.0, 0),
        (15, 18, 10.73, 21.85, 0.0, 0), (15, 23, 10.0, 20.2, 0.0, 0),
        (16, 17, 5.24, 19.23, 0.0, 0), (18, 19, 6.39, 12.92, 0.0, 0),
        (19, 20, 3.40, 6.80, 0.0, 0), (21, 22, 1.16, 2.36, 0.0, 0),
        (22, 24, 11.5, 17.9, 0.0, 0), (23, 24, 13.2, 27.0, 0.0, 0),
        (24, 25, 18.85, 32.92, 0.0, 0), (25, 26, 25.44, 38.0, 0.0, 0),
        (25, 27, 10.93, 20.87, 0.0, 0), (27, 29, 21.98, 41.53, 0.0, 0),
        (27, 30, 32.02, 60.27, 0.0, 0), (28, 27, 0.0, 39.6, 0.0, 0.968),
        (29, 30, 23.99, 45.33, 0.0, 0)
    ]
    s_base = 100.0
    for branch in branches_data:
        f, t, r_pct, x_pct, mvar_ch, tap = branch
        vn_kv = net.bus.at[f, 'vn_kv']
        z_base = (vn_kv**2) / s_base
        r_ohm = (r_pct / 100.0) * z_base
        x_ohm = (x_pct / 100.0) * z_base
        
        if mvar_ch > 0:
            b_siemens = (mvar_ch) / (vn_kv**2) 
            c_nf = (b_siemens / (2 * np.pi * 60)) * 1e9
        else: c_nf = 0.0
        
        if tap > 0:
            vk_pct = np.sqrt(r_pct**2 + x_pct**2)
            
            # --- MODELAGEM COM TAP CHANGER ---
            # Target: Tap 0.932 (ou seja, 93.2% da tensão nominal).
            # Se Neutral = 0 e Step = 1%, então Pos = -6.8.
            # Método mais simples: Definir step = desvio exato e usar pos = 1 (ou -1).
            
            # Se tap < 1.0, estamos reduzindo tensão. Se tap > 1.0, aumentando.
            # Vamos assumir que o Tap Changer está no lado HV.
            # Ratio = 1 + (pos)*step/100.
            # Se tap = 0.932 -> 0.932 = 1 + pos*step/100 -> pos*step = -6.8
            
            # Implementação: Step positivo, Posição negativa.
            step_pct = abs(tap - 1.0) * 100.0
            
            if step_pct < 0.001: # Nominal (Tap ~ 1.0)
                tap_pos = 0
                tap_step = 1.0 # Dummy
            else:
                tap_pos = -1 if tap < 1.0 else 1
                tap_step = step_pct

            pp.create_transformer_from_parameters(net, hv_bus=f, lv_bus=t, sn_mva=100, 
                vn_hv_kv=net.bus.at[f, 'vn_kv'], # Tensão NOMINAL da barra
                vn_lv_kv=net.bus.at[t, 'vn_kv'],
                vkr_percent=r_pct, vk_percent=vk_pct, pfe_kw=0, i0_percent=0,
                
                # Configuração do Tap Changer
                tap_pos=tap_pos, 
                tap_neutral=0, 
                tap_step_percent=tap_step, 
                tap_side="hv", # Padrão ANAREDE
                
                name=f"Trafo-{f}-{t}"
            )
        else:
            pp.create_line_from_parameters(net, from_bus=f, to_bus=t, length_km=1.0,
                r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm, c_nf_per_km=c_nf, max_i_ka=1.0,
                name=f"Line-{f}-{t}")

    export_pwf_anarede(net, "validacao_ieee30_gerado.pwf")
    return net