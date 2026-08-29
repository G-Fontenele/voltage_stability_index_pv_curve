import os
import shutil
import pandas as pd
import glob

def generate_latex_table_from_df(df, output_path, label="tab:mytable", caption="My table"):
    latex_str = df.to_latex(index=False, column_format="c" * len(df.columns), float_format="%.4f")
    latex_wrap = f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n" + latex_str + "\\end{table}\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_wrap)

def parse_initial_report(filepath):
    bus_data = {}
    if not os.path.exists(filepath): return bus_data
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    start_idx = -1
    for i, line in enumerate(lines):
        if "BARRA   | V (pu)  | ANG (deg)" in line:
            start_idx = i + 2
            break
            
    if start_idx == -1: return bus_data
    
    for line in lines[start_idx:]:
        if line.startswith("="): break
        parts = line.split("|")
        if len(parts) >= 8:
            try:
                # Ônibus já são 1-indexed no modelo ANAREDE customizado
                bus_id = int(parts[0].strip())
                v_pu = float(parts[1].strip())
                ang_deg = float(parts[2].strip())
                p_gen = float(parts[3].strip())
                q_gen = float(parts[4].strip())
                # parts[5] and [6] are P_INJ and Q_INJ
                btype = parts[7].strip()
                bus_data[bus_id] = {
                    'v': v_pu, 'ang': ang_deg, 'p': p_gen, 'q': q_gen, 'type': btype
                }
            except: pass
    return bus_data

def generate_voltage_comparison_table(tables_dir, initial_report_path):
    bus_data = parse_initial_report(initial_report_path)
    if not bus_data:
        print("Aviso: Relatório inicial não encontrado para Tabela I.")
        return
        
    # Referência ANAREDE (conforme Tabela I do artigo)
    anarede_ref = {
        1: {'type': 'Ref', 'v': 1.060, 'ang': 0.0},
        2: {'type': 'PV', 'v': 1.043, 'ang': -5.3},
        5: {'type': 'Comp', 'v': 1.010, 'ang': -14.1},
        8: {'type': 'Comp', 'v': 1.010, 'ang': -11.8},
        12: {'type': 'PQ', 'v': 1.057, 'ang': -14.9},
        30: {'type': 'PQ', 'v': 0.992, 'ang': -17.6}
    }
    
    table_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of Voltage and Angle between Python and ANAREDE}",
        "\\label{tab:comp_tensao}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Bus}} & \\multirow{2}{*}{\\textbf{Type}} & \\multicolumn{3}{c}{\\textbf{Voltage (pu)}} & \\multicolumn{3}{c}{\\textbf{Angle (degrees)}} \\\\ \\cmidrule(lr){3-5} \\cmidrule(l){6-8} ",
        " &  & \\textbf{Py} & \\textbf{ANA} & \\textbf{Diff} & \\textbf{Py} & \\textbf{ANA} & \\textbf{Diff} \\\\ \\midrule"
    ]
    
    for b in [1, 2, 5, 8, 12, 30]:
        if b in bus_data and b in anarede_ref:
            py_v = bus_data[b]['v']
            py_ang = bus_data[b]['ang']
            ana_v = anarede_ref[b]['v']
            ana_ang = anarede_ref[b]['ang']
            diff_v = py_v - ana_v
            diff_ang = py_ang - ana_ang
            
            line = f"{b} & {anarede_ref[b]['type']} & {py_v:.3f} & {ana_v:.3f} & {diff_v:.3f} & {py_ang:.1f} & {ana_ang:.1f} & {diff_ang:.1f} \\\\"
            table_lines.append(line)
            
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{table}")
    
    with open(os.path.join(tables_dir, "tab_comp_tensao.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print("Tabela I (Tensão/Ângulo) gerada.")

def generate_generation_comparison_table(tables_dir, initial_report_path):
    bus_data = parse_initial_report(initial_report_path)
    if not bus_data:
        print("Aviso: Relatório inicial não encontrado para Tabela II.")
        return
        
    # Referência ANAREDE (conforme Tabela II do artigo)
    anarede_ref = {
        1: {'type': 'Slack', 'p': 261.0, 'q': -16.5},
        2: {'type': 'PV', 'p': 40.0, 'q': 49.6},
        5: {'type': 'Comp', 'p': 0.0, 'q': 36.0},
        8: {'type': 'Comp', 'p': 0.0, 'q': 37.3},
        11: {'type': 'Comp', 'p': 0.0, 'q': 16.2},
        13: {'type': 'Comp', 'p': 0.0, 'q': 10.6}
    }
    
    table_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of Active and Reactive Generation}",
        "\\label{tab:comp_geracao}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Bus}} & \\multirow{2}{*}{\\textbf{Type}} & \\multicolumn{3}{c}{\\textbf{Active Power (MW)}} & \\multicolumn{3}{c}{\\textbf{Reactive Power (Mvar)}} \\\\ \\cmidrule(lr){3-5} \\cmidrule(l){6-8} ",
        " &  & \\textbf{Py} & \\textbf{ANA} & \\textbf{Diff} & \\textbf{Py} & \\textbf{ANA} & \\textbf{Diff} \\\\ \\midrule"
    ]
    
    for b in [1, 2, 5, 8, 11, 13]:
        if b in bus_data and b in anarede_ref:
            py_p = bus_data[b]['p']
            py_q = bus_data[b]['q']
            ana_p = anarede_ref[b]['p']
            ana_q = anarede_ref[b]['q']
            diff_p = py_p - ana_p
            diff_q = py_q - ana_q
            
            line = f"{b} & {anarede_ref[b]['type']} & {py_p:.1f} & {ana_p:.1f} & {diff_p:+.1f} & {py_q:.1f} & {ana_q:.1f} & {diff_q:+.1f} \\\\"
            table_lines.append(line)
            
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{table}")
    
    with open(os.path.join(tables_dir, "tab_comp_geracao.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print("Tabela II (Geração P/Q) gerada.")

def generate_convergence_summary_table(tables_dir, convergence_report_path):
    if not os.path.exists(convergence_report_path):
        print("Aviso: Relatório de convergência não encontrado para Tabela III.")
        return
        
    with open(convergence_report_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    last_iters = []
    for line in reversed(lines):
        if "---" in line or line.strip() == "": continue
        
        if "Convergente" in line or "Divergente" in line or "Falha" in line or "OK" in line:
            # Novo formato: 1 | 1.02673 | 291.03 | 0.05000 | Convergente
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                try:
                    iter_num = parts[0]
                    lam = float(parts[1])
                    mw_val = parts[2]
                    step_val = float(parts[3])
                    status = parts[4]
                    
                    if "Convergente" in status or "OK" in status:
                        status_tex = "Converged"
                    else:
                        status_tex = "\\textbf{Divergent}"
                        
                    last_iters.append((iter_num, status_tex, f"{step_val:.4f}", f"{lam:.4f}", mw_val))
                except Exception as e:
                    pass
        
        if len(last_iters) >= 5: break
        
    last_iters.reverse()
    
    table_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Summary of the Final Iterations from the Convergence Report}",
        "\\label{tab:convergencia}",
        "\\begin{tabular}{llccr}",
        "\\toprule",
        "\\textbf{Iter} & \\textbf{Status} & \\textbf{Step} & $\\bm{\\lambda}$ \\textbf{(Total)} & \\textbf{Load (MW)} \\\\ \\midrule",
        "... & ... & ... & ... & ... \\\\"
    ]
    
    for (it, st, step, lam, mw) in last_iters:
        table_lines.append(f"{it} & {st} & {step} & {lam} & {mw} \\\\")
        
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{table}")
    
    with open(os.path.join(tables_dir, "tab_convergencia.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print("Tabela III (Convergência Python) gerada.")

def generate_anarede_convergence_table(tables_dir):
    # Texto hardcoded fornecido pelo usuário (ANAREDE original)
    anarede_log = """
   973 Convergente    2  194.600  194.600  194.600   834.90 MW      0.2000
                      2  194.600  194.600  194.600   371.79 Mvar    0.2000
   974 Convergente    3  194.800  194.800  194.800   835.46 MW      0.2000
                      3  194.800  194.800  194.800   372.04 Mvar    0.2000
   975 Convergente    3  195.000  195.000  195.000   836.03 MW      0.2000
                      3  195.000  195.000  195.000   372.29 Mvar    0.2000
   976 Divergente     5                                             0.2000
                      5                                             0.2000
   977 Nao Converg.  31                                             0.1000
                     31                                             0.1000
   978 Nao Converg.  31                                             0.0500
                     31                                             0.0500
"""
    
    last_iters = []
    lines = anarede_log.strip().split('\n')
    for line in lines:
        if "MW" in line or "Divergente" in line or "Nao Converg." in line:
            parts = line.split()
            iter_num = parts[0]
            status = parts[1]
            if status == "Nao": 
                status = "Nao Converg."
            
            if "Convergente" in status:
                status_tex = "Converged"
                lam_val = float(parts[3]) / 100.0  # ANAREDE usa lambda * 100
                mw_val = parts[6]
                step_val = float(parts[-1]) / 100.0
            else:
                status_tex = "\\textbf{Divergent}"
                lam_val = "---"
                mw_val = "---"
                step_val = float(parts[-1]) / 100.0
                
            last_iters.append((iter_num, status_tex, f"{step_val:.4f}", f"{lam_val:.4f}" if isinstance(lam_val, float) else lam_val, mw_val))
            
    table_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Summary of the Final Iterations from ANAREDE Convergence Report}",
        "\\label{tab:convergencia_anarede}",
        "\\begin{tabular}{llccr}",
        "\\toprule",
        "\\textbf{Iter} & \\textbf{Status} & \\textbf{Step} & $\\bm{\\lambda}$ \\textbf{(Total)} & \\textbf{Load (MW)} \\\\ \\midrule",
        "... & ... & ... & ... & ... \\\\"
    ]
    
    for (it, st, step, lam, mw) in last_iters:
        table_lines.append(f"{it} & {st} & {step} & {lam} & {mw} \\\\")
        
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{table}")
    
    with open(os.path.join(tables_dir, "tab_convergencia_anarede.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print("Tabela IV (Convergência ANAREDE) gerada.")

def main(base_dir="outputs"):

    tables_dir = os.path.join(base_dir, "tables")
    figures_dir = os.path.join(base_dir, "figures")
    
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Copiar Figuras (PDFs)
    figures_to_copy = [
        ("ieee_30_anarede_pwf_tcc/pv_figures/curva_pv_sistema_30.pdf", "fig_pv_30.pdf"),
        ("ieee_39_barras_new_england/pv_figures/curva_pv_sistema_39.pdf", "fig_pv_39.pdf"),
        ("ieee_30_anarede_pwf_tcc/reports/heatmap_spearman_ramos_30.pdf", "fig_heatmap_spearman_30.pdf"),
    ]
    
    for src_rel, dst_name in figures_to_copy:
        src = os.path.join(base_dir, src_rel)
        dst = os.path.join(figures_dir, dst_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Figura copiada: {dst_name}")
        else:
            print(f"AVISO: Figura não encontrada: {src}")

    # 2. Gerar Tabela de Correlação (Spearman IEEE 30)
    spearman_path = os.path.join(base_dir, "ieee_30_anarede_pwf_tcc/reports/correlacao_spearman_ramos_30.csv")
    if os.path.exists(spearman_path):
        df_spearman = pd.read_csv(spearman_path, index_col=0)
        df_latex = df_spearman.reset_index()
        generate_latex_table_from_df(
            df_latex, 
            os.path.join(tables_dir, "tab_spearman_30.tex"),
            label="tab:spearman",
            caption="Spearman Rank Correlation between VSIs at Collapse Point (IEEE 30 Bus)"
        )
        print("Tabela Spearman gerada.")

    # 3. Gerar Tabela N-1 Ranking (IEEE 30)
    ranking_path = os.path.join(base_dir, "ieee_30_anarede_pwf_tcc/contingencies/ranking_contingencias.csv")
    if os.path.exists(ranking_path):
        df_ranking = pd.read_csv(ranking_path).head(10) # Top 10
        generate_latex_table_from_df(
            df_ranking, 
            os.path.join(tables_dir, "tab_contingency_ranking.tex"),
            label="tab:n1_ranking",
            caption="Top 10 Most Severe N-1 Contingencies (IEEE 30 Bus)"
        )
        print("Tabela N-1 gerada.")

    # 4. Gerar Tabelas do Artigo (Baseadas no ANAREDE IEEE 30)
    initial_report_path = os.path.join(base_dir, "ieee_30_anarede_pwf_tcc/reports/relatorio_inicial_base_30.txt")
    convergence_report_path = os.path.join(base_dir, "ieee_30_anarede_pwf_tcc/reports/relatorio_convergencia_base_30.txt")
    
    generate_voltage_comparison_table(tables_dir, initial_report_path)
    generate_generation_comparison_table(tables_dir, initial_report_path)
    generate_convergence_summary_table(tables_dir, convergence_report_path)
    generate_anarede_convergence_table(tables_dir)

if __name__ == "__main__":
    main()
