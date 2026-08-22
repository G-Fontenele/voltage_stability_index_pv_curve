# WCNPS 2026 - Review Tracking

| ID | Reviewer | Criticism | Action | Code File | Simulation? | Result | Status | Evidence | Manuscript Section |
|---|---|---|---|---|---|---|---|---|---|
| R1.1 | — | Research gap vago | Reescrever intro com gap quantitativo | — | ❌ | — | Planned | — | Sec. I |
| R1.2 | — | L-Index anomalia | Bug corrigido (bus_to_idx) | `vsi_lib.py`, `analysis_tools.py` | ✅ | Comportamento monotônico | Implemented in code | `tests/test_l_index.py` | Sec. IV-B |
| R1.3 | — | Sem Q-limits | Transição PV→PQ | `simulation_engine.py:110-141` | ✅ | λ_max reduzido | Validated | `outputs_revised/ieee_30/` | Sec. II-A |
| R1.4 | — | Sem N-1 | Loop de contingências | `main.py:122-168` | ✅ | Ranking CSV | Validated | `contingencies/ranking.csv` | Sec. IV-C |
| R1.5 | — | Apenas visual, sem métrica | Spearman + Kendall | `analysis_tools.py:471-516` | ✅ | Matrizes CSV | Validated | `correlacao_spearman.csv` | Sec. IV-B |
| R1.6 | — | NVSI dobra artificial | `np.abs()` removido | `vsi_lib.py:46-51` | ✅ | Sem dobra | Validated | `tests/test_vsi.py` | Sec. II-B |
