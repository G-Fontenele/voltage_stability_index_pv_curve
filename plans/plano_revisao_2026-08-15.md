# Plano de Revisão Técnica - Estabilidade de Tensão (Atualizado em 15/08/2026)

Este documento centraliza o plano detalhado de refatoração e auditoria do projeto de análise de estabilidade de tensão (Voltage Stability Indices - VSIs). O objetivo principal desta força-tarefa é responder às críticas dos revisores do artigo científico, garantindo que a matemática subjacente, o comportamento dos indicadores (singularidades) e as premissas físicas (limites de Q e contingências N-1) sejam rigorosamente defensáveis.

---

## 🟢 O Que Já Foi Concluído (Fases A e B)

**Fase A — Correção e Validação Matemática dos VSIs**
- `[Concluído]` **Criação de suíte de testes unitários** (`tests/test_vsi.py` e `tests/test_l_index.py`).
- `[Concluído]` **Mapeamento robusto `bus_to_idx`:** O L-Index descartava silenciosamente as barras cujos IDs externos não batiam sequencialmente com os internos, pois o código utilizava `in np.ndarray` de forma insegura. Substituído por mapeamento numérico explícito (`bus_lookup`).
- `[Concluído]` **Remoção da dobra artificial no NVSI:** O uso de `np.abs()` no denominador mascarava fisicamente o colapso do sistema (ponto de bifurcação / máximo carregamento), forçando o indicador a continuar com valores positivos após ultrapassar a singularidade. O módulo de cálculo (`vsi_lib.py`) foi limpo dessas maquiagens gráficas.
- `[Concluído]` **Topologia direcional para o FVSI e Lmn:** Ajustado o carregamento da variável `Q_r` baseando-se no sinal físico do fluxo ativo (`is_fwd`), eliminando `np.abs()` em linhas que atuam capacitivamente suportando tensão local.
- `[Concluído]` **Validação do IEEE 30 Padrão:** Executado o mock de validação que convergiu de forma impecável, alocando saídas num diretório temporário `outputs_revised` para preservar o baseline histórico.

**Fase B — Conserto do Pós-processamento de Dados e Figuras**
- `[Concluído]` **Desacoplamento Nodal/Ramo:** O script de simulação gravava os resultados de linha, mas falhava em separar nativamente os resultados Nodais (bus) dos de Ramo (branch). Separados e extraídos individualmente (CSV/DataFrame).
- `[Concluído]` **Recuperação de Barras "Sumidas":** A função `plot_comparative_indices` realizava deduplicação incorreta baseando-se exclusivamente no terminal "To" dos ramos, acarretando perda invisível de barras com injeção de geração e fluxos reversos. A função agora recebe as planilhas Nodais de forma autônoma para criar seus scatter-plots.

---

## 🟡 O Que Vem a Seguir (Para as Próximas Sessões)

As etapas abaixo representam as críticas mais pesadas dos avaliadores científicos (particularmente sobre os limites reativos dos geradores e as contingências N-1).

### Fase C — Robustez do Motor CPF e Limites de Q (Q-Limits)
*O motor de continuação atual assume que o barramento de geração possui capacidade reativa infinita.*
- **Implementação do Checkpoint:** Salvar o estado da rede (`last_good_net = copy.deepcopy(net)`) no início de cada degrau de carregamento no `simulation_engine.py`.
- **Validação de Limites ($Q_{max}$ e $Q_{min}$):** Ao invés de aceitar passos de carga cegamente, a simulação inspecionará a injeção reativa do gerador `net.res_gen['q_mvar']`.
- **Transição PV $\rightarrow$ PQ:** Caso limite estoure, o script deve:
  1. Reverter para a `last_good_net`.
  2. Reduzir o degrau de continuação para encontrar a fronteira exata de limitação (OPCIONAL, dependendo da necessidade de precisão).
  3. Transformar o barramento gerador limitante para tipo PQ fixando a injeção ao máximo ($Q_{max}$).
- **Recálculo Dinâmico:** Uma vez que uma barra vira PQ, a topologia de sensibilidade da rede altera. O projeto deve, a partir desse instante, solicitar o recálculo dinâmico da submatriz Ybus e da F-Matrix (`analysis_tools.py`) a cada violação, em vez de assumir F-matrix estática durante toda a trajetória.

### Fase D — Contingências (N-1)
*O artigo só exibia cenários saudáveis (N-0), o que os revisores apontaram como insuficiente.*
- Refatorar ou clonar o script `main.py` para iterar não apenas o loading, mas um laço externo cortando uma linha vital do sistema por vez.
- Levantar curvas do colapso precoce na contingência severa usando a métrica unificada para provar matematicamente a precisão preditiva do L-Index.

### Fase E — Avaliação Estocástica e Comparação Qualitativa (Spearman)
- O revisor solicitou métricas que justifiquem quantitativamente o porquê de um indicador ser melhor.
- Adicionar pós-processamento utilizando Correlação de Spearman e Kendall Tau para comparar o ranking de criticalidade reportado por cada VSI ao longo da simulação de carregamento e N-1. 

### Fase F — Refação Experimental Final (Batching)
- Realizar a bateria completa de testes nos modelos IEEE 14, 30, 39, 57 e 118 barras após o motor consolidado e a F-Matrix rodarem sem falhas sob stress de limites violados.
- Coletar as tabelas revisadas e reconstruir os painéis do artigo.

---

**NOTA PARA CONTINUIDADE:** 
Para retomar este trabalho a partir da Fase C, o sistema já se encontra funcional do ponto de vista do "núcleo de equações". O foco deve ser unicamente em `simulation_engine.py` (função `run_cpf`) e na sua interface com a função `pre_calculate_matrices` de `analysis_tools.py` para injetar recálculos sob demanda.
