# Simulador de Estabilidade de Tensão (CPF - Python)

Ferramenta computacional avançada para análise de **Estabilidade de Tensão** em sistemas elétricos de potência. O software implementa o método do **Fluxo de Potência Continuado (CPF)** com passo adaptativo e despacho distribuído, desenhado para replicar a metodologia de softwares industriais (ANAREDE) e resultados acadêmicos de referência.

Desenvolvido no âmbito do Mestrado em Engenharia Elétrica (Disciplina de Controle e Estabilidade de Tensão), com validação baseada no TCC de *Madureira (2023)* e na literatura clássica (*Kundur, Van Cutsem*).

## 🚀 Funcionalidades Principais

* **Fluxo de Potência Continuado (CPF) de Alta Resolução:** Algoritmo de incremento de carga com refinamento de passo (Backtracking). Configurado para realizar centenas de iterações com passos finos (0.2%), permitindo o traçado suave da Curva PV e a detecção precisa do Ponto de Colapso.
* **Despacho Distribuído (Distributed Slack):** Redistribuição automática do incremento de carga entre geradores ativos (mantendo os fatores de participação constantes), evitando a sobrecarga irrealista da barra de referência (Slack).
* **Ajuste Fino de Participação:** Funcionalidade específica para o sistema IEEE 30 que ajusta o despacho inicial do Gerador 2 para 13.3% da carga total, reproduzindo exatamente os cenários de referência do TCC.
* **Análise de Múltiplos Sistemas:** Suporte nativo e execução em lote (Bateria de Testes) para os sistemas **IEEE 14, 30, 39, 57 e 118 barras**.
* **Cálculo de Índices de Estabilidade (VSIs):** Biblioteca completa com 17 índices de estabilidade (Linha e Barra) calculados automaticamente para cada cenário, incluindo:
    * **Linha:** FVSI, Lmn, LQP, Lp, NLSI, NVSI, VSLI, VSI_2, VQI, PTSI.
    * **Barra:** L-Index e VCPI (otimizados via álgebra matricial).
* **Relatórios Técnicos:**
    * **Relatório de Colapso:** Estado detalhado do sistema (tensões, fluxos) no ponto crítico.
    * **Relatório de Convergência:** Log passo-a-passo idêntico ao gerado pelo software ANAREDE.

## 📂 Estrutura do Projeto

O código foi modularizado para facilitar a manutenção e escalabilidade:

* **`main.py`**: **Orquestrador Principal**. Gerencia a configuração (`CONFIG`), menu de seleção de sistemas, criação da estrutura de pastas, cronometragem e execução do loop principal.
* **`simulation_engine.py`**: **Motor Numérico**. Contém a lógica do CPF, controle de passo adaptativo (`steps`, `min_step`), aplicação dos fatores de escala (`lambda`) em P e Q, e gerenciamento de divergências.
* **`analysis_tools.py`**: **Pós-processamento**. Responsável pelo pré-cálculo de matrizes (Ybus, Matriz F), geração dos CSVs de resultados, plotagem dos gráficos (Curvas PV e Índices) e escrita dos relatórios `.txt`.
* **`vsi_lib.py`**: **Biblioteca Matemática**. Contém as equações puras de todos os índices de estabilidade implementados (FVSI, Lmn, etc.).

## 🛠️ Instalação e Dependências

Certifique-se de ter o Python 3.8+ instalado. Instale as bibliotecas necessárias:

```bash
pip install pandapower numpy pandas matplotlib scipy
```

## ⚙️ Uso e Configuração

1. Execute o arquivo principal:

```bash
python main.py
```

2. Selecione o sistema desejado no menu interativo:

```bash
SELEÇÃO DO SISTEMA ELÉTRICO:
  [0] TODAS AS REDES (Bateria de Testes)
  [1] IEEE 14 Barras
  [2] IEEE 30 Barras
  ...
```

Digite 0 para rodar todos os sistemas sequencialmente.

# Parâmetros de Simulação (main.py)

A configuração padrão ("Alta Fidelidade") visa replicar o estudo de referência:

```python
CONFIG = {
    'load_scaling_bus_id': None,   # None = Escala carga de todo o sistema (Global)
    'enforce_q_lims': False,       # False = Q Infinito (Curva Teórica/TCC) | True = Realista
    'distributed_slack': True,     # True = Geradores ativos ajudam a Slack (Física correta)
    'max_scale': 5.0,              # Teto de segurança para o Lambda
    'steps': 0.002,                # Passo Fino (0.2%) para alta resolução da curva
    'min_step': 0.00001            # Precisão extrema (1e-5) para o Ponto de Colapso
}
```

## 📊 Saída e Resultados

Os resultados são organizados automaticamente dentro da pasta outputs/, segregados por caso para evitar mistura de dados:

```
outputs/
└── ieee_30_barras/
    ├── index_sheets/        # Tabelas CSV com os índices para cada cenário (0%, 25%...)
    ├── index_figures/       # Gráficos comparativos da evolução de cada índice
    ├── pv_figures/          # Curva PV colorida com destaque para a barra crítica
    └── reports/
        ├── relatorio_colapso.txt       # Raio-X do sistema no ponto de falha
        └── relatorio_convergencia.txt  # Log passo-a-passo (Réplica ANAREDE)
```

## 📝 Nota Metodológica

Este simulador utiliza o método de Incremento de Carga com Refinamento de Passo (Step-wise Load Increase with Refinement). Diferente de métodos de continuação por parametrização completa (que traçam a parte instável da curva PV), esta abordagem foca na determinação exata do Ponto de Máximo Carregamento (PMC) na região estável.

Esta escolha metodológica garante equivalência numérica com os relatórios de convergência de ferramentas comerciais como o ANAREDE e é suficiente para a determinação da Margem de Estabilidade de Tensão.

Autor: Gonçalo Fontenele
Curso: Mestrado em Engenharia Elétrica 
Disciplina: Controle e Estabilidade de Tensão
Instituição: COPPE/UFRJ