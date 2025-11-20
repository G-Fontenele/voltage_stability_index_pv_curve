# Simulador de Estabilidade de Tensão (CPF - Python)

Este repositório contém uma ferramenta desenvolvida em Python para análise de **Estabilidade de Tensão** em sistemas elétricos de potência. O software implementa o método do **Fluxo de Potência Continuado (CPF)** com passo adaptativo, replicando metodologias utilizadas em softwares comerciais (como o ANAREDE) e validadas em literatura acadêmica.

O projeto foi desenvolvido no contexto de um Mestrado em Engenharia Elétrica (Disciplina de Controle e Estabilidade de Tensão), com o objetivo de reproduzir e expandir resultados de referência (TCC Madureira, UFRJ).

## 🚀 Funcionalidades Principais

* **Fluxo de Potência Continuado (CPF):** Algoritmo Preditor-Corretor com passo adaptativo (reduz o passo ao detectar divergência) para traçar a Curva PV completa até o "nariz" (Ponto de Colapso).
* **Alta Resolução:** Configurado para realizar centenas de simulações com passos finos (0.5%), garantindo fidelidade na detecção do limite de estabilidade.
* **Simulação de Despacho Distribuído:** Capacidade de redistribuir o aumento de carga entre os geradores ativos (Distributed Slack), evitando sobrecarga irrealista na barra de referência.
* **Múltiplos Sistemas IEEE:** Suporte nativo para IEEE 14, 30, 39, 57 e 118 barras.
* **Índices de Estabilidade de Tensão (VSIs):** Cálculo automático de 17 índices de estabilidade (Linha e Barra), incluindo:
    * FVSI, Lmn, LQP, Lp, NLSI, NVSI, VSI_2.
    * L-Index e VCPI (Barra) otimizados via álgebra linear.
* **Relatórios Estilo ANAREDE:**
    * `relatorio_colapso.txt`: Estado detalhado do sistema no ponto de máxima carga.
    * `relatorio_convergencia.txt`: Log passo-a-passo de todas as iterações (sucessos e divergências).

## 📂 Estrutura do Projeto

O código foi modularizado para facilitar a manutenção e escalabilidade:

* **`main.py`**: Orquestrador principal. Gerencia a configuração, seleção do sistema, execução do loop principal e cronometragem.
* **`simulation_engine.py`**: "Motor" da simulação. Contém a lógica do CPF, controle de passo adaptativo (`steps`, `min_step`) e aplicação dos fatores de escala (`lambda`).
* **`analysis_tools.py`**: Ferramentas de pós-processamento. Responsável pela álgebra matricial (Ybus, F-matrix), geração dos CSVs de resultados, plotagem de gráficos e criação dos relatórios `.txt`.
* **`vsi_lib.py`**: Biblioteca matemática pura contendo as equações de todos os índices de estabilidade implementados.

## 🛠️ Instalação e Dependências

Certifique-se de ter o Python 3.8+ instalado. Instale as dependências necessárias:

```bash
pip install pandapower numpy pandas matplotlib scipy