# Predição de Risco de Evasão no Ensino Superior

Projeto de **Machine Learning para identificação de cursos de graduação com alto risco de evasão**, utilizando dados administrativos públicos do Censo da Educação Superior do INEP.

> **Projeto final da disciplina CIN0144 — Aprendizado de Máquina e Ciência de Dados, CIn-UFPE.**

---

## Visão geral

A evasão no ensino superior representa um problema relevante para instituições de ensino e para políticas públicas: reduz o aproveitamento das vagas disponíveis, dificulta o planejamento institucional e impacta diretamente a trajetória dos estudantes.

Este projeto investiga uma questão objetiva:

> **É possível identificar cursos com alto risco de evasão utilizando apenas indicadores administrativos agregados, disponíveis publicamente?**

Para responder a essa pergunta, foi desenvolvido um pipeline de Machine Learning capaz de transformar dados do **Censo da Educação Superior 2024 (INEP)** em uma classificação binária de risco.

O modelo trabalha no nível de **curso**, e não no nível individual do aluno.

### O que o sistema faz

```text
Dados administrativos do curso
            ↓
Construção do indicador de evasão
            ↓
Definição do risco
            ↓
Preparação e engenharia de features
            ↓
Treinamento dos modelos
            ↓
Validação cruzada
            ↓
Classificação de risco
```

O objetivo não é substituir políticas de retenção, mas fornecer um **indicador preditivo que possa apoiar a priorização de análises e intervenções**.

---

## Resultado executivo

Os resultados mostram que é possível obter desempenho consistente na classificação de cursos de alto risco, mas também revelam uma limitação importante: **os dados administrativos agregados possuem sinal preditivo limitado**.

Os três principais modelos apresentaram desempenho bastante próximo:

| Indicador                        |                              Resultado |
| -------------------------------- | -------------------------------------: |
| F1 dos três melhores modelos     |                      **0,790 – 0,793** |
| Recall dos três melhores modelos |                      **0,754 – 0,773** |
| Melhor Recall                    |                             **0,7733** |
| Melhor ROC-AUC                   |                             **0,8692** |
| Validação                        | **5-fold Stratified Cross-Validation** |

A diferença entre os dois primeiros colocados não foi estatisticamente significativa pelo teste de Wilcoxon pareado (`p = 0,6250`).

Portanto, o resultado não sustenta a afirmação de que existe um único "melhor modelo". O principal achado é que **diferentes abordagens de modelagem alcançaram desempenho semelhante neste problema e protocolo de avaliação**.

### O que isso significa

O projeto prioriza **Recall** porque, no contexto analisado, deixar de identificar um curso realmente em situação de alto risco pode ser mais prejudicial do que sinalizar um curso que posteriormente não necessitaria de intervenção.

Ao mesmo tempo, os modelos reais superam apenas marginalmente os baselines triviais utilizados como referência:

| Estratégia                  |      Score |
| --------------------------- | ---------: |
| Melhor modelo               | **0,7875** |
| Baseline sempre positivo    |     0,7750 |
| Baseline classe majoritária |     0,7750 |

Essa diferença é pequena e é uma informação importante sobre o problema: **o conjunto de dados utilizado não contém informação suficiente para produzir um classificador altamente discriminativo**.

Isso é coerente com a natureza dos dados. O dataset contém características agregadas dos cursos, mas não informações individuais como desempenho acadêmico, situação socioeconômica ou motivos de evasão.

---

## O problema de Machine Learning

A unidade de análise é um **curso de graduação em uma instituição em determinado ano do Censo**.

A variável alvo é:

```text
alto_risco_evasao
```

com:

* `0` → baixo risco;
* `1` → alto risco.

O dataset utilizado é o **Censo da Educação Superior 2024**, disponibilizado pelo INEP.

| Característica                 | Definição                         |
| ------------------------------ | --------------------------------- |
| Fonte                          | Censo da Educação Superior — INEP |
| Ano                            | 2024                              |
| Unidade de análise             | Curso de graduação                |
| Registros no protocolo oficial | 30.000                            |
| Features brutas utilizadas     | 22                                |
| Target                         | `alto_risco_evasao`               |
| Tipo de problema               | Classificação binária             |

O dataset não é versionado no repositório por possuir mais de 100 MB e deve ser obtido diretamente do INEP.

---

## Construção do target

O target principal é construído a partir da taxa de evasão:

```text
QT_EVADIDOS = QT_SIT_DESVINCULADO + QT_SIT_TRANCADA

taxa_evasao = (QT_EVADIDOS / QT_ING) × 100

alto_risco_evasao =
    1, se taxa_evasao ≥ 20%
    0, caso contrário
```

O limiar de **20%** foi mantido como definição operacional do projeto.

Essa escolha é importante: o modelo não está prevendo "evasão" em termos absolutos. Ele está classificando cursos segundo uma **regra de risco definida a partir da taxa de evasão**.

---

## Construção das features

Uma parte central do projeto foi transformar as variáveis administrativas disponíveis em informações adequadas para modelagem.

Foram utilizadas três categorias principais:

### Características administrativas

Informações relacionadas à organização acadêmica, rede, categoria administrativa, grau acadêmico e modalidade de ensino.

### Indicadores quantitativos

Contagens relacionadas a ingressantes, matriculados, vagas e inscritos.

### Features derivadas

Foram criados indicadores como:

* `TAXA_CONCLUSAO`
* `RAZAO_ING_MAT`
* `PROPORCAO_EAD`
* `PROPORCAO_NOTURNO`
* `INDICE_FINANCIAMENTO`
* `PROPORCAO_FIES`
* `PROPORCAO_PROUNIP`
* `PROPORCAO_18_24`
* `PROPORCAO_FEM`

A engenharia de features foi adaptada ao comportamento de cada abordagem de modelagem, incluindo seleção de features, transformações de escala, interações e discretização quando apropriado.

Isso evita tratar o processo como simplesmente:

```text
Dataset → algoritmo
```

e estrutura o problema como:

```text
Dados brutos
    ↓
Definição do target
    ↓
Remoção de informação inválida
    ↓
Transformação das variáveis
    ↓
Engenharia de features
    ↓
Modelagem
```

---

## Prevenção de Data Leakage

Um dos pontos críticos identificados durante a avaliação foi o risco de **vazamento de informação do target para as features**.

Foram removidos quatro grupos principais de variáveis:

| Grupo                                        | Motivo                                                           |
| -------------------------------------------- | ---------------------------------------------------------------- |
| Identificadores                              | Evitar que o modelo memorize registros específicos               |
| Variáveis usadas na construção do target     | Permitiriam reconstruir diretamente o target                     |
| Variáveis relacionadas a matrícula/conclusão | Podem carregar informação estrutural sobre o resultado da evasão |
| Features derivadas do próprio target         | Representam vazamento direto ou indireto                         |

Um caso particularmente relevante foi `RAZAO_CONCLUSAO_EVASAO`, que apresentava importância de aproximadamente **0,44** na Árvore de Decisão quando presente.

A variável foi identificada como derivada de informação utilizada na construção do target e removida da arquitetura atual.

Esse processo foi fundamental para evitar que uma métrica aparentemente alta representasse apenas **informação vazada**, e não capacidade real de generalização.

---

## Validação

A avaliação foi estruturada para medir o comportamento dos modelos fora dos dados utilizados para treinamento.

### Protocolo

* **5-fold Stratified Cross-Validation**
* `shuffle=True`
* `random_state=42`
* tratamento de desbalanceamento específico para cada abordagem;
* ajuste de threshold;
* métricas calculadas em cada fold;
* agregação dos resultados por média e desvio padrão.

As transformações que aprendem parâmetros dos dados são ajustadas dentro do processo de validação, evitando que informações do conjunto de validação sejam utilizadas durante o treinamento.

### Threshold

O threshold de decisão foi calibrado utilizando **Youden's J**, com uma restrição adicional de prevalência para evitar classificadores degenerados.

Essa etapa foi importante porque uma avaliação inicial do projeto apresentava um comportamento artificial: um threshold extremamente baixo fazia a Regressão Logística classificar praticamente todos os casos como alto risco, produzindo recall próximo de 1,0.

O resultado foi identificado durante a auditoria e não faz parte dos resultados oficiais apresentados neste README.

---

## Resultados da modelagem

Os resultados oficiais foram obtidos após a auditoria metodológica e utilizando o protocolo unificado de avaliação.

| Abordagem                 | Accuracy | Precision |     Recall |         F1 |    ROC-AUC |
| ------------------------- | -------: | --------: | ---------: | ---------: | ---------: |
| Melhor desempenho geral   |   0,8004 |    0,8240 |     0,7685 | **0,7934** | **0,8692** |
| Segunda melhor abordagem  |   0,7962 |    0,8106 | **0,7733** |     0,7914 |     0,8218 |
| Terceira melhor abordagem |   0,7998 |    0,8307 |     0,7536 |     0,7901 |     0,8691 |
| Referência trivial        |        — |         — |          — |          — |          — |

> Os três primeiros resultados possuem desempenho próximo. O teste estatístico realizado não encontrou diferença significativa entre os dois primeiros colocados em recall.

### Principal conclusão dos resultados

O projeto não apresenta um modelo como solução definitiva.

A evidência disponível indica que:

1. diferentes modelos conseguem aprender algum sinal preditivo;
2. os três melhores apresentam desempenho semelhante;
3. o ganho sobre estratégias triviais é pequeno;
4. os dados agregados impõem uma limitação importante ao problema.

Essa conclusão é mais relevante do que simplesmente selecionar o algoritmo com a maior métrica.

---

## O que o modelo aprendeu?

Na análise de importância da Árvore de Decisão, as principais variáveis foram:

| Feature          | Importância |
| ---------------- | ----------: |
| `RAZAO_ING_MAT`  |      0,5112 |
| `QT_ING`         |      0,3294 |
| `QT_MAT`         |      0,1136 |
| `TAXA_CONCLUSAO` |      0,0459 |

As demais features apresentaram importância nula nessa árvore de referência.

Esse resultado indica que grande parte do sinal utilizado pelo modelo está concentrada em **relações entre ingressantes e matriculados e no volume de alunos do curso**, enquanto as demais características administrativas possuem participação muito menor nessa configuração.

---

## Arquitetura

O projeto foi organizado como um pipeline de Machine Learning modular:

```text
                    ┌─────────────────────┐
                    │   Dataset INEP      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   Data Loading      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Target Construction │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Leakage Prevention  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Preprocessing      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │      Training       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Cross Validation    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │    Evaluation       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │      Results        │
                    └─────────────────────┘
```

A arquitetura separa as principais responsabilidades do pipeline:

```text
src/
├── configs/          # configurações e hiperparâmetros
├── data/             # loading, target e leakage
├── pipelines/        # preprocessing e feature engineering
├── models/           # definição dos estimadores
├── training/         # treinamento e validação
├── evaluation/       # métricas e comparação
└── visualization/    # geração de gráficos

experiments/           # experimentação e visualização
scripts/               # execução via linha de comando
docs/                  # documentação técnica
results/               # resultados e análises
```

A separação permite que o processo de treinamento seja executado sem depender da lógica de um notebook e mantém as principais etapas do pipeline isoladas.

---

## Tecnologias

| Tecnologia           | Uso                                    |
| -------------------- | -------------------------------------- |
| Python               | Linguagem principal                    |
| pandas               | Manipulação dos dados                  |
| NumPy                | Computação numérica                    |
| scikit-learn         | Pré-processamento, modelos e validação |
| imbalanced-learn     | Tratamento de desbalanceamento         |
| Matplotlib / Seaborn | Visualização                           |
| Jupyter              | Experimentação                         |
| SciPy                | Análises estatísticas                  |

---

## Como executar

### 1. Criar o ambiente virtual

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

Para executar notebooks e ferramentas de desenvolvimento:

```bash
pip install -r requirements-dev.txt
```

### 3. Baixar o dataset

Baixe os microdados do **Censo da Educação Superior 2024** diretamente do INEP e coloque o arquivo:

```text
MICRODADOS_CADASTRO_CURSOS_2024.CSV
```

na raiz do projeto ou no diretório `data/`.

O dataset não é versionado no Git por exceder 100 MB.

### 4. Executar o treinamento

```bash
python scripts/train_all.py
```

Também é possível especificar parâmetros:

```bash
python scripts/train_all.py \
    --n-splits 5 \
    --random-state 42 \
    --output results/comparacao.csv
```

---

## Limitações

Os resultados precisam ser interpretados dentro das limitações do problema.

### Dados agregados

O dataset trabalha no nível de curso, não de aluno. Portanto, não possui informações individuais que potencialmente explicariam uma parcela importante da evasão.

### Definição do risco

O conceito de "alto risco" foi operacionalizado utilizando o corte:

```text
taxa_evasao ≥ 20%
```

Esse limiar não foi calibrado a partir de uma política institucional real.

### Sinal preditivo limitado

A proximidade entre os modelos e os baselines mostra que existe um limite importante na capacidade preditiva dos dados disponíveis.

Assim, o resultado deve ser interpretado como **um indicador de risco**, e não como uma previsão determinística de que determinado curso irá sofrer evasão.

### Pipelines diferentes

O Naive Bayes utiliza uma definição de target e um universo de features diferente dos demais modelos. Por isso, seus resultados não devem ser interpretados como uma comparação completamente equivalente com os outros quatro pipelines.

---

## Próximos passos

A evolução natural do projeto é aumentar a quantidade e a qualidade do sinal disponível para o modelo.

* Incorporar indicadores socioeconômicos regionais, como **IDH municipal e renda per capita**.
* Recalibrar o limiar de alto risco de acordo com uma política institucional real.
* Unificar o universo de features utilizado nas diferentes abordagens de modelagem.
* Completar a migração das etapas restantes para a arquitetura `src/`.
* Avaliar novamente o desempenho após o enriquecimento dos dados.

---

## Documentação técnica

O README apresenta a visão geral do projeto. Os detalhes metodológicos e de implementação estão separados na documentação técnica:

| Documento                     | Conteúdo                              |
| ----------------------------- | ------------------------------------- |
| `ARCHITECTURE_AUDIT.md`       | Auditoria metodológica e arquitetural |
| `docs/ARCHITECTURE.md`        | Organização da arquitetura            |
| `docs/DATA_PIPELINE.md`       | Dados, target e prevenção de leakage  |
| `docs/FEATURE_ENGINEERING.md` | Engenharia de features                |
| `docs/MODELS.md`              | Configuração dos modelos              |
| `docs/TRAINING.md`            | Treinamento e validação               |
| `docs/EVALUATION.md`          | Métricas e avaliação                  |
| `docs/EXPERIMENTS.md`         | Experimentação                        |
| `docs/FILE_CLEANUP.md`        | Limpeza e organização dos arquivos    |
| `docs/VALIDATION_REPORT.md`   | Validação final da arquitetura        |

---

## Conclusão

Este projeto explora um problema real de **classificação supervisionada** utilizando dados públicos e um pipeline completo de Machine Learning.

Mais do que comparar algoritmos, o trabalho envolve as etapas necessárias para transformar dados administrativos em um problema de modelagem: **definição do target, engenharia de features, prevenção de leakage, tratamento do desbalanceamento, validação cruzada, calibração de decisão e análise crítica dos resultados**.

O principal resultado não é simplesmente encontrar um algoritmo vencedor, mas entender **quanto os dados disponíveis conseguem explicar o fenômeno e quais limitações permanecem para que o modelo seja utilizado em um contexto real**.
