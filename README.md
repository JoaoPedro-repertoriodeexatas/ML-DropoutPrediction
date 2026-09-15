# Predição de Risco de Evasão no Ensino Superior

Projeto final da disciplina **CIN0144 — Aprendizado de Máquina e Ciência de Dados**, CIn-UFPE.

**Equipe:** Gabriel Albertin, Gabriel Fonseca, João Pedro, Luiz Veloso, Victor Conde.

---

## Sumário

1. [Problema](#1-problema)
2. [Resultados](#2-resultados)
3. [Conclusões](#3-conclusões)
4. [Metodologia](#4-metodologia)
5. [Validação](#5-validação)
6. [Arquitetura](#6-arquitetura)
7. [Instalação](#7-instalação)
8. [Execução](#8-execução)

---

## 1. Problema

Evasão em cursos de graduação é um problema estrutural do ensino
superior brasileiro: gera desperdício de vagas públicas e privadas,
prejudica o planejamento institucional e impacta diretamente os alunos
que abandonam. Identificar **quais cursos** apresentam maior risco de
evasão, a partir de dados administrativos que o próprio INEP já coleta
todo ano, permite priorizar ações de retenção antes que o problema se
manifeste em larga escala.

**Objetivo do uso de Machine Learning:** treinar classificadores
binários capazes de prever, a partir de características administrativas
de um curso (perfil dos ingressantes, modalidade de ensino, organização
acadêmica, financiamento estudantil), se aquele curso pertence ao grupo
de **alto risco de evasão** — sem depender de dados individuais de
aluno (não coletados neste dataset), só de indicadores agregados por
curso, já públicos.

**Pergunta central:** *é possível classificar um curso de graduação como
"alto risco de evasão" usando apenas indicadores administrativos
agregados, disponíveis publicamente antes de qualquer aluno concluir ou
abandonar o curso?*

| | |
|---|---|
| **Variável alvo** | `alto_risco_evasao` — binária (0 = baixo risco, 1 = alto risco) |
| **Unidade de análise** | Um **curso de graduação** em uma instituição, em um ano-censo (não é o aluno individual — o dataset é agregado por curso) |
| **Dataset** | Censo da Educação Superior 2024 (INEP) — `MICRODADOS_CADASTRO_CURSOS_2024.CSV` |

A construção exata do alvo está na seção [4.2](#42-target-taxa_evasao-e-alto_risco_evasao).

---

## 2. Resultados

### 2.1 Qual experimento é o resultado oficial

O projeto passou por duas rodadas de auditoria interna, documentadas em
[`ARCHITECTURE_AUDIT.md`](ARCHITECTURE_AUDIT.md). A
primeira rodada de resultados (`models/*/resultados_*.csv`) tinha um
problema metodológico real: o
threshold de decisão de vários modelos era calibrado maximizando F1
**sem nenhum limite de sanidade**, o que inflou artificialmente o recall
da Regressão Logística para ≈1,0 com um threshold ≈0,01 (um padrão
degenerado — o modelo essencialmente prevendo "sempre alto risco").

**Os números desta seção são os oficiais pós-auditoria**
(`results/audit/audited_ranking.csv`, protocolo unificado de 5-fold CV
com threshold calibrado por Youden's J e guarda-corpo de prevalência,
`analysis/audit/`) — a única versão com verificações de sanidade
documentadas (`results/audit/audit_report.md`) e comparação contra
classificadores triviais.

### 2.2 Tabela comparativa (média entre 5 folds ± desvio padrão)

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Árvore de Decisão | 0.8004 ± 0.0040 | 0.8240 ± 0.0376 | 0.7685 ± 0.0484 | 0.7934 ± 0.0122 | 0.8692 ± 0.0033 |
| Naive Bayes (Complement) | 0.5189 ± 0.0028 | 0.8405 ± 0.0182 | 0.0468 ± 0.0069 | 0.0886 ± 0.0125 | 0.5927 ± 0.0140 |
| Regressão Logística | 0.6601 ± 0.0064 | 0.7337 ± 0.0107 | 0.5033 ± 0.0320 | 0.5964 ± 0.0196 | 0.6985 ± 0.0063 |
| SVM (RBF) | 0.7962 ± 0.0035 | 0.8106 ± 0.0096 | 0.7733 ± 0.0132 | 0.7914 ± 0.0044 | 0.8218 ± 0.0058 |
| Rede Neural (MLP) | 0.7998 ± 0.0039 | 0.8307 ± 0.0128 | 0.7536 ± 0.0126 | 0.7901 ± 0.0036 | 0.8691 ± 0.0052 |

*(Média e desvio padrão calculados a partir de `results/audit/<modelo>_folds.csv`, `ddof=1`.)*

### 2.3 Ranking oficial e comparação com baselines triviais

| Rank | Modelo | Score¹ |
|---|---|---:|
| 1 | Árvore de Decisão | 0.7875 |
| 2 | SVM | 0.7866 |
| 3 | Rede Neural | 0.7807 |
| — | *Baseline "sempre positivo"* | *0.7750* |
| — | *Baseline "sempre majoritário"* | *0.7750* |
| 4 | Regressão Logística | 0.5815 |
| 5 | Naive Bayes | 0.2256 |

¹ `score = 0.45·recall + 0.30·f1 + 0.15·precision + 0.10·accuracy` (`analysis/audit/run_audit.py`).

**Teste de significância estatística** (Wilcoxon pareado, `results/audit/audit_report.md`):
entre Árvore de Decisão e SVM em recall, `statistic=5.0000, p-value=0.6250`
— **sem diferença estatisticamente significativa** (p ≥ 0,05) entre os
dois primeiros colocados.

### 2.4 Matriz de confusão (média dos 5 folds)

| Modelo | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Árvore de Decisão | 2497 | 503 | 695 | 2305 |
| SVM | 2457 | 543 | 680 | 2320 |
| Rede Neural | 2538 | 462 | 739 | 2261 |
| Regressão Logística | 2450 | 550 | 1490 | 1510 |
| Naive Bayes | 2973 | 27 | 2860 | 140 |

Fonte: `results/validation/confusion_matrix_summary.md`.

### 2.5 Melhores hiperparâmetros (centralizados em `src/configs/model_hyperparameters.py`)

| Modelo | Hiperparâmetros | Fonte confirmada |
|---|---|---|
| Árvore de Decisão | `criterion=entropy`, `max_depth=15`, `min_samples_split=46`, `min_samples_leaf=13`, `class_weight={0:1,1:2}`, `ccp_alpha=0.00435...`, `min_impurity_decrease=0.00121...` | `models/arvore_de_decisao/melhores_hiperparametros_arvore.txt` (busca `RandomizedSearchCV`, 500 iterações) |
| Regressão Logística | `C=10.0`, `penalty=l1`, `solver=liblinear`, `class_weight=balanced` | `models/regrassao_logisticca/melhores_hiperparametros.txt` (busca `GridSearchCV`) |
| SVM | `C=3.1489`, `gamma=5.6698`, `kernel=rbf`, `class_weight=balanced` | `analysis/audit/evaluator.py` (busca `RandomizedSearchCV`, 12 iterações sobre subamostra de 12.000 registros) |
| Rede Neural | 1 camada oculta (`(n_features+1)//2` neurônios), `activation=relu`, `solver=sgd`, `learning_rate_init=0.01`, `momentum=0.9` | `models/Redes_Neurais/RedesNeurais.ipynb` (arquitetura) + `analysis/audit/evaluator.py` (critério de parada) |
| Naive Bayes (Complement) | `alpha=0.16121...`, `norm=True` | `models/naive_bayes/artifacts/best_params.json` (busca aleatória própria, 90 configurações comparando as 3 variantes) |

### 2.6 Feature importance (Árvore de Decisão)

| Feature | Importância |
|---|---:|
| `RAZAO_ING_MAT` | 0.5112 |
| `QT_ING` | 0.3294 |
| `QT_MAT` | 0.1136 |
| `TAXA_CONCLUSAO` | 0.0459 |
| demais (8 features de organização/modalidade) | 0.0000 |

Fonte: `results/feature_importance/decision_tree_top20.csv` (árvore de
referência com os hiperparâmetros auditados, features comuns sem
seleção adicional — ver `results/feature_importance/feature_importance_analysis.md`).

> Nos notebooks originais, a Árvore de Decisão tem uma feature adicional
> (`RAZAO_CONCLUSAO_EVASAO`) que dominava a importância do modelo. Ela
> foi identificada como vazamento direto de alvo e removida da
> arquitetura atual — ver [seção 4.4](#44-leakage-prevention).

---

## 3. Conclusões

**Não há um vencedor estatisticamente distinguível entre os três
melhores modelos.** Árvore de Decisão, SVM e Rede Neural formam um
grupo próximo (F1 entre 0,790 e 0,793; recall entre 0,754 e 0,773), e o
teste de Wilcoxon (seção 2.3) não encontra diferença significativa entre
os dois primeiros colocados em recall. Afirmar que a Árvore de Decisão
é "o melhor modelo" seria uma leitura mais forte do que os dados
sustentam — o resultado correto é que **esses três modelos têm
desempenho estatisticamente equivalente** neste protocolo.

**Recall é a métrica mais relevante para este problema, não accuracy.**
O objetivo prático é identificar cursos em risco para intervenção — um
falso negativo (curso em risco não identificado) é um custo maior que um
falso positivo (curso sinalizado que não precisava, gerando uma
verificação extra). Por esse critério, SVM (0,7733) e Árvore de Decisão
(0,7685) são os mais adequados; a Regressão Logística, mesmo depois da
correção de threshold, ainda perde quase metade dos casos positivos
(recall 0,5033).

**Achado que exige cautela: os modelos treinados mal superam baselines
triviais.** Os classificadores "sempre positivo" e "sempre majoritário"
pontuam 0,7750 no mesmo score que o melhor modelo real pontua 0,7875 —
uma diferença pequena. Isso não invalida o trabalho (os sanity checks de
`results/audit/audit_report.md` confirmam que nenhum modelo real está
degenerado da mesma forma que os baselines), mas indica que o **sinal
preditivo disponível nos dados administrativos agregados é limitado** —
esperado, já que o dataset não tem informação em nível de aluno
(desempenho acadêmico individual, situação socioeconômica, motivo de
evasão), só características do curso.

**Naive Bayes teve o pior desempenho por um motivo estrutural, não por
ser um algoritmo ruim para o problema.** Ele foi treinado com uma
definição de alvo e um universo de features diferentes dos outros 4
modelos (ver [seção 4.2](#42-target-taxa_evasao-e-alto_risco_evasao)) —
recall de 0,047 significa que o modelo quase nunca prevê a classe
positiva nesse pipeline unificado, mas os `results/audit/naive_bayes_sanity.json`
confirmam que o resultado, embora ruim, não é degenerado (passou nos 6
critérios de sanidade). Comparar o Naive Bayes diretamente aos outros 4
sem essa ressalva seria injusto com o algoritmo.

**Regressão Logística ilustra bem o efeito da correção metodológica do
projeto.** No relatório pré-auditoria, ela aparecia em 1º lugar (recall
≈1,0). Pós-correção do threshold tuning, cai para a penúltima posição
(recall 0,503). A queda não é uma piora do modelo — é a remoção de um
artefato de medição que inflava artificialmente seu desempenho aparente.

### Limitações

- O dataset é agregado por curso, não por aluno — o teto de sinal
  preditivo disponível é inerentemente limitado (ver achado acima sobre
  os baselines).
- O alvo (`taxa_evasao ≥ 20%`) é um limiar fixo, não calibrado
  empiricamente contra nenhuma política institucional real.
- O Naive Bayes não é diretamente comparável aos outros 4 modelos sem a
  ressalva sobre pipelines diferentes.
- A migração de arquitetura mais recente do projeto (ver
  `docs/VALIDATION_REPORT.md`) identificou uma etapa de amostragem
  estratificada dos notebooks originais que ainda não foi portada para
  a nova camada `src/data` — os números desta seção vêm dos notebooks/
  scripts originais (não afetados), mas uma reexecução completa via
  `src/` ainda depende dessa etapa para ser diretamente comparável.

### Trabalhos futuros

- Enriquecer o dataset com indicadores socioeconômicos regionais (ex.:
  IDH municipal, renda per capita) para aumentar o sinal preditivo.
- Recalibrar o limiar de `alto_risco_evasao` contra uma definição de
  política institucional real, em vez do corte fixo de 20%.
- Portar a etapa de amostragem estratificada para `src/data` (ver
  `docs/VALIDATION_REPORT.md`, seção 4.1) antes de qualquer nova rodada
  de treino via a arquitetura migrada.
- Unificar o universo de features do Naive Bayes com o dos outros 4
  modelos, ou documentar formalmente por que isso não é recomendável.

---

## 4. Metodologia

### 4.1 Dataset

| | |
|---|---|
| **Origem** | Censo da Educação Superior — [INEP](https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2024.zip) |
| **Ano** | 2024 |
| **Arquivo** | `MICRODADOS_CADASTRO_CURSOS_2024.CSV` (separador `;`, encoding `latin1`) |
| **Registros usados no protocolo oficial** | 30.000 (amostra estratificada por classe — `SAMPLE_SIZE` em `analysis/audit/data.py`) |
| **Colunas usadas como entrada** | 22 colunas brutas do censo (`COLS_CURSOS`) — categóricas de organização acadêmica/modalidade + contagens de ingressantes, matriculados, vagas, financiamento |
| **Variável alvo** | `alto_risco_evasao` (binária) |

O dataset **não é versionado no Git** (>100MB) — precisa ser baixado
manualmente (ver [seção 7](#7-instalação)).

### 4.2 Target: `taxa_evasao` e `alto_risco_evasao`

**Definição oficial** (usada pelos 4 modelos "notebook-style" e pelo
protocolo de auditoria unificado — `analysis/audit/data.py`,
`src.data.target` com `strategy="threshold_20pct"`):

```text
QT_EVADIDOS   = QT_SIT_DESVINCULADO + QT_SIT_TRANCADA
taxa_evasao   = (QT_EVADIDOS / QT_ING) × 100
alto_risco_evasao = 1  se  taxa_evasao ≥ 20%
                     0  caso contrário
```

**Naive Bayes usa uma definição diferente**, herdada do pipeline
original específico daquele modelo (`data/preprocessamento.py`,
`src.data.target` com `strategy="median_split"`):

```text
taxa_evasao = QT_SIT_DESVINCULADO / (QT_MAT + QT_SIT_DESVINCULADO)
alto_risco_evasao = 1  se  taxa_evasao ≥ mediana(taxa_evasao)
                     0  caso contrário
```

Essa divergência é uma característica **documentada e preservada** do
projeto original (ver [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md)
e [`ARCHITECTURE_AUDIT.md`](ARCHITECTURE_AUDIT.md)), não uma
inconsistência desta documentação — nenhuma tarefa de refatoração
alterou a regra de nenhum dos dois pipelines.

### 4.3 Preprocessing

- **Remoção de identificadores**: nome/código do curso, da instituição,
  do município, da área CINE — não carregam sinal preditivo
  generalizável (risco de "decorar" registros específicos).
- **Tratamento de valores ausentes**: imputação por moda (colunas
  categóricas, notebooks originais) ou por zero (`data/preprocessamento.py`,
  Naive Bayes) — nunca descarte de linha.
- **Conversão de tipos**: colunas de contagem convertidas para numérico
  com `pd.to_numeric(errors="coerce")`, substituindo inválidos por 0
  antes de qualquer razão/proporção.
- **Codificação categórica**: `LabelEncoder` por coluna categórica (8
  colunas de organização acadêmica/modalidade).

### 4.4 Leakage Prevention

Quatro grupos de colunas são removidos antes de qualquer feature entrar
no modelo (`src/data/leakage.py`), cada um com um motivo específico:

| Grupo | Exemplos | Por que não pode entrar como feature |
|---|---|---|
| Identificadores | `NO_CURSO`, `CO_IES`, `NO_MUNICIPIO`, `NU_ANO_CENSO` | Identidade do registro, não característica do curso — risco de o modelo "decorar" em vez de generalizar. |
| Origem do alvo | `QT_SIT_DESVINCULADO`, `QT_SIT_TRANCADA`, `taxa_evasao`, `alto_risco_evasao` | São a matéria-prima do alvo — mantê-las como feature permitiria reconstruir o alvo diretamente (vazamento direto). |
| Prefixos de matrícula/conclusão | `QT_MAT*`, `QT_CONC*` | Estruturalmente correlacionados com quem **não** evadiu — um curso com muitos matriculados/concluintes tem, por definição, poucos evadidos proporcionalmente (vazamento indireto). |
| Features derivadas vazadas | `RAZAO_CONCLUSAO_EVASAO` | Calculada a partir de `taxa_evasao` — a mesma variável usada para construir o alvo. Confirmado empiricamente: quando presente, essa feature tem importância 0,44 (mais que o dobro de qualquer outra) na Árvore de Decisão — ver `docs/VALIDATION_REPORT.md`. |

### 4.5 Feature Engineering

| Modelo | Features / Transformações | Objetivo |
|---|---|---|
| Árvore de Decisão | 21 features comuns + 9 features derivadas adicionais (interações, binning: `EAD_PREDOMINANTE`, `TAMANHO_CATEGORIA`, `CURSO_COMPETITIVO` etc.) → `SelectKBest(mutual_info_classif, k=21)`. Sem scaling. | Árvores particionam por limiar em cada feature isoladamente — invariantes a escala; a seleção reduz as 30 candidatas às 21 mais informativas. |
| Naive Bayes | Universo próprio: todas as colunas numéricas do censo (não as 21 comuns) → `VarianceThreshold(0.01)` → remoção de colunas com correlação par-a-par >0,90 → `StandardScaler` (Gaussian) / `Binarizer` (Bernoulli) / `MinMaxScaler` (Complement, conforme a variante). | Cada variante de Naive Bayes exige um domínio de entrada diferente (contínuo, binário, não negativo). |
| Regressão Logística | 21 features comuns → `StandardScaler`. | Regularização L1 e otimização por gradiente são sensíveis a features em escalas diferentes. |
| SVM | 21 features comuns → seleção por correlação absoluta com o alvo (`\|corr\| ≥ 0,05`) → `RobustScaler`. | Kernel RBF usa distância euclidiana diretamente — scaling é obrigatório; a seleção reduz ruído antes do kernel. |
| Rede Neural | 21 features comuns, sem scaling (replica o notebook original). | Preserva fielmente o comportamento original — a ausência de normalização é uma lacuna documentada, não corrigida propositalmente nesta etapa (ver `docs/FEATURE_ENGINEERING.md`). |

As "21 features comuns" são: 8 categóricas (`TP_ORGANIZACAO_ACADEMICA`,
`TP_REDE`, `TP_CATEGORIA_ADMINISTRATIVA`, `TP_GRAU_ACADEMICO`,
`TP_MODALIDADE_ENSINO`, `TP_DIMENSAO`, `IN_GRATUITO`,
`CO_CINE_AREA_GERAL`) + 4 numéricas brutas (`QT_ING`, `QT_MAT`,
`QT_VG_TOTAL`, `QT_INSCRITO_TOTAL`) + 9 derivadas (`TAXA_CONCLUSAO`,
`RAZAO_ING_MAT`, `PROPORCAO_EAD`, `PROPORCAO_NOTURNO`,
`INDICE_FINANCIAMENTO`, `PROPORCAO_FIES`, `PROPORCAO_PROUNIP`,
`PROPORCAO_18_24`, `PROPORCAO_FEM`).

### 4.6 Modelos

| Modelo | Por que foi usado | Configuração principal |
|---|---|---|
| **Árvore de Decisão** | Interpretável (regras explícitas), lida com não linearidades sem exigir scaling. | `criterion=entropy`, `max_depth=15`, poda por custo-complexidade (`ccp_alpha`). |
| **Naive Bayes** | Baseline probabilístico rápido, três variantes testadas para robustez à natureza das features. | `ComplementNB` venceu a comparação entre Gaussian/Bernoulli/Complement. |
| **Regressão Logística** | Baseline linear interpretável (coeficientes), referência clássica para classificação binária. | Regularização L1 (`penalty=l1`), o que também seleciona features implicitamente. |
| **SVM** | Captura fronteiras de decisão não lineares via kernel RBF, bom para dados de dimensão moderada. | Kernel RBF, `class_weight=balanced`. |
| **Rede Neural (MLP)** | Explora a capacidade de redes rasas de capturar interações não lineares nas features. | 1 camada oculta, tamanho dinâmico (`(n_features+1)//2`), treino via SGD com momentum. |

---

## 5. Validação

- **Estratégia:** `StratifiedKFold`, **5 folds**, `shuffle=True`,
  `random_state=42` — o mesmo protocolo em todos os 5 modelos
  (`src.training.cross_validation.make_stratified_kfold`).
- **Não há um único holdout de teste separado** do CV — o "conjunto de
  teste" de cada modelo é a união dos 5 folds de validação (uma amostra
  nunca é usada para treinar o modelo que a avalia).
- **Busca de hiperparâmetros:** `RandomizedSearchCV` (Árvore de Decisão,
  500 iterações; SVM, 12 iterações sobre subamostra de 12.000) e
  `GridSearchCV` (Regressão Logística) nos notebooks originais; Naive
  Bayes usa uma busca aleatória própria comparando as 3 variantes (90
  configurações). Os hiperparâmetros vencedores estão centralizados em
  `src/configs/model_hyperparameters.py` (seção 2.5); os espaços de
  busca originais, em `src/configs/search_spaces.py`.
- **Desbalanceamento:** estratégia própria de cada modelo, não
  uniformizada em SMOTE — ver `src/training/imbalance.py`:

  | Modelo | Estratégia |
  |---|---|
  | Árvore de Decisão | SMOTE por fold + `class_weight` |
  | Naive Bayes | SMOTE por fold |
  | Regressão Logística | `class_weight="balanced"` |
  | SVM | `class_weight="balanced"` |
  | Rede Neural | peso por amostra (`sample_weight`), calculado por fold |

- **Threshold tuning:** limiar de decisão calibrado por Youden's J com
  guarda-corpo de prevalência (rejeita threshold cuja taxa de positivos
  previstos exceda 1,5× a prevalência real) — `src.training.threshold_tuning`
  / `analysis/audit/threshold.py`. Aplicado aos 5 modelos no protocolo
  oficial.

### Como o leakage é evitado durante a validação

Toda transformação que aprende parâmetro dos dados é ajustada **só** no
fold de treino, nunca no fold de validação nem no dataset inteiro:

| Etapa | Onde é ajustada |
|---|---|
| Imputação/encoding (feature engineering) | Só no fold de treino menos uma fatia de calibração (`FeatureEngineer.for_<modelo>()`, ajustado dentro do laço de fold) |
| Scaling (`StandardScaler`/`RobustScaler`) | Idem — faz parte do mesmo `Pipeline` de feature engineering |
| Seleção de features (`SelectKBest`, correlação) | Idem |
| SMOTE | Só no fold de treino, depois da feature engineering, nunca em calibração/validação |
| Threshold tuning | Calibrado numa fatia separada do fold de treino (nunca vê o fold de validação) |

Uma auditoria final de vazamento (`docs/VALIDATION_REPORT.md`) verificou
7 categorias de risco de leakage no código atual; 6 sem nenhum problema
encontrado, 1 achado real de baixa severidade (imputação por moda
calculada antes da divisão de CV, para a estratégia `"mode"` — a
`"zero"`, usada pelo Naive Bayes, não tem esse risco) — documentado com
recomendação de correção, não corrigido às pressas nesta etapa.

### Reprodutibilidade

| | |
|---|---|
| **Python** | Recomendado 3.10+; testado nesta migração com **3.14.7** (`requirements.txt` não fixa a versão do Python). |
| **Random seed** | `42`, centralizado em `src.configs.settings.RANDOM_STATE` — usado em todo `StratifiedKFold`, split de calibração, SMOTE e nos hiperparâmetros de cada modelo. |
| **Folds de CV** | 5 (`src.configs.settings.CV_FOLDS`). |
| **Arquivo de lock** | **Não existe** nenhum arquivo de lock (`poetry.lock`, `Pipfile.lock` etc.) neste repositório — `requirements.txt` fixa só versões mínimas (`>=`), não exatas. |
| **Versões usadas para testar esta documentação** | `scikit-learn 1.9.0`, `pandas 3.0.5`, `numpy 2.5.2`, `matplotlib 3.11.1`, `seaborn 0.13.2`, `pytest 9.1.1` — instaladas manualmente neste ambiente; **não fixadas** em `requirements.txt` além do mínimo. |
| **Dependências além de `requirements.txt`** | `matplotlib`, `seaborn`, `jupyter`, `nbformat`, `nbclient`, `ipykernel` (visualização/notebooks) e `pytest`, `scipy` (testes) estão em `requirements-dev.txt` (ver [seção 7](#7-instalação)). |

---

## 6. Arquitetura

### 6.1 Fluxo de dados

```text
Raw Dataset (MICRODADOS_CADASTRO_CURSOS_2024.CSV)
     ↓
Data Loader            src/data/loading.py
     ↓
Target Construction    src/data/target.py         (taxa_evasao, alto_risco_evasao)
     ↓
Leakage Removal        src/data/leakage.py         (4 grupos de colunas removidas)
     ↓
Preprocessing          src/pipelines/base_preprocessing.py   (comum aos 5 modelos)
     ↓
Feature Engineering    src/pipelines/feature_engineering.py  (específico por modelo)
     ↓
Model Pipeline         src/models/                 (5 factories, hiperparâmetros centralizados)
     ↓
Cross Validation       src/training/               (StratifiedKFold + threshold tuning + imbalance)
     ↓
Evaluation             src/evaluation/             (métricas, comparação entre modelos)
     ↓
Results                TrainingResult / tabelas comparativas (em memória, sem gráfico)
     ↓
Visualization           src/visualization/ + experiments/*.ipynb
```

`src/data/pipeline.py::run_data_pipeline(...)` encadeia as 4 primeiras
etapas; `src/training/trainer.py::Trainer.train(...)` encadeia Feature
Engineering → Model Pipeline → Cross Validation, por fold.

### 6.2 Estrutura de diretórios (real, pós-refatoração)

```text
Projeto-Final-Machine-Learning/
│
├── src/                        # arquitetura nova (Tarefas 2-7)
│   ├── configs/                 # paths, seeds, hiperparâmetros, espaços de busca
│   ├── data/                    # loading, target, leakage, orquestração do pipeline
│   ├── pipelines/                # preprocessing comum + feature engineering por modelo
│   ├── models/                   # definição dos 5 estimadores (sem treino/avaliação)
│   ├── training/                 # Trainer, cross-validation, imbalance, threshold tuning
│   ├── evaluation/                # métricas, comparação entre modelos
│   └── visualization/             # gráficos (matplotlib), sem lógica de dados
│
├── experiments/                 # notebooks de experimentação (Tarefa 8) — 1 por modelo
├── scripts/                     # train_all.py — treina os 5 modelos via linha de comando
├── tests/                       # 190 testes automatizados (pytest)
├── docs/                        # documentação técnica detalhada de cada etapa
│
├── models/                      # artefatos de resultado ORIGINAIS por modelo (notebooks e scripts removidos — ver docs/FILE_CLEANUP.md)
├── analysis/                    # pipeline de auditoria científica ORIGINAL (fonte dos resultados oficiais)
├── results/                     # resultados oficiais: audit/, validation/, feature_importance/
│
├── ARCHITECTURE_AUDIT.md         # auditoria inicial que motivou a refatoração
├── README.md
└── requirements.txt
```

**Responsabilidade de cada diretório:**

| Diretório | Responsabilidade |
|---|---|
| `src/configs/` | Único lugar com caminhos, sementes, hiperparâmetros e espaços de busca — nenhum valor mágico solto pelo resto do código. |
| `src/data/` | Carrega o CSV, constrói o alvo, remove vazamento — nunca faz feature engineering ou treino. |
| `src/pipelines/` | Transformações de feature: comuns (`base_preprocessing.py`) e específicas por modelo (`feature_engineering.py`), sempre como `Pipeline` do scikit-learn, nunca ajustadas fora de um fold de treino. |
| `src/models/` | Só define/configura o estimador (`create_<modelo>(...)`) — não treina, não avalia, não plota. |
| `src/training/` | `Trainer.train(...)`: cross-validation, desbalanceamento, threshold tuning. Não gera gráfico, não salva arquivo. |
| `src/evaluation/` | Cálculo de métrica centralizado (fonte única, também usada por `src/training/`) e comparação tabular entre modelos. |
| `src/visualization/` | Funções de gráfico — recebem dado já calculado, devolvem `Figure`, nunca leem dataset. |
| `experiments/` | Um notebook por modelo — só chama a pipeline, carrega resultados, gera gráficos e interpreta. Nenhuma lógica de ML. |
| `scripts/` | Pontos de entrada de linha de comando (`train_all.py`) — treinar sem depender de notebook. |
| `tests/` | Testes automatizados de cada camada de `src/`. |
| `docs/` | Documentação técnica detalhada — uma por etapa da refatoração (ver lista completa no final deste README). |
| `models/`, `analysis/`, `results/` | Estrutura **original** do projeto. `analysis/audit/` é a fonte dos resultados oficiais desta seção; `models/<modelo>/` mantém os artefatos de resultado/hiperparâmetro gerados pelos notebooks e scripts originais, que foram removidos numa limpeza posterior por já estarem inteiramente substituídos por `experiments/` e `src/` — ver [`docs/FILE_CLEANUP.md`](docs/FILE_CLEANUP.md) para a lista completa e a justificativa de cada remoção. |

---

## 7. Instalação

**Requisitos:** Python 3.10 ou superior (testado com 3.14.7).

```bash
# 1. Criar o ambiente virtual
python -m venv .venv
```

Ativação no Windows:

```bash
.venv\Scripts\activate
```

Ativação no Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
# 2. Instalar as dependências principais
pip install -r requirements.txt
```

`requirements.txt` cobre o núcleo de ML (`pandas`, `numpy`,
`scikit-learn`, `imbalanced-learn`, `joblib`, `reportlab`). Para rodar
notebooks e testes, instale também `requirements-dev.txt` (inclui
`requirements.txt` via `-r` e adiciona `pytest`, `matplotlib`,
`seaborn`, `scipy`, `jupyter`, `nbformat`, `nbclient`, `ipykernel`):

```bash
pip install -r requirements-dev.txt
```

**Dataset:** baixe `MICRODADOS_CADASTRO_CURSOS_2024.CSV` do
[INEP](https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2024.zip)
e coloque na **raiz do repositório** ou em `data/` — `src.configs.paths.DATASET_PATH_CANDIDATES`
procura nos dois lugares (e também em
`~/Downloads/microdados_censo_da_educacao_superior_2024/dados/`). O
arquivo não é versionado no Git (>100MB, listado em `.gitignore`).

---

## 8. Execução

### Treinar todos os modelos

```bash
python scripts/train_all.py
```

Executa os 5 modelos (`arvore_de_decisao`, `naive_bayes`,
`regressao_logistica`, `svm`, `rede_neural`) via `src.training.trainer.Trainer`
e imprime a tabela comparativa. Opções:

```bash
python scripts/train_all.py --n-splits 5 --random-state 42 --output results/comparacao.csv
```

### Treinar um único modelo (via Python)

Não há, nesta versão do projeto, um script de linha de comando dedicado
a um único modelo — a API em Python já cobre esse caso diretamente:

```python
from src.data.pipeline import run_data_pipeline
from src.training.trainer import Trainer

data = run_data_pipeline(
    target_strategy="threshold_20pct",   # "median_split" para naive_bayes
    missing_strategy="mode",
    include_prefix_leakage_group=False,
)
result = Trainer().train("svm", data.X, data.y)   # ou: arvore_de_decisao, regressao_logistica,
                                                    #     rede_neural, naive_bayes (+ variant=...)
print(result.metrics)
```

### Testes

```bash
pytest
```

190 testes cobrindo cada camada de `src/`: carregamento de dados,
construção do alvo, remoção de vazamento, feature engineering por
modelo, definição dos 5 estimadores, treino/cross-validation/threshold
tuning, cálculo de métricas, comparação entre modelos, geração de
gráficos e a estrutura dos notebooks de `experiments/` (sem executá-los
— ver abaixo).

### Visualização — notebooks de experimentação

Os notebooks ficam em `experiments/`, um por modelo:

| Notebook | Modelo |
|---|---|
| `experiments/decision_tree.ipynb` | Árvore de Decisão |
| `experiments/naive_bayes.ipynb` | Naive Bayes (as 3 variantes) |
| `experiments/logistic_regression.ipynb` | Regressão Logística |
| `experiments/svm.ipynb` | SVM |
| `experiments/neural_network.ipynb` | Rede Neural |

Cada um só chama `run_data_pipeline` + `Trainer.train` (a mesma
arquitetura do script de linha de comando), carrega o `TrainingResult`
resultante e gera os gráficos chamando `src.visualization.*` — nenhuma
lógica de treino/métrica está escrita no notebook. Requer o dataset real
(ver [seção 7](#7-instalação)).

```bash
jupyter notebook experiments/decision_tree.ipynb
```

Ou, do zero: `Kernel → Restart & Run All` — nenhuma célula depende de
execução anterior fora de ordem.

Os 4 notebooks **originais** (`models/*/*.ipynb`) e os scripts originais
do Naive Bayes (`models/naive_bayes/*.py`) foram removidos numa limpeza
posterior à publicação inicial deste README, por já estarem inteiramente
substituídos por `experiments/` e `src/` — ver
[`docs/FILE_CLEANUP.md`](docs/FILE_CLEANUP.md). Os artefatos que eles
geraram (hiperparâmetros, métricas por fold, `best_model.pkl` do Naive
Bayes) permanecem em `models/<modelo>/` e continuam sendo a fonte dos
resultados oficiais da [seção 2](#2-resultados), junto com
`analysis/audit/` (que os reavaliou de forma unificada).

---

## Documentação técnica detalhada

Cada etapa da refatoração tem um documento próprio em `docs/`:

| Documento | Conteúdo |
|---|---|
| `ARCHITECTURE_AUDIT.md` | Auditoria que motivou toda a refatoração |
| `docs/ARCHITECTURE.md` | Estrutura de `src/` e decisões de organização |
| `docs/DATA_PIPELINE.md` | Data loading, target, leakage removal |
| `docs/FEATURE_ENGINEERING.md` | Engenharia de features por modelo, classificação de cada transformação |
| `docs/MODELS.md` | Definição dos 5 modelos, hiperparâmetros centralizados |
| `docs/TRAINING.md` | Cross-validation, desbalanceamento, threshold tuning |
| `docs/EVALUATION.md` | Métricas centralizadas, comparação entre modelos |
| `docs/EXPERIMENTS.md` | Notebooks de experimentação, verificação de execução |
| `docs/FILE_CLEANUP.md` | Auditoria de arquivos obsoletos |
| `docs/VALIDATION_REPORT.md` | Validação final de reprodutibilidade — o que foi confirmado, o que ainda precisa de correção |
