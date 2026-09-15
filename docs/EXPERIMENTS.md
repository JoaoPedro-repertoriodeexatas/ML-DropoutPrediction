# Notebooks como camada de experimentação e visualização (Tarefa 8)

**Branch:** `audit/architecture-review`
**Baseado em:** [`docs/TRAINING.md`](TRAINING.md), [`docs/EVALUATION.md`](EVALUATION.md)

Cria `experiments/*.ipynb` — um notebook por modelo, que só orquestra a
arquitetura já construída nas Tarefas 3-7. Os 4 notebooks originais
(`models/*/​*.ipynb`) e `models/naive_bayes/*.py` **não foram alterados
nem removidos** — continuam existindo lado a lado com os novos.

---

## 1. O que cada notebook faz (e não faz)

```text
experiments/
├── decision_tree.ipynb
├── logistic_regression.ipynb
├── svm.ipynb
├── neural_network.ipynb
└── naive_bayes.ipynb
```

Cada um segue exatamente a estrutura pedida:

| Seção | Conteúdo |
|---|---|
| 1. Objetivo | O que o experimento avalia, com que modelo/hiperparâmetros, e as ressalvas metodológicas relevantes (ex.: vazamento corrigido, threshold com guarda-corpo). |
| 2. Configuração do experimento | Localiza a raiz do projeto, importa de `src.*`, define as constantes do experimento (`MODEL_NAME`, `TARGET_STRATEGY`, `RANDOM_STATE` etc.). |
| 3. Execução do pipeline | `run_data_pipeline(...)` + `Trainer.train(...)` — 2 chamadas, nenhuma lógica própria. |
| 4. Resultados | `training_result.metrics`/`metrics_std` e `fold_metrics_table(...)` — carregados, não recalculados. |
| 5. Visualizações | Chamadas a `src.visualization.*` — o notebook nunca desenha nada com `matplotlib` diretamente. |
| 6. Análise | Texto interpretativo, com o que observar em cada gráfico e por quê (fundamentado na arquitetura/auditoria, não em números específicos — ver seção 4). |
| 7. Conclusão | Checklist do que preencher depois de rodar com o dataset real (comparação com os resultados históricos). |

O que **não** existe em nenhuma célula: leitura de CSV, cálculo de
`taxa_evasao`, remoção manual de coluna, label encoding, criação de
`DecisionTreeClassifier`/`LogisticRegression`/`SVC`/`MLPClassifier`/
`GaussianNB`/`BernoulliNB`/`ComplementNB`, `StratifiedKFold`,
`GridSearchCV`/`RandomizedSearchCV`, `SMOTE`, ou qualquer chamada a
`accuracy_score`/`precision_score`/`recall_score`/`f1_score`/
`roc_auc_score`. Tudo isso é responsabilidade de `src/` (Tarefas 3-7).
`tests/test_experiment_notebooks.py::test_notebook_has_no_forbidden_ml_implementation_logic`
verifica isso automaticamente, varrendo o código de cada notebook por
esses padrões.

---

## 2. Reprodutibilidade — nenhuma célula depende de execução anterior

Cada notebook é auto-contido: a célula de configuração localiza a raiz
do projeto subindo diretórios até achar `src/configs/` (funciona tanto
se o Jupyter foi iniciado na raiz do repositório quanto dentro de
`experiments/`), e toda semente aleatória vem de uma única variável
`RANDOM_STATE = 42`, usada em todas as chamadas subsequentes — nunca um
literal solto repetido. `Kernel > Restart & Run All` reproduz o mesmo
resultado do início ao fim.

`tests/test_experiment_notebooks.py::test_notebook_does_not_hardcode_random_state_outside_config_cell`
confirma que `RANDOM_STATE` é atribuído uma única vez por notebook.

---

## 3. Visualizações — `src/visualization/`

Implementado nesta tarefa, em 4 módulos:

| Módulo | Funções |
|---|---|
| `training_plots.py` | `plot_metrics_by_fold`, `plot_metrics_boxplot`, `plot_threshold_by_fold` |
| `evaluation_plots.py` | `plot_confusion_matrix`, `plot_roc_curve`, `plot_precision_recall_curve` |
| `comparison_plots.py` | `plot_model_comparison_bars`, `plot_model_ranking`, `plot_metric_heatmap` |
| `tree_plots.py` | `plot_feature_importance` (específico da Árvore de Decisão) |

Toda função recebe dado **já calculado** (`TrainingResult`,
`ConfusionMatrixSummary`, um `DataFrame` de `fold_metrics_table`/
`compare_models`) e devolve uma `matplotlib.figure.Figure` — nenhuma lê
dataset, treina, ou calcula métrica; nenhuma chama `plt.show()` (o
notebook decide). Cada notebook só importa e chama essas funções.

---

## 4. Análise e conclusão sem números reais — por quê

As seções "Análise" e "Conclusão" de cada notebook não citam valores
específicos de métrica. Isso é intencional, não uma lacuna: este
ambiente não tem o CSV real do INEP (>100MB, não versionado — ver
`README.md`, seção "Dados"), então qualquer número que os notebooks
produzissem aqui seria sobre dados sintéticos, não sobre o Censo da
Educação Superior real. Publicar esses números como se fossem o
resultado do experimento violaria diretamente "não altere os resultados
propositalmente".

Em vez disso, "Análise" explica **o que observar** em cada gráfico e por
quê (fundamentado na arquitetura e na auditoria — ex.: "um threshold
perto do extremo da busca seria um sinal de alerta"), e "Conclusão" é um
checklist do que preencher depois de rodar com o dataset real, incluindo
contra qual artefato histórico comparar (`results/audit/*_folds.csv`,
`models/*/resultados_*.csv`).

---

## 5. Verificação de execução

**O que foi verificado nesta etapa:** os 5 notebooks foram executados de
ponta a ponta, célula a célula, via `nbclient` (o mesmo motor de
execução do `jupyter nbconvert --execute`), contra um **CSV sintético**
gerado localmente com o mesmo schema de colunas do Censo (categóricas
como códigos numéricos, as colunas de contagem que alimentam as 9
features derivadas comuns, `QT_SIT_DESVINCULADO`/`QT_SIT_TRANCADA` para
o alvo). As 5 execuções terminaram **sem erro em nenhuma célula**:

```text
Executando: decision_tree.ipynb       -> OK
Executando: logistic_regression.ipynb -> OK
Executando: svm.ipynb                 -> OK
Executando: neural_network.ipynb      -> OK
Executando: naive_bayes.ipynb         -> OK

TODOS OS 5 NOTEBOOKS EXECUTARAM COM SUCESSO.
```

Isso prova que a composição `run_data_pipeline` → `Trainer.train` →
`fold_metrics_table`/`compare_models` → `src.visualization.*` funciona
de ponta a ponta para os 5 modelos, com as 21/features-próprias
corretas, sem erro de tipo, de coluna ausente, ou de `Pipeline`. O CSV
sintético e os scripts que o geraram/executaram os notebooks **não
foram commitados** (eram ferramentas de verificação, não parte do
projeto) — os `.ipynb` entregues tiveram os *outputs* dessa execução
sintética removidos antes de finalizar, para que ninguém confunda esses
números com resultado real.

**O que não foi (e não podia ser) verificado aqui:** os valores
numéricos reais — recall, F1, ROC-AUC do Censo de verdade — porque o
dataset real não está disponível neste ambiente. Essa é exatamente a
comparação pedida por "após migrar cada notebook, compare os resultados
com os resultados originais": só pode ser feita por quem rodar os
notebooks com `MICRODADOS_CADASTRO_CURSOS_2024.CSV` de verdade. Cada
notebook já aponta, na seção 7 (Conclusão), para qual artefato histórico
comparar (`results/audit/*_folds.csv` para os 4 modelos "notebook-style";
`models/naive_bayes/artifacts/best_params.json` e
`results/audit/naive_bayes_folds.csv` para o Naive Bayes).

`tests/test_experiment_notebooks.py` cobre a parte que **não** depende
do dataset real: estrutura das 7 seções, ausência de lógica de
implementação de ML, uso comprovado da arquitetura migrada, semente
única — 26 testes, todos passando.

---

## 6. Como cada notebook monta o `X` de entrada

Reflete exatamente `docs/FEATURE_ENGINEERING.md`, seção 8:

| Notebook | `target_strategy` | `missing_strategy` | `include_prefix_leakage_group` |
|---|---|---|---|
| Árvore de Decisão, Regressão Logística, SVM, Rede Neural | `"threshold_20pct"` | `"mode"` | `False` (preserva `QT_MAT`/`QT_CONC` para as 9 features derivadas comuns) |
| Naive Bayes | `"median_split"` (padrão) | `"zero"` (padrão) | `True` (padrão) |

---

## 7. Testes

26 testes novos (`tests/test_experiment_notebooks.py` + 11 de
`tests/test_visualization.py`), num total de **190** (164 → 190, sem
regressão):

- `test_visualization.py` — cada função de `src/visualization/` devolve
  uma `Figure` válida a partir de dados sintéticos já calculados
  (`TrainingResult`, `ConfusionMatrixSummary`, tabelas de comparação);
  confere que salvar em disco (`save_path=...`) funciona.
- `test_experiment_notebooks.py` — os 5 `.ipynb` existem e são JSON
  válido; têm as 7 seções, na ordem certa; não contêm nenhum dos padrões
  proibidos (leitura de CSV, modelos scikit-learn instanciados
  diretamente, `StratifiedKFold`/busca de hiperparâmetros/SMOTE manuais,
  métricas calculadas na mão); usam de fato
  `run_data_pipeline`/`Trainer.train`; `RANDOM_STATE` definido uma única
  vez; os 5 nomes canônicos de modelo (`src.configs.constants.MODEL_NAMES`)
  estão todos cobertos, um por notebook.

### Execução real

```text
python -m pytest tests/ -v
...
====================== 190 passed in 34.59s =======================
```

---

## 8. O que fica fora do escopo desta etapa

- **Os 4 notebooks originais e `models/naive_bayes/*.py`** continuam
  existindo, inalterados — nenhum arquivo foi removido.
- **Resultados reais com o Censo de verdade** — dependem de quem rodar
  o notebook localmente com o CSV baixado (`README.md`).
- **Comparação numérica com os resultados originais** — preparada
  (seção 7 de cada notebook aponta o artefato certo), mas não executada
  aqui, pela mesma razão.
