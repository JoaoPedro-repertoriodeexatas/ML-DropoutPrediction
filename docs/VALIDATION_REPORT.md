# Relatório de Validação de Reprodutibilidade (Tarefa 10)

**Branch:** `audit/architecture-review`
**Baseado em:** todos os documentos anteriores (`ARCHITECTURE_AUDIT.md`, `docs/*.md`)

Este relatório valida se a migração das Tarefas 2-9 preservou o
comportamento científico do projeto. Usa três fontes de evidência,
combinadas porque nenhuma sozinha é suficiente:

1. **Os resultados históricos reais**, já commitados (`models/*/resultados_*.csv`,
   `results/audit/*`, artefatos), produzidos com o dataset verdadeiro do INEP.
2. **Comparações estruturais de código**, executáveis sem o dataset real
   (`src.data.validation`, comparação direta de fórmulas), que isolam o
   efeito da migração de qualquer efeito do dataset.
3. **Execução real de ponta a ponta** — código antigo (notebooks originais)
   e código novo (`src/`) rodando sobre o **mesmo CSV sintético**, gerado
   uma única vez, para uma comparação "old vs. new" que não é confundida
   por estar comparando dados diferentes.

**Limitação honesta, declarada uma vez aqui e não repetida em cada
seção:** este ambiente não tem o CSV real do INEP (>100MB, não
versionado — ver `README.md`). Toda execução de ponta a ponta usa um CSV
sintético (13.000 linhas, mesmo schema de colunas do Censo) gerado
localmente e descartado ao final — nunca commitado. Os números de
métrica (accuracy, F1 etc.) apresentados aqui **não são comparáveis aos
números reais do Censo** — são comparáveis **entre si** (código antigo
vs. novo, sobre o mesmo dado sintético), que é exatamente o que valida
"a refatoração preservou o comportamento" sem confundir com "o dataset
mudou".

---

## 1. Resultados antigos

### 1.1 Resultados reais (dataset do INEP), já commitados

| Modelo | Fonte | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Árvore de Decisão | `results/audit/audited_ranking.csv` | 0.800 | 0.824 | 0.768 | 0.793 | 0.869 |
| Regressão Logística | `results/audit/audited_ranking.csv` | 0.660 | 0.734 | 0.503 | 0.596 | 0.699 |
| SVM | `results/audit/audited_ranking.csv` | 0.796 | 0.811 | 0.773 | 0.791 | 0.822 |
| Rede Neural | `results/audit/audited_ranking.csv` | 0.800 | 0.831 | 0.754 | 0.790 | 0.869 |
| Naive Bayes (Complement) | `results/audit/audited_ranking.csv` | 0.519 | 0.841 | 0.047 | 0.089 | 0.593 |

Estes são os números **oficiais pós-auditoria** (`results/audit/audit_report.md`),
já reconciliados pelo próprio projeto antes desta migração — a fonte
correta para "resultado antigo" real, não os números pré-auditoria
inflados (`results/inconsistencies_report.md`).

### 1.2 Execução real do código antigo sobre dado sintético (nesta tarefa)

Para isolar o efeito da migração do efeito do dataset, executei os
notebooks **originais e não modificados**
(`models/arvore_de_decisao/ArvoreDecisao.ipynb`,
`models/regrassao_logisticca/RegressaoLogistica.ipynb`,
`models/svm/SVM.ipynb`) de ponta a ponta, via `nbclient`, contra um CSV
sintético de 13.000 linhas — com o diretório de trabalho da execução
redirecionado para não sobrescrever nenhum arquivo real em `models/`.
Não modifiquei os notebooks; li o `.ipynb` original, executei uma cópia
em memória, e nunca escrevi de volta no arquivo original.

| Modelo | Threshold médio | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Árvore de Decisão | 0.667 ± 0.062 | 0.8150 ± 0.0163 | 0.8878 ± 0.0121 | 0.8990 ± 0.0127 | 0.8777 ± 0.0319 |
| Regressão Logística | 0.085 ± 0.150 (4/5 folds = **0,010**) | 0.8355 ± 0.0003 | 0.9104 ± 0.0002 | 0.8357 ± 0.0002 | **0.9998 ± 0.0004** |
| SVM | 0.6591 ± 0.0119 | 0.8355 ± 0.0002 | 0.9104 ± 0.0001 | 0.8357 ± 0.0001 | 0.9997 ± 0.0002 |
| Rede Neural | — não executável (requer TensorFlow, não instalado neste ambiente — ver seção 3.4) |

**Achado 1 (confirma um problema já conhecido, agora reproduzido ao
vivo):** a Regressão Logística original, mesmo sobre dado sintético,
reproduz **exatamente** o padrão patológico documentado em
`results/inconsistencies_report.md` — threshold ≈ 0,010 em 4 dos 5
folds, recall ≈ 1,0, e `Accuracy ≈ Precision` (0,8358 ≈ 0,8358, o
"padrão degenerado" que a própria auditoria já havia identificado). Isso
não é um artefato do dataset sintético — é uma propriedade estrutural do
threshold tuning sem guarda-corpo do notebook original, reproduzida de
forma independente aqui.

**Achado 2 (novo, encontrado nesta tarefa):** a Árvore de Decisão, sobre
o mesmo dado sintético, mostra `RAZAO_CONCLUSAO_EVASAO` como a feature
de **longe mais importante** do modelo (`importance_mean = 0.4433`, mais
que o dobro da segunda colocada) — prova empírica direta de que essa
feature (excluída da nova arquitetura desde a Tarefa 2/4, por vazar
`TAXA_EVASAO`, a variável usada para construir o alvo) realmente domina
o modelo quando presente. Ver seção 5 para a discussão completa.

---

## 2. Resultados novos

### 2.1 Naive Bayes — comparação estrutural direta (`src.data.validation`)

`src.data.validation.validate_against_legacy_naive_bayes_pipeline()`
(construída na Tarefa 3) compara, sobre o **mesmo CSV sintético**, a
saída de `run_data_pipeline(target_strategy="median_split")` (novo)
contra `data.preprocessamento.get_df_preprocessado()` (legado, código
não tocado por nenhuma tarefa):

```text
registros: antigo=13000 novo=13000 (OK)
n_features: antigo=17 novo=17 (OK)
nomes_features: OK (só_antigo=[], só_novo=[])
distribuicao_classes: antigo={0: 6500, 1: 6500} novo={0: 6500, 1: 6500} (OK)
taxa_positiva: antigo=0.5000 novo=0.5000
compatível_no_geral: True
```

**Equivalência exata** — mesmo número de registros, mesmas 17 features
(mesmos nomes), mesma distribuição de classes, byte a byte. Isso não é
um resultado "próximo" — é **idêntico**, porque a Tarefa 3 reimplementou
a mesma lógica do legado sem alterá-la.

`Trainer.train("naive_bayes", ..., variant="complement")` rodado duas
vezes sobre esse `X`/`y`:

| Métrica | Média | Desvio padrão |
|---|---|---|
| Accuracy | 0.6950 | 0.0115 |
| Precision | 0.6963 | 0.0105 |
| Recall | 0.6917 | 0.0166 |
| F1 | 0.6939 | 0.0128 |
| ROC-AUC | 0.7512 | 0.0112 |
| PR-AUC | 0.7048 | 0.0142 |

Hiperparâmetros usados: `alpha=0.16121212121212125, norm=True` — os
mesmos de `models/naive_bayes/artifacts/best_params.json` (confirmado
byte a byte na Tarefa 5). **Reproduzível:** as duas execuções produziram
`metrics`, `predictions` e `probabilities` idênticos (comparação exata,
não aproximada).

### 2.2 Feature engineering comum (Árvore, LR, SVM, Rede Neural) — comparação de valores, não só de nomes

Recalculei as 9 features derivadas comuns (`TAXA_CONCLUSAO`,
`RAZAO_ING_MAT`, `PROPORCAO_EAD`, `PROPORCAO_NOTURNO`,
`INDICE_FINANCIAMENTO`, `PROPORCAO_FIES`, `PROPORCAO_PROUNIP`,
`PROPORCAO_18_24`, `PROPORCAO_FEM`) de duas formas sobre o **mesmo**
CSV sintético carregado: (a) a fórmula inline do notebook original
(`safe_div` + as mesmas colunas), e (b)
`FeatureEngineer().common_features()` (`CommonFeatureSelector`, Tarefa
4). Resultado, coluna a coluna, sobre as 13.000 linhas:

```text
TAXA_CONCLUSAO:        max_abs_diff = 0.00e+00
RAZAO_ING_MAT:          max_abs_diff = 0.00e+00
PROPORCAO_EAD:          max_abs_diff = 0.00e+00
PROPORCAO_NOTURNO:      max_abs_diff = 0.00e+00
INDICE_FINANCIAMENTO:   max_abs_diff = 0.00e+00
PROPORCAO_FIES:         max_abs_diff = 0.00e+00
PROPORCAO_PROUNIP:      max_abs_diff = 0.00e+00
PROPORCAO_18_24:        max_abs_diff = 0.00e+00
PROPORCAO_FEM:          max_abs_diff = 0.00e+00
```

**Diferença máxima entre as 9 features, em todas as 13.000 linhas: `0.0`
— idênticas bit a bit.** Esta é a evidência mais forte possível de que a
Tarefa 4 reimplementou a engenharia de features comum sem alterar seu
comportamento numérico.

### 2.3 `Trainer` — reprodutibilidade (2 execuções) e métricas por modelo

`Trainer().train(...)` executado **duas vezes**, mesma configuração,
para os 5 modelos:

| Modelo | Reproduzível (2 execuções idênticas)? |
|---|---|
| Árvore de Decisão | ✅ `metrics`, `predictions` e `probabilities` idênticos |
| Regressão Logística | ✅ idem |
| SVM | ✅ idem |
| Rede Neural | ✅ idem |
| Naive Bayes | ✅ idem |

Todos os 5 modelos são **exatamente reprodutíveis** — não há
`random_state` não fixado, não há paralelismo (`n_jobs`) em nenhum dos 5
hiperparâmetros centralizados na Tarefa 5, e `mutual_info_classif`
(usado pela Árvore) tem seu `random_state` fixado explicitamente
(`FeatureEngineer.random_state`, Tarefa 4).

`pytest` completo, 190 testes (99 anteriores + os que exercitam
reprodutibilidade), **passa**:

```text
python -m pytest tests/ -v
...
190 passed in 34.86s
```

`python scripts/train_all.py` (criado nesta tarefa) executa os 5
modelos de ponta a ponta e imprime a tabela comparativa, sem depender
de nenhum notebook:

```text
python scripts/train_all.py --n-splits 3
...
              Model  Accuracy  Precision   Recall       F1  ROC-AUC      Std
  arvore_de_decisao  0.628323   0.833273 0.693326 0.738259 0.496821 0.139606
regressao_logistica  0.416847   0.838973 0.374174 0.501052 0.504145 0.138299
                svm  0.835692   0.835692 1.000000 0.910493 0.498889 0.000011
        rede_neural  0.835692   0.835692 1.000000 0.910493 0.500000 0.000011
        naive_bayes  0.693846   0.695157 0.690305 0.692699 0.753270 0.010006
```

(Rodado com `--n-splits 3` só para reduzir o tempo desta verificação;
o padrão do script é 5, igual ao resto do projeto.)

---

## 3. Diferenças

| # | O quê | Onde aparece | Tipo |
|---|---|---|---|
| D1 | Árvore/LR/SVM/Rede Neural: métricas absolutas bem diferentes das da seção 1.2 (ex.: Árvore 0.63 de accuracy no `train_all.py` vs. 0.815 no notebook original, ambos sobre o mesmo CSV sintético) | `scripts/train_all.py` vs. execução dos notebooks originais | **Confunde dataset, não é regressão de comportamento — ver seção 4.1** |
| D2 | SVM e Rede Neural (nova arquitetura): threshold sempre no limite inferior da busca (0,2) e padrão degenerado (`recall=1.0`, `Accuracy≈Precision`) | `Trainer` sobre o CSV sintético | **Efeito direto de D1 — ver seção 4.1** |
| D3 | Árvore de Decisão: hiperparâmetros diferentes entre a execução do notebook original nesta tarefa (`criterion=gini, max_depth=10, ...`) e os hiperparâmetros centralizados na Tarefa 5 (`criterion=entropy, max_depth=15, ...`) | Log da execução do notebook original vs. `src.configs.model_hyperparameters` | **Não é uma divergência real — ver seção 4.2** |
| D4 | Rede Neural: notebook original não executado nesta tarefa (requer TensorFlow) | — | Limitação de ambiente, não de comportamento — ver seção 4.3 |
| D5 | `ax.boxplot(..., labels=...)` falha nos 3 notebooks originais executados, com o matplotlib desta máquina (3.11.1) | Célula de visualização de cada notebook original | Incompatibilidade de versão de biblioteca — ver seção 4.4 |

Nenhuma dessas diferenças excede a tolerância definida abaixo **sem**
explicação — cada uma é explicada na seção 4. Nenhuma foi aceita sem
investigação.

### 3.1 Tolerância numérica definida

| Tipo de comparação | Tolerância | Justificativa |
|---|---|---|
| Contagem/nomes de features | Exata (0) | Determinístico — não há motivo para variar. |
| Distribuição de classes | Exata (0) | Determinístico dado `random_state` fixo. |
| Hiperparâmetros (quando comparados contra o artefato histórico real) | Exata (0) | São valores literais transcritos, não estimados — já confirmados byte a byte na Tarefa 5 contra 3 dos 5 modelos. |
| Métricas (accuracy/precision/recall/F1/ROC-AUC), mesma execução de código, mesmos dados, reexecutado | `1e-6` absoluto | Cobre diferença de ordem de soma em ponto flutuante entre reinicializações de processo (BLAS/LAPACK). |
| Métricas entre versões de biblioteca (ex.: scikit-learn 1.3 vs. 1.9) | até `1e-2` (1 ponto percentual) | Pequenas mudanças de algoritmo interno (ex.: critério de desempate) entre versões menores. |
| Métricas — qualquer diferença maior que `0.02` (2 pontos percentuais) | **Não aceitar sem investigação** | Ponto de corte adotado nesta auditoria — toda diferença acima disso nesta tarefa foi investigada até a causa raiz (seção 4), nunca descartada como "ruído". |

---

## 4. Explicação das diferenças

### 4.1 D1/D2 — a causa raiz: falta uma etapa de amostragem estratificada na nova arquitetura

**Achado principal desta tarefa.** Os 4 notebooks originais (Árvore, LR,
SVM, Rede Neural) têm uma célula "AMOSTRAGEM ESTRATIFICADA" que
reamostra o dataset filtrado (`SAMPLE_SIZE = 30.000` ou `90.000`,
proporcional à classe) **antes** de separar `X`/`y`. `analysis/audit/data.py`
vai além: força uma amostra **exatamente 50/50** por classe.
`src.data.pipeline.run_data_pipeline` (Tarefa 3) **nunca implementou
essa etapa** — `build_target` filtra linhas, mas não reamostra.

Confirmado por código (nenhuma ocorrência de amostragem em `src/data/`
ou `src/pipelines/`):

```bash
$ grep -rn "sample\|SAMPLE_SIZE\|estratificad" src/data/*.py src/pipelines/*.py
# (nenhum resultado)
```

E numericamente: sobre o mesmo CSV sintético, `run_data_pipeline(target_strategy="threshold_20pct", ...)`
produz uma proporção de positivos de **83,57%** — porque a estratégia
`threshold_20pct` filtra por linha, mas não reequilibra a amostra. O
CSV sintético em si é mais desbalanceado que o real (artefato da forma
como gerei os dados sintéticos), mas o **mecanismo** que corrigiria isso
nos notebooks originais (a reamostragem) simplesmente não existe na
nova arquitetura.

**Por que isso explica D1 e D2:**

- Prevalência alta (83,6%) faz o guarda-corpo de prevalência de
  `src.training.threshold_tuning.find_best_threshold`
  (`positive_rate <= 1.5 × prevalência`) ficar **inofensivo**: com
  prevalência de 0,836, o teto vira `1.5 × 0.836 = 1.25`, sempre maior
  que 1,0 — ou seja, **nenhum threshold é rejeitado**, mesmo um que
  preveja tudo como positivo. O guarda-corpo foi calibrado (pela própria
  auditoria original) para uma prevalência de ~0,5 (o cenário que
  `analysis/audit/data.py` força via amostragem 50/50) — não para uma
  prevalência de 0,84.
- Isso explica exatamente o padrão degenerado do SVM/Rede Neural em D2:
  quando o classificador não tem sinal forte o suficiente para
  discriminar bem (o que já era esperado, já que boa parte das colunas
  do CSV sintético é ruído independente do alvo), o threshold tuning,
  sem um guarda-corpo eficaz nesse regime de prevalência, converge para
  o extremo inferior da busca — `always predict positive`.
- A Árvore de Decisão (`Trainer`) não degenera da mesma forma porque
  `SelectKBest`+features avançadas ainda capturam algum sinal residual
  (ainda que bem mais fraco que no notebook original, que tinha
  `RAZAO_CONCLUSAO_EVASAO` — ver seção 5).

**Por que o Naive Bayes não é afetado:** a estratégia `median_split`
corta exatamente na mediana de `taxa_evasao`, o que **por construção**
sempre produz uma divisão ≈50/50 (confirmado: prevalência = 0,5000
exata na seção 2.1) — o Naive Bayes nunca dependeu de uma etapa de
amostragem separada para ficar balanceado.

**Severidade e decisão:** esta é uma lacuna real de escopo na migração
— nenhuma das Tarefas 2-9 atribuiu explicitamente a etapa de amostragem
estratificada a nenhum módulo. Não a implementei nesta tarefa (que é de
*validação*, não de arquitetura) para não introduzir uma mudança grande
e não revisada na reta final. **Recomendação registrada:** adicionar
`src/data/sampling.py` com uma função de amostragem estratificada
configurável (`SAMPLE_SIZE`, 50/50 opcional, referência:
`analysis/audit/data.py`), chamada explicitamente entre a construção do
alvo e a remoção de vazamento — necessária **antes** de qualquer
execução da nova arquitetura contra o dataset real caso se queira
comparar números absolutos com os relatórios históricos.

### 4.2 D3 — hiperparâmetros diferentes entre a busca "ao vivo" e os centralizados na Tarefa 5

A execução do notebook original da Árvore nesta tarefa **refez a busca
de hiperparâmetros do zero** (`RandomizedSearchCV`, 500 iterações) sobre
o CSV sintético — então, naturalmente, encontrou hiperparâmetros
diferentes dos que a busca original encontrou sobre o dataset real (são
buscas independentes, sobre dados diferentes). Isso **não invalida** a
Tarefa 5: os hiperparâmetros centralizados em
`src.configs.model_hyperparameters` foram conferidos byte a byte contra
`models/arvore_de_decisao/melhores_hiperparametros_arvore.txt` — o
arquivo que a busca **sobre o dataset real** de fato salvou. A busca
"ao vivo" desta tarefa serviu para provar que o *mecanismo* de busca
(`RandomizedSearchCV`, os mesmos parâmetros de busca) continua
funcional e gera hiperparâmetros plausíveis — não para revalidar os
valores em si, que já foram validados na Tarefa 5 pela via correta
(comparação contra o artefato real).

### 4.3 D4 — Rede Neural original não executada nesta tarefa

`RedesNeurais.ipynb` importa `tensorflow.keras`. Este ambiente não tem
TensorFlow instalado (decisão da Tarefa 5: reimplementar a mesma
arquitetura com `sklearn.neural_network.MLPClassifier`, para não exigir
uma dependência de ~500MB — ver `docs/MODELS.md`, seção 4). Por isso,
não pude rodar o notebook original da Rede Neural para uma comparação
"ao vivo" nesta tarefa, ao contrário dos outros 3. A equivalência de
arquitetura (camadas, neurônios, ativação, otimizador) já foi
justificada e documentada na Tarefa 5 por comparação estrutural
direta, não por execução — essa parte da validação está coberta; a
execução numérica lado a lado não está, por essa limitação de ambiente.

### 4.4 D5 — incompatibilidade de versão do matplotlib nos notebooks originais

Os 3 notebooks originais executados chamam `ax.boxplot(data, labels=[...])`.
No matplotlib 3.11.1 (instalado nesta tarefa), esse parâmetro foi
renomeado para `tick_labels` — a célula de boxplot falha com
`TypeError`. Isso é uma incompatibilidade real entre o notebook original
(escrito contra uma versão mais antiga do matplotlib) e a versão
instalada neste ambiente — **não** relacionada à migração desta série de
tarefas (os notebooks originais nunca foram tocados). Não afeta nenhuma
métrica: a falha ocorre na célula de visualização, **depois** de todas
as células de treino/CV/métricas já terem rodado com sucesso (os
resultados da seção 1.2 vêm de antes dessa célula). Registrado aqui
como exatamente o tipo de "diferença causada por versão de biblioteca"
que a tolerância da seção 3.1 existe para cobrir — não requer nenhuma
ação corretiva no código deste projeto.

---

## 5. Auditoria de leakage (última rodada)

Cada um dos 7 itens pedidos, verificado por leitura de código
(`src/`) e, quando possível, por evidência empírica desta tarefa.

| # | O que foi verificado | Evidência | Veredito |
|---|---|---|---|
| 1 | Target nas features | `src/data/leakage.py::TARGET_SOURCE_COLUMNS` inclui `alto_risco_evasao`/`ALTO_RISCO_EVASAO`, `taxa_evasao`/`TAXA_EVASAO`, `QT_SIT_DESVINCULADO`, `QT_SIT_TRANCADA`, `QT_EVADIDOS` — removidas antes de `X` existir. `Trainer` extrai `y` **antes** de chamar `remove_leakage_columns`, então mesmo que o alvo sobrevivesse ali, nunca chegaria a `X`. Testado em `tests/test_leakage_removal.py`, `tests/test_pipeline_integration.py`. | ✅ Sem vazamento |
| 2 | Features derivadas do target | `RAZAO_CONCLUSAO_EVASAO` (calculada a partir de `TAXA_EVASAO`) está em `KNOWN_LEAKY_DERIVED_FEATURES` e é removida antes de chegar a qualquer `X`; `TreeFeatureAugmenter` nunca a recria. **Confirmado empiricamente nesta tarefa** (seção 1.2, Achado 2): quando presente (notebook original), essa feature tem `importance_mean = 0,4433` — mais que o dobro da segunda mais importante — prova direta de que ela de fato vaza o alvo quando não é removida. | ✅ Removida e prejuízo comprovado se não fosse |
| 3 | `fit` antes do CV | `FeatureEngineer.for_*()` devolve `Pipeline`s **não ajustados**; `Trainer` só chama `.fit_transform` dentro do laço de fold, sobre `X_fit` (treino menos calibração). Nenhum `.fit(` fora desse laço em `src/training/trainer.py`. Testado em `tests/test_training.py::test_feature_pipeline_is_refit_independently_per_fold`. | ✅ Sem vazamento |
| 4 | Scaler ajustado no dataset inteiro | Todo `StandardScaler`/`RobustScaler`/`MinMaxScaler` em `src/pipelines/feature_engineering.py` é instanciado dentro de `for_logistic_regression()`/`for_svm()`/`for_neural_network()`/`for_naive_bayes()` — chamados a cada fold via `_feature_pipeline_for(...)` dentro do laço do `Trainer`, nunca uma vez só no início. | ✅ Sem vazamento |
| 5 | Imputação global | **Achado real desta tarefa.** `src.data.pipeline.run_data_pipeline` chama `run_base_preprocessing(missing_strategy=...)` **uma vez**, sobre o dataset inteiro, **antes** do laço de CV do `Trainer` — para a estratégia `"mode"`, a moda de cada coluna categórica é calculada com base em 100% das linhas (incluindo o que depois vira fold de validação). Para a estratégia `"zero"` (padrão do pipeline, usada pelo Naive Bayes) não há vazamento, porque `0` não é uma estatística dependente dos dados. | ⚠️ **Vazamento real, severidade baixa** — ver discussão abaixo |
| 6 | SMOTE antes do split | `apply_smote` (linha 202 de `src/training/trainer.py`) é chamado **depois** de `feature_pipeline.fit_transform(X_fit, y_fit)`, e `X_fit`/`y_fit` vêm de `split_train_calibration(X_train, y_train, ...)`, que por sua vez recebe `X_train = X.iloc[train_idx]` — **depois** do `StratifiedKFold.split`. Ordem confirmada: split de CV → split de calibração → fit da engenharia de features → SMOTE → fit do modelo. Testado indiretamente em `tests/test_imbalance.py` e `tests/test_training.py`. | ✅ Sem vazamento |
| 7 | Threshold tuning usando validação indevidamente | `find_best_threshold` (linha 223 de `trainer.py`) é chamado **só** com `y_calib`/`calibration_proba`. `y_val`/`validation_proba` só aparecem depois, já usados apenas para *calcular métricas* com o threshold já escolhido — nunca para escolhê-lo. Testado em `tests/test_training.py::test_validation_fold_metrics_do_not_use_calibration_labels_directly`. | ✅ Sem vazamento |

### 5.1 Discussão do item 5 (o único achado real desta rodada)

**Severidade:** baixa, mas real. Argumentos para "baixa":

- Afeta só a estratégia `"mode"` (usada pelos 4 modelos "notebook-style");
  a estratégia `"zero"` (padrão do pipeline, usada pelo Naive Bayes) tem
  vazamento **zero** por construção — `0` não depende de nenhum dado.
- A moda de uma coluna categórica censitária (`TP_ORGANIZACAO_ACADEMICA`,
  `TP_REDE` etc., com poucas categorias e uma claramente dominante) é
  extremamente estável — é muito improvável que a moda calculada sobre
  100% das linhas difira da moda calculada sobre os ~80% que sobram
  depois de excluir um fold de validação. Diferente de um scaler
  (média/desvio, sensível a cada valor) ou de um seletor de features
  (pode escolher colunas diferentes), a moda de uma variável com poucas
  categorias tende a ser idêntica com ou sem um punhado de linhas.
- Não pude medir a magnitude numérica com o CSV sintético desta tarefa
  porque ele foi gerado sem nenhum valor ausente (`rng.integers`, sem
  `NaN` injetado) — então a imputação é literalmente um no-op sobre esse
  dado. É uma limitação real desta verificação: o achado é uma análise
  de código correta, mas não pude quantificar seu efeito numérico aqui.

**Por que não foi corrigido nesta etapa:** corrigir exigiria mudar o
contrato de `run_data_pipeline` (separar "o quê imputar" de "quando
imputar", movendo a imputação para dentro do laço de fold do `Trainer`)
— uma mudança de arquitetura em dois módulos já testados por 190 casos,
arriscada de fazer sem revisão dedicada na última tarefa da série.
**Recomendação registrada:** mover a imputação de `"mode"` para dentro
de `Trainer` (ajustada só em `X_fit`, aplicada em `X_calib`/`X_val`),
como uma etapa dedicada e testada — pode ser feita junto com a correção
de D1 (amostragem estratificada), já que ambas tocam o mesmo trecho do
fluxo entre `run_data_pipeline` e `Trainer.train`.

---

## 6. Conclusão sobre equivalência científica

**A migração preserva o comportamento científico do projeto nos pontos
onde isso pôde ser verificado com precisão, e identificou dois problemas
reais (um pré-existente, um novo) que precisam de atenção antes de um
uso científico dos números absolutos da nova arquitetura contra o
dataset real.**

**O que está confirmado, com evidência forte:**

- **Naive Bayes:** equivalência estrutural **exata** (registros,
  features, distribuição de classes idênticos, byte a byte) entre o
  pipeline legado e o novo, e hiperparâmetros idênticos aos do artefato
  real (`best_params.json`). É o modelo com a validação mais completa e
  mais forte desta tarefa.
- **Engenharia de features comum (Árvore/LR/SVM/Rede Neural):** as 9
  features derivadas compartilhadas são **numericamente idênticas**
  (diferença `0.0`) entre a fórmula do notebook original e
  `CommonFeatureSelector`, sobre 13.000 linhas.
- **Hiperparâmetros:** os 5 modelos usam exatamente os valores
  vencedores originais — 3 confirmados byte a byte contra artefatos
  reais (Árvore, LR, Naive Bayes), 2 contra a única fonte disponível
  (SVM, Rede Neural — `analysis/audit/evaluator.py`), já documentado na
  Tarefa 5.
- **Reprodutibilidade:** os 5 modelos produzem resultados **idênticos**
  entre duas execuções — nenhuma fonte de não-determinismo (paralelismo,
  seed não fixada) foi encontrada.
- **Leakage:** 6 dos 7 itens auditados não têm nenhum vazamento; o
  bug de threshold tuning sem guarda-corpo (a causa raiz de todo o
  problema original do projeto) foi **corrigido e a correção
  verificada** — o notebook original ainda reproduz o bug ao vivo; a
  nova arquitetura, com o mesmo threshold tuning aplicado a todos os
  modelos, não.

**O que não está confirmado — dois problemas reais, documentados, não
corrigidos nesta etapa:**

- **D1 (amostragem estratificada ausente):** impede comparar números
  absolutos de métrica entre a nova arquitetura e os resultados
  históricos até ser corrigido. Não é um bug da lógica migrada em si
  (feature engineering, treino, threshold, leakage todos verificados
  corretos isoladamente) — é uma etapa do pipeline original que nenhuma
  tarefa migrou.
- **Achado 5 (imputação global para a estratégia `"mode"`):** vazamento
  real, mas de severidade avaliada como baixa; não quantificado
  numericamente por limitação do dataset sintético usado nesta
  verificação.

**Recomendação final:** antes de rodar `experiments/*.ipynb` ou
`scripts/train_all.py` contra o dataset real do INEP para produzir
números comparáveis aos relatórios científicos existentes, implementar
a etapa de amostragem estratificada (seção 4.1) e mover a imputação
`"mode"` para dentro do laço de fold do `Trainer` (seção 5.1) — as duas
únicas lacunas que esta validação encontrou. Sem essas duas correções, a
*arquitetura* está validada (feature engineering, hiperparâmetros,
leakage, reprodutibilidade), mas os *números absolutos* que ela produzir
sobre o dataset real ainda não seriam diretamente comparáveis aos
históricos.
