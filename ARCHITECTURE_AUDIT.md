# Auditoria de Arquitetura — Projeto-Final-Machine-Learning

**Branch:** `audit/architecture-review` (criada a partir de `main`, nenhum arquivo do projeto foi modificado)
**Escopo:** somente análise. Nenhum notebook foi convertido, nenhum arquivo foi removido ou reorganizado.

---

## 1. Resumo executivo

O projeto compara cinco famílias de modelos (Árvore de Decisão, Regressão Logística, SVM, Rede Neural, Naive Bayes) para prever `alto_risco_evasao` a partir do Censo da Educação Superior 2024 (INEP). A base de código já passou por **duas rodadas de auditoria interna** (`analysis/audit/` e `analysis/final_validation/`), documentadas em `results/inconsistencies_report.md` e `results/audit/audit_report.md`. Essas auditorias corrigiram um problema grave de *threshold tuning* sem guarda-corpo (Regressão Logística "vencia" com recall ≈ 1,0 por um limiar patológico ≈0,01) e reavaliaram todos os modelos num protocolo unificado.

Esta auditoria de arquitetura confirma esses achados e identifica problemas adicionais, não documentados nas auditorias anteriores:

1. **Vazamento de dados direto e não corrigido no notebook da Árvore de Decisão**: a feature `RAZAO_CONCLUSAO_EVASAO` é construída a partir de `TAXA_EVASAO`, a mesma variável contínua usada para definir o alvo `ALTO_RISCO_EVASAO`. Isso não foi pego pela auditoria unificada porque `analysis/audit/data.py` reconstrói as features do zero e não usa as 10 features "avançadas" do notebook — ou seja, os números oficiais pós-auditoria da Árvore *não* têm esse vazamento, mas os números originalmente reportados pelo notebook (`models/arvore_de_decisao/resultados_arvore_decisao_5fold.csv`) têm.
2. **Duas definições de alvo e de feature set coexistem no repositório**: o pipeline dos 4 notebooks (`TAXA_EVASAO >= 20%`, 21 features manuais) e o pipeline do Naive Bayes (`data/preprocessamento.py`, mediana de `taxa_evasao`, todas as colunas numéricas). Isso já foi identificado nas auditorias internas, mas nunca foi resolvido na arquitetura — o Naive Bayes segue sem um caminho de "reavaliação justa" versionado como código de produção (só existe dentro de `analysis/audit/evaluator.py`, como uma reimplementação paralela).
3. **Ausência de nested cross-validation**: em DT, LR e SVM, a busca de hiperparâmetros (`RandomizedSearchCV`/`GridSearchCV`) roda **uma vez sobre o dataset inteiro**, antes do loop de 5-fold CV. Os hiperparâmetros escolhidos, portanto, "viram" todos os folds antes de serem avaliados neles — um viés otimista sutil, mas real, distinto do vazamento de alvo do item 1.
4. **Triplicação de lógica**: a leitura de dados, a engenharia de features base (21 features) e o cálculo do alvo aparecem, quase idênticos, em pelo menos **6 lugares** (4 notebooks + `analysis/audit/data.py` + implicitamente em `analysis/final_validation`). Qualquer correção metodológica futura precisa ser replicada manualmente em todos eles.
5. **Três sistemas de ranking concorrentes** (`results/ranking.csv`, `results/audit/audited_ranking.csv`, `results/final_validated_ranking.csv`), com metodologias de score diferentes (pesos 45/30/15/10 vs. pesos iguais de 20%), sem que o README indique qual é a fonte oficial para quem chega ao projeto agora.
6. **README desatualizado**: a árvore de diretórios do `README.md` não menciona `analysis/`, `results/`, `slides/` nem `report/` — cerca de metade do código e dos artefatos do projeto não está documentada.

O projeto tem valor científico real e já demonstra maturidade incomum (auditoria própria, sanity checks, testes de significância estatística). O principal risco não é falta de rigor — é que o rigor está espalhado em camadas sucessivas de "correção sobre correção" sem uma arquitetura única que impeça a régressão dos mesmos erros.

---

## 2. Arquitetura atual

```text
Projeto-Final-Machine-Learning/
├── data/
│   ├── __init__.py
│   └── preprocessamento.py          # pipeline PRÓPRIO do Naive Bayes (mediana, all-numeric)
├── models/
│   ├── arvore_de_decisao/
│   │   ├── ArvoreDecisao.ipynb      # pipeline duplicado nº1 (20%, 90k amostras)
│   │   └── *.csv/*.png/*.txt/*.pdf  # artefatos
│   ├── regrassao_logisticca/
│   │   ├── RegressaoLogistica.ipynb # pipeline duplicado nº2 (20%, 30k amostras)
│   │   └── *.csv/*.png/*.txt/*.pdf
│   ├── svm/
│   │   ├── SVM.ipynb                # pipeline duplicado nº3 (20%, 30k amostras)
│   │   └── *.csv/*.pdf
│   ├── Redes_Neurais/
│   │   ├── RedesNeurais.ipynb       # pipeline duplicado nº4 (20%, 30k amostras)
│   │   └── *.csv/*.pdf
│   └── naive_bayes/                 # único modelo com pipeline .py real
│       ├── train.py / evaluate.py / model_factory.py / preprocessing.py
│       ├── report_generator.py
│       ├── artifacts/ (pkl, json, csv)
│       └── report/ (pdf, html)
├── analysis/
│   ├── compare_models.py            # ranking "ingênuo" via scan de models/*/resultados_*.csv
│   ├── metrics_discovery.py
│   ├── report_generator.py
│   ├── scientific_analysis.py
│   ├── visualizations.py
│   ├── run_audit.py                 # 2ª camada: reavaliação unificada
│   ├── audit/
│   │   ├── data.py                  # pipeline unificado Nº5 (reimplementação dos notebooks)
│   │   ├── evaluator.py             # reimplementa os 5 modelos com hiperparâmetros hardcoded
│   │   ├── threshold.py             # Youden's J com guarda-corpo (a versão "correta")
│   │   ├── sanity_checks.py
│   │   └── final_report_pdf.py
│   └── final_validation/            # 3ª camada: validação científica final
│       ├── run_final_validation.py
│       ├── consistency.py, ranking.py, plots.py, roc_plot.py, decision_tree_audit.py
│       ├── scientific_discussion.py, latex_report.py
│       └── templates/ieee_report.tex.tpl
├── results/
│   ├── ranking.csv, final_comparison.csv, extracted_metrics.csv   # pré-auditoria
│   ├── audit/                       # pós-auditoria (oficial nº1)
│   ├── validation/, feature_importance/, decision_tree/           # validação final
│   ├── final_validated_ranking.csv  # pós-validação final (oficial nº2)
│   ├── visualizations/  vs.  audit/visualizations/                # possível duplicação
│   └── final_report.pdf / final_report.tex / IEEEtran.cls
├── report/final_report.pdf          # 3ª cópia de "relatório final"
├── slides/
├── EDA/.gitkeep                     # pasta vazia, sem uso aparente
├── requirements.txt                 # não inclui tensorflow nem jupyter (README pede instalação manual)
└── README.md                        # árvore de diretórios não reflete a estrutura real
```

### 2.1 Problemas arquiteturais principais

| # | Problema | Evidência |
|---|----------|-----------|
| A1 | Lógica de carregamento + engenharia de features base duplicada em 5 lugares (4 notebooks + `analysis/audit/data.py`) | células idênticas de `safe_div`, `COLS_CURSOS`, `ALTO_RISCO_EVASAO` copiadas em cada `.ipynb` |
| A2 | Dois universos de features/alvo incompatíveis (notebooks vs. Naive Bayes) sem adaptador oficial versionado fora de `analysis/audit/` | `data/preprocessamento.py` vs. `models/*/*.ipynb` |
| A3 | Notebooks importam de `analysis.audit.threshold` (ex.: `ArvoreDecisao.ipynb`), criando acoplamento circular entre "camada de exploração" e "camada de auditoria" | `encontrar_melhor_threshold` na célula 3 do notebook da Árvore |
| A4 | Três pipelines de scoring/ranking com pesos e fórmulas diferentes, nenhum marcado como deprecated | `analysis/metrics_discovery.py`, `analysis/audit/run_audit.py` (score 45/30/15/10), `analysis/final_validation/ranking.py` (pesos iguais 20%) |
| A5 | Nenhuma camada de "modelo" reutilizável — hiperparâmetros finais da Árvore/SVM/LR/MLP estão hardcoded dentro de `analysis/audit/evaluator.py`, não derivados automaticamente do resultado dos notebooks | `get_model_specs()` em `evaluator.py` |
| A6 | Sem testes automatizados (não há `tests/`, nem em `models/naive_bayes` nem em `analysis/`) | busca no repositório |
| A7 | Dataset lido por caminho relativo em cada notebook (`./MICRODADOS_...CSV`), exigindo cópia manual do arquivo de >100MB para cada pasta de modelo | células "CARREGAMENTO DOS DADOS" dos 4 notebooks |
| A8 | README não documenta `analysis/`, `results/`, `report/`, `slides/` | comparar `README.md` seção "Estrutura do repositório" com a árvore real |

---

## 3. Análise individual dos cinco modelos

### 3.1 Árvore de Decisão (`models/arvore_de_decisao/ArvoreDecisao.ipynb`)

- **Data loading:** inline, `pd.read_csv('./MICRODADOS_CADASTRO_CURSOS_2024.CSV', sep=';', encoding='latin1', usecols=COLS_CURSOS)` (22 colunas whitelisted).
- **Target:** `ALTO_RISCO_EVASAO = (TAXA_EVASAO >= 20).astype(int)`, com `TAXA_EVASAO = (QT_SIT_DESVINCULADO+QT_SIT_TRANCADA)/QT_ING*100`.
- **Amostra:** **90.000 registros** — o único notebook que usa 3× a amostra dos demais (30.000), o que por si só invalida qualquer comparação direta com os outros modelos a partir dos números *originais* do notebook (a auditoria em `analysis/audit/` corrige isso ao usar 30.000 para todos).
- **Leakage prevention:** nenhuma remoção explícita de colunas de vazamento — depende apenas de `ALL_FEATURES` não incluir `QT_SIT_DESVINCULADO`/`QT_SIT_TRANCADA`/`QT_CONC` diretamente. **Porém, a feature derivada `RAZAO_CONCLUSAO_EVASAO = TAXA_CONCLUSAO / (TAXA_EVASAO + 1)` usa `TAXA_EVASAO` diretamente — a mesma variável contínua binarizada no alvo.** Isso é vazamento direto de alvo→feature, presente **somente** neste notebook (célula "ENGENHARIA DE FEATURES", bloco "🔥 FEATURE ENGINEERING AVANÇADO"). Como é selecionada por `SelectKBest(mutual_info_classif, k=21)` a partir de 31 candidatas, é praticamente certo que essa feature tenha sido escolhida (MI altíssima por construção).
- **Feature engineering:** 8 categóricas (label encoding) + 4 numéricas brutas + 9 derivadas "base" (compartilhadas com os outros 3 notebooks) + **10 features "avançadas" exclusivas** (`RAZAO_CONCLUSAO_EVASAO` [vazada], `EAD_PREDOMINANTE`, `FINANC_ALTO`, `CURSO_PEQUENO`, `RAZAO_FEM_18_24`, `TAMANHO_CATEGORIA`, `RAZAO_FINANC_EAD`, `TAXA_PREENCHIMENTO`, `RAZAO_INSCRITOS_VAGAS`, `CURSO_COMPETITIVO`) → seleção `SelectKBest(mutual_info_classif, k=21)`.
- **Desbalanceamento:** SMOTE aplicado **dentro de cada fold**, sobre o treino apenas — correto.
- **Model:** `DecisionTreeClassifier`; busca `RandomizedSearchCV` com `param_distributions` amplo (criterion, max_depth, min_samples_split/leaf, max_features, class_weight, min_impurity_decrease). **Inconsistência textual:** os prints dizem "200 iterações" mas o código define `n_iter=500`.
- **Validação:** `StratifiedKFold(5, shuffle=True, rs=42)`; calibração de threshold em split interno (20% do treino balanceado) usando a função "segura" (`Youden's J` com guarda-corpo) importada de `analysis.audit.threshold` — é o único dos 4 notebooks que já usa a versão corrigida.
- **Pós-poda:** cost-complexity pruning (`ccp_alpha`) calculado a posteriori sobre o primeiro fold e aplicado retroativamente a `best_params`.
- **Evaluation:** accuracy, F1, precision, recall (por fold e agregado), profundidade/nº de folhas da árvore, importância de features.
- **Outputs:** `arvore_decisao_feature_importance.{csv,png}`, `arvore_decisao_metricas.png`, `arvore_decisao_visualizacao.png`, `resultados_arvore_decisao_5fold*.csv`, `melhores_hiperparametros_arvore.txt`, `Relatorio_Arvore_Decisao.pdf`.

### 3.2 Regressão Logística (`models/regrassao_logisticca/RegressaoLogistica.ipynb`)

- **Data loading:** idêntico ao da Árvore (mesmo bloco de código, copiado).
- **Target:** idêntico (`TAXA_EVASAO >= 20`), amostra de **30.000** (estratificada 50/50 aproximada por proporção real).
- **Leakage prevention:** mesma whitelist de 22 colunas; sem as features avançadas da Árvore, logo sem o vazamento direto do item 3.1. Ainda assim, `QT_MAT` (matriculados) permanece como feature bruta e é usado também como denominador de `TAXA_CONCLUSAO`, `RAZAO_ING_MAT` e `INDICE_FINANCIAMENTO` — ver seção 5 sobre vazamento indireto por definição do alvo.
- **Feature engineering:** apenas as 21 features "base" comuns (8 cat + 4 num + 9 derivadas). Nenhuma feature extra, nenhuma seleção.
- **Scaling:** `StandardScaler`, ajustado **uma vez sobre todo o dataset antes do CV** (`X_scaled = scaler_grid.fit_transform(X_processed)`) só para a etapa de busca de hiperparâmetros; dentro do loop de 5-fold o scaler é reajustado corretamente por fold.
- **Model:** `LogisticRegression`; `GridSearchCV` em grade pequena (`C`, `penalty` l1/l2, `solver='liblinear'`, `class_weight='balanced'`), `cv=3`.
- **Threshold tuning:** função local `encontrar_melhor_threshold` — varre 100 pontos entre 0,01 e 0,99 maximizando F1 **sem nenhum guarda-corpo de prevalência**. **Esta é a causa raiz do recall≈1,0/threshold≈0,01 documentado em `results/inconsistencies_report.md` item 1** — a versão do notebook não foi corrigida; só a reavaliação em `analysis/audit/` usa a versão segura.
- **Validation:** `StratifiedKFold(5)`, split de calibração interno 20%, sem nested CV (busca de hiperparâmetros roda uma vez fora do loop de folds).
- **Evaluation:** accuracy, F1, precision, recall.
- **Outputs:** `regressao_logistica_resultados.png`, `resultados_regressao_logistica_5fold*.csv`, `melhores_hiperparametros.txt`, `Relatorio_Regressao_Logistica.pdf`.

### 3.3 SVM (`models/svm/SVM.ipynb`)

- **Data loading:** mesmo bloco duplicado, mesma amostra de 30.000.
- **Target:** idêntico.
- **Leakage prevention:** mesma whitelist; sem features avançadas.
- **Feature engineering:** seleção por correlação absoluta com o alvo (`|corr| >= 0.05`) sobre as 21 features base — critério próprio, diferente do `SelectKBest`/MI da Árvore. Depois roda um **benchmark de pré-processamento** (5 combinações fixas de missing/encoding/scaling/seleção, não as 48 combinações teoricamente possíveis descritas nos comentários) para escolher a config final: `mode + label + standard`, mas a busca de hiperparâmetros final na prática usa `RobustScaler`, não `StandardScaler` — **o resultado do benchmark não é o que é efetivamente usado na etapa seguinte** (inconsistência entre a "melhor configuração" reportada e a configuração aplicada).
- **Model:** `SVC`; busca em duas etapas — `param_dist` amplo definido na célula 5 (inclui `kernel: ['rbf','linear','poly','sigmoid']`, `degree`, `coef0`) nunca é usado; a busca real (célula 8) usa um `param_dist` menor (`C`, `gamma`, `shrinking`, `tol`) **sem `kernel`**, portanto o kernel fica sempre no default do pipeline. `RandomizedSearchCV(n_iter=12, cv=2)` roda sobre uma subamostra de 12.000 (não o dataset de treino completo) — busca bem mais leve que a dos demais modelos.
- **Validation:** `StratifiedKFold(5)` com split de calibração interno; threshold via varredura da `decision_function` (sem guarda-corpo, mesmo risco metodológico da Regressão Logística, mitigado apenas porque o SVM com kernel RBF não degenera tão facilmente). Há ainda um holdout adicional de 20% (célula 9) só para gerar a matriz de confusão "final", criado com re-treino específico — introduz uma terceira divisão de dados não usada em nenhuma métrica agregada.
- **Evaluation:** accuracy, F1, precision, recall, matriz de confusão, classification report, distribuição de hiperparâmetros por fold.
- **Outputs:** `resultados_svm_5fold*.csv`, `resultados_svm_benchmark_preprocessamento.csv`, `Relatorio_SVM.pdf`.

### 3.4 Rede Neural (`models/Redes_Neurais/RedesNeurais.ipynb`)

- **Data loading:** mesmo bloco duplicado (variável `df` em vez de `cursos_df`, mas idêntico), amostra 30.000.
- **Target:** idêntico.
- **Leakage prevention:** mesma whitelist; nenhuma feature avançada.
- **Feature engineering:** as 21 features base, **label-encoded, sem nenhum scaling** — diferente de LR/SVM, que normalizam. Isso é relevante porque redes neurais treinadas com Adam/SGD são sensíveis à escala das entradas; a ausência de `StandardScaler` aqui é uma lacuna metodológica real, não documentada nas auditorias anteriores.
- **Exploração adicional (não usada no modelo final):** `RandomForestClassifier` + `mutual_info_classif` só para ranking de importância; `PCA(2)` só para visualizar fronteiras de decisão 2D.
- **Model:** 5 arquiteturas Keras comparadas informalmente (perceptron linear, perceptron não-linear, MLP 1 camada, MLP 2 camadas, MLP 1 camada + SGD/momentum) — só a última é validada com 5-fold CV; as outras 4 recebem apenas um único split 80/20, **sem repetição nem CV** (inconsistência de rigor dentro do mesmo notebook). `EarlyStopping` é importado mas nunca passado a `model.fit(...)` — callback morto.
- **Desbalanceamento:** `class_weight` calculado via `compute_class_weight('balanced', ...)` dentro de cada fold (correto), mais um teste anterior de "sem balanceamento" vs. "com balanceamento" só a título de comparação exploratória.
- **Validation:** `StratifiedKFold(5)` apenas para o modelo final escolhido (`build_mlp_backprop`); threshold tuning via split de calibração interno, com varredura simples (sem guarda-corpo Youden).
- **Evaluation:** accuracy, F1, precision, recall; matrizes de confusão comparativas entre as 5 arquiteturas (com threshold 0,5 fixo nessa comparação, mas threshold calibrado no resultado final reportado).
- **Outputs:** `nn_cv_metrics.png`, `resultados_rede_neural_5fold*.csv`, `Redes_Neurais-1.pdf`.

### 3.5 Naive Bayes (`models/naive_bayes/*.py`)

- **Data loading:** `data/preprocessamento.py::get_df_preprocessado()` — o único pipeline do projeto que é módulo Python real (não notebook), com busca de caminho em 3 candidatos, tratamento de erro explícito e logging.
- **Target:** `alto_risco_evasao = (taxa_evasao >= mediana(taxa_evasao))`, com `taxa_evasao = QT_SIT_DESVINCULADO / (QT_MAT + QT_SIT_DESVINCULADO)` — **definição diferente** da usada pelos outros 4 modelos (limiar relativo/mediana vs. limiar absoluto de 20%; denominador diferente; sem `QT_SIT_TRANCADA` no numerador). Filtros adicionais: `QT_ING >= 10`, `QT_MAT > 0`.
- **Leakage prevention:** **a mais rigorosa do projeto** — remove explicitamente `COLUNAS_IDENTIFICADORAS`, `COLUNAS_VAZAMENTO` (inclui `taxa_evasao`, `QT_SIT_*`) e qualquer coluna com prefixo `QT_MAT`/`QT_CONC` (`PREFIXOS_VAZAMENTO`). Isso evita justamente o problema que os outros 4 notebooks têm de manter `QT_MAT` como feature bruta.
- **Feature engineering:** usa **todas as colunas numéricas restantes** do censo (não as 21 features manuais dos notebooks) — universo de features totalmente diferente. Por variante:
  - Gaussian: `VarianceThreshold(0.01)` → `CorrelationRemover(0.90)` (implementação própria, remove uma de cada par com `|corr|>0.90`) → `StandardScaler`.
  - Bernoulli: mesma seleção de variância/correlação → `Binarizer(threshold configurável)`.
  - Complement: apenas `MinMaxScaler` (exige valores não-negativos).
- **Desbalanceamento:** SMOTE aplicado só no treino, fora do pipeline sklearn (função separada `aplicar_smote_treino`), tanto na busca quanto na avaliação final — disciplina correta.
- **Model/busca:** não usa `GridSearchCV`/`RandomizedSearchCV` — implementa a própria varredura aleatória (`_gerar_configuracoes_aleatorias`, 30 configs por variante × 3 variantes = 90 experimentos), cada uma avaliada com StratifiedKFold(5) completo. Seleção final por **recall** (não F1), com desempate por F1/ROC-AUC/menor desvio de recall.
- **Validation:** `StratifiedKFold(5)`, sem split de calibração — não há threshold tuning no pipeline original do Naive Bayes; usa o threshold padrão (0,5) do `predict()`.
- **Evaluation:** accuracy, precision, recall, F1, ROC-AUC (por fold e resumo), matriz de confusão agregada.
- **Outputs:** `artifacts/all_experiments.csv`, `best_params.json`, `best_model.pkl` (contém preprocessador + modelo serializados juntos), `confusion_matrix.csv`, `metrics_by_fold.csv`, `metrics_summary.csv`, `report/naive_bayes_report.pdf`, `report/project_naive_bayes_architecture.html`.
- **Nota:** como o feature set e o alvo diferem dos outros 4 modelos, o Naive Bayes original **não é comparável** ao resto do ranking sem reavaliação — que é exatamente o que `analysis/audit/evaluator.py` faz (reimplementando o Naive Bayes com `MinMaxFoldPreprocessor` sobre as 21 features dos notebooks). O resultado dessa reavaliação (recall ≈ 0,047, o pior do projeto) é drasticamente diferente do resultado original do pipeline próprio (recall ≈ 0,973 documentado em `inconsistencies_report.md`), porque o alvo e as features são outros.

---

## 4. Análise de feature engineering

### 4.1 Tabela de transformações

| Modelo | Feature/Etapa | Transformação | Motivo | Pode ser compartilhada? |
|---|---|---|---|---|
| Comum (DT/LR/SVM/NN) | 8 colunas `TP_*`/`IN_GRATUITO`/`CO_CINE_AREA_GERAL` | Label Encoding | tornar categóricas numéricas | **Sim** — idêntico nos 4 notebooks |
| Comum | `QT_ING`,`QT_MAT`,`QT_VG_TOTAL`,`QT_INSCRITO_TOTAL` | uso bruto (com imputação por moda) | contagens diretas do censo | **Sim**, mas revisar `QT_MAT` (risco de vazamento indireto, seção 5) |
| Comum | `TAXA_CONCLUSAO`, `RAZAO_ING_MAT`, `INDICE_FINANCIAMENTO` | razão (`safe_div`) usando `QT_MAT` como denominador | proxy de desempenho do curso | **Sim**, com a mesma ressalva acima |
| Comum | `PROPORCAO_EAD`, `PROPORCAO_NOTURNO`, `PROPORCAO_FIES`, `PROPORCAO_PROUNIP`, `PROPORCAO_18_24`, `PROPORCAO_FEM` | proporções (`safe_div`) | perfil demográfico/modalidade do curso | **Sim** — preprocessing comum, sem relação com o alvo |
| Árvore (exclusiva) | `RAZAO_CONCLUSAO_EVASAO` | `TAXA_CONCLUSAO / (TAXA_EVASAO + 1)` | "interação" pretendida | **Não — vazamento de alvo, deve ser removida, não compartilhada** |
| Árvore (exclusiva) | `EAD_PREDOMINANTE`, `FINANC_ALTO`, `CURSO_PEQUENO`, `CURSO_COMPETITIVO` | flags binárias por limiar | discretização de sinal contínuo | Sim, como feature engineering opcional (não vazada) |
| Árvore (exclusiva) | `TAMANHO_CATEGORIA` | `pd.cut` em 4 faixas de `QT_ING` | binning estratégico | Sim, específica de árvore mas sem problema de vazamento |
| Árvore (exclusiva) | `RAZAO_FEM_18_24`, `RAZAO_FINANC_EAD`, `TAXA_PREENCHIMENTO`, `RAZAO_INSCRITOS_VAGAS` | razões compostas | interações de 2ª ordem | Sim, avaliar valor preditivo isoladamente |
| Árvore (seleção) | `SelectKBest(mutual_info_classif, k=21)` | seleção estatística | reduzir dimensionalidade | Transformação específica do modelo (não compartilhar a seleção, só a função) |
| Árvore (balanceamento) | SMOTE por fold | oversampling sintético | classes desbalanceadas dentro da amostra | Sim, como utilitário comum (`aplicar_smote_treino` do NB já cumpre esse papel) |
| Regressão Logística | `StandardScaler` | normalização z-score | GLM sensível a escala | Sim — comum a LR e potencialmente NN (que hoje não escala!) |
| SVM | seleção por correlação (`|corr|>=0.05`) | filtro univariado simples | reduzir dimensionalidade | Transformação específica do modelo (critério diferente do MI da Árvore) |
| SVM | `RobustScaler` | normalização robusta a outliers | kernel RBF sensível a escala | Específica do modelo, mas reaproveitável como opção comum |
| Rede Neural | nenhum scaling aplicado | — | (lacuna) | **Deveria ser comum** — ausência é um problema, não uma escolha |
| Rede Neural | `PCA(2)` | redução de dimensionalidade só para plot | visualização de fronteira de decisão | Não é usada para treino — não compartilhar como feature, só como utilitário de visualização |
| Naive Bayes | `VarianceThreshold(0.01)` | remove baixa variância | reduzir ruído | Transformação estatística comum, reaproveitável pelos outros modelos |
| Naive Bayes | `CorrelationRemover(0.90)` | remove colunas redundantes | reduzir multicolinearidade | Transformação estatística comum, reaproveitável |
| Naive Bayes | `Binarizer`/`MinMaxScaler` | conforme variante (Bernoulli/Complement exigem domínios específicos) | requisito do algoritmo | Específica do modelo |
| Naive Bayes | universo de features = todas as colunas numéricas menos vazamento | — | diferente do resto do projeto | **Não compartilhável sem decisão explícita de unificação de feature set** |

### 4.2 Classificação em grupos

- **`common_features`** (21): as 8 categóricas + 4 numéricas brutas + 9 derivadas usadas identicamente por DT/LR/SVM/NN. Candidatas naturais a um módulo único `feature_engineering/common.py`.
- **`tree_features`** (10, exclusivas da Árvore): as "features avançadas" — **excluir `RAZAO_CONCLUSAO_EVASAO`** antes de qualquer reuso; as outras 9 são candidatas legítimas.
- **`linear_features`**: nenhuma feature nova — só a etapa de `StandardScaler` sobre `common_features`.
- **`svm_features`**: nenhuma feature nova — seleção por correlação + `RobustScaler` sobre `common_features`.
- **`neural_network_features`**: nenhuma feature nova além de `common_features`; falta scaling (lacuna a corrigir na migração, não a preservar).
- **`naive_bayes_features`**: universo próprio (todas as colunas numéricas do censo, exceto identificadores e vazamento), incompatível por definição com os outros grupos — deve ser tratado como um "modo" de feature engineering separado, não como subconjunto de `common_features`.

---

## 5. Análise de vazamento de dados (leakage)

| Risco | Onde | Severidade | Detalhe |
|---|---|---|---|
| **Feature construída a partir do alvo contínuo** | `ArvoreDecisao.ipynb`, feature `RAZAO_CONCLUSAO_EVASAO` | **Crítico** | `TAXA_EVASAO` (usada para binarizar o alvo com corte em 20%) é usada como denominador de uma feature preditiva. O modelo pode aprender a "inverter" essa razão para reconstruir o alvo quase perfeitamente. Não corrigido nos números originais do notebook; a reavaliação em `analysis/audit/` não é afetada porque não usa essa feature. |
| **`QT_MAT` como feature bruta + denominador de derivadas** | DT/LR/SVM/NN, feature `QT_MAT` e derivadas `TAXA_CONCLUSAO`, `RAZAO_ING_MAT`, `INDICE_FINANCIAMENTO` | **Moderado** | O alvo é função de `QT_SIT_DESVINCULADO`/`QT_ING`; `QT_MAT` (matriculados) é definicionalmente relacionado a quem *não* evadiu. Não é vazamento direto (a coluna não entra na fórmula do alvo), mas é uma variável estruturalmente correlacionada com a definição de evasão, o que pode inflar artificialmente o poder preditivo de forma não generalizável a outros cortes de dados. `data/preprocessamento.py` já trata isso corretamente ao excluir qualquer coluna com prefixo `QT_MAT`/`QT_CONC`. |
| **Hiperparâmetros ajustados sobre o dataset inteiro antes do CV** | DT (`RandomizedSearchCV` na célula 8), LR (`GridSearchCV` na célula 8), SVM (`RandomizedSearchCV` na célula 8 sobre subamostra) | **Moderado** | Não é vazamento de *dados* no sentido clássico, mas é uma violação de nested cross-validation: os folds de validação usados depois já influenciaram a escolha do modelo. As métricas por fold são otimisticamente enviesadas em algum grau (provavelmente pequeno, dado que os hiperparâmetros são fixos entre folds, mas metodologicamente incorreto). |
| **Ausência de holdout verdadeiramente cego** | Todos os notebooks (exceto o holdout extra do SVM, que não é usado nas métricas agregadas) | **Baixo/informativo** | Todas as métricas reportadas vêm de CV, sem um conjunto de teste final nunca tocado durante desenvolvimento. Aceitável para fins acadêmicos com CV disciplinado, mas deve ser explicitado no relatório final. |
| **Naive Bayes: alvo e features diferentes do resto** | `data/preprocessamento.py` vs. notebooks | **Não é leakage, mas é um risco de comparação inválida** | Já identificado e corrigido pela auditoria (`analysis/audit/evaluator.py` reavalia o NB no pipeline unificado). Mantido aqui porque é a causa raiz do item 2 do resumo executivo. |
| **Threshold tuning sem guarda-corpo (LR, SVM, NN)** | funções locais `encontrar_melhor_threshold`/`find_best_threshold_svm`/`find_best_threshold` nos respectivos notebooks | **Alto (metodológico, não leakage de dados)** | Já documentado em `results/inconsistencies_report.md`. Vale reafirmar aqui porque é a mesma classe de problema (superestimação de métrica) e porque só a Árvore usa a versão corrigida (`analysis.audit.threshold`) dentro do próprio notebook. |

---

## 6. Análise de arquivos

| Arquivo/Pasta | Motivo | Referenciado? | Pode remover? | Risco |
|---|---|---|---|---|
| `EDA/.gitkeep` | pasta vazia, nenhum notebook/script de EDA versionado | Não | Avaliar — se a EDA existiu apenas localmente e nunca foi versionada, ou popular a pasta ou removê-la | Baixo |
| `results/ranking.csv`, `results/final_comparison.csv`, `results/extracted_metrics.csv` | métricas **pré-auditoria**, já documentadas como não-oficiais em `results/inconsistencies_report.md` | Sim, citados no relatório de inconsistências como "fonte pré" | **Não remover** — são parte do rastro de auditoria (mostram o antes/depois) | Médio se removidos sem contexto — perderiam a prova do problema corrigido |
| `results/visualizations/*` vs `results/audit/visualizations/*` | mesmos nomes de arquivo (`metric_heatmap.png`, `metrics_bar_chart.png`, `radar_chart.png`, `ranking_heatmap.png`) em dois diretórios | Sim, gerados por pipelines diferentes (`analysis/visualizations.py` vs. auditoria) | Não remover sem antes confirmar qual representa "pré" e qual "pós" auditoria — hoje o nome sozinho não deixa isso claro | Médio (risco de confusão, não de perda de informação) |
| `report/final_report.pdf`, `results/final_report.pdf`, `results/audit/final_report_audited.pdf` | três PDFs de "relatório final" em estágios diferentes do pipeline | Sim, cada um gerado por uma etapa distinta (`analysis/report_generator.py`, `analysis/audit/final_report_pdf.py`, `analysis/final_validation/latex_report.py`→`results/final_report.tex`) | Não remover — mas a nomenclatura idêntica (`final_report.pdf`) em dois lugares diferentes é um problema de nomenclatura a resolver na migração | Alto risco de confusão para quem abre o repositório pela primeira vez |
| `models/naive_bayes/report/project_naive_bayes_architecture.html` | HTML de arquitetura específico do Naive Bayes, propósito não documentado no README | Não claramente referenciado fora da própria pasta | Avaliar com a equipe — pode ser material de apresentação ou artefato órfão | Baixo |
| `.venv`/`env` (se existirem localmente) | já ignorados via `.gitignore` (`env/`, mas não `.venv/`) | — | N/A (não versionado) | Nota: `.gitignore` cobre `env/` mas não `.venv/`; o README recomenda `.venv`. Ajustar `.gitignore` é uma correção segura, não uma remoção de arquivo |
| `results/audit/*_sanity.json`, `results/audit/*_folds.csv` | artefatos do protocolo de sanidade, únicos e não duplicados em outro lugar | Sim, fonte primária citada em `audit_report.md` | Não remover — são a evidência bruta do protocolo de auditoria | Alto se removidos |

Nenhum arquivo do projeto foi identificado como seguramente removível sem revisão humana — o padrão dominante é "existem versões pré e pós auditoria com nomes parecidos", que é um problema de **nomenclatura e organização**, não de arquivos mortos.

---

## 7. Arquitetura proposta (não implementada)

```text
Projeto-Final-Machine-Learning/
├── src/
│   ├── data/
│   │   ├── loading.py          # 1 função: localizar e ler o CSV (substitui as 5 cópias)
│   │   └── target.py           # construção EXPLÍCITA do alvo, com 1 estratégia por vez,
│   │                            # nomeada (ex.: "threshold_20pct" vs "median_split"),
│   │                            # nunca implícita dentro de um notebook
│   ├── features/
│   │   ├── common.py           # as 21 features "common_features" (seção 4.2)
│   │   ├── tree_features.py    # as 9 features avançadas da Árvore SEM RAZAO_CONCLUSAO_EVASAO
│   │   ├── naive_bayes_features.py  # universo de features do NB (todas numéricas - vazamento)
│   │   └── leakage_guard.py    # lista central de colunas proibidas (equivalente a
│   │                            # COLUNAS_VAZAMENTO/PREFIXOS_VAZAMENTO), usada por TODOS os
│   │                            # pipelines de feature engineering, sem exceção
│   ├── preprocessing/
│   │   ├── pipelines.py        # scalers/encoders/seletores como sklearn Pipeline nomeados
│   │   └── balancing.py        # SMOTE / class_weight como utilitário único
│   ├── models/
│   │   ├── registry.py         # 1 dict central: nome → (factory, hiperparâmetros, feature_set)
│   │   ├── decision_tree.py, logistic_regression.py, svm.py, neural_network.py, naive_bayes.py
│   ├── training/
│   │   ├── cross_validation.py # protocolo único de 5-fold + split de calibração
│   │   ├── hyperparameter_search.py  # busca DENTRO de cada fold externo (nested CV real)
│   │   └── threshold_tuning.py # única implementação: Youden's J com guarda-corpo
│   │                            # (hoje é `analysis/audit/threshold.py`; deveria ser a única)
│   ├── evaluation/
│   │   ├── metrics.py          # accuracy/precision/recall/F1/ROC-AUC/PR-AUC unificados
│   │   ├── sanity_checks.py    # as 6 verificações já existentes em analysis/audit/sanity_checks.py
│   │   └── ranking.py          # ÚNICA fórmula de score, com pesos documentados e justificados
│   ├── visualization/
│   │   └── plots.py            # gráficos comparativos e por-modelo, com destino único
│   └── artifacts.py             # convenção única de nomes/pastas de saída (csv/png/pdf/json/pkl)
├── experiments/
│   ├── run_model.py             # `python -m experiments.run_model --model arvore_decisao`
│   ├── run_all.py                # roda os 5 modelos e gera o comparativo
│   └── configs/                  # 1 arquivo de config por modelo (hiperparâmetros, feature_set)
├── notebooks/                    # SOMENTE leitura/visualização — chamam src/, não reimplementam
│   ├── 01_eda.ipynb
│   ├── 02_arvore_de_decisao.ipynb
│   ├── 03_regressao_logistica.ipynb
│   ├── 04_svm.ipynb
│   ├── 05_rede_neural.ipynb
│   └── 06_naive_bayes.ipynb
├── tests/
│   ├── test_target_construction.py     # garante que o alvo é determinístico e sem leakage
│   ├── test_leakage_guard.py           # garante que nenhuma feature colide com leakage_guard
│   ├── test_feature_engineering.py
│   ├── test_threshold_tuning.py        # garante que o threshold nunca fica no extremo da busca
│   └── test_cross_validation.py        # garante ausência de vazamento entre folds
├── artifacts/                    # única árvore de saída (substitui results/, report/, models/*/*.csv)
│   ├── <model>/metrics/, <model>/plots/, <model>/models/
│   └── comparison/ranking.csv, comparison/report.pdf
├── docs/
│   └── audit_history.md          # histórico das rodadas de auditoria (o que hoje está espalhado
│                                   # em inconsistencies_report.md, audit_report.md etc.)
├── data_raw/ (gitignored)
├── requirements.txt (com tensorflow e jupyter incluídos, ou requirements-notebooks.txt separado)
└── README.md (árvore de diretórios sempre sincronizada com a estrutura real)
```

### 7.1 Como a proposta atende aos critérios pedidos

- **Evita duplicação:** carregamento, alvo, features comuns e threshold tuning passam a existir em um único lugar cada; os 4 notebooks hoje quase idênticos colapsam para chamadas a `src/`.
- **Notebooks só para análise/visualização:** cada notebook chamaria `src.training.cross_validation.run(model="arvore_decisao")` e depois plotaria os resultados — sem reimplementar pipeline.
- **Treinar qualquer modelo via Python:** `experiments/run_model.py --model X` cobre isso; hoje só o Naive Bayes tem essa capacidade (`train.py`).
- **Adicionar novos modelos:** basta um novo arquivo em `src/models/` registrado em `registry.py` + um config em `experiments/configs/`.
- **Preserva resultados científicos:** `docs/audit_history.md` mantém o histórico completo (pré-auditoria → auditoria → validação final) como narrativa, em vez de três pastas de resultado concorrentes; os CSVs originais de `results/` podem ser movidos para `artifacts/legacy/` em vez de apagados.
- **Minimiza risco de leakage:** `leakage_guard.py` centralizado, testado (`test_leakage_guard.py`), usado por todos os pipelines — teria pego a `RAZAO_CONCLUSAO_EVASAO` antes de qualquer treino.
- **Testável:** módulo `tests/` cobre construção de alvo, guard de leakage, threshold tuning e ausência de vazamento entre folds — nada disso existe hoje.
- **Reprodutível:** um único caminho de dataset resolvido centralmente (`src/data/loading.py`), sem exigir cópia manual do CSV em 4 pastas diferentes.

---

## 8. Plano de migração (proposto, não executado)

1. **Fase 0 — Congelar a linha de base.** Rodar `analysis/run_audit.py` e `analysis/final_validation/run_final_validation.py` uma última vez sobre o estado atual e arquivar os outputs em `docs/audit_history.md` como "resultado de referência pré-migração", para servir de teste de regressão.
2. **Fase 1 — Extrair `src/data/` e `src/features/common.py`** a partir do código já triplicado nos notebooks e em `analysis/audit/data.py`, escrevendo testes que comparem a saída da nova função com a saída manual dos notebooks atuais (bit-a-bit, mesma seed) antes de tocar em qualquer notebook.
3. **Fase 2 — Corrigir o vazamento da Árvore.** Remover `RAZAO_CONCLUSAO_EVASAO` do conjunto `tree_features`, re-treinar e comparar métricas com e sem a feature — documentar a diferença no `docs/audit_history.md` como parte da auditoria, não silenciosamente.
4. **Fase 3 — Unificar threshold tuning.** Substituir as 3 implementações "inseguras" (LR, SVM, NN) pela função já existente em `analysis/audit/threshold.py`, promovida a `src/training/threshold_tuning.py`.
5. **Fase 4 — Corrigir nested CV.** Mover a busca de hiperparâmetros para dentro do loop externo de folds (ou aceitar explicitamente o custo computacional de re-buscar por fold, documentando a escolha).
6. **Fase 5 — Decidir o destino do Naive Bayes.** Escolher, com a equipe, se o projeto mantém dois universos de features/alvo documentados lado a lado (acadêmico, ambos válidos) ou se migra o Naive Bayes para o feature set comum como os outros 4 modelos — hoje essa decisão está implícita em `analysis/audit/evaluator.py`, sem registro explícito da justificativa.
7. **Fase 6 — Migrar os notebooks para consumir `src/`,** um de cada vez, validando que os números batem com os já publicados em `results/audit/` antes de apagar o código antigo do notebook.
8. **Fase 7 — Consolidar ranking.** Escolher uma única fórmula de score (recomenda-se a com guarda-corpos de `analysis/audit/`, por já incorporar os sanity checks) e mover as outras duas para `artifacts/legacy/` com nota explicando por que foram substituídas.
9. **Fase 8 — Atualizar o README** para refletir a estrutura final, incluindo instruções de reprodução via `experiments/run_model.py`.

## 9. Riscos da migração

| Risco | Mitigação |
|---|---|
| Re-treinar a Árvore sem `RAZAO_CONCLUSAO_EVASAO` muda o ranking dos 5 modelos (ela pode estar inflando a posição da Árvore mesmo na versão "oficial" do notebook) | Tratar como resultado científico esperado e documentar explicitamente — é o objetivo da correção, não um efeito colateral indesejado |
| Servir dois pipelines de feature/alvo (notebooks vs. Naive Bayes) dentro da mesma arquitetura pode gerar confusão para quem só olha `src/models/naive_bayes.py` | Nomear explicitamente os dois "modos" de dataset (`target=threshold_20pct` vs `target=median_split`) em vez de deixar implícito |
| Extrair `src/data/loading.py` pode introduzir divergências sutis de tipo/arredondamento em relação ao código legado dos notebooks | Testes de regressão bit-a-bit (Fase 1) antes de qualquer refatoração subsequente |
| Perda de artefatos históricos (`results/ranking.csv` etc.) durante a reorganização de pastas | Mover, nunca apagar, para `artifacts/legacy/`; manter no controle de versão |
| Corrigir o nested CV pode aumentar substancialmente o tempo de execução (busca de hiperparâmetros por fold em vez de uma vez só) | Avaliar custo/benefício explicitamente; para um projeto acadêmico, pode ser aceitável manter a busca única fora do loop, desde que documentado como limitação conhecida em vez de bug silencioso |
| Equipe (5 pessoas, ver `README.md`) trabalhando em paralelo em notebooks separados pode reintroduzir duplicação depois da migração | Regra de revisão: nenhum PR pode adicionar lógica de carregamento/target/feature engineering fora de `src/` |

---

*Relatório gerado por auditoria automatizada em `audit/architecture-review`. Nenhum arquivo do projeto original foi alterado.*
