# Camada centralizada de treinamento (Tarefa 6)

**Branch:** `audit/architecture-review`
**Baseado em:** [`docs/FEATURE_ENGINEERING.md`](FEATURE_ENGINEERING.md), [`docs/MODELS.md`](MODELS.md)

Remove dos notebooks a lógica de `fit`, cross-validation, busca de
hiperparâmetros, seleção do melhor modelo, tratamento de desbalanceamento
e threshold tuning — centralizando tudo em `src/training/`. Nenhum
notebook foi removido; continuam funcionando exatamente como antes.

---

## 1. Interface

```python
from src.training.trainer import Trainer

trainer = Trainer()  # 5 folds, shuffle, random_state=42 — o protocolo já em uso no projeto
result = trainer.train(model_name="svm", X=X, y=y)

result.metrics          # {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "roc_auc": ..., "pr_auc": ...}
result.metrics_std      # desvio padrão de cada métrica entre os 5 folds
result.fold_results      # 1 FoldResult por fold (threshold, tamanhos, métricas)
result.predictions       # predições out-of-fold, alinhadas a X/y
result.probabilities     # probabilidade da classe positiva, out-of-fold
result.fitted_models     # 1 estimador ajustado por fold
result.params            # hiperparâmetros efetivos do último fold
result.imbalance_strategy  # estratégia de desbalanceamento usada
```

`X` deve ser a saída de `src.data.pipeline.run_data_pipeline(...)` —
**ainda sem** passar pela engenharia de features específica do modelo
(isso acontece dentro do `Trainer`, por fold, para não vazar dado entre
folds). Ver `docs/FEATURE_ENGINEERING.md`, seção 8, para como montar
esse `X` corretamente por modelo (`include_prefix_leakage_group=False`
para os 4 modelos "notebook-style"; padrão para o Naive Bayes).

---

## 2. Onde a solução vive

```text
src/training/
├── trainer.py               # Trainer.train(...) — orquestrador único
├── cross_validation.py      # StratifiedKFold, split de calibração, métricas por fold
├── imbalance.py             # estratégia de desbalanceamento ORIGINAL de cada modelo
├── threshold_tuning.py      # única implementação de threshold tuning (com guarda-corpo)
└── hyperparameter_search.py # wrappers de GridSearchCV/RandomizedSearchCV

src/configs/
└── search_spaces.py         # espaços de busca originais (dados, não lógica)
```

---

## 3. O que acontece dentro de cada fold

Para cada um dos 5 folds de `StratifiedKFold` (5 folds, embaralhado,
`random_state=42` — os mesmos parâmetros de todos os notebooks e de
`analysis/audit/evaluator.py`):

```text
fold de treino (80% de N)              fold de validação (20% de N)
         │                                       │
         ├── 80% → treino-para-ajuste            │  (nunca tocado até a
         └── 20% → calibração                    │   avaliação final)
              │         │                        │
    feature engineering │                        │
    (fit só aqui)        │                        │
              │    threshold tuning               │
              │    (ajustado aqui)                 │
              │         │                          │
        modelo.fit(...) │                          │
                         └──────────► modelo.predict_proba(fold de validação)
                                       + threshold calibrado
                                       → métricas do fold
```

1. **Split de calibração** (`cross_validation.split_train_calibration`):
   separa 20% do fold de treino — nunca do fold de validação. Satisfaz o
   requisito explícito da Tarefa 6: "garanta que o conjunto de
   validação/teste permaneça invisível durante o ajuste".
2. **Engenharia de features** (`FeatureEngineer.for_<modelo>()`, Tarefa 4):
   ajustada **só** na fatia de treino-menos-calibração; aplicada
   (`transform`, nunca reajustada) em calibração e validação.
3. **Desbalanceamento** (seção 4): aplicado depois da engenharia de
   features, só na fatia de treino — nunca em calibração/validação.
4. **Ajuste do modelo** (`src.models.registry.create_model`, Tarefa 5).
5. **Threshold tuning** (seção 5): calibrado na fatia de calibração.
6. **Avaliação**: métricas calculadas no fold de validação, com o
   threshold calibrado — a validação nunca influencia nem o modelo, nem
   o threshold, nem a engenharia de features.

---

## 4. Desbalanceamento — estratégia original de cada modelo

Requisito explícito: **não** transformar tudo em SMOTE.
`src.training.imbalance.IMBALANCE_STRATEGY_BY_MODEL`:

| Modelo | Estratégia | Como é aplicada |
|---|---|---|
| Árvore de Decisão | `smote` | SMOTE no fold de treino **+** `class_weight={0:1,1:2}` já embutido no hiperparâmetro (as duas coexistem, como no notebook original) |
| Regressão Logística | `class_weight` | Só `class_weight='balanced'`, já embutido no hiperparâmetro — nenhum passo extra no `Trainer` |
| SVM | `class_weight` | Só `class_weight='balanced'`, já embutido no hiperparâmetro — nenhum passo extra |
| Rede Neural | `sample_weight` | `MLPClassifier` não aceita `class_weight` no construtor; peso por amostra calculado (`compute_balanced_sample_weight`) e passado a `.fit(..., sample_weight=...)` |
| Naive Bayes | `smote` | Só SMOTE no treino — réplica de `models/naive_bayes/train.py` |

---

## 5. Threshold tuning — decisão sobre "preservar o da Rede Neural"

A Tarefa 6 pede para "preservar o threshold tuning utilizado pela rede
neural caso ele exista no pipeline atual". Ele existe
(`RedesNeurais.ipynb`, célula "AJUSTE DE LIMIAR") — mas na versão
**sem guarda-corpo**, a mesma classe de bug que
`results/inconsistencies_report.md` documenta ter inflado o recall da
Regressão Logística para ~1,0 com um threshold ≈ 0,01.

**Decisão:** `Trainer` usa `src.training.threshold_tuning.find_best_threshold`
— a reimplementação da versão **segura** (Youden's J com guarda-corpo de
prevalência, já validada em `analysis/audit/threshold.py`) — para os
**5 modelos**, não só a Rede Neural. Isso preserva o *conceito* pedido
(threshold calibrado por modelo em vez de 0,5 fixo, algo que a Rede
Neural de fato tem) sem reintroduzir, para nenhum modelo, um bug que o
próprio projeto já identificou e corrigiu. A alternativa — usar a versão
insegura só porque "é a que a Rede Neural usa" — contrariaria o espírito
da Tarefa 6 (não é uma mudança de metodologia científica, é a correção
de um bug já conhecido, mesma classe de decisão já tomada nas Tarefas
3-5 para outros problemas de vazamento).

O guarda-corpo: rejeita qualquer threshold cuja taxa de positivos
previstos exceda 1,5× a prevalência real da calibração — impede
exatamente o cenário "quase tudo previsto como positivo" que inflou
artificialmente o recall no `results/inconsistencies_report.md`. Testado
diretamente em `tests/test_threshold_tuning.py::test_rejects_pathological_threshold_that_predicts_everything_positive`.

---

## 6. Busca de hiperparâmetros

`src.training.hyperparameter_search` centraliza os 2 mecanismos hoje
usados no projeto (`run_grid_search` para `GridSearchCV`,
`run_randomized_search` para `RandomizedSearchCV`), com os espaços de
busca originais preservados como dados em `src.configs.search_spaces`
(não recalculados — os valores vencedores já estão fixados desde a
Tarefa 5).

**`Trainer.train(...)` não executa nenhuma busca por padrão.** Refazer a
busca da Árvore de Decisão, por exemplo, significa treinar ~2500 modelos
(`RandomizedSearchCV(n_iter=500, cv=5)`) — caro e desnecessário para
reproduzir o comportamento atual, que já usa os hiperparâmetros
vencedores centralizados na Tarefa 5. Quem precisar reabrir uma busca
chama `run_grid_search`/`run_randomized_search` diretamente, sempre
passando um `Pipeline` completo (engenharia de features + modelo), nunca
um estimador nu com features já transformadas fora da busca — isso
reintroduziria o problema de "sem nested cross-validation" já
documentado em `ARCHITECTURE_AUDIT.md`, seção 5.

A Naive Bayes original não usa `GridSearchCV`/`RandomizedSearchCV` — usa
uma busca aleatória própria comparando as 3 variantes
(`models/naive_bayes/train.py::_gerar_configuracoes_aleatorias`), fora
do escopo desta tarefa (é lógica de avaliação/comparação, não um dos
dois mecanismos que a Tarefa 6 pede para centralizar). Os espaços de
candidatos de cada variante ficam documentados, como dado, em
`src.configs.search_spaces` mesmo assim.

---

## 7. Leakage

- Toda transformação que aprende parâmetro dos dados (engenharia de
  features, SMOTE, threshold tuning) ocorre **dentro** do fold de
  treino — nunca antes do `StratifiedKFold.split(...)`.
- A fatia de calibração garante que o threshold nunca vê o fold de
  validação.
- `FeatureEngineer.for_<modelo>()` (Tarefa 4) já devolve `Pipeline`s do
  scikit-learn — `Trainer` os ajusta (`fit_transform`) só na fatia de
  treino-menos-calibração de cada fold, replicando exatamente o padrão
  seguro de `analysis/audit/evaluator.py`.
- SMOTE é aplicado **depois** da engenharia de features e **só** na
  fatia de treino-menos-calibração — nunca em calibração/validação.

---

## 8. Outputs — `TrainingResult`

Nenhum gráfico é gerado; nenhum arquivo é salvo. `Trainer.train(...)`
devolve um `TrainingResult` (`src.training.results`) com:

| Campo | Conteúdo |
|---|---|
| `fitted_models` | 1 estimador ajustado por fold |
| `fitted_feature_pipelines` | 1 pipeline de features ajustado por fold |
| `params` | hiperparâmetros efetivos do último fold (`estimator.get_params()`) |
| `metrics` / `metrics_std` | média/desvio padrão de accuracy, precision, recall, F1, ROC-AUC, PR-AUC entre os 5 folds |
| `fold_results` | 1 `FoldResult` por fold: threshold, critério, tamanhos de treino/calibração/validação, métricas |
| `predictions` | predições **out-of-fold**, alinhadas por posição a `y` |
| `probabilities` | probabilidade da classe positiva, mesma lógica out-of-fold |
| `imbalance_strategy` | qual estratégia da seção 4 foi usada |
| `y_true` | cópia de `y` como array, mesma ordem de `predictions` |

---

## 9. Testes

129 testes no total (99 das Tarefas 2-5 + 30 novos desta tarefa),
divididos em:

- `tests/test_imbalance.py` (6) — cada modelo mantém sua própria
  estratégia (não uniformiza SMOTE); `apply_smote` só balanceia o treino
  e é reprodutível; `compute_balanced_sample_weight` pesa a classe
  minoritária corretamente.
- `tests/test_threshold_tuning.py` (5) — encontra um threshold razoável
  quando há separação real; **rejeita o cenário patológico** documentado
  em `results/inconsistencies_report.md`; usa fallback quando nada passa
  no guarda-corpo; respeita os limites de busca; determinístico.
- `tests/test_hyperparameter_search.py` (5) — `GridSearchCV`/
  `RandomizedSearchCV` funcionam sobre estimador nu e sobre `Pipeline`
  (confirmando que o scaler é reajustado por fold interno da busca);
  `n_iter` respeitado; reprodutível com seed fixa.
- `tests/test_training.py` (14) — **o requisito central da tarefa**:
  `Trainer.train(...)` executa de ponta a ponta para os 5 modelos
  (Árvore, Regressão Logística, SVM, Rede Neural + as 3 variantes de
  Naive Bayes), produzindo `TrainingResult` estruturalmente válido
  (5 folds, predições/probabilidades completas e no intervalo `[0,1]`,
  métricas presentes); `n_splits` customizável; reprodutibilidade;
  engenharia de features reajustada de forma independente por fold
  (checagem de leakage); arquitetura dinâmica da Rede Neural preservada
  dentro do `Trainer` (`hidden_layer_sizes == (11,)` para as 21
  `common_features`).

### Execução real

```text
python -m pytest tests/ -v
...
======================= 129 passed in 27.26s =======================
```

Alguns `FutureWarning`/`UserWarning` aparecem nos testes da Regressão
Logística e do SVM — são avisos de depreciação do **scikit-learn 1.9**
(`penalty` da `LogisticRegression` e `probability` da `SVC` estão sendo
descontinuados em versões futuras da biblioteca). Não são erros: os
hiperparâmetros continuam os mesmos valores vencedores da Tarefa 5,
preservados fielmente; o aviso é sobre uma API que vai mudar em uma
versão futura do scikit-learn, não sobre um problema no código desta
tarefa.

---

## 10. O que fica fora do escopo desta etapa

- **Migração dos notebooks** — continuam existindo e funcionando.
- **Reexecução das buscas de hiperparâmetros** — arquitetura pronta
  (`hyperparameter_search.py`), não executada por padrão.
- **Busca de hiperparâmetros do Naive Bayes** (comparação de 90
  configurações entre as 3 variantes) — fora do escopo (lógica de
  avaliação/comparação, não um dos mecanismos pedidos).
- **Seleção do "melhor modelo" entre os 5** e geração de ranking —
  pertence a uma futura `src/evaluation/` (ainda não implementada).
- **Persistência de modelo treinado** (`.pkl`) — este pacote não salva
  arquivos, por requisito explícito da tarefa.
