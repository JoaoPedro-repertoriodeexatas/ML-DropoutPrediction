# Camada centralizada de avaliação (Tarefa 7)

**Branch:** `audit/architecture-review`
**Baseado em:** [`docs/TRAINING.md`](TRAINING.md)

Remove dos notebooks toda lógica de cálculo de métrica, centralizando-a
em `src/evaluation/`. Nenhum gráfico é gerado; nenhum notebook foi
alterado ou removido.

---

## 1. Interface

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator()
result = evaluator.evaluate(model=model, X=X_test, y=y_test)

result.metrics.accuracy
result.metrics.precision
result.metrics.recall
result.metrics.f1
result.metrics.roc_auc            # None se indisponível (ver seção 4)
result.metrics.confusion_matrix   # ConfusionMatrixSummary (tp/fp/fn/tn)
```

`Evaluator.evaluate` recebe um modelo **já ajustado** e qualquer par
`(X, y)` — não treina, não faz cross-validation (isso é
`src.training.trainer.Trainer`, Tarefa 6), não sabe nada sobre
notebooks. Funciona com qualquer estimador scikit-learn-compatível, não
só os 5 deste projeto.

---

## 2. Onde a solução vive

```text
src/evaluation/
├── metrics.py       # fonte única de verdade: compute_metrics, aggregate_metric_sets
├── evaluator.py      # Evaluator.evaluate(model, X, y) -> EvaluationResult
└── comparison.py     # compare_models(...) -> tabela; fold_metrics_table(...)
```

**Decisão de consistência** (seção "Garanta que todos os modelos
utilizem a mesma definição de métricas"): `src.training.cross_validation`
(Tarefa 6) foi refatorado nesta tarefa para **delegar** em
`src.evaluation.metrics.compute_metrics` em vez de ter sua própria cópia
do cálculo. Antes da Tarefa 7, `Trainer` calculava métricas com uma
chamada local a `accuracy_score`/`precision_score`/etc.; agora chama a
mesma função que `Evaluator` usa — uma métrica só pode ter uma
definição em todo o projeto, nunca duas cópias que podem divergir com o
tempo (o mesmo princípio já aplicado a vazamento de dados na Tarefa 3 e
a threshold tuning na Tarefa 6).

---

## 3. Métricas centralizadas

`src.evaluation.metrics.compute_metrics(y_true, y_pred, y_proba)` calcula,
de uma vez, o conjunto completo pedido pela Tarefa 7:

| Métrica | Sempre calculável? | Fonte |
|---|---|---|
| Accuracy | Sim | `sklearn.metrics.accuracy_score` |
| Precision | Sim (`zero_division=0`) | `sklearn.metrics.precision_score` |
| Recall | Sim (`zero_division=0`) | `sklearn.metrics.recall_score` |
| F1-score | Sim (`zero_division=0`) | `sklearn.metrics.f1_score` |
| ROC-AUC | Só com `y_proba` e as 2 classes presentes | `sklearn.metrics.roc_auc_score` |
| Matriz de confusão | Sim | `sklearn.metrics.confusion_matrix`, envolvida em `ConfusionMatrixSummary` (tp/fp/fn/tn nomeados) |

PR-AUC (`average_precision_score`) também é calculada, além do pedido
mínimo da tarefa — é a mesma métrica que `analysis/audit/evaluator.py`
já usava ao lado do ROC-AUC para julgar se a AUC "corrobora" as métricas
com threshold (ver `ARCHITECTURE_AUDIT.md`, seção verificações de
sanidade); manter as duas juntas evita perder essa capacidade de
checagem cruzada.

`aggregate_metric_sets(...)` calcula **média e desvio padrão** de uma
lista de `MetricSet` (tipicamente um por fold) — a peça de "média;
desvio padrão; métricas por fold" pedida explicitamente pela tarefa.

---

## 4. Consistência e tratamento explícito de métrica indisponível

Requisito literal da tarefa: "quando uma métrica não puder ser
calculada para determinado modelo, trate explicitamente o caso".
`MetricSet.roc_auc`/`pr_auc` ficam **`None`** (nunca `NaN` silencioso)
em dois cenários, cada um com o motivo registrado em
`MetricSet.unavailable`:

1. **O modelo não expõe probabilidade utilizável.**
   `Evaluator.extract_positive_class_probability` tenta
   `predict_proba` primeiro (o caminho dos 5 modelos deste projeto — SVM
   incluso, porque `SVMHyperparameters.probability=True`, Tarefa 5);
   cai para `decision_function` reescalado para `[0,1]` só se
   `predict_proba` não existir; devolve `None` se nenhum dos dois
   existir ou falhar. `compute_metrics` recebe `y_proba=None` nesse
   caso e marca `roc_auc`/`pr_auc` como indisponíveis, com o motivo.
2. **`y_true` tem uma única classe.** ROC-AUC não é matematicamente
   definido nesse caso — `compute_metrics` detecta e marca como
   indisponível em vez de deixar `roc_auc_score` levantar exceção.

`aggregate_metric_sets` propaga essa mesma disciplina: se uma métrica
está disponível em só parte dos `MetricSet` agregados, a média/desvio
usam só os disponíveis (`AggregatedMetrics.n_available` registra
quantos); se estiver indisponível em **todos**, o resultado agregado é
`None`, não `NaN`. `compare_models` (seção 5) propaga `None` até a
tabela final — nunca omite a coluna nem preenche com zero.

Accuracy, precision, recall e F1 são **sempre** calculáveis a partir de
`y_true`/`y_pred` — não dependem de probabilidade, então nunca ficam
indisponíveis (mesmo definição usada pelos 5 modelos, sem exceção).

---

## 5. Comparação tabular entre os 5 modelos

```python
from src.evaluation.comparison import comparison_row_from_training_result, compare_models

rows = [comparison_row_from_training_result(training_results[name]) for name in MODEL_NAMES]
table = compare_models(rows)
```

Colunas exatamente como pedido: `Model | Accuracy | Precision | Recall |
F1 | ROC-AUC | Std`. `Std` é o desvio padrão do F1 entre folds por
padrão (`std_metric="f1"`, configurável) — F1 é a métrica usada como
`scoring` nas buscas de hiperparâmetros de 3 dos 5 modelos
(`src.configs.search_spaces`), por isso é a referência natural de
variabilidade para comparação.

`compare_models_detailed(...)` é a variante mais completa, com uma
coluna `<Métrica>_Std` por métrica (inclui PR-AUC) para quem precisar de
mais detalhe do que as 7 colunas pedidas.

`fold_metrics_table(training_result)` monta a tabela "métricas por
fold" — uma linha por fold, com accuracy/precision/recall/F1/ROC-AUC/
PR-AUC/threshold — direto de `training_result.fold_results` (Tarefa 6).

`comparison_row_from_training_result`/`comparison_row_from_aggregated_metrics`
aceitam tanto a saída do `Trainer` (Tarefa 6) quanto um `AggregatedMetrics`
calculado manualmente com `Evaluator` — `compare_models` não depende de
nenhuma das duas origens especificamente, só de uma lista de
`ModelComparisonRow`.

---

## 6. Testes

24 testes novos (129 → 153 no total, sem regressão):

- `tests/test_evaluation_metrics.py` (10) — matriz de confusão,
  `compute_metrics` com predição perfeita e com erros, as 6 métricas
  cobertas, indisponibilidade explícita (sem `y_proba`, com uma única
  classe), `zero_division` não levanta exceção,
  `aggregate_metric_sets` com disponibilidade total/parcial/nula,
  lista vazia rejeitada.
- `tests/test_evaluator.py` (6) — `Evaluator.evaluate` devolve
  `EvaluationResult` estruturado; funciona **sem** nenhum `Trainer`/CV
  envolvido (modelo ajustado manualmente); indisponibilidade explícita
  de ROC-AUC para um modelo sem `predict_proba`; fallback para
  `decision_function`; `y_proba` pré-computado é aceito sem recálculo.
- `tests/test_comparison.py` (8) — linha de comparação a partir de
  `TrainingResult` reaproveita exatamente os números do `Trainer`
  (prova da consistência da seção 2); tabela com as colunas exatas
  pedidas; `Std` configurável; métrica indisponível aparece como `None`/
  `NaN` na tabela, não omitida; variante detalhada; tabela por fold.

### Execução real

```text
python -m pytest tests/ -v
...
======================= 153 passed in 27.94s =======================
```

---

## 7. O que fica fora do escopo desta etapa

- **Geração de gráficos** (curvas ROC/PR, matrizes de confusão
  visuais) — por requisito explícito, fica para `src/visualization/`
  (ainda não implementado).
- **Ranking/seleção do "melhor modelo"** — `compare_models` monta a
  tabela; decidir um critério de ranking (como o score ponderado de
  `analysis/audit/run_audit.py` ou os pesos iguais de
  `analysis/final_validation/ranking.py`) é uma decisão de produto,
  não uma responsabilidade desta camada.
- **Sanity checks metodológicos** (as 6 verificações de
  `analysis/audit/sanity_checks.py`, ex.: detectar threshold no extremo
  da busca) — complementares às métricas, mas com objetivo diferente
  (detectar resultado implausível, não medir desempenho); candidatas a
  um módulo próprio futuro.
