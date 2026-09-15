"""Camada centralizada de avaliação (Tarefa 7).

- ``metrics.py`` — fonte única de verdade para o cálculo de métricas
  (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, matriz de
  confusão) e sua agregação (média/desvio padrão). Usada também por
  `src.training.cross_validation` (Tarefa 6), para que treino e
  avaliação nunca divirjam na definição de uma métrica.
- ``evaluator.py`` — `Evaluator.evaluate(model, X, y)`: avalia qualquer
  modelo já ajustado, independente de notebook.
- ``comparison.py`` — `compare_models(...)`: tabela comparativa entre
  modelos (`Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Std`)
  e `fold_metrics_table(...)` para métricas por fold.

Ver `docs/EVALUATION.md`. Nenhum gráfico é gerado por este pacote.
"""
