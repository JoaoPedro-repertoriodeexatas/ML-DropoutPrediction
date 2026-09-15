"""Visualizações — implementado na Tarefa 8.

Toda função aqui recebe dado **já calculado** (por
`src.training`/`src.evaluation`) e devolve uma `matplotlib.figure.Figure`
— nenhuma lê o dataset, treina, ou calcula métrica; nenhuma chama
`plt.show()` (quem chama decide se mostra/salva).

- ``training_plots.py`` — métricas por fold, boxplot entre folds,
  threshold calibrado por fold.
- ``evaluation_plots.py`` — matriz de confusão, curva ROC, curva
  Precision-Recall.
- ``comparison_plots.py`` — barras comparativas entre modelos, ranking,
  heatmap de métricas.
- ``tree_plots.py`` — importância de features (específico da Árvore de
  Decisão).

Os notebooks em `experiments/` (Tarefa 8) só chamam estas funções — não
implementam nenhuma lógica de plot própria.
"""
