"""Camada centralizada de treinamento (Tarefa 6).

- ``trainer.py`` — `Trainer.train(model_name, X, y, ...)`, o ponto único
  de treino: cross-validation + desbalanceamento + threshold tuning.
- ``cross_validation.py`` — `StratifiedKFold`, split de calibração,
  cálculo e agregação de métricas por fold.
- ``imbalance.py`` — estratégia de desbalanceamento **original** de cada
  modelo (não uniformiza tudo em SMOTE).
- ``threshold_tuning.py`` — a única implementação de threshold tuning
  com guarda-corpo (Youden's J), usada por todos os modelos.
- ``hyperparameter_search.py`` — wrappers de `GridSearchCV`/
  `RandomizedSearchCV`, não chamados por padrão (`Trainer` usa os
  hiperparâmetros já vencedores da Tarefa 5).

Ver `docs/TRAINING.md`. Nenhum notebook foi migrado; nenhum gráfico é
gerado; nenhum arquivo é salvo por este pacote.
"""
