"""Espaços de busca de hiperparâmetros usados pelos notebooks originais.

Dados de referência (não lógica) para reproduzir, se necessário, as
buscas que já foram executadas (cujo resultado vencedor está
centralizado em `src.configs.model_hyperparameters`, Tarefa 5) via
`src.training.hyperparameter_search` (Tarefa 6). Nomes de hiperparâmetro
sem prefixo — quem chama a busca decide se precisa prefixar para uso
dentro de um `Pipeline` (ver
`src.training.hyperparameter_search.prefix_param_names`).

Não são executados automaticamente por `Trainer.train(...)`. Refazer
essas buscas custa caro (a da Árvore, por exemplo, treina ~2500 modelos)
e não é necessário para reproduzir o comportamento atual — os
hiperparâmetros vencedores já estão fixados. Ficam aqui para quem
precisar reabrir a busca (Rede Neural: sem espaço de busca — o notebook
original compara 5 arquiteturas fixas, não faz `GridSearchCV`/
`RandomizedSearchCV`).
"""

from __future__ import annotations

from typing import Any

from scipy.stats import loguniform, randint, uniform

# --------------------------------------------------------------------------
# Árvore de Decisão — RandomizedSearchCV(n_iter=500, cv=5, scoring='f1')
# Fonte: ArvoreDecisao.ipynb, célula "HIPERPARÂMETROS AMPLIADOS"
# --------------------------------------------------------------------------

DECISION_TREE_PARAM_DISTRIBUTIONS: dict[str, Any] = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, *range(10, 41, 5)],
    "min_samples_split": randint(10, 50),
    "min_samples_leaf": randint(5, 20),
    "max_features": ["sqrt", "log2", None, 0.5, 0.75],
    "class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
    "min_impurity_decrease": uniform(0.0, 0.005),
}
DECISION_TREE_SEARCH_N_ITER = 500
DECISION_TREE_SEARCH_CV = 5
DECISION_TREE_SEARCH_SCORING = "f1"

# --------------------------------------------------------------------------
# Regressão Logística — GridSearchCV(cv=3, scoring='f1')
# Fonte: RegressaoLogistica.ipynb, célula "OTIMIZAÇÃO DE HIPERPARÂMETROS"
# --------------------------------------------------------------------------

LOGISTIC_REGRESSION_PARAM_GRID: dict[str, Any] = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "class_weight": ["balanced"],
    "max_iter": [1000],
}
LOGISTIC_REGRESSION_SEARCH_CV = 3
LOGISTIC_REGRESSION_SEARCH_SCORING = "f1"

# --------------------------------------------------------------------------
# SVM — RandomizedSearchCV(n_iter=12, cv=2, scoring='f1')
# Fonte: SVM.ipynb, célula 8 (busca efetivamente usada para o modelo final)
# --------------------------------------------------------------------------

SVM_PARAM_DISTRIBUTIONS: dict[str, Any] = {
    "C": loguniform(1e-1, 1e3),
    "gamma": loguniform(1e-4, 1e1),
    "shrinking": [True, False],
    "tol": loguniform(1e-5, 1e-2),
}
SVM_SEARCH_N_ITER = 12
SVM_SEARCH_CV = 2
SVM_SEARCH_SCORING = "f1"

# --------------------------------------------------------------------------
# Naive Bayes — busca aleatória própria (não GridSearchCV/RandomizedSearchCV),
# 30 configurações por variante, escolha pelo maior recall.
# Fonte: models/naive_bayes/train.py::_gerar_configuracoes_aleatorias
# --------------------------------------------------------------------------

NAIVE_BAYES_GAUSSIAN_VAR_SMOOTHING_SPACE = {"low": -12, "high": -1, "n_candidates": 100}
NAIVE_BAYES_BERNOULLI_ALPHA_SPACE = {"low": 0.01, "high": 5.0, "n_candidates": 100}
NAIVE_BAYES_BERNOULLI_BINARIZE_SPACE = {"low": 0.0, "high": 1.0, "n_candidates": 20}
NAIVE_BAYES_COMPLEMENT_ALPHA_SPACE = {"low": 0.01, "high": 5.0, "n_candidates": 100}
NAIVE_BAYES_EXPERIMENTS_PER_VARIANT = 30
NAIVE_BAYES_SELECTION_METRIC = "recall"


__all__ = [
    "DECISION_TREE_PARAM_DISTRIBUTIONS",
    "DECISION_TREE_SEARCH_N_ITER",
    "DECISION_TREE_SEARCH_CV",
    "DECISION_TREE_SEARCH_SCORING",
    "LOGISTIC_REGRESSION_PARAM_GRID",
    "LOGISTIC_REGRESSION_SEARCH_CV",
    "LOGISTIC_REGRESSION_SEARCH_SCORING",
    "SVM_PARAM_DISTRIBUTIONS",
    "SVM_SEARCH_N_ITER",
    "SVM_SEARCH_CV",
    "SVM_SEARCH_SCORING",
    "NAIVE_BAYES_GAUSSIAN_VAR_SMOOTHING_SPACE",
    "NAIVE_BAYES_BERNOULLI_ALPHA_SPACE",
    "NAIVE_BAYES_BERNOULLI_BINARIZE_SPACE",
    "NAIVE_BAYES_COMPLEMENT_ALPHA_SPACE",
    "NAIVE_BAYES_EXPERIMENTS_PER_VARIANT",
    "NAIVE_BAYES_SELECTION_METRIC",
]
