"""Busca de hiperparâmetros centralizada (Tarefa 6).

Wrappers finos sobre `GridSearchCV`/`RandomizedSearchCV` do
scikit-learn — os dois mecanismos de busca hoje usados no projeto
(Árvore de Decisão e SVM usam `RandomizedSearchCV`; Regressão Logística
usa `GridSearchCV`). Os espaços de busca ficam em
`src.configs.search_spaces` (dados, não lógica).

`Trainer.train(...)` (`src.training.trainer`) **não chama estas funções
por padrão** — usa os hiperparâmetros já vencedores, centralizados em
`src.configs.model_hyperparameters` (Tarefa 5). Refazer a busca é uma
ação explícita (`Trainer.train(..., search=True, ...)` ou chamando estas
funções diretamente), porque é uma operação cara (a busca da Árvore, por
exemplo, treina ~2500 modelos) e não é necessária para reproduzir o
comportamento atual.

**Leakage:** ``estimator`` deve normalmente ser um `Pipeline` completo
(engenharia de features + modelo), não um estimador nu — assim, cada
fold interno do `GridSearchCV`/`RandomizedSearchCV` reajusta a
engenharia de features só no seu próprio treino. Passar um estimador nu,
com features já transformadas fora da busca, reintroduz o problema de
"sem nested cross-validation" documentado em `ARCHITECTURE_AUDIT.md`,
seção 5.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.configs.settings import RANDOM_STATE


def prefix_param_names(params: Mapping[str, Any], step_name: str) -> dict[str, Any]:
    """Prefixa nomes de hiperparâmetro para uso dentro de um `Pipeline`.

    ``{"C": [1, 10]}`` com ``step_name="model"`` vira
    ``{"model__C": [1, 10]}`` — a convenção do scikit-learn para
    endereçar o hiperparâmetro de um passo específico do pipeline.
    """
    return {f"{step_name}__{key}": value for key, value in params.items()}


def run_grid_search(
    estimator: BaseEstimator,
    param_grid: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: int = 3,
    scoring: str = "f1",
    n_jobs: int = -1,
    verbose: int = 0,
) -> GridSearchCV:
    """Busca exaustiva de hiperparâmetros (mecanismo usado pela Regressão
    Logística — ver `src.configs.search_spaces.LOGISTIC_REGRESSION_PARAM_GRID`).

    Ajusta (`fit`) e devolve o `GridSearchCV`; `search.best_estimator_` e
    `search.best_params_` ficam disponíveis depois da chamada.
    """
    search = GridSearchCV(
        estimator,
        param_grid=dict(param_grid),
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    search.fit(X, y)
    return search


def run_randomized_search(
    estimator: BaseEstimator,
    param_distributions: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 50,
    cv: int = 5,
    scoring: str = "f1",
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
    verbose: int = 0,
) -> RandomizedSearchCV:
    """Busca aleatória de hiperparâmetros (mecanismo usado pela Árvore de
    Decisão e pelo SVM — ver `src.configs.search_spaces`).

    Ajusta (`fit`) e devolve o `RandomizedSearchCV`; `search.best_estimator_`
    e `search.best_params_` ficam disponíveis depois da chamada.
    """
    search = RandomizedSearchCV(
        estimator,
        param_distributions=dict(param_distributions),
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )
    search.fit(X, y)
    return search


__all__ = ["prefix_param_names", "run_grid_search", "run_randomized_search"]
