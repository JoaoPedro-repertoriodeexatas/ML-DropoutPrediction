"""Testes de `src.training.hyperparameter_search` (Tarefa 6).

Usa espaços de busca minúsculos e dados sintéticos pequenos — o
objetivo é provar que os wrappers funcionam e respeitam o contrato de
`Pipeline` (sem vazamento), não reproduzir as buscas reais de 500/12
iterações dos notebooks (isso é caro e não é o que este teste precisa
garantir).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.hyperparameter_search import (
    prefix_param_names,
    run_grid_search,
    run_randomized_search,
)


def _sample_data(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    return X, y


def test_prefix_param_names_adds_step_prefix() -> None:
    prefixed = prefix_param_names({"C": [1, 10], "penalty": ["l1", "l2"]}, "model")
    assert prefixed == {"model__C": [1, 10], "model__penalty": ["l1", "l2"]}


def test_run_grid_search_on_bare_estimator() -> None:
    X, y = _sample_data()
    search = run_grid_search(
        LogisticRegression(max_iter=200),
        {"C": [0.1, 1.0]},
        X,
        y,
        cv=2,
        scoring="f1",
    )
    assert search.best_params_["C"] in (0.1, 1.0)
    assert hasattr(search, "best_estimator_")


def test_run_grid_search_on_pipeline_refits_scaler_per_fold() -> None:
    """Passar um Pipeline (não um estimador nu) garante que o scaler é
    reajustado em cada fold interno da busca — sem vazamento."""
    X, y = _sample_data(seed=1)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))])
    param_grid = prefix_param_names({"C": [0.1, 1.0]}, "model")

    search = run_grid_search(pipeline, param_grid, X, y, cv=2, scoring="f1")

    assert "model__C" in search.best_params_
    assert isinstance(search.best_estimator_, Pipeline)


def test_run_randomized_search_respects_n_iter() -> None:
    X, y = _sample_data(seed=2)
    search = run_randomized_search(
        LogisticRegression(max_iter=200),
        {"C": [0.01, 0.1, 1.0, 10.0]},
        X,
        y,
        n_iter=2,
        cv=2,
        scoring="f1",
        random_state=42,
    )
    assert len(search.cv_results_["params"]) == 2


def test_run_randomized_search_is_reproducible_with_fixed_seed() -> None:
    X, y = _sample_data(seed=3)
    search_a = run_randomized_search(
        LogisticRegression(max_iter=200), {"C": [0.01, 0.1, 1.0, 10.0]}, X, y,
        n_iter=3, cv=2, random_state=42,
    )
    search_b = run_randomized_search(
        LogisticRegression(max_iter=200), {"C": [0.01, 0.1, 1.0, 10.0]}, X, y,
        n_iter=3, cv=2, random_state=42,
    )
    assert search_a.best_params_ == search_b.best_params_


if __name__ == "__main__":
    test_functions = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"OK   - {test_fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL - {test_fn.__name__}: {exc}")
    total = len(test_functions)
    print(f"\n{total - failures}/{total} testes passaram.")
    sys.exit(1 if failures else 0)
