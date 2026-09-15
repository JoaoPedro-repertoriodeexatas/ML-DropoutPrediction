"""Testes de `src.training.imbalance` (Tarefa 6)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.training.imbalance import (
    IMBALANCE_STRATEGY_BY_MODEL,
    apply_smote,
    compute_balanced_sample_weight,
    get_imbalance_strategy,
)


def test_each_model_has_its_own_original_strategy_not_all_smote() -> None:
    """Requisito explícito da Tarefa 6: não uniformizar tudo em SMOTE."""
    assert get_imbalance_strategy("arvore_de_decisao") == "smote"
    assert get_imbalance_strategy("naive_bayes") == "smote"
    assert get_imbalance_strategy("regressao_logistica") == "class_weight"
    assert get_imbalance_strategy("svm") == "class_weight"
    assert get_imbalance_strategy("rede_neural") == "sample_weight"

    strategies_used = set(IMBALANCE_STRATEGY_BY_MODEL.values())
    assert strategies_used == {"smote", "class_weight", "sample_weight"}


def test_get_imbalance_strategy_unknown_model_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_imbalance_strategy("modelo_que_nao_existe")


def test_apply_smote_balances_minority_class_on_train_only() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    y = pd.Series([0] * 80 + [1] * 20)

    X_balanced, y_balanced = apply_smote(X, y, random_state=42)

    counts = y_balanced.value_counts()
    assert counts[0] == counts[1]  # SMOTE equaliza as classes
    assert len(X_balanced) == len(y_balanced)
    assert len(X_balanced) > len(X)  # gerou exemplos sintéticos


def test_apply_smote_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = pd.Series([0] * 45 + [1] * 15)

    X_a, y_a = apply_smote(X, y, random_state=42)
    X_b, y_b = apply_smote(X, y, random_state=42)

    pd.testing.assert_frame_equal(X_a.reset_index(drop=True), X_b.reset_index(drop=True))
    pd.testing.assert_series_equal(
        pd.Series(y_a).reset_index(drop=True), pd.Series(y_b).reset_index(drop=True)
    )


def test_compute_balanced_sample_weight_upweights_minority_class() -> None:
    y = pd.Series([0] * 90 + [1] * 10)
    weights = compute_balanced_sample_weight(y)

    minority_weight = weights[y.to_numpy() == 1][0]
    majority_weight = weights[y.to_numpy() == 0][0]
    assert minority_weight > majority_weight
    assert len(weights) == len(y)


def test_compute_balanced_sample_weight_uniform_for_balanced_classes() -> None:
    y = pd.Series([0] * 50 + [1] * 50)
    weights = compute_balanced_sample_weight(y)
    assert np.allclose(weights, 1.0)


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
