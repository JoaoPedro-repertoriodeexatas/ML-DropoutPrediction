"""Testes de `src.training.threshold_tuning` (Tarefa 6)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.training.threshold_tuning import find_best_threshold


def test_finds_threshold_close_to_perfect_separator() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.array([0.1] * 50 + [0.9] * 50)

    selection = find_best_threshold(y_true, y_proba)

    assert 0.1 < selection.threshold < 0.9
    assert selection.criterion == "youden_j_constrained"
    assert not selection.fallback_used


def test_rejects_pathological_threshold_that_predicts_everything_positive() -> None:
    """Réplica do caso documentado em `results/inconsistencies_report.md`:
    threshold ~0.01 que preveria quase tudo como positivo (recall~1.0,
    mas sem sustentação real)."""
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 500 + [1] * 500)
    # Probabilidades fracamente informativas — quase todas > 0.01.
    y_proba = np.clip(rng.normal(loc=0.5, scale=0.1, size=1000), 0.02, 0.98)

    selection = find_best_threshold(y_true, y_proba)

    # O guarda-corpo de prevalência nunca deveria aceitar um threshold
    # que classifique quase tudo como positivo.
    predicted_positive_rate = float((y_proba >= selection.threshold).mean())
    assert predicted_positive_rate <= 1.5 * 0.5 + 1e-6  # prevalence_cap padrão = 1.5


def test_uses_fallback_when_no_candidate_passes_prevalence_guard() -> None:
    # Todas as probabilidades são altas -> qualquer threshold na faixa de
    # busca (0.2-0.8) classificaria quase tudo como positivo.
    y_true = np.array([0] * 10 + [1] * 90)
    y_proba = np.full(100, 0.95)

    selection = find_best_threshold(y_true, y_proba, prevalence_cap=1.01)

    assert selection.fallback_used
    assert selection.criterion == "fallback"
    assert selection.threshold == 0.5


def test_threshold_search_stays_within_configured_bounds() -> None:
    y_true = np.array([0, 1] * 50)
    y_proba = np.tile([0.3, 0.7], 50)

    selection = find_best_threshold(y_true, y_proba, search_min=0.2, search_max=0.8)

    assert 0.2 <= selection.threshold <= 0.8


def test_is_deterministic_across_calls() -> None:
    rng = np.random.default_rng(7)
    y_true = np.array([0] * 60 + [1] * 40)
    y_proba = rng.uniform(size=100)

    first = find_best_threshold(y_true, y_proba)
    second = find_best_threshold(y_true, y_proba)

    assert first == second


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
