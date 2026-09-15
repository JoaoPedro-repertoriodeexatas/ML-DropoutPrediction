"""Testes de `src.evaluation.metrics` (Tarefa 7)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.evaluation.metrics import (
    METRIC_NAMES,
    aggregate_metric_sets,
    compute_confusion_matrix,
    compute_metrics,
)


def test_compute_confusion_matrix_counts_all_four_quadrants() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])  # 1 FP, 1 FN

    cm = compute_confusion_matrix(y_true, y_pred)

    assert cm.true_negative == 1
    assert cm.false_positive == 1
    assert cm.false_negative == 1
    assert cm.true_positive == 2
    assert cm.as_array().tolist() == [[1, 1], [1, 2]]


def test_compute_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.unavailable == {}


def test_compute_metrics_all_six_metric_names_covered() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.2, 0.9, 0.6, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_proba)
    as_dict = metrics.as_dict()

    assert set(as_dict) == set(METRIC_NAMES)


def test_compute_metrics_without_y_proba_marks_roc_auc_unavailable_explicitly() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = compute_metrics(y_true, y_pred, y_proba=None)

    assert metrics.roc_auc is None
    assert metrics.pr_auc is None
    assert "roc_auc" in metrics.unavailable
    assert "pr_auc" in metrics.unavailable
    # accuracy/precision/recall/f1 continuam calculáveis mesmo sem proba.
    assert metrics.accuracy == 0.75


def test_compute_metrics_single_class_marks_roc_auc_unavailable_explicitly() -> None:
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    y_proba = np.array([0.9, 0.8, 0.3, 0.7])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics.roc_auc is None
    assert "apenas uma classe" in metrics.unavailable["roc_auc"]


def test_compute_metrics_zero_division_does_not_raise() -> None:
    """Nenhuma predição positiva -> precision indefinida deveria ser 0,
    não levantar exceção (zero_division=0)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_aggregate_metric_sets_computes_mean_and_std() -> None:
    common = dict(y_true=np.array([0, 0, 1, 1]), y_pred=np.array([0, 0, 1, 1]))
    metric_sets = [
        compute_metrics(common["y_true"], common["y_pred"], np.array([0.1, 0.2, 0.8, 0.9])),
        compute_metrics(common["y_true"], common["y_pred"], np.array([0.2, 0.3, 0.7, 0.8])),
    ]

    aggregated = aggregate_metric_sets(metric_sets)

    assert aggregated.n == 2
    assert aggregated.mean["accuracy"] == 1.0
    assert aggregated.std["accuracy"] == 0.0
    assert aggregated.n_available["roc_auc"] == 2


def test_aggregate_metric_sets_handles_partial_availability_of_roc_auc() -> None:
    with_proba = compute_metrics(np.array([0, 1]), np.array([0, 1]), np.array([0.2, 0.8]))
    without_proba = compute_metrics(np.array([0, 1]), np.array([0, 1]), None)

    aggregated = aggregate_metric_sets([with_proba, without_proba])

    assert aggregated.n_available["roc_auc"] == 1  # só 1 dos 2 tinha ROC-AUC
    assert aggregated.mean["roc_auc"] == with_proba.roc_auc  # média de só quem tinha
    assert aggregated.mean["accuracy"] is not None  # accuracy sempre disponível nos 2


def test_aggregate_metric_sets_all_unavailable_yields_none_not_nan() -> None:
    metric_sets = [
        compute_metrics(np.array([1, 1]), np.array([1, 1]), None),
        compute_metrics(np.array([1, 1]), np.array([1, 0]), None),
    ]

    aggregated = aggregate_metric_sets(metric_sets)

    assert aggregated.mean["roc_auc"] is None
    assert aggregated.std["roc_auc"] is None
    assert aggregated.n_available["roc_auc"] == 0


def test_aggregate_metric_sets_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        aggregate_metric_sets([])


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
