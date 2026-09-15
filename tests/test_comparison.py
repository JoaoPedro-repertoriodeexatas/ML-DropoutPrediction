"""Testes de `src.evaluation.comparison` (Tarefa 7)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.comparison import (
    COMPARISON_COLUMNS,
    ModelComparisonRow,
    comparison_row_from_aggregated_metrics,
    comparison_row_from_training_result,
    compare_models,
    compare_models_detailed,
    fold_metrics_table,
)
from src.evaluation.metrics import AggregatedMetrics
from src.training.trainer import Trainer


def _sample_notebook_style_dataset(n: int = 150, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y_values = np.array([0, 1] * (n // 2))
    rng.shuffle(y_values)
    signal = y_values * 80

    df = pd.DataFrame(
        {
            "TP_ORGANIZACAO_ACADEMICA": rng.integers(1, 5, n),
            "TP_REDE": rng.integers(1, 3, n),
            "TP_CATEGORIA_ADMINISTRATIVA": rng.integers(1, 8, n),
            "TP_GRAU_ACADEMICO": rng.integers(1, 4, n),
            "TP_MODALIDADE_ENSINO": rng.integers(1, 3, n),
            "TP_DIMENSAO": rng.integers(1, 6, n),
            "IN_GRATUITO": rng.integers(0, 2, n),
            "CO_CINE_AREA_GERAL": rng.integers(1, 10, n),
            "QT_ING": rng.integers(10, 100, n) + signal,
            "QT_MAT": rng.integers(10, 100, n) + signal // 2,
            "QT_VG_TOTAL": rng.integers(10, 500, n),
            "QT_INSCRITO_TOTAL": rng.integers(10, 1000, n),
            "QT_CONC": rng.integers(0, 200, n),
            "QT_VG_TOTAL_EAD": rng.integers(0, 200, n),
            "QT_VG_TOTAL_NOTURNO": rng.integers(0, 200, n),
            "QT_MAT_FINANC": rng.integers(0, 200, n),
            "QT_ING_FIES": rng.integers(0, 100, n),
            "QT_ING_PROUNIP": rng.integers(0, 100, n),
            "QT_ING_18_24": rng.integers(0, 300, n),
            "QT_ING_FEM": rng.integers(0, 300, n),
        }
    )
    y = pd.Series(y_values, name="alto_risco_evasao")
    return df, y


def test_comparison_row_from_training_result_reuses_trainer_metrics() -> None:
    X, y = _sample_notebook_style_dataset()
    result = Trainer().train("regressao_logistica", X, y)

    row = comparison_row_from_training_result(result)

    assert row.model_name == "regressao_logistica"
    assert row.mean_metrics == result.metrics
    assert row.std_metrics == result.metrics_std


def test_comparison_row_from_aggregated_metrics() -> None:
    aggregated = AggregatedMetrics(
        mean={"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "roc_auc": 0.75, "pr_auc": 0.7},
        std={"accuracy": 0.02, "precision": 0.03, "recall": 0.04, "f1": 0.03, "roc_auc": 0.05, "pr_auc": 0.04},
        n=5,
        n_available={name: 5 for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
    )
    row = comparison_row_from_aggregated_metrics("svm", aggregated)
    assert row.model_name == "svm"
    assert row.mean_metrics["accuracy"] == 0.8


def test_compare_models_has_exact_columns_requested() -> None:
    rows = [
        ModelComparisonRow(
            model_name="arvore_de_decisao",
            mean_metrics={"accuracy": 0.8, "precision": 0.79, "recall": 0.77, "f1": 0.78, "roc_auc": 0.85, "pr_auc": 0.8},
            std_metrics={"accuracy": 0.01, "precision": 0.02, "recall": 0.02, "f1": 0.02, "roc_auc": 0.03, "pr_auc": 0.03},
        ),
        ModelComparisonRow(
            model_name="naive_bayes",
            mean_metrics={"accuracy": 0.55, "precision": 0.6, "recall": 0.1, "f1": 0.17, "roc_auc": 0.58, "pr_auc": 0.5},
            std_metrics={"accuracy": 0.02, "precision": 0.05, "recall": 0.03, "f1": 0.04, "roc_auc": 0.04, "pr_auc": 0.04},
        ),
    ]

    table = compare_models(rows)

    assert list(table.columns) == list(COMPARISON_COLUMNS)
    assert list(table["Model"]) == ["arvore_de_decisao", "naive_bayes"]
    assert table.loc[0, "F1"] == 0.78
    assert table.loc[0, "Std"] == 0.02  # std do F1 (padrão)


def test_compare_models_std_metric_is_configurable() -> None:
    rows = [
        ModelComparisonRow(
            model_name="svm",
            mean_metrics={"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8, "roc_auc": 0.82, "pr_auc": 0.8},
            std_metrics={"accuracy": 0.01, "precision": 0.02, "recall": 0.03, "f1": 0.04, "roc_auc": 0.09, "pr_auc": 0.05},
        )
    ]
    table = compare_models(rows, std_metric="roc_auc")
    assert table.loc[0, "Std"] == 0.09


def test_compare_models_rejects_unknown_std_metric() -> None:
    with pytest.raises(ValueError):
        compare_models([], std_metric="nao_existe")


def test_compare_models_explicitly_surfaces_unavailable_metric_as_none() -> None:
    """Requisito da Tarefa 7: métrica indisponível para um modelo deve
    ficar explícita na tabela, não omitida nem zerada."""
    rows = [
        ModelComparisonRow(
            model_name="modelo_sem_proba",
            mean_metrics={"accuracy": 0.7, "precision": 0.6, "recall": 0.5, "f1": 0.55, "roc_auc": None, "pr_auc": None},
            std_metrics={"accuracy": 0.01, "precision": 0.01, "recall": 0.01, "f1": 0.01, "roc_auc": None, "pr_auc": None},
        )
    ]
    table = compare_models(rows)
    assert table.loc[0, "ROC-AUC"] is None or pd.isna(table.loc[0, "ROC-AUC"])

    # A coluna Std também respeita a indisponibilidade quando configurada
    # para a métrica que faltou (aqui, ROC-AUC).
    table_roc_std = compare_models(rows, std_metric="roc_auc")
    assert table_roc_std.loc[0, "Std"] is None or pd.isna(table_roc_std.loc[0, "Std"])


def test_compare_models_detailed_includes_std_per_metric() -> None:
    rows = [
        ModelComparisonRow(
            model_name="rede_neural",
            mean_metrics={"accuracy": 0.79, "precision": 0.8, "recall": 0.75, "f1": 0.77, "roc_auc": 0.86, "pr_auc": 0.83},
            std_metrics={"accuracy": 0.02, "precision": 0.02, "recall": 0.03, "f1": 0.025, "roc_auc": 0.02, "pr_auc": 0.02},
        )
    ]
    table = compare_models_detailed(rows)

    assert "F1_Std" in table.columns
    assert "ROC-AUC_Std" in table.columns
    assert "PR-AUC" in table.columns  # detalhado inclui PR-AUC, que compare_models não inclui
    assert table.loc[0, "F1_Std"] == 0.025


def test_fold_metrics_table_has_one_row_per_fold() -> None:
    X, y = _sample_notebook_style_dataset()
    result = Trainer().train("regressao_logistica", X, y)

    table = fold_metrics_table(result)

    assert len(table) == 5
    assert set(table.columns) >= {"fold", "accuracy", "precision", "recall", "f1", "roc_auc", "threshold"}
    assert list(table["fold"]) == [1, 2, 3, 4, 5]


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
