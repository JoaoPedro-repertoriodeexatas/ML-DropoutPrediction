"""Testes de `src.visualization` (Tarefa 8).

Usa o backend `Agg` do matplotlib (sem display) — os testes verificam
que cada função devolve uma `Figure` válida a partir de dados sintéticos
já calculados, sem ler nenhum dataset nem treinar nada.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.evaluation.comparison import compare_models, fold_metrics_table, ModelComparisonRow
from src.evaluation.metrics import compute_metrics
from src.training.trainer import Trainer
from src.visualization.comparison_plots import (
    plot_metric_heatmap,
    plot_model_comparison_bars,
    plot_model_ranking,
)
from src.visualization.evaluation_plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.visualization.training_plots import plot_metrics_boxplot, plot_metrics_by_fold, plot_threshold_by_fold
from src.visualization.tree_plots import plot_feature_importance


def _sample_notebook_style_dataset(n: int = 90, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
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


def _sample_comparison_table() -> pd.DataFrame:
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
    return compare_models(rows)


def _close(fig: Figure) -> None:
    plt.close(fig)


# --------------------------------------------------------------------------
# training_plots
# --------------------------------------------------------------------------


def test_plot_metrics_by_fold_returns_figure() -> None:
    X, y = _sample_notebook_style_dataset()
    result = Trainer().train("regressao_logistica", X, y)
    table = fold_metrics_table(result)

    fig = plot_metrics_by_fold(table)
    assert isinstance(fig, Figure)
    _close(fig)


def test_plot_metrics_boxplot_returns_figure() -> None:
    X, y = _sample_notebook_style_dataset()
    result = Trainer().train("regressao_logistica", X, y)
    table = fold_metrics_table(result)

    fig = plot_metrics_boxplot(table)
    assert isinstance(fig, Figure)
    _close(fig)


def test_plot_threshold_by_fold_returns_figure() -> None:
    X, y = _sample_notebook_style_dataset()
    result = Trainer().train("regressao_logistica", X, y)
    table = fold_metrics_table(result)

    fig = plot_threshold_by_fold(table)
    assert isinstance(fig, Figure)
    _close(fig)


def test_plot_functions_can_save_to_disk() -> None:
    X, y = _sample_notebook_style_dataset()
    result = Trainer().train("regressao_logistica", X, y)
    table = fold_metrics_table(result)

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "metrics_by_fold.png"
        fig = plot_metrics_by_fold(table, save_path=save_path)
        _close(fig)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


# --------------------------------------------------------------------------
# evaluation_plots
# --------------------------------------------------------------------------


def test_plot_confusion_matrix_returns_figure() -> None:
    metrics = compute_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([0.2, 0.6, 0.7, 0.9])
    )
    fig = plot_confusion_matrix(metrics.confusion_matrix)
    assert isinstance(fig, Figure)
    _close(fig)


def test_plot_roc_curve_returns_figure() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.clip(y_true + rng.normal(scale=0.3, size=100), 0, 1)

    fig = plot_roc_curve(y_true, y_proba, label="Regressão Logística")
    assert isinstance(fig, Figure)
    _close(fig)


def test_plot_precision_recall_curve_returns_figure() -> None:
    rng = np.random.default_rng(1)
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.clip(y_true + rng.normal(scale=0.3, size=100), 0, 1)

    fig = plot_precision_recall_curve(y_true, y_proba, label="SVM")
    assert isinstance(fig, Figure)
    _close(fig)


# --------------------------------------------------------------------------
# comparison_plots
# --------------------------------------------------------------------------


def test_plot_model_comparison_bars_returns_figure() -> None:
    table = _sample_comparison_table()
    fig = plot_model_comparison_bars(table)
    assert isinstance(fig, Figure)
    _close(fig)


def test_plot_model_ranking_returns_figure() -> None:
    table = _sample_comparison_table()
    fig = plot_model_ranking(table, metric="F1")
    assert isinstance(fig, Figure)
    _close(fig)


def test_plot_metric_heatmap_handles_missing_values_gracefully() -> None:
    table = _sample_comparison_table()
    table.loc[0, "ROC-AUC"] = None  # métrica indisponível para um modelo

    fig = plot_metric_heatmap(table)
    assert isinstance(fig, Figure)
    _close(fig)


# --------------------------------------------------------------------------
# tree_plots
# --------------------------------------------------------------------------


def test_plot_feature_importance_returns_figure() -> None:
    feature_names = [f"feature_{i}" for i in range(10)]
    importances = np.random.default_rng(0).random(10)

    fig = plot_feature_importance(feature_names, importances, top_n=5)
    assert isinstance(fig, Figure)
    _close(fig)


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
