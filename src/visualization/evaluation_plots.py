"""Gráficos de avaliação: matriz de confusão, ROC, precision-recall (Tarefa 8).

Consomem as estruturas de `src.evaluation.metrics`
(`ConfusionMatrixSummary`) e arrays de rótulo/probabilidade — nunca
recalculam métrica (isso é `src.evaluation.metrics`), nunca leem dado,
nunca chamam `plt.show()`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.evaluation.metrics import ConfusionMatrixSummary


def _maybe_save(fig: Figure, save_path: Path | str | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_confusion_matrix(
    confusion_matrix: ConfusionMatrixSummary,
    *,
    labels: tuple[str, str] = ("Baixo Risco", "Alto Risco"),
    title: str = "Matriz de Confusão",
    figsize: tuple[float, float] = (5, 4.5),
    save_path: Path | str | None = None,
) -> Figure:
    """Heatmap 2x2 a partir de um `ConfusionMatrixSummary` já calculado."""
    fig, ax = plt.subplots(figsize=figsize)
    matrix = confusion_matrix.as_array()
    im = ax.imshow(matrix, cmap="Blues")

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=13)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    label: str | None = None,
    title: str = "Curva ROC",
    figsize: tuple[float, float] = (5.5, 5),
    save_path: Path | str | None = None,
) -> Figure:
    """Curva ROC com a diagonal de referência (classificador aleatório)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_value = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    curve_label = f"{label} (AUC = {roc_auc_value:.3f})" if label else f"AUC = {roc_auc_value:.3f}"
    ax.plot(fpr, tpr, linewidth=2, label=curve_label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    label: str | None = None,
    title: str = "Curva Precision-Recall",
    figsize: tuple[float, float] = (5.5, 5),
    save_path: Path | str | None = None,
) -> Figure:
    """Curva Precision-Recall — mais informativa que a ROC sob desbalanceamento."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_value = auc(recall, precision)

    fig, ax = plt.subplots(figsize=figsize)
    curve_label = f"{label} (AUC = {pr_auc_value:.3f})" if label else f"AUC = {pr_auc_value:.3f}"
    ax.plot(recall, precision, linewidth=2, label=curve_label)
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Baseline (prevalência = {baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


__all__ = ["plot_confusion_matrix", "plot_roc_curve", "plot_precision_recall_curve"]
