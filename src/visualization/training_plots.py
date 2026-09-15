"""Gráficos de treinamento: métricas e threshold por fold (Tarefa 8).

Cada função recebe dados já calculados (`src.training.results.TrainingResult`
via `src.evaluation.comparison.fold_metrics_table`, ou o `DataFrame`
diretamente) e devolve uma `matplotlib.figure.Figure` — nunca chama
`plt.show()`, nunca lê dado, nunca treina nada. Quem chama (tipicamente
um notebook em `experiments/`) decide se mostra, salva ou compõe a
figura com outras.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

DEFAULT_FOLD_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")


def _maybe_save(fig: Figure, save_path: Path | str | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_metrics_by_fold(
    fold_table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_FOLD_METRICS,
    title: str = "Métricas por fold",
    figsize: tuple[float, float] = (8, 5),
    save_path: Path | str | None = None,
) -> Figure:
    """Linha por métrica, um ponto por fold.

    Args:
        fold_table: saída de `src.evaluation.comparison.fold_metrics_table`
            (colunas ``fold`` + uma por métrica).
        metrics: quais colunas de métrica plotar.
        title: título do gráfico.
        figsize: tamanho da figura.
        save_path: se fornecido, salva a figura nesse caminho.

    Returns:
        A `Figure` do matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for metric in metrics:
        ax.plot(fold_table["fold"], fold_table[metric], marker="o", label=metric.replace("_", " ").title())
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(fold_table["fold"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_metrics_boxplot(
    fold_table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_FOLD_METRICS,
    title: str = "Distribuição das métricas entre folds",
    figsize: tuple[float, float] = (7, 5),
    save_path: Path | str | None = None,
) -> Figure:
    """Boxplot da distribuição de cada métrica entre os folds."""
    fig, ax = plt.subplots(figsize=figsize)
    data = [fold_table[metric] for metric in metrics]
    labels = [metric.replace("_", " ").title() for metric in metrics]
    ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_threshold_by_fold(
    fold_table: pd.DataFrame,
    *,
    title: str = "Threshold calibrado por fold",
    figsize: tuple[float, float] = (7, 4),
    save_path: Path | str | None = None,
) -> Figure:
    """Barra com o threshold escolhido em cada fold, com a média marcada.

    Útil para inspecionar se o threshold ficou perto do extremo da busca
    (ver `src.training.threshold_tuning` — guarda-corpo de prevalência).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(fold_table["fold"], fold_table["threshold"], color="steelblue", alpha=0.8, edgecolor="black")
    mean_threshold = fold_table["threshold"].mean()
    ax.axhline(mean_threshold, color="red", linestyle="--", label=f"Média: {mean_threshold:.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Threshold")
    ax.set_title(title)
    ax.set_xticks(fold_table["fold"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


__all__ = [
    "DEFAULT_FOLD_METRICS",
    "plot_metrics_by_fold",
    "plot_metrics_boxplot",
    "plot_threshold_by_fold",
]
