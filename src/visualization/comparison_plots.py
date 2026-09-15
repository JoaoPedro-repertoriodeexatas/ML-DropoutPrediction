"""Gráficos comparativos entre modelos (Tarefa 8).

Consomem a tabela de `src.evaluation.comparison.compare_models(...)` —
não recalculam nada, não leem dado, não chamam `plt.show()`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

DEFAULT_COMPARISON_METRICS: tuple[str, ...] = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")


def _maybe_save(fig: Figure, save_path: Path | str | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_model_comparison_bars(
    comparison_table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_COMPARISON_METRICS,
    title: str = "Comparação entre modelos",
    figsize: tuple[float, float] = (10, 5.5),
    save_path: Path | str | None = None,
) -> Figure:
    """Barras agrupadas: uma cor por métrica, um grupo por modelo.

    Args:
        comparison_table: saída de `src.evaluation.comparison.compare_models`
            (colunas ``Model`` + as métricas).
    """
    fig, ax = plt.subplots(figsize=figsize)
    models = comparison_table["Model"].tolist()
    x_positions = np.arange(len(models))
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics

    for index, metric in enumerate(metrics):
        offset = (index - (n_metrics - 1) / 2) * bar_width
        values = comparison_table[metric].to_numpy(dtype=float)
        ax.bar(x_positions + offset, values, width=bar_width, label=metric)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_model_ranking(
    comparison_table: pd.DataFrame,
    *,
    metric: str = "F1",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    save_path: Path | str | None = None,
) -> Figure:
    """Barras horizontais de um único metric, ordenadas do maior para o
    menor, com a barra de erro da coluna ``Std`` quando presente."""
    ordered = comparison_table.sort_values(metric, ascending=True)
    errors = ordered["Std"].to_numpy(dtype=float) if "Std" in ordered.columns else None

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(ordered["Model"], ordered[metric], xerr=errors, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1.05)
    ax.set_title(title or f"Ranking por {metric}")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_metric_heatmap(
    comparison_table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_COMPARISON_METRICS,
    title: str = "Mapa de calor — métricas por modelo",
    figsize: tuple[float, float] | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """Heatmap modelo × métrica, com o valor anotado em cada célula."""
    data = comparison_table.set_index("Model")[list(metrics)]
    figsize = figsize or (7, max(3, len(data) * 0.7))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)

    for row in range(len(data)):
        for col in range(len(metrics)):
            value = data.iloc[row, col]
            text = "—" if pd.isna(value) else f"{value:.3f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=9)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


__all__ = [
    "DEFAULT_COMPARISON_METRICS",
    "plot_model_comparison_bars",
    "plot_model_ranking",
    "plot_metric_heatmap",
]
