"""Gráficos específicos da Árvore de Decisão (Tarefa 8).

`plot_feature_importance` substitui a célula de plot de importância de
features que hoje vive em `ArvoreDecisao.ipynb` — recebe a importância
já calculada (`model.feature_importances_`), não recalcula nada.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _maybe_save(fig: Figure, save_path: Path | str | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    *,
    top_n: int = 15,
    title: str = "Importância das Features — Árvore de Decisão",
    figsize: tuple[float, float] = (8, 6),
    save_path: Path | str | None = None,
) -> Figure:
    """Barras horizontais com as `top_n` features mais importantes.

    Args:
        feature_names: nomes das colunas na mesma ordem de ``importances``
            (ex.: `feature_pipeline.get_feature_names_out()` de um fold
            de `TrainingResult.fitted_feature_pipelines`).
        importances: `model.feature_importances_` do estimador ajustado.
        top_n: quantas features mostrar (as de maior importância).
    """
    series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(series.index[::-1], series.to_numpy()[::-1], color="steelblue", alpha=0.85)
    ax.set_xlabel("Importância")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


__all__ = ["plot_feature_importance"]
