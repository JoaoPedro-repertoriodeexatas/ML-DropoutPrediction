"""Comparação tabular entre modelos e métricas por fold (Tarefa 7).

```python
from src.evaluation.comparison import comparison_row_from_training_result, compare_models

rows = [comparison_row_from_training_result(result) for result in training_results]
table = compare_models(rows)
```

`compare_models` não sabe treinar nem avaliar nada — só organiza
resultados já calculados (por `src.training.trainer.Trainer`, Tarefa 6,
ou por `src.evaluation.evaluator.Evaluator` + agregação manual) em uma
tabela. Nenhum gráfico é gerado.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.evaluation.metrics import METRIC_NAMES, AggregatedMetrics

COMPARISON_COLUMNS: tuple[str, ...] = ("Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Std")

_METRIC_NAME_TO_COLUMN: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


@dataclass(frozen=True)
class ModelComparisonRow:
    """Uma linha de comparação: nome do modelo + métricas agregadas.

    Attributes:
        model_name: rótulo do modelo (ex.: `src.configs.constants.MODEL_NAMES`
            ou qualquer outro identificador de quem chama).
        mean_metrics: média de cada métrica (``None`` = indisponível).
        std_metrics: desvio padrão de cada métrica entre as unidades
            agregadas (tipicamente folds).
    """

    model_name: str
    mean_metrics: dict[str, float | None]
    std_metrics: dict[str, float | None]


def comparison_row_from_aggregated_metrics(
    model_name: str, aggregated: AggregatedMetrics
) -> ModelComparisonRow:
    """Constrói uma linha a partir de `src.evaluation.metrics.AggregatedMetrics`."""
    return ModelComparisonRow(model_name=model_name, mean_metrics=aggregated.mean, std_metrics=aggregated.std)


def comparison_row_from_training_result(training_result) -> ModelComparisonRow:
    """Constrói uma linha a partir de um `TrainingResult` (Tarefa 6).

    Reaproveita ``training_result.metrics``/``metrics_std`` — calculados
    com a mesma definição de métrica desta camada, já que
    `src.training.cross_validation` delega em `src.evaluation.metrics`
    desde esta tarefa (fonte única de verdade, ver `docs/EVALUATION.md`).
    """
    return ModelComparisonRow(
        model_name=training_result.model_name,
        mean_metrics=dict(training_result.metrics),
        std_metrics=dict(training_result.metrics_std),
    )


def fold_metrics_table(training_result) -> pd.DataFrame:
    """Tabela com uma linha por fold (accuracy, precision, recall, F1,
    ROC-AUC, PR-AUC, threshold calibrado) — a partir de
    ``training_result.fold_results`` (`src.training.results.FoldResult`,
    Tarefa 6)."""
    rows = [
        {
            "fold": fold.fold,
            "accuracy": fold.accuracy,
            "precision": fold.precision,
            "recall": fold.recall,
            "f1": fold.f1,
            "roc_auc": fold.roc_auc,
            "pr_auc": fold.pr_auc,
            "threshold": fold.threshold,
            "threshold_criterion": fold.threshold_criterion,
        }
        for fold in training_result.fold_results
    ]
    return pd.DataFrame(rows)


def compare_models(rows: list[ModelComparisonRow], *, std_metric: str = "f1") -> pd.DataFrame:
    """Tabela comparativa entre modelos: ``Model | Accuracy | Precision |
    Recall | F1 | ROC-AUC | Std``.

    Args:
        rows: uma `ModelComparisonRow` por modelo (ver
            `comparison_row_from_training_result`/
            `comparison_row_from_aggregated_metrics`).
        std_metric: qual métrica usar na coluna ``Std`` (padrão:
            ``"f1"`` — a métrica usada como `scoring` nas buscas de
            hiperparâmetros originais, ver `src.configs.search_spaces`).

    Returns:
        `DataFrame` com exatamente as colunas de `COMPARISON_COLUMNS`.
        Uma métrica indisponível para um modelo aparece como `None`
        (`NaN` no pandas) na célula correspondente — nunca omitida
        silenciosamente, nem substituída por zero.

    Raises:
        ValueError: ``std_metric`` não é uma métrica reconhecida.
    """
    if std_metric not in METRIC_NAMES:
        raise ValueError(f"std_metric inválido: {std_metric!r}. Use um de {METRIC_NAMES}.")

    records = [
        {
            "Model": row.model_name,
            "Accuracy": row.mean_metrics.get("accuracy"),
            "Precision": row.mean_metrics.get("precision"),
            "Recall": row.mean_metrics.get("recall"),
            "F1": row.mean_metrics.get("f1"),
            "ROC-AUC": row.mean_metrics.get("roc_auc"),
            "Std": row.std_metrics.get(std_metric),
        }
        for row in rows
    ]
    return pd.DataFrame(records, columns=list(COMPARISON_COLUMNS))


def compare_models_detailed(rows: list[ModelComparisonRow]) -> pd.DataFrame:
    """Como `compare_models`, mas com uma coluna ``<Métrica>_Std`` por
    métrica (inclui PR-AUC) em vez de uma única coluna ``Std``."""
    records = []
    for row in rows:
        record: dict[str, str | float | None] = {"Model": row.model_name}
        for metric_name in METRIC_NAMES:
            column = _METRIC_NAME_TO_COLUMN[metric_name]
            record[column] = row.mean_metrics.get(metric_name)
            record[f"{column}_Std"] = row.std_metrics.get(metric_name)
        records.append(record)
    return pd.DataFrame(records)


__all__ = [
    "COMPARISON_COLUMNS",
    "ModelComparisonRow",
    "comparison_row_from_aggregated_metrics",
    "comparison_row_from_training_result",
    "fold_metrics_table",
    "compare_models",
    "compare_models_detailed",
]
