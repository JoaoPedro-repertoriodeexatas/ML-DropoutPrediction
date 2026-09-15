"""Cálculo de métricas de classificação, centralizado (Tarefa 7).

Fonte única de verdade para "como uma métrica é calculada" — usada tanto
pela camada de avaliação (`src.evaluation.evaluator`) quanto, desde esta
tarefa, pela camada de treinamento (`src.training.cross_validation`,
Tarefa 6), para que as duas nunca divirjam na definição de accuracy,
precision, recall, F1, ROC-AUC ou PR-AUC — requisito explícito da
Tarefa 7 ("garanta que todos os modelos utilizem a mesma definição de
métricas sempre que metodologicamente possível").

Métricas que dependem de probabilidade (ROC-AUC, PR-AUC) nem sempre
podem ser calculadas — um modelo sem `predict_proba`/`decision_function`
utilizável, ou um fold de validação com uma única classe presente. Esses
casos são tratados **explicitamente**: o valor correspondente fica
``None`` e o motivo entra em `MetricSet.unavailable`, nunca um `NaN`
silencioso ou uma métrica omitida sem explicação.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
PROBABILITY_DEPENDENT_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc")


@dataclass(frozen=True)
class ConfusionMatrixSummary:
    """Matriz de confusão 2x2, com os quatro contadores nomeados."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def as_array(self) -> np.ndarray:
        """Matriz 2x2 no formato `sklearn.metrics.confusion_matrix`
        (linhas = real, colunas = previsto, ordem de classe `[0, 1]`)."""
        return np.array(
            [
                [self.true_negative, self.false_positive],
                [self.false_negative, self.true_positive],
            ]
        )


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrixSummary:
    """Matriz de confusão binária, sempre calculável a partir de `predict`."""
    tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ConfusionMatrixSummary(
        true_negative=int(tn), false_positive=int(fp), false_negative=int(fn), true_positive=int(tp)
    )


@dataclass(frozen=True)
class MetricSet:
    """Conjunto padrão de métricas de um único modelo/avaliação.

    Attributes:
        accuracy, precision, recall, f1: sempre calculáveis a partir de
            `y_true`/`y_pred` — nunca `None`.
        roc_auc, pr_auc: dependem de probabilidade; `None` quando não
            calculáveis (ver `unavailable`).
        confusion_matrix: `ConfusionMatrixSummary`, sempre calculável.
        unavailable: ``{nome_da_métrica: motivo}`` para toda métrica que
            ficou `None` — nunca fica vazio inconsistentemente com um
            `None` presente em outro campo.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: ConfusionMatrixSummary
    unavailable: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | None]:
        """As 6 métricas escalares, sem a matriz de confusão nem os motivos."""
        return {name: getattr(self, name) for name in METRIC_NAMES}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> MetricSet:
    """Calcula o conjunto padrão de métricas para uma predição binária.

    Args:
        y_true: rótulos verdadeiros (0/1).
        y_pred: rótulos previstos (0/1) — já com o threshold aplicado,
            quando houver (ver `src.training.threshold_tuning`).
        y_proba: probabilidade da classe positiva. ``None`` se o modelo
            não expõe probabilidade utilizável — nesse caso, `roc_auc` e
            `pr_auc` ficam `None`, com o motivo em `unavailable`.

    Returns:
        `MetricSet` com as 6 métricas, a matriz de confusão e os motivos
        de qualquer métrica indisponível.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unavailable: dict[str, str] = {}

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    confusion = compute_confusion_matrix(y_true, y_pred)

    roc_auc: float | None = None
    pr_auc: float | None = None
    has_both_classes = len(np.unique(y_true)) > 1

    if y_proba is None:
        reason = "y_proba não fornecido (modelo sem predict_proba/decision_function utilizável)"
        unavailable["roc_auc"] = reason
        unavailable["pr_auc"] = reason
    elif not has_both_classes:
        reason = "y_true contém apenas uma classe — ROC-AUC/PR-AUC não são definidos"
        unavailable["roc_auc"] = reason
        unavailable["pr_auc"] = reason
    else:
        y_proba_array = np.asarray(y_proba, dtype=float)
        roc_auc = float(roc_auc_score(y_true, y_proba_array))
        pr_auc = float(average_precision_score(y_true, y_proba_array))

    return MetricSet(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=confusion,
        unavailable=unavailable,
    )


@dataclass(frozen=True)
class AggregatedMetrics:
    """Média e desvio padrão de um conjunto de `MetricSet` (ex.: entre folds).

    Attributes:
        mean: média de cada métrica. `None` se nenhum `MetricSet` de
            entrada tinha essa métrica disponível.
        std: desvio padrão de cada métrica, calculado só sobre os
            `MetricSet` em que ela estava disponível.
        n: quantos `MetricSet` entraram na agregação (ex.: nº de folds).
        n_available: quantos desses tinham cada métrica disponível — útil
            para saber se a média de `roc_auc`, por exemplo, veio de
            todos os folds ou só de parte deles.
    """

    mean: dict[str, float | None]
    std: dict[str, float | None]
    n: int
    n_available: dict[str, int]


def aggregate_metric_sets(metric_sets: list[MetricSet]) -> AggregatedMetrics:
    """Agrega uma lista de `MetricSet` (tipicamente um por fold) em média/desvio.

    Uma métrica indisponível em alguns `MetricSet` (mas não todos) entra
    na média/desvio só com os valores disponíveis — nunca vira `0` nem
    derruba a agregação inteira. Se estiver indisponível em **todos**,
    a média/desvio ficam `None`, não `NaN`.

    Raises:
        ValueError: ``metric_sets`` vazio.
    """
    if not metric_sets:
        raise ValueError("Não é possível agregar uma lista vazia de MetricSet.")

    mean: dict[str, float | None] = {}
    std: dict[str, float | None] = {}
    n_available: dict[str, int] = {}

    for name in METRIC_NAMES:
        values = [getattr(m, name) for m in metric_sets if getattr(m, name) is not None]
        n_available[name] = len(values)
        if values:
            mean[name] = float(np.mean(values))
            std[name] = float(np.std(values))
        else:
            mean[name] = None
            std[name] = None

    return AggregatedMetrics(mean=mean, std=std, n=len(metric_sets), n_available=n_available)


__all__ = [
    "METRIC_NAMES",
    "PROBABILITY_DEPENDENT_METRICS",
    "ConfusionMatrixSummary",
    "MetricSet",
    "AggregatedMetrics",
    "compute_confusion_matrix",
    "compute_metrics",
    "aggregate_metric_sets",
]
