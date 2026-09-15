"""Protocolo de validação cruzada (Tarefa 6).

Utilitários reaproveitáveis pelo `Trainer` (`src.training.trainer`):
construção do `StratifiedKFold`, separação do fold de treino em
treino-para-ajuste/calibração (para o threshold tuning nunca ver o fold
de validação), cálculo de métricas por fold e agregação entre folds.

Nenhuma função aqui ajusta um modelo — só organiza índices e calcula
métricas a partir de predições já feitas.

Desde a Tarefa 7, `compute_fold_metrics`/`aggregate_fold_metrics`
delegam em `src.evaluation.metrics` — a fonte única de verdade para a
definição de cada métrica, compartilhada entre treino e avaliação (ver
`docs/EVALUATION.md`). A assinatura e o formato de retorno (dict simples
com `NaN` para métrica indisponível) continuam os mesmos da Tarefa 6,
para não quebrar `Trainer`/`TrainingResult`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.configs.settings import CALIBRATION_SPLIT_SIZE, CV_FOLDS, CV_SHUFFLE, RANDOM_STATE
from src.evaluation.metrics import METRIC_NAMES, compute_metrics

FOLD_METRIC_NAMES: tuple[str, ...] = METRIC_NAMES


def make_stratified_kfold(
    *,
    n_splits: int = CV_FOLDS,
    shuffle: bool = CV_SHUFFLE,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """`StratifiedKFold` com os parâmetros já em uso no projeto (5 folds,
    embaralhado, `random_state` fixo) — os mesmos de todos os notebooks e
    de `analysis/audit/evaluator.py`."""
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def split_train_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    test_size: float = CALIBRATION_SPLIT_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa uma fatia de calibração dentro do fold de treino.

    O threshold tuning é ajustado nessa fatia, nunca no fold de
    validação — garante que "o conjunto de validação/teste permaneça
    invisível durante o ajuste" (requisito explícito da Tarefa 6).
    """
    return train_test_split(
        X_train, y_train, test_size=test_size, stratify=y_train, random_state=random_state
    )


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Métricas de um fold de validação: accuracy, precision, recall, F1,
    ROC-AUC, PR-AUC — calculadas por `src.evaluation.metrics.compute_metrics`
    (fonte única de verdade desde a Tarefa 7). Retorna um dict simples
    com `NaN` para métrica indisponível (em vez de `None`), para manter
    o contrato já usado por `Trainer`/`FoldResult` (Tarefa 6)."""
    metric_set = compute_metrics(y_true, y_pred, y_score)
    return {name: (value if value is not None else float("nan")) for name, value in metric_set.as_dict().items()}


def aggregate_fold_metrics(
    fold_metric_dicts: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Média e desvio padrão de cada métrica entre folds (ignora `NaN`)."""
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for metric_name in FOLD_METRIC_NAMES:
        values = np.array([fold[metric_name] for fold in fold_metric_dicts], dtype=float)
        means[metric_name] = float(np.nanmean(values))
        stds[metric_name] = float(np.nanstd(values))
    return means, stds


__all__ = [
    "FOLD_METRIC_NAMES",
    "make_stratified_kfold",
    "split_train_calibration",
    "compute_fold_metrics",
    "aggregate_fold_metrics",
]
