"""Estruturas de saída do treinamento (Tarefa 6).

Nenhuma classe aqui executa lógica — só armazena o resultado estruturado
que `src.training.trainer.Trainer.train(...)` devolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FoldResult:
    """Resultado de um único fold da validação cruzada.

    Attributes:
        fold: número do fold (1-indexado).
        threshold: limiar de decisão escolhido na calibração deste fold.
        threshold_criterion: critério usado para escolher o limiar (ver
            `src.training.threshold_tuning`).
        train_size: nº de amostras usadas para ajustar o modelo (treino
            menos a fatia de calibração).
        calibration_size: nº de amostras usadas só para calibrar o limiar.
        validation_size: nº de amostras do fold de validação (nunca
            usadas para ajustar nada — nem modelo, nem limiar).
        accuracy, precision, recall, f1, roc_auc, pr_auc: métricas
            calculadas no fold de validação, com o limiar calibrado.
    """

    fold: int
    threshold: float
    threshold_criterion: str
    train_size: int
    calibration_size: int
    validation_size: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


@dataclass
class TrainingResult:
    """Saída completa de `Trainer.train(...)`.

    Attributes:
        model_name: nome canônico do modelo treinado.
        imbalance_strategy: estratégia de desbalanceamento aplicada (ver
            `src.training.imbalance.IMBALANCE_STRATEGY_BY_MODEL`).
        params: hiperparâmetros efetivamente usados pelo último fold
            (via `estimator.get_params()`) — inclui valores dependentes
            de dado, como `hidden_layer_sizes` da Rede Neural.
        fold_results: um `FoldResult` por fold.
        metrics: média de cada métrica entre os folds.
        metrics_std: desvio padrão de cada métrica entre os folds.
        predictions: predições **out-of-fold** — para cada amostra de
            `X`/`y` passada a `train(...)`, a predição do fold em que ela
            caiu como validação (nunca usada para treinar aquele modelo).
            Alinhado por posição a `y_true`.
        probabilities: probabilidade da classe positiva, mesma lógica
            out-of-fold. ``None`` só se o modelo não suportar
            `predict_proba` (nenhum dos 5 modelos deste projeto cai nesse
            caso).
        fitted_feature_pipelines: um pipeline de feature engineering
            ajustado por fold (na fatia de treino menos calibração).
        fitted_models: um estimador ajustado por fold.
        y_true: cópia de ``y`` como array, na mesma ordem de `predictions`.
    """

    model_name: str
    imbalance_strategy: str
    params: dict[str, Any]
    fold_results: list[FoldResult]
    metrics: dict[str, float]
    metrics_std: dict[str, float]
    predictions: np.ndarray
    probabilities: np.ndarray | None
    fitted_feature_pipelines: list[Any] = field(default_factory=list)
    fitted_models: list[Any] = field(default_factory=list)
    y_true: np.ndarray | None = None


__all__ = ["FoldResult", "TrainingResult"]
