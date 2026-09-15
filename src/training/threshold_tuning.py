"""Seleção de threshold com guarda-corpo metodológico (Tarefa 6).

Reimplementação de `analysis/audit/threshold.py::encontrar_melhor_threshold`
— a única versão "segura" de threshold tuning hoje presente no projeto,
já validada pela auditoria (`ARCHITECTURE_AUDIT.md`, seção 5) como
correção do problema documentado em `results/inconsistencies_report.md`
("Regressão Logística — Recall inflado", causado por maximização cega de
F1 sem guarda-corpo, threshold ≈ 0,01).

`Trainer` (`src.training.trainer`) usa esta implementação **para todos
os 5 modelos**, não só a Rede Neural — decisão documentada em
`docs/TRAINING.md`. O enunciado da Tarefa 6 pede para preservar
"o threshold tuning utilizado pela rede neural caso ele exista": ele
existe (`RedesNeurais.ipynb`, seção "AJUSTE DE LIMIAR"), mas a versão do
notebook é a variante *sem* guarda-corpo — a mesma classe de problema já
corrigida para os outros modelos pela auditoria. Usar a versão segura
para todos, incluindo a Rede Neural, evita reintroduzir um bug já
conhecido e corrigido, sem mudar o conceito (limiar calibrado em vez de
0,5 fixo) que a Tarefa 6 pede para preservar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from src.configs.settings import (
    THRESHOLD_FALLBACK,
    THRESHOLD_PREVALENCE_CAP,
    THRESHOLD_SEARCH_MAX,
    THRESHOLD_SEARCH_MIN,
    THRESHOLD_SEARCH_STEPS,
)


@dataclass(frozen=True)
class ThresholdSelection:
    """Resultado da seleção de threshold.

    Attributes:
        threshold: limiar de decisão escolhido.
        criterion: nome do critério usado (``"youden_j_constrained"`` ou
            ``"fallback"``).
        calibration_f1: F1-score obtido no conjunto de calibração com
            este limiar.
        positive_rate: proporção de positivos previstos na calibração
            com este limiar.
        rejected_extreme: se algum candidato foi rejeitado por exceder o
            guarda-corpo de prevalência.
        fallback_used: se nenhum candidato passou no guarda-corpo e o
            limiar padrão (`THRESHOLD_FALLBACK`) foi usado.
    """

    threshold: float
    criterion: str
    calibration_f1: float
    positive_rate: float
    rejected_extreme: bool
    fallback_used: bool


def _youden_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity + specificity - 1.0


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    search_min: float = THRESHOLD_SEARCH_MIN,
    search_max: float = THRESHOLD_SEARCH_MAX,
    n_thresholds: int = THRESHOLD_SEARCH_STEPS,
    prevalence_cap: float = THRESHOLD_PREVALENCE_CAP,
    fallback: float = THRESHOLD_FALLBACK,
) -> ThresholdSelection:
    """Seleciona o threshold por Youden's J, com guarda-corpo de prevalência.

    Rejeita candidatos cuja taxa de positivos previstos exceda
    ``prevalence_cap`` vezes a prevalência real observada em ``y_true``
    — o mecanismo que impede o "threshold patológico" (ex.: 0,01,
    prevendo quase tudo como positivo) documentado em
    `results/inconsistencies_report.md`. Se nenhum candidato passar,
    usa ``fallback`` (0,5 por padrão).

    Args:
        y_true: rótulos verdadeiros do conjunto de **calibração** (nunca
            do conjunto de validação/teste — isso é responsabilidade de
            quem chama, ver `src.training.cross_validation.split_train_calibration`).
        y_proba: probabilidade prevista da classe positiva, mesmo
            conjunto de `y_true`.

    Returns:
        `ThresholdSelection` com o limiar escolhido e metadados da busca.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    prevalence = float(y_true.mean()) if len(y_true) else 0.5

    thresholds = np.linspace(search_min, search_max, n_thresholds)
    best_j = -np.inf
    best_threshold = fallback
    best_f1 = 0.0
    best_positive_rate = float((y_proba >= fallback).mean())
    rejected_extreme = False
    fallback_used = True

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        positive_rate = float(y_pred.mean())

        if positive_rate > prevalence_cap * max(prevalence, 1e-6):
            rejected_extreme = True
            continue

        j_statistic = _youden_j(y_true, y_pred)
        if j_statistic > best_j:
            best_j = j_statistic
            best_threshold = float(threshold)
            best_f1 = float(f1_score(y_true, y_pred, zero_division=0))
            best_positive_rate = positive_rate
            fallback_used = False

    if fallback_used:
        y_pred_fallback = (y_proba >= fallback).astype(int)
        best_f1 = float(f1_score(y_true, y_pred_fallback, zero_division=0))
        best_positive_rate = float(y_pred_fallback.mean())

    return ThresholdSelection(
        threshold=best_threshold,
        criterion="youden_j_constrained" if not fallback_used else "fallback",
        calibration_f1=best_f1,
        positive_rate=best_positive_rate,
        rejected_extreme=rejected_extreme,
        fallback_used=fallback_used,
    )


__all__ = ["ThresholdSelection", "find_best_threshold"]
