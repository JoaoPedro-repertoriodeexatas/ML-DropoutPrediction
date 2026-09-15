"""Tratamento de desbalanceamento de classes (Tarefa 6).

Preserva a **diversidade metodológica original** do projeto — cada
notebook trata classes desbalanceadas de um jeito diferente, e a Tarefa
6 pede explicitamente para não uniformizar tudo em SMOTE:

| Modelo | Estratégia original | Onde |
|---|---|---|
| Árvore de Decisão | SMOTE por fold **+** `class_weight` já embutido no hiperparâmetro | `ArvoreDecisao.ipynb` |
| Regressão Logística | só `class_weight='balanced'` (embutido no hiperparâmetro) | `RegressaoLogistica.ipynb` |
| SVM | só `class_weight='balanced'` (embutido no hiperparâmetro) | `SVM.ipynb` |
| Rede Neural | peso por amostra (`sample_weight`) calculado a partir da frequência de classe — `MLPClassifier` não aceita `class_weight` no construtor | `RedesNeurais.ipynb` |
| Naive Bayes | só SMOTE, aplicado no treino | `models/naive_bayes/train.py` |

`class_weight` já está embutido nos hiperparâmetros de
`src.configs.model_hyperparameters` (Tarefa 5) — por isso os modelos com
essa estratégia não precisam de nenhum passo extra aqui, só o registro
de que "a estratégia já está resolvida no hiperparâmetro".
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from src.configs.settings import SMOTE_RANDOM_STATE

ImbalanceStrategyName = Literal["smote", "class_weight", "sample_weight"]

IMBALANCE_STRATEGY_BY_MODEL: dict[str, ImbalanceStrategyName] = {
    "arvore_de_decisao": "smote",
    "regressao_logistica": "class_weight",
    "svm": "class_weight",
    "rede_neural": "sample_weight",
    "naive_bayes": "smote",
}


def get_imbalance_strategy(model_name: str) -> ImbalanceStrategyName:
    """Estratégia de desbalanceamento original do modelo ``model_name``.

    Raises:
        KeyError: ``model_name`` não é um nome canônico de modelo.
    """
    if model_name not in IMBALANCE_STRATEGY_BY_MODEL:
        raise KeyError(
            f"Modelo desconhecido: {model_name!r}. Use um de {tuple(IMBALANCE_STRATEGY_BY_MODEL)}."
        )
    return IMBALANCE_STRATEGY_BY_MODEL[model_name]


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = SMOTE_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica SMOTE **só** no fold de treino (nunca em calibração/validação).

    Deve ser chamada depois da engenharia de features (SMOTE precisa de
    entrada puramente numérica) e antes de `.fit(...)` — nunca antes do
    split de cross-validation, para não vazar informação da validação
    para os exemplos sintéticos gerados.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def compute_balanced_sample_weight(y_train: pd.Series | np.ndarray) -> np.ndarray:
    """Peso por amostra, inversamente proporcional à frequência da classe.

    Réplica de `compute_class_weight('balanced', ...)` usado em
    `RedesNeurais.ipynb` e `analysis/audit/evaluator.py` para dar a
    `MLPClassifier.fit(..., sample_weight=...)` — a única forma de
    balancear classes nesse estimador, que não aceita `class_weight` no
    construtor.
    """
    y_array = np.asarray(y_train)
    classes, counts = np.unique(y_array, return_counts=True)
    weight_by_class = {
        label: len(y_array) / (len(classes) * count) for label, count in zip(classes, counts)
    }
    return np.array([weight_by_class[label] for label in y_array])


__all__ = [
    "ImbalanceStrategyName",
    "IMBALANCE_STRATEGY_BY_MODEL",
    "get_imbalance_strategy",
    "apply_smote",
    "compute_balanced_sample_weight",
]
