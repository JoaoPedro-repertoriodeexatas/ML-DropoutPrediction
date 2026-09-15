"""API de avaliação independente de notebook (Tarefa 7).

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator()
result = evaluator.evaluate(model=model, X=X_test, y=y_test)
result.metrics.accuracy
result.metrics.confusion_matrix.as_array()
```

Avalia qualquer modelo **já ajustado** (não treina, não faz
cross-validation — isso é `src.training.trainer.Trainer`, Tarefa 6) e
qualquer par `(X, y)`. Não gera gráficos, não salva nada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import MetricSet, compute_metrics


@dataclass(frozen=True)
class EvaluationResult:
    """Resultado de uma avaliação pontual (`Evaluator.evaluate(...)`).

    Attributes:
        model_name: rótulo opcional informado por quem chamou (não
            inspecionado do modelo — o `Evaluator` não sabe nada sobre
            nomes canônicos de `src.configs.constants.MODEL_NAMES`,
            propositalmente, para funcionar com qualquer estimador).
        metrics: `MetricSet` — accuracy, precision, recall, F1, ROC-AUC,
            PR-AUC e matriz de confusão.
        n_samples: nº de amostras avaliadas.
        y_pred: predições do modelo em `X`.
        y_proba: probabilidade da classe positiva, quando disponível
            (ver `extract_positive_class_probability`); `None` caso
            contrário — refletido em `metrics.unavailable`.
    """

    model_name: str | None
    metrics: MetricSet
    n_samples: int
    y_pred: np.ndarray
    y_proba: np.ndarray | None


def extract_positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Extrai a probabilidade da classe positiva de um modelo, tolerando
    estimadores que não a expõem.

    Tenta `predict_proba` primeiro — o caminho usado pelos 5 modelos
    deste projeto (todos suportam, ver `docs/MODELS.md`). Cai para
    `decision_function`, reescalado para `[0, 1]`, só se `predict_proba`
    não existir — para que a camada de avaliação também funcione com um
    modelo externo que só exponha `decision_function` (ex.: um `SVC`
    criado sem `probability=True`). Se nenhum dos dois existir, ou se a
    extração falhar, devolve ``None`` em vez de propagar a exceção —
    quem chama trata isso como "métrica indisponível", não como erro.
    """
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X))[:, 1]
        except Exception:
            return None

    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(X), dtype=float)
            score_range = scores.max() - scores.min()
            if score_range == 0:
                return None
            return (scores - scores.min()) / score_range
        except Exception:
            return None

    return None


class Evaluator:
    """Avalia um modelo já ajustado, de forma independente de notebook."""

    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        model_name: str | None = None,
        y_proba: np.ndarray | None = None,
    ) -> EvaluationResult:
        """Avalia ``model`` em ``(X, y)``.

        Args:
            model: estimador já ajustado (com `.predict(X)` disponível).
            X: features de entrada, já no formato que ``model`` espera
                (ex.: já passadas pela engenharia de features da Tarefa 4
                — este método não faz nenhuma transformação de dado).
            y: rótulos verdadeiros, mesmo índice/ordem de ``X``.
            model_name: rótulo opcional para identificar o resultado
                (útil ao juntar vários `EvaluationResult` em
                `compare_models`).
            y_proba: probabilidade da classe positiva, se já calculada
                por quem chama; ``None`` (padrão) faz o `Evaluator`
                extrair via `extract_positive_class_probability`.

        Returns:
            `EvaluationResult` com as métricas, a matriz de confusão e
            as predições.
        """
        y_true = np.asarray(y)
        y_pred = np.asarray(model.predict(X))

        if y_proba is None:
            y_proba = extract_positive_class_probability(model, X)

        metric_set = compute_metrics(y_true, y_pred, y_proba)

        return EvaluationResult(
            model_name=model_name,
            metrics=metric_set,
            n_samples=len(y_true),
            y_pred=y_pred,
            y_proba=y_proba,
        )


__all__ = ["EvaluationResult", "Evaluator", "extract_positive_class_probability"]
