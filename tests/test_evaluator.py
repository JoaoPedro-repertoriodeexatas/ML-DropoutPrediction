"""Testes de `src.evaluation.evaluator` (Tarefa 7)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.evaluation.evaluator import EvaluationResult, Evaluator, extract_positive_class_probability


def _fitted_logistic_regression(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + rng.normal(scale=0.3, size=n) > 0).astype(int))
    model = LogisticRegression().fit(X, y)
    return model, X, y


class _PredictOnlyModel:
    """Estimador mínimo sem `predict_proba` nem `decision_function` — para
    testar o tratamento explícito de métrica indisponível."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X["a"].to_numpy() > self.threshold).astype(int)


class _DecisionFunctionOnlyModel:
    """Estimador com `decision_function` mas sem `predict_proba` — testa
    o caminho alternativo de `extract_positive_class_probability`."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X["a"].to_numpy() > 0).astype(int)

    def decision_function(self, X):
        return X["a"].to_numpy()


def test_evaluate_returns_structured_result_with_all_metrics() -> None:
    model, X, y = _fitted_logistic_regression()
    result = Evaluator().evaluate(model, X, y, model_name="regressao_logistica")

    assert isinstance(result, EvaluationResult)
    assert result.model_name == "regressao_logistica"
    assert result.n_samples == len(y)
    assert result.y_pred.shape == (len(y),)
    assert result.y_proba is not None
    assert result.metrics.roc_auc is not None
    assert result.metrics.unavailable == {}


def test_evaluate_is_independent_of_any_notebook_or_training_loop() -> None:
    """Não depende de `Trainer`/CV — um modelo ajustado manualmente, fora
    de qualquer pipeline de treino, ainda pode ser avaliado."""
    X = pd.DataFrame({"a": [1.0, 2.0, -1.0, -2.0, 0.5, -0.5]})
    y = pd.Series([1, 1, 0, 0, 1, 0])
    model = SVC(probability=True, random_state=42).fit(X, y)

    result = Evaluator().evaluate(model, X, y)

    assert 0.0 <= result.metrics.accuracy <= 1.0


def test_evaluate_explicit_none_when_model_has_no_probability_output() -> None:
    """Requisito da Tarefa 7: quando uma métrica não pode ser calculada,
    trate o caso explicitamente — nunca um NaN silencioso ou um crash."""
    X = pd.DataFrame({"a": [1.0, 2.0, -1.0, -2.0]})
    y = pd.Series([1, 1, 0, 0])
    model = _PredictOnlyModel().fit(X, y)

    result = Evaluator().evaluate(model, X, y)

    assert result.y_proba is None
    assert result.metrics.roc_auc is None
    assert result.metrics.pr_auc is None
    assert "roc_auc" in result.metrics.unavailable
    # accuracy continua calculável mesmo sem probabilidade.
    assert result.metrics.accuracy == 1.0


def test_extract_positive_class_probability_falls_back_to_decision_function() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, -1.0, -2.0]})
    model = _DecisionFunctionOnlyModel().fit(X, None)

    proba = extract_positive_class_probability(model, X)

    assert proba is not None
    assert (proba >= 0).all() and (proba <= 1).all()


def test_extract_positive_class_probability_returns_none_when_unavailable() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0]})
    model = _PredictOnlyModel().fit(X, None)

    assert extract_positive_class_probability(model, X) is None


def test_evaluate_accepts_precomputed_y_proba() -> None:
    """Quem chama pode passar `y_proba` já calculado (ex.: probabilidade
    out-of-fold de um `TrainingResult`), sem que o Evaluator recalcule."""
    model, X, y = _fitted_logistic_regression(seed=3)
    precomputed = np.full(len(y), 0.5)

    result = Evaluator().evaluate(model, X, y, y_proba=precomputed)

    assert np.array_equal(result.y_proba, precomputed)


if __name__ == "__main__":
    test_functions = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"OK   - {test_fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL - {test_fn.__name__}: {exc}")
    total = len(test_functions)
    print(f"\n{total - failures}/{total} testes passaram.")
    sys.exit(1 if failures else 0)
