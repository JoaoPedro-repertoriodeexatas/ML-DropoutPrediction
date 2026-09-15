"""Testes de `src.training.trainer.Trainer` (Tarefa 6).

Cobrem o requisito explícito da tarefa: "criar testes para garantir que
o treinamento é executável para os cinco modelos". Usa dados sintéticos
pequenos com sinal real (não puramente aleatório), para que os modelos
aprendam algo e o threshold tuning não degenere.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.training.results import TrainingResult
from src.training.trainer import Trainer


def _sample_notebook_style_dataset(n: int = 150, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Dados para Árvore, Regressão Logística, SVM e Rede Neural — mesmo
    formato esperado por `FeatureEngineer.common_features()` (Tarefa 4):
    colunas brutas do censo, categóricas como códigos numéricos."""
    rng = np.random.default_rng(seed)
    y_values = np.array([0, 1] * (n // 2))
    rng.shuffle(y_values)
    signal = y_values * 80  # correlaciona QT_ING/QT_MAT com o alvo p/ o modelo aprender algo real

    df = pd.DataFrame(
        {
            "TP_ORGANIZACAO_ACADEMICA": rng.integers(1, 5, n),
            "TP_REDE": rng.integers(1, 3, n),
            "TP_CATEGORIA_ADMINISTRATIVA": rng.integers(1, 8, n),
            "TP_GRAU_ACADEMICO": rng.integers(1, 4, n),
            "TP_MODALIDADE_ENSINO": rng.integers(1, 3, n),
            "TP_DIMENSAO": rng.integers(1, 6, n),
            "IN_GRATUITO": rng.integers(0, 2, n),
            "CO_CINE_AREA_GERAL": rng.integers(1, 10, n),
            "QT_ING": rng.integers(10, 100, n) + signal,
            "QT_MAT": rng.integers(10, 100, n) + signal // 2,
            "QT_VG_TOTAL": rng.integers(10, 500, n),
            "QT_INSCRITO_TOTAL": rng.integers(10, 1000, n),
            "QT_CONC": rng.integers(0, 200, n),
            "QT_VG_TOTAL_EAD": rng.integers(0, 200, n),
            "QT_VG_TOTAL_NOTURNO": rng.integers(0, 200, n),
            "QT_MAT_FINANC": rng.integers(0, 200, n),
            "QT_ING_FIES": rng.integers(0, 100, n),
            "QT_ING_PROUNIP": rng.integers(0, 100, n),
            "QT_ING_18_24": rng.integers(0, 300, n),
            "QT_ING_FEM": rng.integers(0, 300, n),
        }
    )
    y = pd.Series(y_values, name="alto_risco_evasao")
    return df, y


def _sample_naive_bayes_dataset(n: int = 150, seed: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """Dados para Naive Bayes — universo próprio de features numéricas
    (Tarefa 4, `FeatureEngineer.for_naive_bayes`)."""
    rng = np.random.default_rng(seed)
    y_values = np.array([0, 1] * (n // 2))
    rng.shuffle(y_values)
    signal = y_values * 2.0

    df = pd.DataFrame(
        {
            "feature_1": rng.normal(size=n) + signal,
            "feature_2": rng.normal(size=n) + signal * 0.5,
            "feature_3": rng.normal(size=n),
            "feature_4": rng.normal(size=n),
        }
    )
    y = pd.Series(y_values, name="alto_risco_evasao")
    return df, y


def _assert_valid_training_result(result: TrainingResult, *, n_samples: int, n_splits: int) -> None:
    assert isinstance(result, TrainingResult)
    assert len(result.fold_results) == n_splits
    assert len(result.fitted_feature_pipelines) == n_splits
    assert len(result.fitted_models) == n_splits

    # Toda amostra recebeu exatamente uma predição out-of-fold.
    assert result.predictions.shape == (n_samples,)
    assert (result.predictions != -1).all()
    assert set(np.unique(result.predictions)).issubset({0, 1})

    assert result.probabilities is not None
    assert result.probabilities.shape == (n_samples,)
    assert not np.isnan(result.probabilities).any()
    assert (result.probabilities >= 0).all() and (result.probabilities <= 1).all()

    expected_metric_keys = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert set(result.metrics) == expected_metric_keys
    assert set(result.metrics_std) == expected_metric_keys
    for value in result.metrics.values():
        assert 0.0 <= value <= 1.0

    assert result.params  # não vazio

    for fold in result.fold_results:
        assert fold.threshold_criterion in ("youden_j_constrained", "fallback")
        assert 0.0 <= fold.threshold <= 1.0
        assert fold.train_size > 0
        assert fold.calibration_size > 0
        assert fold.validation_size > 0


# --------------------------------------------------------------------------
# O treinamento é executável para os 5 modelos
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected_imbalance_strategy"),
    [
        ("arvore_de_decisao", "smote"),
        ("regressao_logistica", "class_weight"),
        ("svm", "class_weight"),
        ("rede_neural", "sample_weight"),
    ],
)
def test_trainer_trains_each_aligned_model(model_name, expected_imbalance_strategy) -> None:
    X, y = _sample_notebook_style_dataset()
    trainer = Trainer()

    result = trainer.train(model_name, X, y)

    _assert_valid_training_result(result, n_samples=len(y), n_splits=5)
    assert result.model_name == model_name
    assert result.imbalance_strategy == expected_imbalance_strategy


@pytest.mark.parametrize("variant", ["gaussian", "bernoulli", "complement"])
def test_trainer_trains_naive_bayes_all_variants(variant) -> None:
    X, y = _sample_naive_bayes_dataset()
    trainer = Trainer()

    result = trainer.train("naive_bayes", X, y, variant=variant)

    _assert_valid_training_result(result, n_samples=len(y), n_splits=5)
    assert result.imbalance_strategy == "smote"


def test_trainer_train_unknown_model_raises_key_error() -> None:
    X, y = _sample_notebook_style_dataset(n=20)
    with pytest.raises(KeyError):
        Trainer().train("modelo_que_nao_existe", X, y)


# --------------------------------------------------------------------------
# Cross-validation: StratifiedKFold, 5 folds, random_state, configurável
# --------------------------------------------------------------------------


def test_trainer_respects_custom_n_splits() -> None:
    X, y = _sample_notebook_style_dataset()
    trainer = Trainer(n_splits=3)

    result = trainer.train("regressao_logistica", X, y)

    _assert_valid_training_result(result, n_samples=len(y), n_splits=3)


def test_trainer_default_matches_project_protocol() -> None:
    trainer = Trainer()
    assert trainer.n_splits == 5
    assert trainer.shuffle is True
    assert trainer.random_state == 42


def test_trainer_is_reproducible_with_fixed_random_state() -> None:
    X, y = _sample_notebook_style_dataset()

    result_a = Trainer(random_state=123).train("regressao_logistica", X, y)
    result_b = Trainer(random_state=123).train("regressao_logistica", X, y)

    assert result_a.metrics == result_b.metrics
    assert np.array_equal(result_a.predictions, result_b.predictions)


# --------------------------------------------------------------------------
# Leakage: engenharia de features e threshold ajustados só no fold de treino
# --------------------------------------------------------------------------


def test_feature_pipeline_is_refit_independently_per_fold() -> None:
    X, y = _sample_notebook_style_dataset()
    result = Trainer().train("arvore_de_decisao", X, y)

    pipelines = result.fitted_feature_pipelines
    assert len({id(p) for p in pipelines}) == len(pipelines)  # 5 instâncias distintas


def test_validation_fold_metrics_do_not_use_calibration_labels_directly() -> None:
    """Checagem estrutural: `train_size + calibration_size` de cada fold
    bate com o tamanho do fold de treino do StratifiedKFold (80% de N),
    e `validation_size` bate com os 20% restantes — confirma que a
    calibração sai de dentro do treino, nunca da validação."""
    X, y = _sample_notebook_style_dataset(n=150)
    result = Trainer().train("regressao_logistica", X, y)

    for fold in result.fold_results:
        assert fold.train_size + fold.calibration_size in range(115, 125)  # ~80% de 150
        assert fold.validation_size in range(25, 35)  # ~20% de 150


# --------------------------------------------------------------------------
# Rede Neural: arquitetura dinâmica preservada dentro do Trainer
# --------------------------------------------------------------------------


def test_neural_network_hidden_layer_size_matches_common_features_count() -> None:
    X, y = _sample_notebook_style_dataset()
    result = Trainer().train("rede_neural", X, y)

    # common_features() produz 21 colunas -> (21+1)//2 = 11 neurônios.
    assert result.params["hidden_layer_sizes"] == (11,)


if __name__ == "__main__":
    test_functions = [
        obj
        for name, obj in list(globals().items())
        if name.startswith("test_") and not hasattr(obj, "pytestmark")
    ]
    failures = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"OK   - {test_fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL - {test_fn.__name__}: {exc}")
    total = len(test_functions)
    print(f"\n{total - failures}/{total} testes passaram (parametrizados exigem pytest).")
    sys.exit(1 if failures else 0)
