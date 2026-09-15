"""Testes de `src.pipelines.feature_engineering` (Tarefa 4).

Cobrem os cinco requisitos explícitos da Tarefa 4:

1. cada modelo recebe as features esperadas;
2. nenhuma feature proibida entra;
3. não existem NaNs inesperados;
4. o número de features é determinístico;
5. transformações são reproduzíveis.

Usa dados sintéticos pequenos (sem depender do CSV real do INEP),
seguindo o mesmo padrão dos testes da Tarefa 3.
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

from src.configs.constants import COMMON_FEATURES
from src.configs.settings import DECISION_TREE_SELECT_K_BEST
from src.data.leakage import KNOWN_LEAKY_DERIVED_FEATURES, TARGET_SOURCE_COLUMNS
from src.pipelines.feature_engineering import (
    CommonFeatureSelector,
    CorrelationThresholdSelector,
    FeatureEngineer,
    HighCorrelationRemover,
    TreeFeatureAugmenter,
)


def _balanced_target(n: int, seed: int) -> pd.Series:
    """Vetor binário com exatamente n//2 positivos, embaralhado deterministicamente."""
    values = np.array([0, 1] * (n // 2))
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return pd.Series(values, name="alto_risco_evasao")


def _sample_notebook_style_input(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Simula o X que chega em `CommonFeatureSelector`: colunas brutas do
    censo (categóricas já como códigos numéricos, como no CSV real do
    INEP) mais as colunas-fonte das 9 features derivadas."""
    rng = np.random.default_rng(seed)
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
            "QT_ING": rng.integers(10, 500, n),
            "QT_MAT": rng.integers(10, 500, n),
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
    y = _balanced_target(n, seed=seed + 1)
    return df, y


def _sample_naive_bayes_input(n: int = 30, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Frame com uma coluna de variância zero e um par altamente
    correlacionado, para exercitar `VarianceThreshold` e
    `HighCorrelationRemover` de forma não aleatória."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "col_a": base,
            "col_b": base + rng.normal(scale=1e-6, size=n),  # correlação ~1.0 com col_a
            "col_c": rng.normal(size=n),
            "col_zero_var": np.zeros(n),
        }
    )
    y = _balanced_target(n, seed=seed + 1)
    return df, y


# --------------------------------------------------------------------------
# 1. Cada modelo recebe as features esperadas
# --------------------------------------------------------------------------


def test_common_features_output_matches_declared_common_features() -> None:
    X, y = _sample_notebook_style_input()
    selector = CommonFeatureSelector()
    result = selector.fit_transform(X, y)

    assert list(result.columns) == list(COMMON_FEATURES)
    assert len(result.columns) == 21


def test_for_logistic_regression_keeps_all_common_features() -> None:
    X, y = _sample_notebook_style_input()
    pipeline = FeatureEngineer().for_logistic_regression()
    result = pipeline.fit_transform(X, y)

    assert set(result.columns) == set(COMMON_FEATURES)


def test_for_neural_network_keeps_all_common_features_without_scaling() -> None:
    X, y = _sample_notebook_style_input()
    pipeline = FeatureEngineer().for_neural_network()
    result = pipeline.fit_transform(X, y)

    assert set(result.columns) == set(COMMON_FEATURES)
    # Sem scaling: valores de QT_ING devem permanecer na escala original (10-500).
    assert result["QT_ING"].min() >= 10
    assert result["QT_ING"].max() <= 500


def test_for_neural_network_scale_true_applies_standard_scaler() -> None:
    X, y = _sample_notebook_style_input()
    pipeline = FeatureEngineer(random_state=0).for_neural_network(scale=True)
    result = pipeline.fit_transform(X, y)

    assert set(result.columns) == set(COMMON_FEATURES)
    # Com StandardScaler, média ~0 e desvio padrão ~1 por coluna.
    assert abs(result["QT_ING"].mean()) < 1e-6
    assert result["QT_ING"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_for_decision_tree_selects_exactly_k_best_features() -> None:
    X, y = _sample_notebook_style_input()
    pipeline = FeatureEngineer().for_decision_tree()
    result = pipeline.fit_transform(X, y)

    assert result.shape[1] == DECISION_TREE_SELECT_K_BEST
    # Todas as selecionadas devem vir do pool de 30 candidatas (21 comuns + 9 avançadas).
    tree_augmented_columns = set(COMMON_FEATURES) | {
        "EAD_PREDOMINANTE", "FINANC_ALTO", "CURSO_PEQUENO", "RAZAO_FEM_18_24",
        "TAMANHO_CATEGORIA", "RAZAO_FINANC_EAD", "TAXA_PREENCHIMENTO",
        "RAZAO_INSCRITOS_VAGAS", "CURSO_COMPETITIVO",
    }
    assert set(result.columns).issubset(tree_augmented_columns)


def test_for_svm_selects_subset_of_common_features() -> None:
    X, y = _sample_notebook_style_input()
    pipeline = FeatureEngineer().for_svm()
    result = pipeline.fit_transform(X, y)

    assert set(result.columns).issubset(set(COMMON_FEATURES))
    assert result.shape[1] >= 1  # nunca fica vazio (fallback do notebook original)


def test_tree_feature_augmenter_adds_nine_columns() -> None:
    X, y = _sample_notebook_style_input()
    common = CommonFeatureSelector().fit_transform(X, y)
    augmented = TreeFeatureAugmenter().fit_transform(common)

    new_columns = set(augmented.columns) - set(common.columns)
    assert new_columns == {
        "EAD_PREDOMINANTE", "FINANC_ALTO", "CURSO_PEQUENO", "RAZAO_FEM_18_24",
        "TAMANHO_CATEGORIA", "RAZAO_FINANC_EAD", "TAXA_PREENCHIMENTO",
        "RAZAO_INSCRITOS_VAGAS", "CURSO_COMPETITIVO",
    }


def test_for_naive_bayes_gaussian_applies_variance_and_correlation_filters() -> None:
    X, y = _sample_naive_bayes_input()
    pipeline = FeatureEngineer().for_naive_bayes("gaussian")
    result = pipeline.fit_transform(X, y)

    assert "col_zero_var" not in result.columns  # removida por VarianceThreshold
    # col_a e col_b são quase idênticas: só uma das duas deve sobreviver.
    assert len({"col_a", "col_b"} & set(result.columns)) == 1
    assert "col_c" in result.columns


def test_for_naive_bayes_bernoulli_output_is_binary() -> None:
    X, y = _sample_naive_bayes_input()
    pipeline = FeatureEngineer().for_naive_bayes("bernoulli", binarize_threshold=0.0)
    result = pipeline.fit_transform(X, y)

    assert set(np.unique(result.to_numpy())).issubset({0.0, 1.0})


def test_for_naive_bayes_complement_scales_to_unit_interval() -> None:
    X, y = _sample_naive_bayes_input()
    pipeline = FeatureEngineer().for_naive_bayes("complement")
    result = pipeline.fit_transform(X, y)

    assert set(result.columns) == set(X.columns)  # MinMaxScaler não remove colunas
    assert (result.to_numpy() >= 0).all()
    assert (result.to_numpy() <= 1).all()


def test_for_naive_bayes_unknown_variant_raises_value_error() -> None:
    with pytest.raises(ValueError):
        FeatureEngineer().for_naive_bayes("nao_existe")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 2. Nenhuma feature proibida entra
# --------------------------------------------------------------------------


def test_known_leaky_feature_never_appears_in_decision_tree_output() -> None:
    X, y = _sample_notebook_style_input()
    result = FeatureEngineer().for_decision_tree().fit_transform(X, y)

    for leaky_feature in KNOWN_LEAKY_DERIVED_FEATURES:
        assert leaky_feature not in result.columns


@pytest.mark.parametrize(
    "pipeline_factory",
    [
        lambda engineer: engineer.for_decision_tree(),
        lambda engineer: engineer.for_logistic_regression(),
        lambda engineer: engineer.for_svm(),
        lambda engineer: engineer.for_neural_network(),
    ],
)
def test_target_source_columns_never_appear_in_any_pipeline_output(pipeline_factory) -> None:
    X, y = _sample_notebook_style_input()
    engineer = FeatureEngineer()
    result = pipeline_factory(engineer).fit_transform(X, y)

    forbidden = set(TARGET_SOURCE_COLUMNS) | set(KNOWN_LEAKY_DERIVED_FEATURES)
    assert set(result.columns) & forbidden == set()


def test_common_feature_selector_raises_on_missing_required_columns() -> None:
    incomplete_df = pd.DataFrame({"QT_ING": [10, 20]})
    with pytest.raises(ValueError):
        CommonFeatureSelector().fit(incomplete_df)


# --------------------------------------------------------------------------
# 3. Não existem NaNs inesperados
# --------------------------------------------------------------------------


def test_common_features_never_produce_nan_even_with_zero_denominators() -> None:
    X, y = _sample_notebook_style_input()
    # Zera denominadores usados nas 9 razões derivadas (QT_MAT, QT_VG_TOTAL, QT_ING).
    X.loc[0, "QT_MAT"] = 0
    X.loc[1, "QT_VG_TOTAL"] = 0
    X.loc[2, "QT_ING"] = 0

    result = CommonFeatureSelector().fit_transform(X, y)
    assert not result.isna().any().any()


def test_all_pipelines_produce_no_nan() -> None:
    X, y = _sample_notebook_style_input()
    engineer = FeatureEngineer()

    for pipeline in (
        engineer.for_decision_tree(),
        engineer.for_logistic_regression(),
        engineer.for_svm(),
        engineer.for_neural_network(),
    ):
        result = pipeline.fit_transform(X, y)
        assert not result.isna().any().any(), f"NaN inesperado em {pipeline.steps[-1][0]}"


def test_naive_bayes_pipelines_produce_no_nan() -> None:
    X, y = _sample_naive_bayes_input()
    engineer = FeatureEngineer()

    for variant in ("gaussian", "bernoulli", "complement"):
        result = engineer.for_naive_bayes(variant).fit_transform(X, y)
        assert not pd.DataFrame(result).isna().any().any()


# --------------------------------------------------------------------------
# 4. O número de features é determinístico
# --------------------------------------------------------------------------


def test_decision_tree_feature_count_is_deterministic_across_calls() -> None:
    X, y = _sample_notebook_style_input()
    pipeline = FeatureEngineer().for_decision_tree()

    result_first = pipeline.fit_transform(X, y)
    result_second = FeatureEngineer().for_decision_tree().fit_transform(X, y)

    assert result_first.shape[1] == result_second.shape[1] == DECISION_TREE_SELECT_K_BEST


def test_common_feature_count_is_always_21_regardless_of_input_size() -> None:
    for n in (20, 40, 80):
        X, y = _sample_notebook_style_input(n=n, seed=n)
        result = CommonFeatureSelector().fit_transform(X, y)
        assert result.shape[1] == 21


# --------------------------------------------------------------------------
# 5. Transformações são reproduzíveis
# --------------------------------------------------------------------------


def test_common_feature_selector_is_reproducible() -> None:
    X, y = _sample_notebook_style_input()

    result_a = CommonFeatureSelector().fit_transform(X, y)
    result_b = CommonFeatureSelector().fit_transform(X, y)

    pd.testing.assert_frame_equal(result_a, result_b)


def test_decision_tree_pipeline_is_reproducible_with_fixed_random_state() -> None:
    """`mutual_info_classif` tem componente aleatório interno; sem fixar
    `random_state`, `SelectKBest` poderia variar a seleção entre execuções."""
    X, y = _sample_notebook_style_input()

    result_a = FeatureEngineer(random_state=123).for_decision_tree().fit_transform(X, y)
    result_b = FeatureEngineer(random_state=123).for_decision_tree().fit_transform(X, y)

    assert list(result_a.columns) == list(result_b.columns)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_svm_pipeline_is_reproducible() -> None:
    X, y = _sample_notebook_style_input()

    result_a = FeatureEngineer().for_svm().fit_transform(X, y)
    result_b = FeatureEngineer().for_svm().fit_transform(X, y)

    pd.testing.assert_frame_equal(result_a, result_b)


def test_naive_bayes_pipeline_is_reproducible() -> None:
    X, y = _sample_naive_bayes_input()

    result_a = FeatureEngineer().for_naive_bayes("gaussian").fit_transform(X, y)
    result_b = FeatureEngineer().for_naive_bayes("gaussian").fit_transform(X, y)

    pd.testing.assert_frame_equal(pd.DataFrame(result_a), pd.DataFrame(result_b))


def test_fit_on_train_transform_on_validation_does_not_refit() -> None:
    """Verifica o contrato de leakage: `transform` de um fold de validação
    usa os parâmetros aprendidos no treino, não recalcula nada."""
    X_train, y_train = _sample_notebook_style_input(n=40, seed=10)
    X_val, _ = _sample_notebook_style_input(n=10, seed=99)

    selector = CorrelationThresholdSelector(threshold=0.05)
    common_train = CommonFeatureSelector().fit_transform(X_train, y_train)
    selector.fit(common_train, y_train)
    selected_after_fit = list(selector.selected_features_)

    common_val = CommonFeatureSelector().fit(X_train, y_train).transform(X_val)
    transformed_val = selector.transform(common_val)

    # As colunas selecionadas não mudam ao transformar dados novos.
    assert list(transformed_val.columns) == selected_after_fit
    assert selector.selected_features_ == selected_after_fit


def test_high_correlation_remover_is_reproducible_and_deterministic_count() -> None:
    X, y = _sample_naive_bayes_input()

    remover_a = HighCorrelationRemover(threshold=0.90).fit(X, y)
    remover_b = HighCorrelationRemover(threshold=0.90).fit(X, y)

    assert remover_a.columns_to_keep_ == remover_b.columns_to_keep_
    assert len(remover_a.columns_to_keep_) == 3  # col_a-ou-col_b, col_c, col_zero_var


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
