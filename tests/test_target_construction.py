"""Testes de `src.data.target` — construção de `taxa_evasao`/`alto_risco_evasao`."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.data.target import (
    TARGET_COLUMN,
    TAXA_EVASAO_COLUMN,
    THRESHOLD_20PCT_CUTOFF,
    build_target,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "QT_ING": [100, 50, 20, 5, 200],
            "QT_MAT": [90, 10, 15, 4, 150],
            "QT_SIT_DESVINCULADO": [10, 40, 5, 1, 50],
            "QT_SIT_TRANCADA": [0, 0, 0, 0, 0],
        }
    )


def test_median_split_formula_and_columns() -> None:
    df = _sample_df()
    result = build_target(df, strategy="median_split")

    assert TAXA_EVASAO_COLUMN in result.df.columns
    assert TARGET_COLUMN in result.df.columns
    assert result.strategy == "median_split"

    # taxa_evasao = QT_SIT_DESVINCULADO / (QT_MAT + QT_SIT_DESVINCULADO)
    first_row = result.df.iloc[0]
    expected = 10 / (90 + 10)
    assert first_row[TAXA_EVASAO_COLUMN] == pytest.approx(expected)


def test_median_split_filters_min_ingressantes_and_matriculados() -> None:
    df = _sample_df()
    result = build_target(df, strategy="median_split")

    # A linha com QT_ING=5 (< 10) deve ser filtrada.
    assert (result.df["QT_ING"] >= 10).all()
    assert (result.df["QT_MAT"] > 0).all()


def test_median_split_threshold_is_the_median() -> None:
    df = _sample_df()
    result = build_target(df, strategy="median_split")

    assert result.threshold == pytest.approx(float(result.df[TAXA_EVASAO_COLUMN].median()))
    # alto_risco_evasao == 1 exatamente quando taxa_evasao >= mediana
    expected_target = (result.df[TAXA_EVASAO_COLUMN] >= result.threshold).astype(int)
    assert (result.df[TARGET_COLUMN] == expected_target).all()


def test_threshold_20pct_formula_and_cutoff() -> None:
    df = _sample_df()
    result = build_target(df, strategy="threshold_20pct")

    assert result.strategy == "threshold_20pct"
    assert result.threshold == THRESHOLD_20PCT_CUTOFF == 20.0

    # taxa_evasao (%) = (QT_SIT_DESVINCULADO + QT_SIT_TRANCADA) / QT_ING * 100
    first_row = result.df.iloc[0]
    expected = (10 + 0) / 100 * 100
    assert first_row[TAXA_EVASAO_COLUMN] == pytest.approx(expected)


def test_threshold_20pct_does_not_filter_by_qt_mat() -> None:
    """A estratégia dos notebooks não exige QT_MAT>0 — diferença documentada."""
    df = _sample_df()
    df.loc[0, "QT_MAT"] = 0  # não deveria ser removido por esta estratégia
    result = build_target(df, strategy="threshold_20pct")

    assert len(result.df) == len(df)  # nenhuma linha removida por causa de QT_MAT


def test_unknown_strategy_raises_value_error() -> None:
    df = _sample_df()
    with pytest.raises(ValueError):
        build_target(df, strategy="nao_existe")  # type: ignore[arg-type]


def test_missing_required_columns_raises_value_error() -> None:
    df = pd.DataFrame({"QT_ING": [10, 20]})
    with pytest.raises(ValueError):
        build_target(df, strategy="median_split")
    with pytest.raises(ValueError):
        build_target(df, strategy="threshold_20pct")


def test_empty_result_after_filters_raises_value_error() -> None:
    df = pd.DataFrame(
        {
            "QT_ING": [1, 2],  # abaixo do mínimo de 10 para median_split
            "QT_MAT": [10, 10],
            "QT_SIT_DESVINCULADO": [1, 1],
            "QT_SIT_TRANCADA": [0, 0],
        }
    )
    with pytest.raises(ValueError):
        build_target(df, strategy="median_split")


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
