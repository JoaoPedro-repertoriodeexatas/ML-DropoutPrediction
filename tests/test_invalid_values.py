"""Testes de tratamento de valores inválidos: infinito e ausente."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.pipelines.base_preprocessing import (
    impute_missing_with_mode,
    impute_missing_with_zero,
    replace_infinite_with_nan,
    run_base_preprocessing,
)


def test_replace_infinite_with_nan_handles_positive_and_negative_inf() -> None:
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0]})
    result = replace_infinite_with_nan(df)

    assert result["a"].isna().sum() == 2
    assert not np.isinf(result["a"]).any()
    assert result["a"].iloc[0] == 1.0
    assert result["a"].iloc[3] == 4.0


def test_replace_infinite_with_nan_does_not_mutate_input() -> None:
    df = pd.DataFrame({"a": [1.0, np.inf]})
    replace_infinite_with_nan(df)
    assert np.isinf(df["a"]).any()  # original inalterado


def test_impute_missing_with_zero_fills_all_nan() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    result = impute_missing_with_zero(df)
    assert result["a"].tolist() == [1.0, 0.0, 3.0]


def test_impute_missing_with_mode_uses_most_frequent_value() -> None:
    df = pd.DataFrame({"a": [1, 1, 1, np.nan]})
    result = impute_missing_with_mode(df)
    assert result["a"].iloc[3] == 1


def test_impute_missing_with_mode_falls_back_to_zero_when_all_nan() -> None:
    df = pd.DataFrame({"a": [np.nan, np.nan]})
    result = impute_missing_with_mode(df)
    assert (result["a"] == 0).all()


def test_run_base_preprocessing_end_to_end_zero_strategy() -> None:
    df = pd.DataFrame(
        {
            "QT_ING": [10, 20, np.inf],
            "TAXA_CONCLUSAO": [80.0, np.nan, -np.inf],
            "NO_CURSO": ["Engenharia", "Direito", "Medicina"],  # não numérica
        }
    )
    result = run_base_preprocessing(df, missing_strategy="zero")

    assert "NO_CURSO" not in result.columns  # restrição a numéricas
    assert not result.isna().any().any()  # nenhum NaN restante
    assert not np.isinf(result.to_numpy(dtype=float)).any()  # nenhum inf restante
    assert result.loc[2, "QT_ING"] == 0.0  # inf -> NaN -> 0
    assert result.loc[1, "TAXA_CONCLUSAO"] == 0.0  # NaN -> 0


def test_run_base_preprocessing_end_to_end_mode_strategy() -> None:
    df = pd.DataFrame({"QT_ING": [10, 10, np.nan]})
    result = run_base_preprocessing(df, missing_strategy="mode")
    assert result["QT_ING"].iloc[2] == 10


def test_run_base_preprocessing_rejects_unknown_missing_strategy() -> None:
    df = pd.DataFrame({"QT_ING": [10, 20]})
    with pytest.raises(ValueError):
        run_base_preprocessing(df, missing_strategy="media")  # type: ignore[arg-type]


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
