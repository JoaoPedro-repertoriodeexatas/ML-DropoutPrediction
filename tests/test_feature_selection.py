"""Testes de seleção de features: `select_numeric_columns` (Tarefa 3).

Os testes da engenharia de features específica de cada modelo
(`src.pipelines.feature_engineering`, Tarefa 4) vivem em
`tests/test_feature_engineering.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.pipelines.base_preprocessing import run_base_preprocessing, select_numeric_columns


def test_select_numeric_columns_drops_text_columns() -> None:
    df = pd.DataFrame(
        {
            "QT_ING": [10, 20],
            "NO_CURSO": ["Engenharia", "Direito"],
            "TAXA_CONCLUSAO": [80.0, 90.0],
        }
    )
    result = select_numeric_columns(df)

    assert set(result.columns) == {"QT_ING", "TAXA_CONCLUSAO"}
    assert "NO_CURSO" not in result.columns


def test_select_numeric_columns_keeps_all_numeric() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    result = select_numeric_columns(df)
    assert list(result.columns) == ["a", "b"]


def test_run_base_preprocessing_selects_numeric_by_default() -> None:
    df = pd.DataFrame(
        {
            "QT_ING": [10, 20],
            "NO_CURSO": ["Engenharia", "Direito"],
        }
    )
    result = run_base_preprocessing(df)
    assert "NO_CURSO" not in result.columns


def test_run_base_preprocessing_can_skip_numeric_restriction() -> None:
    df = pd.DataFrame(
        {
            "QT_ING": [10, 20],
            "NO_CURSO": ["Engenharia", "Direito"],
        }
    )
    result = run_base_preprocessing(df, restrict_to_numeric=False)
    assert "NO_CURSO" in result.columns


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
