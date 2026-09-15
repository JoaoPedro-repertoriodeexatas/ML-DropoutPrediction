"""Testes de `src.data.leakage` — remoção de colunas com risco de vazamento."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.leakage import (
    IDENTIFIER_COLUMNS,
    KNOWN_LEAKY_DERIVED_FEATURES,
    LEAKAGE_PREFIXES,
    TARGET_SOURCE_COLUMNS,
    explain_removed_columns,
    leakage_columns_for,
    remove_leakage_columns,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "NO_CURSO": ["Engenharia"],  # grupo 1: identificador
            "QT_SIT_DESVINCULADO": [10],  # grupo 2: origem do alvo
            "taxa_evasao": [0.2],  # grupo 2: origem do alvo (nome legado)
            "QT_MAT_FINANC": [5],  # grupo 3: prefixo QT_MAT
            "QT_CONC": [3],  # grupo 3: prefixo QT_CONC
            "RAZAO_CONCLUSAO_EVASAO": [1.5],  # grupo 4: feature vazada conhecida
            "TAXA_CONCLUSAO": [80.0],  # feature legítima (não deve ser removida)
            "QT_ING": [100],  # feature legítima (não deve ser removida)
        }
    )


def test_leakage_columns_for_covers_all_four_groups() -> None:
    df = _sample_df()
    removed = leakage_columns_for(df.columns)

    assert "NO_CURSO" in removed  # grupo 1
    assert "QT_SIT_DESVINCULADO" in removed  # grupo 2
    assert "taxa_evasao" in removed  # grupo 2
    assert "QT_MAT_FINANC" in removed  # grupo 3 (prefixo)
    assert "QT_CONC" in removed  # grupo 3 (prefixo)
    assert "RAZAO_CONCLUSAO_EVASAO" in removed  # grupo 4

    # Features legítimas nunca devem ser marcadas para remoção.
    assert "TAXA_CONCLUSAO" not in removed
    assert "QT_ING" not in removed


def test_remove_leakage_columns_keeps_legitimate_features() -> None:
    df = _sample_df()
    cleaned = remove_leakage_columns(df)

    assert set(cleaned.columns) == {"TAXA_CONCLUSAO", "QT_ING"}


def test_remove_leakage_columns_does_not_mutate_input() -> None:
    df = _sample_df()
    original_columns = list(df.columns)
    remove_leakage_columns(df)
    assert list(df.columns) == original_columns


def test_prefix_group_can_be_disabled_for_inspection_only() -> None:
    df = _sample_df()
    cleaned = remove_leakage_columns(df, include_prefix_group=False)

    # Sem o grupo de prefixos, QT_MAT_FINANC e QT_CONC sobrevivem.
    assert "QT_MAT_FINANC" in cleaned.columns
    assert "QT_CONC" in cleaned.columns
    # Os outros três grupos continuam sendo removidos.
    assert "NO_CURSO" not in cleaned.columns
    assert "QT_SIT_DESVINCULADO" not in cleaned.columns
    assert "RAZAO_CONCLUSAO_EVASAO" not in cleaned.columns


def test_absent_columns_are_silently_ignored() -> None:
    df = pd.DataFrame({"QT_ING": [1, 2], "TAXA_CONCLUSAO": [10.0, 20.0]})
    cleaned = remove_leakage_columns(df)  # nenhuma coluna de vazamento presente
    assert list(cleaned.columns) == ["QT_ING", "TAXA_CONCLUSAO"]


def test_explain_removed_columns_groups_by_reason() -> None:
    df = _sample_df()
    explanation = explain_removed_columns(df.columns)

    assert explanation["identificadores"] == ["NO_CURSO"]
    assert set(explanation["origem_do_alvo"]) == {"QT_SIT_DESVINCULADO", "taxa_evasao"}
    assert explanation["features_derivadas_vazadas"] == ["RAZAO_CONCLUSAO_EVASAO"]
    assert set(explanation["prefixos_matricula_conclusao"]) == {"QT_MAT_FINANC", "QT_CONC"}


def test_known_leaky_feature_is_the_one_found_in_the_audit() -> None:
    """Trava de regressão: a feature identificada em ARCHITECTURE_AUDIT.md
    (seção 5) como vazamento crítico continua na lista central."""
    assert "RAZAO_CONCLUSAO_EVASAO" in KNOWN_LEAKY_DERIVED_FEATURES


def test_target_source_columns_cover_both_naming_conventions() -> None:
    """A lista cobre tanto a convenção dos notebooks (maiúsculas) quanto
    a de `data/preprocessamento.py` (minúsculas)."""
    assert "TAXA_EVASAO" in TARGET_SOURCE_COLUMNS
    assert "taxa_evasao" in TARGET_SOURCE_COLUMNS
    assert "ALTO_RISCO_EVASAO" in TARGET_SOURCE_COLUMNS
    assert "alto_risco_evasao" in TARGET_SOURCE_COLUMNS


def test_leakage_prefixes_match_naive_bayes_pipeline() -> None:
    assert LEAKAGE_PREFIXES == ("QT_MAT", "QT_CONC")


def test_identifier_columns_non_empty() -> None:
    assert len(IDENTIFIER_COLUMNS) > 0


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
