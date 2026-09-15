"""Testes de integração: `src.data.pipeline` (orquestração completa) e
`src.data.validation` (comparação de compatibilidade).

Não usa o CSV real do INEP — gera um CSV sintético pequeno cobrindo as
colunas exigidas pelas duas estratégias de alvo, para exercitar as
quatro etapas (Loading → Target → Leakage Removal → Base Preprocessing)
de ponta a ponta.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.leakage import KNOWN_LEAKY_DERIVED_FEATURES, TARGET_SOURCE_COLUMNS
from src.data.pipeline import run_data_pipeline
from src.data.target import TARGET_COLUMN
from src.data.validation import compare_preprocessing_outputs


def _write_sample_csv(directory: Path) -> Path:
    csv_path = directory / "amostra.csv"
    rows = []
    # 12 registros com QT_ING/QT_MAT variados para sobreviver aos filtros
    # de ambas as estratégias e gerar as duas classes do alvo.
    for i in range(12):
        qt_ing = 20 + i * 5
        qt_mat = 15 + i * 3
        qt_sit_desvinculado = i  # cresce de 0 a 11
        qt_sit_trancada = 0
        rows.append(f"{qt_ing};{qt_mat};{qt_sit_desvinculado};{qt_sit_trancada};{80.0 + i}")
    header = "QT_ING;QT_MAT;QT_SIT_DESVINCULADO;QT_SIT_TRANCADA;TAXA_CONCLUSAO"
    csv_path.write_text("\n".join([header, *rows]), encoding="latin1")
    return csv_path


def test_run_data_pipeline_end_to_end_median_split() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_sample_csv(Path(tmp))
        result = run_data_pipeline(dataset_path=csv_path, target_strategy="median_split")

    assert result.target_strategy == "median_split"
    assert result.n_records_final <= result.n_records_raw
    assert TARGET_COLUMN not in result.X.columns  # alvo nunca vaza para X
    assert len(result.y) == result.n_records_final
    assert set(result.X.columns) & set(TARGET_SOURCE_COLUMNS) == set()
    assert set(result.X.columns) & set(KNOWN_LEAKY_DERIVED_FEATURES) == set()
    assert 0.0 <= result.positive_rate <= 1.0


def test_run_data_pipeline_end_to_end_threshold_20pct() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_sample_csv(Path(tmp))
        result = run_data_pipeline(dataset_path=csv_path, target_strategy="threshold_20pct")

    assert result.target_strategy == "threshold_20pct"
    assert result.target_threshold == 20.0
    assert TARGET_COLUMN not in result.X.columns


def test_run_data_pipeline_reports_removed_leakage_columns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_sample_csv(Path(tmp))
        result = run_data_pipeline(dataset_path=csv_path, target_strategy="median_split")

    assert "QT_SIT_DESVINCULADO" in result.removed_leakage_columns["origem_do_alvo"]


def test_compare_preprocessing_outputs_detects_identical_pipelines() -> None:
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.Series([0, 1, 0], name="alto_risco_evasao")

    report = compare_preprocessing_outputs(X, y, X.copy(), y.copy())

    assert report.is_fully_compatible
    assert report.n_records_old == report.n_records_new == 3
    assert report.features_only_in_old == []
    assert report.features_only_in_new == []


def test_compare_preprocessing_outputs_detects_feature_name_divergence() -> None:
    X_old = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    X_new = pd.DataFrame({"a": [1, 2], "c": [3, 4]})
    y = pd.Series([0, 1], name="alto_risco_evasao")

    report = compare_preprocessing_outputs(X_old, y, X_new, y)

    assert not report.is_fully_compatible
    assert report.matches["feature_names"] is False
    assert report.features_only_in_old == ["b"]
    assert report.features_only_in_new == ["c"]


def test_compare_preprocessing_outputs_detects_record_count_divergence() -> None:
    X_old = pd.DataFrame({"a": [1, 2, 3]})
    X_new = pd.DataFrame({"a": [1, 2]})
    y_old = pd.Series([0, 1, 0])
    y_new = pd.Series([0, 1])

    report = compare_preprocessing_outputs(X_old, y_old, X_new, y_new)

    assert not report.is_fully_compatible
    assert report.matches["records"] is False
    assert "DIVERGE" in report.summary()


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
