"""Testes de `src.data.loading` — carregamento do dataset bruto.

Não depende do CSV real do INEP (>100MB, não versionado — ver
`.gitignore`). Usa um CSV sintético minúsculo criado em um diretório
temporário, com o mesmo separador/encoding do arquivo real.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.data.loading import CSV_READ_KWARGS, load_raw_dataset


def _write_sample_csv(directory: Path) -> Path:
    """Cria um CSV minúsculo no mesmo formato do dataset real (`;`, latin1)."""
    csv_path = directory / "amostra.csv"
    conteudo = (
        "QT_ING;QT_MAT;NO_CURSO\n"
        "100;90;Engenharia da Computação\n"
        "50;10;Administração\n"
    )
    csv_path.write_text(conteudo, encoding="latin1")
    return csv_path


def test_csv_read_kwargs_match_dataset_format() -> None:
    assert CSV_READ_KWARGS["sep"] == ";"
    assert CSV_READ_KWARGS["encoding"] == "latin1"
    assert CSV_READ_KWARGS["low_memory"] is False


def test_load_raw_dataset_reads_explicit_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_sample_csv(Path(tmp))
        df = load_raw_dataset(csv_path)

    assert list(df.columns) == ["QT_ING", "QT_MAT", "NO_CURSO"]
    assert len(df) == 2
    assert df.loc[0, "QT_ING"] == 100


def test_load_raw_dataset_missing_path_raises_runtime_error() -> None:
    """Um caminho explícito que não existe deve falhar de forma clara.

    `pandas.read_csv` levanta `FileNotFoundError` (que é uma subclasse de
    `OSError`) para um arquivo inexistente; `load_raw_dataset` a
    encapsula em `RuntimeError` para dar contexto (caminho tentado).
    """
    with tempfile.TemporaryDirectory() as tmp:
        missing_path = Path(tmp) / "nao-existe.csv"
        with pytest.raises(RuntimeError):
            load_raw_dataset(missing_path)


def test_load_raw_dataset_no_path_resolves_via_config() -> None:
    """Sem `path`, deve tentar `resolve_dataset_path()` (config central).

    Não asseramos sucesso (o CSV real de >100MB normalmente não está
    presente neste ambiente) — só que a falha, quando ocorre, vem da
    ausência do dataset e não de um caminho hardcoded incorreto no
    próprio módulo de carregamento.
    """
    from src.configs.paths import DATASET_PATH_CANDIDATES

    try:
        df = load_raw_dataset()
    except FileNotFoundError as exc:
        for candidate in DATASET_PATH_CANDIDATES:
            assert str(candidate) in str(exc)
    else:
        assert not df.empty


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
