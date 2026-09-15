"""Verifica que a nova árvore `src/` é importável e internamente consistente.

Não testa comportamento de dados/modelos (nada disso foi migrado ainda).
Serve como o "smoke test" pedido na Tarefa 2: garantir que criar a
estrutura não quebrou nada que já existia e que os novos pacotes podem
ser importados de ponta a ponta.

Execução:
    python -m pytest tests/test_imports.py -v
ou, sem pytest instalado:
    python tests/test_imports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_config_layer_imports() -> None:
    from src.configs import constants, paths, settings

    assert paths.PROJECT_ROOT.is_dir()
    assert paths.PROJECT_ROOT == PROJECT_ROOT
    assert settings.RANDOM_STATE == 42
    assert constants.TARGET_COLUMN == "ALTO_RISCO_EVASAO"


def test_legacy_paths_still_exist() -> None:
    """Estrutura legada que continua presente após a limpeza final
    (ver docs/FILE_CLEANUP.md).

    `LEGACY_DATA_DIR` não entra aqui: `data/` só continha
    `preprocessamento.py`/`__init__.py`, já totalmente substituídos por
    `src/data/`, e foi removido nessa limpeza — `LEGACY_DATA_DIR`
    continua definido em `src/configs/paths.py` apenas como candidato de
    localização do CSV bruto (mesma convenção de `analysis/audit/data.py`),
    não como diretório que deva existir.
    """
    from src.configs.paths import (
        LEGACY_ANALYSIS_DIR,
        LEGACY_MODELS_DIR,
        LEGACY_RESULTS_DIR,
    )

    for legacy_dir in (
        LEGACY_MODELS_DIR,
        LEGACY_ANALYSIS_DIR,
        LEGACY_RESULTS_DIR,
    ):
        assert legacy_dir.is_dir(), f"Diretório legado ausente: {legacy_dir}"


def test_legacy_model_dirs_registered() -> None:
    from src.configs.constants import MODEL_NAMES
    from src.configs.paths import LEGACY_MODEL_DIRS

    assert set(LEGACY_MODEL_DIRS) == set(MODEL_NAMES)
    for model_name, model_dir in LEGACY_MODEL_DIRS.items():
        assert model_dir.is_dir(), f"Pasta do modelo '{model_name}' ausente: {model_dir}"


def test_new_subpackages_import() -> None:
    import src.data
    import src.evaluation
    import src.models
    import src.pipelines
    import src.training
    import src.visualization

    for module in (
        src.data,
        src.evaluation,
        src.models,
        src.pipelines,
        src.training,
        src.visualization,
    ):
        assert module.__doc__, f"{module.__name__} deveria ter docstring de módulo"


def test_model_registry_has_all_canonical_models() -> None:
    from src.configs.constants import MODEL_NAMES
    from src.models.registry import MODEL_REGISTRY, get_model_spec

    assert set(MODEL_REGISTRY) == set(MODEL_NAMES)
    for name in MODEL_NAMES:
        spec = get_model_spec(name)
        assert spec.name == name


def test_feature_groups_have_no_known_leakage() -> None:
    from src.pipelines.feature_groups import (
        NEURAL_NETWORK_FEATURE_SET,
        SVM_FEATURE_SET,
        TREE_FEATURE_SET,
        assert_no_known_leakage,
    )

    for feature_set in (TREE_FEATURE_SET, SVM_FEATURE_SET, NEURAL_NETWORK_FEATURE_SET):
        assert_no_known_leakage(feature_set)  # não deve levantar

    assert "RAZAO_CONCLUSAO_EVASAO" not in TREE_FEATURE_SET


def test_common_feature_set_matches_audit_count() -> None:
    """21 features = 8 categóricas + 4 numéricas brutas + 9 derivadas."""
    from src.pipelines.feature_groups import COMMON_FEATURE_SET

    assert len(COMMON_FEATURE_SET) == 21


def test_experiments_and_scripts_packages_import() -> None:
    import experiments  # noqa: F401
    import scripts  # noqa: F401


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
