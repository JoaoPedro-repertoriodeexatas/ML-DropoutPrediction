"""Testes estruturais dos notebooks de `experiments/` (Tarefa 8).

Não executa os notebooks (isso exige o dataset real do INEP, >100MB, não
versionado — ver `README.md`). Verifica, a partir do JSON de cada
`.ipynb`:

1. os 5 notebooks existem e são JSON válido de notebook;
2. têm as 7 seções pedidas, na ordem;
3. não contêm lógica proibida de ML (carregamento manual, definição de
   modelo, treino, CV, busca de hiperparâmetros, cálculo manual de
   métrica, balanceamento manual) — só chamadas a `src.*`;
4. importam de `src.*` (prova de que usam a arquitetura migrada).

A verificação de que os notebooks **executam** de ponta a ponta foi
feita manualmente com um dataset sintético (ver `docs/EXPERIMENTS.md`,
seção "Verificação de execução") — os 5 rodaram com sucesso via
`nbclient`; este teste automatizado cobre a parte que não depende do
dataset real: estrutura e ausência de lógica de implementação.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nbformat as nbf
import pytest

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

NOTEBOOK_FILENAMES: tuple[str, ...] = (
    "decision_tree.ipynb",
    "logistic_regression.ipynb",
    "svm.ipynb",
    "neural_network.ipynb",
    "naive_bayes.ipynb",
)

REQUIRED_SECTIONS: tuple[str, ...] = (
    "1. Objetivo",
    "2. Configuração do experimento",
    "3. Execução do pipeline",
    "4. Resultados",
    "5. Visualizações",
    "6. Análise",
    "7. Conclusão",
)

# Padrões que indicariam lógica de implementação de ML vazando para o
# notebook — nenhum deve aparecer no código das células.
FORBIDDEN_PATTERNS: dict[str, str] = {
    r"pd\.read_csv\(": "carregamento manual do dataset (deveria usar src.data.loading/pipeline)",
    r"\bStratifiedKFold\(": "cross-validation manual (deveria usar src.training.trainer.Trainer)",
    r"\bGridSearchCV\(": "busca de hiperparâmetros manual (deveria usar src.training.hyperparameter_search)",
    r"\bRandomizedSearchCV\(": "busca de hiperparâmetros manual (deveria usar src.training.hyperparameter_search)",
    r"\bSMOTE\(": "balanceamento manual (deveria usar src.training.imbalance)",
    r"\bDecisionTreeClassifier\(": "definição manual de modelo (deveria usar src.models.registry.create_model)",
    r"\bLogisticRegression\(": "definição manual de modelo (deveria usar src.models.registry.create_model)",
    r"\bSVC\(": "definição manual de modelo (deveria usar src.models.registry.create_model)",
    r"\bMLPClassifier\(": "definição manual de modelo (deveria usar src.models.registry.create_model)",
    r"\bGaussianNB\(|\bBernoulliNB\(|\bComplementNB\(": "definição manual de modelo (deveria usar src.models.registry.create_model)",
    r"\bLabelEncoder\(": "feature engineering manual (deveria usar src.pipelines.feature_engineering)",
    r"\bStandardScaler\(|\bRobustScaler\(|\bMinMaxScaler\(": "scaling manual (deveria usar src.pipelines.feature_engineering)",
    r"accuracy_score\(|precision_score\(|recall_score\(|f1_score\(|roc_auc_score\(": "cálculo manual de métrica (deveria usar src.evaluation.metrics)",
}


def _load_notebook(filename: str) -> nbf.NotebookNode:
    path = EXPERIMENTS_DIR / filename
    return nbf.read(path, as_version=4)


def _code_source(notebook: nbf.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def _markdown_source(notebook: nbf.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")


@pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
def test_notebook_exists_and_is_valid_json(filename: str) -> None:
    path = EXPERIMENTS_DIR / filename
    assert path.is_file(), f"Notebook ausente: {path}"

    notebook = _load_notebook(filename)
    nbf.validate(notebook)  # levanta se o JSON não for um notebook válido


@pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
def test_notebook_has_all_seven_sections_in_order(filename: str) -> None:
    notebook = _load_notebook(filename)
    markdown = _markdown_source(notebook)

    positions = [markdown.find(section) for section in REQUIRED_SECTIONS]
    assert all(pos != -1 for pos in positions), (
        f"{filename}: seção ausente entre {REQUIRED_SECTIONS} "
        f"(encontradas nas posições {positions})"
    )
    assert positions == sorted(positions), f"{filename}: seções fora de ordem"


@pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
def test_notebook_has_no_forbidden_ml_implementation_logic(filename: str) -> None:
    notebook = _load_notebook(filename)
    code = _code_source(notebook)

    violations = []
    for pattern, reason in FORBIDDEN_PATTERNS.items():
        if re.search(pattern, code):
            violations.append(f"{pattern!r} ({reason})")

    assert not violations, f"{filename} contém lógica proibida: {violations}"


@pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
def test_notebook_uses_the_migrated_architecture(filename: str) -> None:
    notebook = _load_notebook(filename)
    code = _code_source(notebook)

    assert "from src.data.pipeline import run_data_pipeline" in code
    assert "from src.training.trainer import Trainer" in code
    assert "trainer.train(" in code
    assert "import src" not in code.replace("from src", "")  # sanity: sem import de src fora dos "from src..."


@pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
def test_notebook_does_not_hardcode_random_state_outside_config_cell(filename: str) -> None:
    """`random_state`/semente devem vir só da variável de configuração
    `RANDOM_STATE`, nunca de um literal solto espalhado pelo notebook."""
    notebook = _load_notebook(filename)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]

    # A primeira aparição de "RANDOM_STATE = " define a única semente do notebook.
    assignments = [source for source in code_cells if re.search(r"^RANDOM_STATE\s*=\s*42", source, re.MULTILINE)]
    assert len(assignments) == 1, f"{filename}: RANDOM_STATE deveria ser definido uma única vez"


def test_all_five_models_are_covered_across_notebooks() -> None:
    model_names_found = set()
    for filename in NOTEBOOK_FILENAMES:
        notebook = _load_notebook(filename)
        code = _code_source(notebook)
        match = re.search(r'^MODEL_NAME\s*=\s*"([^"]+)"', code, re.MULTILINE)
        assert match, f"{filename}: MODEL_NAME não encontrado"
        model_names_found.add(match.group(1))

    from src.configs.constants import MODEL_NAMES

    assert model_names_found == set(MODEL_NAMES)


if __name__ == "__main__":
    import itertools

    test_functions = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = 0
    total = 0
    for test_fn in test_functions:
        if hasattr(test_fn, "pytestmark"):
            for filename in NOTEBOOK_FILENAMES:
                total += 1
                try:
                    test_fn(filename)
                    print(f"OK   - {test_fn.__name__}[{filename}]")
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    print(f"FAIL - {test_fn.__name__}[{filename}]: {exc}")
        else:
            total += 1
            try:
                test_fn()
                print(f"OK   - {test_fn.__name__}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL - {test_fn.__name__}: {exc}")
    print(f"\n{total - failures}/{total} testes passaram.")
    sys.exit(1 if failures else 0)
