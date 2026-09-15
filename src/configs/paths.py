"""Caminhos centrais do projeto.

Todos os caminhos são resolvidos a partir de ``PROJECT_ROOT`` com
``pathlib.Path``, para funcionar de forma idêntica em qualquer sistema
operacional e independente do diretório de onde o código é executado.

Os caminhos ``LEGACY_*`` apontam para a estrutura antiga (notebooks,
scripts e resultados já existentes) e são somente leitura a partir daqui:
nada neste módulo cria, move ou apaga arquivos dessas pastas. Os
caminhos ``NEW_*`` descrevem onde a futura pipeline unificada (fora do
escopo desta etapa) deverá escrever seus artefatos.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Raiz do projeto
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Estrutura nova (esta auditoria/etapa) — código em `src/`
# --------------------------------------------------------------------------

SRC_DIR: Path = PROJECT_ROOT / "src"
CONFIGS_DIR: Path = SRC_DIR / "configs"
DATA_PKG_DIR: Path = SRC_DIR / "data"
PIPELINES_DIR: Path = SRC_DIR / "pipelines"
MODELS_PKG_DIR: Path = SRC_DIR / "models"
TRAINING_DIR: Path = SRC_DIR / "training"
EVALUATION_DIR: Path = SRC_DIR / "evaluation"
VISUALIZATION_DIR: Path = SRC_DIR / "visualization"

EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Onde a futura pipeline unificada deverá escrever seus artefatos.
# Definido aqui apenas como referência — o diretório não é criado por
# este módulo (ver ARCHITECTURE_AUDIT.md, seção 7, "artifacts/").
NEW_ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

# --------------------------------------------------------------------------
# Estrutura legada (não tocar — apenas referenciar)
# --------------------------------------------------------------------------

# `data/` (que só continha `preprocessamento.py`, já substituído por
# `src/data/`) foi removido na limpeza final (docs/FILE_CLEANUP.md).
# `LEGACY_DATA_DIR` continua definido só como candidato de localização
# do CSV bruto abaixo — não se assume mais que o diretório exista.
LEGACY_DATA_DIR: Path = PROJECT_ROOT / "data"
LEGACY_MODELS_DIR: Path = PROJECT_ROOT / "models"
LEGACY_ANALYSIS_DIR: Path = PROJECT_ROOT / "analysis"
LEGACY_RESULTS_DIR: Path = PROJECT_ROOT / "results"
LEGACY_REPORT_DIR: Path = PROJECT_ROOT / "report"
LEGACY_SLIDES_DIR: Path = PROJECT_ROOT / "slides"

LEGACY_MODEL_DIRS: dict[str, Path] = {
    "arvore_de_decisao": LEGACY_MODELS_DIR / "arvore_de_decisao",
    "regressao_logistica": LEGACY_MODELS_DIR / "regrassao_logisticca",
    "svm": LEGACY_MODELS_DIR / "svm",
    "rede_neural": LEGACY_MODELS_DIR / "Redes_Neurais",
    "naive_bayes": LEGACY_MODELS_DIR / "naive_bayes",
}

# --------------------------------------------------------------------------
# Dataset bruto (INEP) — mesmos candidatos usados hoje em
# `data/preprocessamento.py` e `analysis/audit/data.py`, centralizados
# aqui para que as duas cópias não precisem mais divergir no futuro.
# --------------------------------------------------------------------------

DATASET_FILENAME: str = "MICRODADOS_CADASTRO_CURSOS_2024.CSV"

DATASET_PATH_CANDIDATES: tuple[Path, ...] = (
    PROJECT_ROOT / DATASET_FILENAME,
    LEGACY_DATA_DIR / DATASET_FILENAME,
    Path.home()
    / "Downloads"
    / "microdados_censo_da_educacao_superior_2024"
    / "dados"
    / DATASET_FILENAME,
)


def resolve_dataset_path() -> Path:
    """Retorna o primeiro caminho existente entre ``DATASET_PATH_CANDIDATES``.

    Não realiza nenhuma leitura do arquivo — apenas resolve o caminho.
    Levanta ``FileNotFoundError`` se nenhum candidato existir, com a
    mesma mensagem informativa hoje duplicada em `data/preprocessamento.py`
    e `analysis/audit/data.py`.
    """
    for candidate in DATASET_PATH_CANDIDATES:
        if candidate.exists():
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in DATASET_PATH_CANDIDATES)
    raise FileNotFoundError(
        f"Arquivo {DATASET_FILENAME} não encontrado. Caminhos verificados:\n{checked}"
    )
