"""Nomes de colunas, listas de features e nomes de arquivo de saída.

Estas listas hoje existem, quase idênticas, em pelo menos 5 lugares do
repositório (4 notebooks + `analysis/audit/data.py`) — ver
`ARCHITECTURE_AUDIT.md`, seção 2.1 (problema A1). Elas são reproduzidas
aqui apenas como **dados de referência** (nomes, não transformações) para
a futura migração; nenhuma leitura de CSV ou cálculo acontece neste
módulo.

``ADVANCED_TREE_FEATURES`` reproduz a lista de features "avançadas" do
notebook da Árvore de Decisão, **excluindo deliberadamente**
``RAZAO_CONCLUSAO_EVASAO`` — a feature identificada como vazamento direto
de alvo em `ARCHITECTURE_AUDIT.md`, seção 5.

As regras de remoção de vazamento (incluindo a lista de features vazadas
conhecidas) foram centralizadas em ``src.data.leakage`` a partir da
Tarefa 3 — ver `docs/DATA_PIPELINE.md`. Este módulo não duplica mais
essas listas; importe de ``src.data.leakage`` quando precisar delas.

Nota de nomenclatura: ``TARGET_COLUMN`` abaixo usa o nome em maiúsculas
(``ALTO_RISCO_EVASAO``) porque é a convenção dos 4 notebooks e do grupo
``COMMON_FEATURES`` (Tarefa 2). A nova camada de dados (``src.data``,
Tarefa 3) padroniza em ``snake_case`` (``alto_risco_evasao``,
``taxa_evasao``), como pedido explicitamente na Tarefa 3 e alinhado à
convenção já usada em `data/preprocessamento.py`. As duas convenções
coexistem até a unificação dos pipelines (ver ARCHITECTURE_AUDIT.md,
seção 8, fase 5).
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Alvo
# --------------------------------------------------------------------------

TARGET_COLUMN: str = "ALTO_RISCO_EVASAO"

# --------------------------------------------------------------------------
# Features comuns aos 4 notebooks (Árvore, Regressão Logística, SVM,
# Rede Neural) — grupo `common_features` da auditoria.
# --------------------------------------------------------------------------

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_REDE",
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "TP_DIMENSAO",
    "IN_GRATUITO",
    "CO_CINE_AREA_GERAL",
)

RAW_NUMERIC_FEATURES: tuple[str, ...] = (
    "QT_ING",
    "QT_MAT",
    "QT_VG_TOTAL",
    "QT_INSCRITO_TOTAL",
)

DERIVED_FEATURES: tuple[str, ...] = (
    "TAXA_CONCLUSAO",
    "RAZAO_ING_MAT",
    "PROPORCAO_EAD",
    "PROPORCAO_NOTURNO",
    "INDICE_FINANCIAMENTO",
    "PROPORCAO_FIES",
    "PROPORCAO_PROUNIP",
    "PROPORCAO_18_24",
    "PROPORCAO_FEM",
)

COMMON_FEATURES: tuple[str, ...] = (
    CATEGORICAL_FEATURES + RAW_NUMERIC_FEATURES + DERIVED_FEATURES
)

# --------------------------------------------------------------------------
# Features específicas por modelo (ver ARCHITECTURE_AUDIT.md, seção 4.2)
# --------------------------------------------------------------------------

# Features "avançadas" exclusivas do notebook da Árvore de Decisão,
# EXCLUINDO `RAZAO_CONCLUSAO_EVASAO` (vazamento de alvo).
ADVANCED_TREE_FEATURES: tuple[str, ...] = (
    "EAD_PREDOMINANTE",
    "FINANC_ALTO",
    "CURSO_PEQUENO",
    "RAZAO_FEM_18_24",
    "TAMANHO_CATEGORIA",
    "RAZAO_FINANC_EAD",
    "TAXA_PREENCHIMENTO",
    "RAZAO_INSCRITOS_VAGAS",
    "CURSO_COMPETITIVO",
)

# Movido para `src.data.leakage` na Tarefa 3 (fonte única de verdade para
# regras de remoção de vazamento — ver docs/DATA_PIPELINE.md):
#   - KNOWN_LEAKY_FEATURES      -> src.data.leakage.KNOWN_LEAKY_DERIVED_FEATURES
#   - TARGET_SOURCE_COLUMNS     -> src.data.leakage.TARGET_SOURCE_COLUMNS

# --------------------------------------------------------------------------
# Nomes de modelos (usados como chaves em registries e nomes de pasta)
# --------------------------------------------------------------------------

MODEL_NAMES: tuple[str, ...] = (
    "arvore_de_decisao",
    "regressao_logistica",
    "svm",
    "rede_neural",
    "naive_bayes",
)

# --------------------------------------------------------------------------
# Nomes de arquivo de saída (referência para a futura convenção única —
# ver ARCHITECTURE_AUDIT.md, seção 7, item "artifacts.py")
# --------------------------------------------------------------------------

METRICS_SUMMARY_FILENAME: str = "metrics_summary.csv"
METRICS_BY_FOLD_FILENAME: str = "metrics_by_fold.csv"
BEST_PARAMS_FILENAME: str = "best_params.json"
CONFUSION_MATRIX_FILENAME: str = "confusion_matrix.csv"
MODEL_ARTIFACT_FILENAME: str = "best_model.pkl"
RANKING_FILENAME: str = "ranking.csv"
