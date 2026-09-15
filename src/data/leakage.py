"""Regras centralizadas de remoção de variáveis com risco de vazamento.

Fonte única de verdade para "o que nunca pode entrar como feature".
Antes da Tarefa 3, essas regras existiam em dois lugares parcialmente
divergentes: `data/preprocessamento.py::COLUNAS_VAZAMENTO`/`PREFIXOS_VAZAMENTO`
(usado só pelo Naive Bayes) e, implicitamente, na lista fixa de colunas
lidas pelos 4 notebooks (`COLS_CURSOS`), que nunca declarava por que
certas colunas ficavam de fora. Ver `ARCHITECTURE_AUDIT.md`, seção 5.

Cada grupo abaixo documenta o motivo da remoção — requisito explícito da
Tarefa 3 ("documente claramente o motivo de cada grupo de colunas
removido").
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------
# Grupo 1 — Identificadores e texto livre
# --------------------------------------------------------------------------
# Motivo: não carregam sinal preditivo generalizável — são identidade do
# registro (nome/código do curso, da instituição, do município etc.), não
# uma característica do curso. Mantê-las como feature permite ao modelo
# "decorar" registros específicos em vez de aprender um padrão que
# generalize para cursos fora da amostra de treino.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "NU_ANO_CENSO",
    "NO_REGIAO",
    "NO_UF",
    "SG_UF",
    "NO_MUNICIPIO",
    "NO_CURSO",
    "CO_CURSO",
    "CO_IES",
    "NO_CINE_ROTULO",
    "CO_CINE_ROTULO",
    "NO_CINE_AREA_GERAL",
    "NO_CINE_AREA_ESPECIFICA",
    "NO_CINE_AREA_DETALHADA",
)

# --------------------------------------------------------------------------
# Grupo 2 — Colunas usadas para CONSTRUIR o alvo
# --------------------------------------------------------------------------
# Motivo: são a matéria-prima de `taxa_evasao`/`alto_risco_evasao` (ver
# `src.data.target`). Se permanecerem como feature, o modelo pode
# reconstruir o alvo quase perfeitamente a partir delas — vazamento
# direto de alvo→feature, o problema mais grave que a Tarefa 3 precisa
# evitar. Cobre as duas convenções de nome usadas no projeto (maiúsculas
# nos notebooks, minúsculas em `data/preprocessamento.py`).
TARGET_SOURCE_COLUMNS: tuple[str, ...] = (
    "QT_SIT_DESVINCULADO",
    "QT_SIT_TRANCADA",
    "QT_SIT_TRANSFERIDO",
    "QT_SIT_FALECIDO",
    "QT_EVADIDOS",
    "TAXA_EVASAO",
    "taxa_evasao",
    "ALTO_RISCO_EVASAO",
    "alto_risco_evasao",
)

# --------------------------------------------------------------------------
# Grupo 3 — Prefixos de matrícula e conclusão
# --------------------------------------------------------------------------
# Motivo: `QT_MAT*` (matriculados) e `QT_CONC*` (concluintes) são
# estruturalmente correlacionados com quem NÃO evadiu — por definição do
# problema, um curso com muitos matriculados/concluintes tem,
# necessariamente, poucos evadidos proporcionalmente. Mantê-las como
# feature bruta infla o poder preditivo de forma que não generaliza para
# um cenário real de predição (o número de matriculados de um período
# futuro não está disponível no momento em que a predição seria útil).
# Esta é a mesma regra já aplicada em `data/preprocessamento.py`
# (`PREFIXOS_VAZAMENTO`); a auditoria (seção 5) mostrou que os 4
# notebooks NÃO aplicam essa regra e mantêm `QT_MAT` como feature bruta —
# um vazamento indireto documentado, mas não corrigido nos notebooks.
LEAKAGE_PREFIXES: tuple[str, ...] = ("QT_MAT", "QT_CONC")

# --------------------------------------------------------------------------
# Grupo 4 — Features derivadas com vazamento direto confirmado
# --------------------------------------------------------------------------
# Motivo: não vêm do CSV bruto — são calculadas por notebooks específicos
# a partir de outras colunas. `RAZAO_CONCLUSAO_EVASAO` (exclusiva do
# notebook da Árvore de Decisão) usa `TAXA_EVASAO` diretamente como
# denominador, ou seja, é construída a partir da própria variável que
# depois é binarizada no alvo. Ver ARCHITECTURE_AUDIT.md, seção 5
# ("Feature construída a partir do alvo contínuo" — severidade crítica).
KNOWN_LEAKY_DERIVED_FEATURES: tuple[str, ...] = ("RAZAO_CONCLUSAO_EVASAO",)


def leakage_columns_for(
    columns: pd.Index | list[str],
    *,
    include_prefix_group: bool = True,
) -> set[str]:
    """Calcula quais colunas de ``columns`` devem ser removidas por vazamento.

    Args:
        columns: colunas presentes no DataFrame a ser filtrado.
        include_prefix_group: se ``True`` (padrão), também remove qualquer
            coluna cujo nome comece com um dos ``LEAKAGE_PREFIXES``
            (grupo 3). Desligar isso só é aceitável para inspeção/debug —
            nunca antes de treinar um modelo.

    Returns:
        Conjunto de nomes de coluna, restrito ao que de fato existe em
        ``columns`` (nunca levanta erro por coluna ausente).
    """
    present = set(columns)
    to_remove = (
        set(IDENTIFIER_COLUMNS)
        | set(TARGET_SOURCE_COLUMNS)
        | set(KNOWN_LEAKY_DERIVED_FEATURES)
    ) & present

    if include_prefix_group:
        to_remove |= {
            column
            for column in present
            if column.startswith(LEAKAGE_PREFIXES)
        }

    return to_remove


def remove_leakage_columns(
    df: pd.DataFrame,
    *,
    include_prefix_group: bool = True,
) -> pd.DataFrame:
    """Retorna uma cópia de ``df`` sem as colunas de vazamento conhecidas.

    Não modifica ``df`` in-place. Colunas listadas nas regras acima que
    não existem em ``df`` são simplesmente ignoradas (nenhum erro).
    """
    columns_to_drop = leakage_columns_for(df.columns, include_prefix_group=include_prefix_group)
    return df.drop(columns=sorted(columns_to_drop))


def explain_removed_columns(
    df_columns_before: pd.Index | list[str],
    *,
    include_prefix_group: bool = True,
) -> dict[str, list[str]]:
    """Detalha, por grupo, quais colunas de ``df_columns_before`` seriam removidas.

    Útil para logging/documentação — mostra exatamente o motivo de cada
    coluna removida, em vez de só o conjunto final.
    """
    present = set(df_columns_before)
    result: dict[str, list[str]] = {
        "identificadores": sorted(set(IDENTIFIER_COLUMNS) & present),
        "origem_do_alvo": sorted(set(TARGET_SOURCE_COLUMNS) & present),
        "features_derivadas_vazadas": sorted(set(KNOWN_LEAKY_DERIVED_FEATURES) & present),
    }
    if include_prefix_group:
        result["prefixos_matricula_conclusao"] = sorted(
            column for column in present if column.startswith(LEAKAGE_PREFIXES)
        )
    return result


__all__ = [
    "IDENTIFIER_COLUMNS",
    "TARGET_SOURCE_COLUMNS",
    "LEAKAGE_PREFIXES",
    "KNOWN_LEAKY_DERIVED_FEATURES",
    "leakage_columns_for",
    "remove_leakage_columns",
    "explain_removed_columns",
]
