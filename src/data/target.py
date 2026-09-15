"""Construção centralizada de ``taxa_evasao`` e ``alto_risco_evasao``.

O projeto hoje calcula essas duas colunas de duas formas diferentes e
incompatíveis (ver `ARCHITECTURE_AUDIT.md`, seções 3 e 5):

- **``median_split``** — usada em `data/preprocessamento.py` (Naive
  Bayes): ``taxa_evasao = QT_SIT_DESVINCULADO / (QT_MAT + QT_SIT_DESVINCULADO)``;
  alvo = 1 se ``taxa_evasao`` for maior ou igual à **mediana** da própria
  coluna, após filtrar ``QT_ING >= 10`` e ``QT_MAT > 0``.
- **``threshold_20pct``** — usada nos 4 notebooks (Árvore, Regressão
  Logística, SVM, Rede Neural) e em `analysis/audit/data.py`:
  ``taxa_evasao (%) = (QT_SIT_DESVINCULADO + QT_SIT_TRANCADA) / QT_ING * 100``;
  alvo = 1 se ``taxa_evasao >= 20``, sem filtro de ``QT_MAT``.

A Tarefa 3 pede para transformar "a lógica atualmente existente em
`data/preprocessamento.py`" — por isso ``median_split`` é a estratégia
**padrão** deste módulo. A estratégia dos notebooks é mantida como
alternativa nomeada (não escolhida silenciosamente) para que a validação
de compatibilidade (`src.data.validation`) e uma futura unificação de
pipeline (ARCHITECTURE_AUDIT.md, seção 8, fase 5) possam usar qualquer
uma das duas de forma explícita.

Nomenclatura: as colunas produzidas por este módulo são sempre
``snake_case`` (``taxa_evasao``, ``alto_risco_evasao``), como pedido
explicitamente na Tarefa 3 — mesmo quando a estratégia escolhida é a dos
notebooks (que originalmente usam maiúsculas). Isso corrige, para a nova
camada de dados, o problema de nomenclatura inconsistente apontado em
ARCHITECTURE_AUDIT.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.configs.settings import (
    EVASION_RATE_THRESHOLD_PCT,
    LEGACY_NAIVE_BAYES_MIN_INGRESSANTES,
)

TAXA_EVASAO_COLUMN: str = "taxa_evasao"
TARGET_COLUMN: str = "alto_risco_evasao"

TargetStrategyName = Literal["median_split", "threshold_20pct"]

_VALID_STRATEGIES: tuple[TargetStrategyName, ...] = ("median_split", "threshold_20pct")

# Limiar percentual da estratégia `threshold_20pct` (mesmo valor fixo dos
# 4 notebooks), centralizado em `src.configs.settings`.
THRESHOLD_20PCT_CUTOFF: float = EVASION_RATE_THRESHOLD_PCT

# Filtro mínimo de `QT_ING` da estratégia `median_split`, centralizado em
# `src.configs.settings` (mesmo valor usado em `data/preprocessamento.py`).
_MEDIAN_SPLIT_MIN_INGRESSANTES: int = LEGACY_NAIVE_BAYES_MIN_INGRESSANTES


@dataclass(frozen=True)
class TargetResult:
    """Resultado da construção do alvo.

    Attributes:
        df: DataFrame de entrada com as colunas ``taxa_evasao`` e
            ``alto_risco_evasao`` adicionadas, e com as linhas filtradas
            de acordo com a estratégia (ex.: ``QT_ING`` mínimo).
        strategy: nome da estratégia usada.
        threshold: valor de corte efetivamente aplicado (a mediana
            calculada, no caso de ``median_split``; o valor fixo de 20.0,
            no caso de ``threshold_20pct``).
        positive_rate: proporção de registros com ``alto_risco_evasao == 1``
            após a construção do alvo — útil para a rotina de validação
            comparar distribuição de classes entre pipelines.
    """

    df: pd.DataFrame
    strategy: TargetStrategyName
    threshold: float
    positive_rate: float


def _require_columns(df: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes para construir o alvo: {missing}"
        )


def _compute_taxa_evasao_median_strategy(df: pd.DataFrame) -> pd.Series:
    """``taxa_evasao = QT_SIT_DESVINCULADO / (QT_MAT + QT_SIT_DESVINCULADO)``."""
    _require_columns(df, ("QT_MAT", "QT_SIT_DESVINCULADO"))
    denominador = df["QT_MAT"] + df["QT_SIT_DESVINCULADO"]
    return pd.Series(
        np.where(denominador > 0, df["QT_SIT_DESVINCULADO"] / denominador, np.nan),
        index=df.index,
        name=TAXA_EVASAO_COLUMN,
    )


def _compute_taxa_evasao_threshold_strategy(df: pd.DataFrame) -> pd.Series:
    """``taxa_evasao (%) = (QT_SIT_DESVINCULADO + QT_SIT_TRANCADA) / QT_ING * 100``."""
    _require_columns(df, ("QT_ING", "QT_SIT_DESVINCULADO", "QT_SIT_TRANCADA"))
    evadidos = df["QT_SIT_DESVINCULADO"] + df["QT_SIT_TRANCADA"]
    return pd.Series(
        np.where(df["QT_ING"] > 0, evadidos / df["QT_ING"] * 100, np.nan),
        index=df.index,
        name=TAXA_EVASAO_COLUMN,
    )


def _build_median_split(df: pd.DataFrame) -> TargetResult:
    df = df.copy()
    df[TAXA_EVASAO_COLUMN] = _compute_taxa_evasao_median_strategy(df)

    df = df[df["QT_ING"] >= _MEDIAN_SPLIT_MIN_INGRESSANTES]
    df = df[df["QT_MAT"] > 0]
    df = df[df[TAXA_EVASAO_COLUMN].notna()]

    if df.empty:
        raise ValueError(
            "Nenhum registro restante após os filtros da estratégia "
            "'median_split' (QT_ING>=10, QT_MAT>0, taxa_evasao válida)."
        )

    threshold = float(df[TAXA_EVASAO_COLUMN].median())
    df[TARGET_COLUMN] = (df[TAXA_EVASAO_COLUMN] >= threshold).astype(int)

    return TargetResult(
        df=df,
        strategy="median_split",
        threshold=threshold,
        positive_rate=float(df[TARGET_COLUMN].mean()),
    )


def _build_threshold_20pct(df: pd.DataFrame) -> TargetResult:
    df = df.copy()
    df[TAXA_EVASAO_COLUMN] = _compute_taxa_evasao_threshold_strategy(df)

    df = df[df["QT_ING"] > 0]
    df = df[df[TAXA_EVASAO_COLUMN].notna()]

    if df.empty:
        raise ValueError(
            "Nenhum registro restante após o filtro da estratégia "
            "'threshold_20pct' (QT_ING>0)."
        )

    df[TARGET_COLUMN] = (df[TAXA_EVASAO_COLUMN] >= THRESHOLD_20PCT_CUTOFF).astype(int)

    return TargetResult(
        df=df,
        strategy="threshold_20pct",
        threshold=THRESHOLD_20PCT_CUTOFF,
        positive_rate=float(df[TARGET_COLUMN].mean()),
    )


def build_target(
    df: pd.DataFrame,
    strategy: TargetStrategyName = "median_split",
) -> TargetResult:
    """Constrói ``taxa_evasao`` e ``alto_risco_evasao`` em ``df``.

    Args:
        df: DataFrame bruto (ainda não filtrado), contendo pelo menos as
            colunas exigidas pela estratégia escolhida.
        strategy: ``"median_split"`` (padrão, replica
            `data/preprocessamento.py`) ou ``"threshold_20pct"`` (replica
            os 4 notebooks e `analysis/audit/data.py`).

    Returns:
        ``TargetResult`` com o DataFrame filtrado e as colunas de alvo
        adicionadas.

    Raises:
        ValueError: estratégia desconhecida, colunas obrigatórias
            ausentes, ou nenhum registro restante após os filtros.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Estratégia de alvo inválida: {strategy!r}. Use uma de {_VALID_STRATEGIES}."
        )

    if strategy == "median_split":
        return _build_median_split(df)
    return _build_threshold_20pct(df)


__all__ = [
    "TAXA_EVASAO_COLUMN",
    "TARGET_COLUMN",
    "TargetStrategyName",
    "THRESHOLD_20PCT_CUTOFF",
    "TargetResult",
    "build_target",
]
