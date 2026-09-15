"""Orquestra a camada de dados: Loading → Target → Leakage → Base Preprocessing.

Este módulo é o único ponto de entrada que encadeia as quatro primeiras
etapas do diagrama da Tarefa 3:

    Data Loading → Target Construction → Leakage Removal → Base Preprocessing → (Feature Engineering)

A quinta etapa (engenharia de features específica de cada modelo) tem
arquitetura própria desde a Tarefa 4 — ver `src.pipelines.feature_engineering.
FeatureEngineer`. Este orquestrador continua **não a chamando
automaticamente**: `run_data_pipeline` devolve ``X``/``y`` já prontos
para receber, fora daqui, um `Pipeline` de
`FeatureEngineer.for_<modelo>()` ajustado só no fold de treino (ver
`docs/FEATURE_ENGINEERING.md`). O parâmetro opcional ``feature_engineer``
existe para compor isso quando fizer sentido fazê-lo fora de um loop de
CV (ex.: inspeção manual, testes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.leakage import explain_removed_columns, remove_leakage_columns
from src.data.loading import load_raw_dataset
from src.data.target import TARGET_COLUMN, TargetResult, TargetStrategyName, build_target
from src.pipelines.base_preprocessing import MissingValueStrategy, run_base_preprocessing


@dataclass(frozen=True)
class DataPipelineResult:
    """Saída completa do pipeline de dados, com metadados para auditoria.

    Attributes:
        X: matriz de features, já sem colunas de vazamento e já com o
            pré-processamento comum aplicado (mas sem scaling/seleção
            específica de modelo — ver `src.pipelines.feature_engineering`).
        y: vetor alvo (``alto_risco_evasao``), alinhado ao índice de ``X``.
        target_strategy: estratégia usada para construir o alvo.
        target_threshold: limiar efetivamente aplicado (ver `TargetResult`).
        positive_rate: proporção de positivos em ``y``.
        n_records_raw: número de linhas do dataset bruto, antes de
            qualquer filtro.
        n_records_final: número de linhas em ``X``/``y``.
        removed_leakage_columns: nomes das colunas removidas por
            vazamento, agrupadas por motivo (ver
            `src.data.leakage.explain_removed_columns`).
        missing_strategy: estratégia de imputação usada no pré-processamento.
    """

    X: pd.DataFrame
    y: pd.Series
    target_strategy: TargetStrategyName
    target_threshold: float
    positive_rate: float
    n_records_raw: int
    n_records_final: int
    removed_leakage_columns: dict[str, list[str]]
    missing_strategy: MissingValueStrategy


def run_data_pipeline(
    *,
    dataset_path: Path | None = None,
    target_strategy: TargetStrategyName = "median_split",
    missing_strategy: MissingValueStrategy = "zero",
    feature_engineer: Pipeline | None = None,
    include_prefix_leakage_group: bool = True,
) -> DataPipelineResult:
    """Executa Data Loading → Target → Leakage Removal → Base Preprocessing.

    Args:
        dataset_path: caminho explícito do CSV; ``None`` resolve
            automaticamente (ver `src.data.loading.load_raw_dataset`).
        target_strategy: ``"median_split"`` (padrão) ou ``"threshold_20pct"``
            — ver `src.data.target`.
        missing_strategy: ``"zero"`` (padrão) ou ``"mode"`` — ver
            `src.pipelines.base_preprocessing`.
        feature_engineer: `Pipeline` **não ajustado** (ex.: o resultado de
            ``FeatureEngineer().for_decision_tree()`` — ver
            `src.pipelines.feature_engineering`). Se fornecido, é ajustado
            e aplicado aqui via ``fit_transform(X, y)`` sobre **todo** o
            resultado deste pipeline — por isso só é seguro usar este
            parâmetro fora de um loop de cross-validation (inspeção
            manual, testes). Dentro de CV de verdade, ajuste o `Pipeline`
            de feature engineering separadamente, só no fold de treino, e
            não passe nada aqui (``None``, o padrão).
        include_prefix_leakage_group: repassado para
            `src.data.leakage.remove_leakage_columns`. Os 4 modelos
            "notebook-style" (Árvore, Regressão Logística, SVM, Rede
            Neural) esperam ``False`` aqui — as features comuns incluem
            razões que dependem de colunas com prefixo `QT_MAT`/`QT_CONC`
            (ex.: `TAXA_CONCLUSAO`, `INDICE_FINANCIAMENTO`), removidas por
            padrão como vazamento indireto (ver ARCHITECTURE_AUDIT.md,
            seção 5, e `docs/FEATURE_ENGINEERING.md`, seção "Como montar
            o X de entrada"). O Naive Bayes usa o padrão ``True``.

    Returns:
        `DataPipelineResult` com ``X``, ``y`` e metadados de auditoria.
    """
    raw_df = load_raw_dataset(dataset_path)
    n_records_raw = len(raw_df)

    target_result: TargetResult = build_target(raw_df, strategy=target_strategy)
    df_with_target = target_result.df

    y = df_with_target[TARGET_COLUMN].copy()

    removed_columns = explain_removed_columns(
        df_with_target.columns,
        include_prefix_group=include_prefix_leakage_group,
    )
    df_no_leakage = remove_leakage_columns(
        df_with_target,
        include_prefix_group=include_prefix_leakage_group,
    )

    X = run_base_preprocessing(df_no_leakage, missing_strategy=missing_strategy)

    if feature_engineer is not None:
        X = feature_engineer.fit_transform(X, y)

    return DataPipelineResult(
        X=X,
        y=y,
        target_strategy=target_result.strategy,
        target_threshold=target_result.threshold,
        positive_rate=target_result.positive_rate,
        n_records_raw=n_records_raw,
        n_records_final=len(X),
        removed_leakage_columns=removed_columns,
        missing_strategy=missing_strategy,
    )


__all__ = ["DataPipelineResult", "run_data_pipeline"]
