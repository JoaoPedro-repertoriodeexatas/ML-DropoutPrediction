"""Pré-processamento comum a todos os modelos.

Contém **somente** as transformações que os 5 modelos aplicam da mesma
forma, antes de qualquer engenharia de features específica:

- substituir ``inf``/``-inf`` por ``NaN`` (necessário porque `taxa_evasao`
  e outras razões usam divisão, que pode gerar infinito);
- tratar valores ausentes (duas estratégias nomeadas, replicando o que
  já existe no projeto — ver ``MissingValueStrategy``);
- restringir a colunas numéricas (o formato que todo estimador scikit-learn
  espera).

**Não incluído de propósito** (é responsabilidade de `src/pipelines/`
específico de cada modelo, ainda não implementado — ver
`feature_engineering.py`):

- ``StandardScaler``/``RobustScaler``/``MinMaxScaler`` (usados só por
  Regressão Logística, SVM e Naive Bayes, cada um de um jeito);
- ``SelectKBest``/seleção por correlação (Árvore e SVM, com critérios
  diferentes);
- SMOTE ou ``class_weight`` (balanceamento é uma decisão de treino, não
  de pré-processamento — ver `src/training/`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

MissingValueStrategy = Literal["mode", "zero"]

_VALID_MISSING_STRATEGIES: tuple[MissingValueStrategy, ...] = ("mode", "zero")


def replace_infinite_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui ``inf``/``-inf`` por ``NaN`` em todas as colunas numéricas.

    Não modifica ``df`` in-place.
    """
    return df.replace([np.inf, -np.inf], np.nan)


def impute_missing_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche ausentes com a moda de cada coluna.

    Replica ``handle_missing_mode`` dos 4 notebooks. Se uma coluna não
    tiver moda definida (todos os valores ausentes), usa 0 como
    alternativa — mesmo comportamento do código original.
    """
    df_filled = df.copy()
    for column in df_filled.columns:
        if df_filled[column].isnull().any():
            mode_values = df_filled[column].mode()
            fill_value = mode_values.iloc[0] if len(mode_values) > 0 else 0
            df_filled[column] = df_filled[column].fillna(fill_value)
    return df_filled


def impute_missing_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche todos os valores ausentes com 0.

    Replica o comportamento de `data/preprocessamento.py::_selecionar_features`.
    """
    return df.fillna(0)


def select_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Restringe ``df`` às colunas de tipo numérico.

    Colunas categóricas (texto) devem ser codificadas antes desta etapa
    pela engenharia de features específica de cada modelo — este módulo
    não faz encoding, porque o encoding usado hoje já diverge entre
    modelos (label encoding nos 4 notebooks; nenhum encoding categórico
    no Naive Bayes, que já opera só sobre colunas numéricas do censo).
    """
    return df.select_dtypes(include=[np.number])


def run_base_preprocessing(
    df: pd.DataFrame,
    *,
    missing_strategy: MissingValueStrategy = "zero",
    restrict_to_numeric: bool = True,
) -> pd.DataFrame:
    """Executa o pré-processamento comum, na ordem: inf→NaN, imputação, dtype.

    Args:
        df: DataFrame já sem colunas de vazamento (ver `src.data.leakage`).
        missing_strategy: ``"zero"`` (padrão, replica
            `data/preprocessamento.py`) ou ``"mode"`` (replica os 4
            notebooks). Ver módulo `MissingValueStrategy`.
        restrict_to_numeric: se ``True`` (padrão), aplica
            `select_numeric_columns` como última etapa.

    Returns:
        Novo DataFrame pré-processado. Não modifica ``df`` in-place.

    Raises:
        ValueError: ``missing_strategy`` desconhecida.
    """
    if missing_strategy not in _VALID_MISSING_STRATEGIES:
        raise ValueError(
            f"Estratégia de valores ausentes inválida: {missing_strategy!r}. "
            f"Use uma de {_VALID_MISSING_STRATEGIES}."
        )

    result = replace_infinite_with_nan(df)

    if missing_strategy == "mode":
        result = impute_missing_with_mode(result)
    else:
        result = impute_missing_with_zero(result)

    if restrict_to_numeric:
        result = select_numeric_columns(result)

    return result


__all__ = [
    "MissingValueStrategy",
    "replace_infinite_with_nan",
    "impute_missing_with_mode",
    "impute_missing_with_zero",
    "select_numeric_columns",
    "run_base_preprocessing",
]
