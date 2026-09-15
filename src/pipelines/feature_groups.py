"""Definição declarativa dos grupos de features por modelo.

Reexporta, com nomes por grupo, as listas já centralizadas em
``src.configs.constants`` — ver ARCHITECTURE_AUDIT.md, seção 4.2. Não
contém nenhuma lógica de transformação; é só a organização das constantes
por "quem usa o quê", para servir de referência à futura implementação
dos pipelines reais de cada modelo.
"""

from __future__ import annotations

from src.configs.constants import (
    ADVANCED_TREE_FEATURES,
    CATEGORICAL_FEATURES,
    COMMON_FEATURES,
    DERIVED_FEATURES,
    RAW_NUMERIC_FEATURES,
)
from src.data.leakage import KNOWN_LEAKY_DERIVED_FEATURES as KNOWN_LEAKY_FEATURES

# Features usadas por Árvore de Decisão, Regressão Logística, SVM e Rede
# Neural sem nenhuma transformação adicional.
COMMON_FEATURE_SET: tuple[str, ...] = COMMON_FEATURES

# Features adicionais exclusivas da Árvore de Decisão, já sem a feature
# de vazamento `RAZAO_CONCLUSAO_EVASAO` (ver KNOWN_LEAKY_FEATURES).
TREE_FEATURE_SET: tuple[str, ...] = COMMON_FEATURES + ADVANCED_TREE_FEATURES

# Regressão Logística e SVM não adicionam features novas; a diferença
# entre eles está na etapa de scaling/seleção (StandardScaler + nenhuma
# seleção vs. seleção por correlação + RobustScaler), a ser implementada
# em `src/pipelines/` na próxima etapa.
LOGISTIC_REGRESSION_FEATURE_SET: tuple[str, ...] = COMMON_FEATURES
SVM_FEATURE_SET: tuple[str, ...] = COMMON_FEATURES

# A Rede Neural também usa as features comuns; a auditoria identificou
# que o notebook original não aplica scaling — isso é uma lacuna a
# corrigir na migração, não um comportamento a preservar (ver
# ARCHITECTURE_AUDIT.md, seção 3.4).
NEURAL_NETWORK_FEATURE_SET: tuple[str, ...] = COMMON_FEATURES

# O Naive Bayes usa um universo de features totalmente diferente (todas
# as colunas numéricas do censo, exceto identificadores e vazamento) —
# não é um subconjunto de COMMON_FEATURE_SET. A lista completa depende
# das colunas do CSV e será resolvida em tempo de execução por
# `src/data/loading.py`, não aqui.
NAIVE_BAYES_FEATURE_SET: tuple[str, ...] = ()


def assert_no_known_leakage(feature_set: tuple[str, ...]) -> None:
    """Levanta ``ValueError`` se alguma feature vazada estiver presente.

    Utilitário simples para os testes e para os futuros pipelines
    validarem um conjunto de features antes do treino.
    """
    leaked = set(feature_set) & set(KNOWN_LEAKY_FEATURES)
    if leaked:
        raise ValueError(f"Features com vazamento de dados detectadas: {sorted(leaked)}")


__all__ = [
    "COMMON_FEATURE_SET",
    "TREE_FEATURE_SET",
    "LOGISTIC_REGRESSION_FEATURE_SET",
    "SVM_FEATURE_SET",
    "NEURAL_NETWORK_FEATURE_SET",
    "NAIVE_BAYES_FEATURE_SET",
    "assert_no_known_leakage",
    "CATEGORICAL_FEATURES",
    "RAW_NUMERIC_FEATURES",
    "DERIVED_FEATURES",
]
