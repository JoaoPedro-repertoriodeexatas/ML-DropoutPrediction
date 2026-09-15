"""Parâmetros gerais de execução.

Valores hoje espalhados (e às vezes divergentes) entre os 4 notebooks,
`data/preprocessamento.py` e `analysis/audit/`. Centralizá-los aqui não
altera o comportamento de nenhum pipeline existente — é só a referência
que a futura migração (fora do escopo desta etapa) deverá usar.

Onde os pipelines legados divergem entre si (ex.: tamanho de amostra),
isso é registrado explicitamente em vez de escolhido silenciosamente —
ver `ARCHITECTURE_AUDIT.md`, seções 3 e 5.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Reprodutibilidade
# --------------------------------------------------------------------------

RANDOM_STATE: int = 42

# --------------------------------------------------------------------------
# Validação cruzada
# --------------------------------------------------------------------------

CV_FOLDS: int = 5
CV_SHUFFLE: bool = True

# Fração do treino de cada fold reservada para calibração de threshold
# (mesmo valor usado em todos os notebooks e em `analysis/audit/evaluator.py`).
CALIBRATION_SPLIT_SIZE: float = 0.2

# --------------------------------------------------------------------------
# Amostragem do dataset
# --------------------------------------------------------------------------

# Tamanho de amostra usado por LR, SVM e Rede Neural (e pelo pipeline
# unificado de auditoria em `analysis/audit/data.py`).
DEFAULT_SAMPLE_SIZE: int = 30_000

# A Árvore de Decisão usa 90_000 no notebook original — divergência já
# documentada em ARCHITECTURE_AUDIT.md (seção 3.1) e propositalmente NÃO
# adotada aqui como padrão, para não repetir a inconsistência.
LEGACY_DECISION_TREE_SAMPLE_SIZE: int = 90_000

# --------------------------------------------------------------------------
# Construção do alvo
# --------------------------------------------------------------------------

# Limiar percentual usado pelos 4 notebooks (Árvore, LR, SVM, Rede Neural)
# e por `analysis/audit/data.py` para `TAXA_EVASAO >= EVASION_RATE_THRESHOLD_PCT`.
EVASION_RATE_THRESHOLD_PCT: float = 20.0

# Filtro mínimo de ingressantes usado pelos mesmos pipelines.
MIN_INGRESSANTES: int = 0

# Filtro mínimo de ingressantes usado por `data/preprocessamento.py`
# (pipeline específico do Naive Bayes — alvo por mediana, não por 20%).
LEGACY_NAIVE_BAYES_MIN_INGRESSANTES: int = 10

# --------------------------------------------------------------------------
# Threshold tuning
# --------------------------------------------------------------------------

# Faixa de busca com guarda-corpo (Youden's J), hoje implementada apenas
# em `analysis/audit/threshold.py`.
THRESHOLD_SEARCH_MIN: float = 0.2
THRESHOLD_SEARCH_MAX: float = 0.8
THRESHOLD_SEARCH_STEPS: int = 81
THRESHOLD_FALLBACK: float = 0.5
THRESHOLD_PREVALENCE_CAP: float = 1.5

# --------------------------------------------------------------------------
# Balanceamento de classes
# --------------------------------------------------------------------------

SMOTE_RANDOM_STATE: int = RANDOM_STATE

# --------------------------------------------------------------------------
# Feature engineering específica por modelo (Tarefa 4 — ver
# src.pipelines.feature_engineering e docs/FEATURE_ENGINEERING.md)
# --------------------------------------------------------------------------

# SelectKBest(mutual_info_classif, k=...) da Árvore de Decisão. Mesmo
# valor numérico do notebook original; o pool de candidatas mudou de 31
# para 30 porque `RAZAO_CONCLUSAO_EVASAO` (vazamento) foi removida — ver
# ARCHITECTURE_AUDIT.md, seção 5.
DECISION_TREE_SELECT_K_BEST: int = 21

# Limiar de |correlação| com o alvo para seleção de features do SVM.
SVM_CORRELATION_THRESHOLD: float = 0.05

# VarianceThreshold e limiar de correlação par-a-par usados pelas
# variantes Gaussian/Bernoulli do Naive Bayes (replica
# `models/naive_bayes/preprocessing.py`).
NAIVE_BAYES_VARIANCE_THRESHOLD: float = 0.01
NAIVE_BAYES_CORRELATION_THRESHOLD: float = 0.90
