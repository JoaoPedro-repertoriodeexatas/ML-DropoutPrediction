"""Validação de compatibilidade entre o novo pipeline e o legado.

Requisito explícito da Tarefa 3: "garanta que o resultado do novo
preprocessing possa ser comparado com o resultado anterior". Compara:

- número de registros;
- número de features;
- nomes das features;
- coluna alvo (nome e valores);
- distribuição das classes.

Não decide sozinho se uma divergência é "aceitável" — apenas relata.
Divergências são esperadas e ok quando explicadas pela documentação da
estratégia usada (ex.: `median_split` filtra `QT_ING>=10`, o pipeline
legado dos notebooks não filtra por `QT_ING` mínimo); o objetivo desta
rotina é tornar essas diferenças visíveis, não escondê-las.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ComparisonReport:
    """Resultado da comparação entre um pipeline "antigo" e um "novo".

    Attributes:
        n_records_old / n_records_new: número de linhas de cada resultado.
        n_features_old / n_features_new: número de colunas de features
            (excluindo o alvo) de cada resultado.
        feature_names_old / feature_names_new: nomes das colunas de
            features de cada resultado.
        features_only_in_old / features_only_in_new: diferença simétrica
            entre os dois conjuntos de nomes de feature.
        target_name_old / target_name_new: nome da série/coluna alvo
            passada para a comparação (informativo — o valor em si é
            comparado via ``class_distribution_*``).
        class_distribution_old / class_distribution_new: contagem de cada
            classe do alvo (``{0: n, 1: n}``).
        positive_rate_old / positive_rate_new: proporção da classe
            positiva em cada alvo.
        matches: dicionário por critério (``records``, ``n_features``,
            ``feature_names``, ``class_distribution_shape``) indicando se
            os dois lados são idênticos nesse critério.
        is_fully_compatible: ``True`` somente se todos os critérios em
            ``matches`` forem ``True``. Divergência não é necessariamente
            um bug — ver docstring do módulo.
    """

    n_records_old: int
    n_records_new: int
    n_features_old: int
    n_features_new: int
    feature_names_old: list[str]
    feature_names_new: list[str]
    features_only_in_old: list[str]
    features_only_in_new: list[str]
    target_name_old: str
    target_name_new: str
    class_distribution_old: dict[int, int]
    class_distribution_new: dict[int, int]
    positive_rate_old: float
    positive_rate_new: float
    matches: dict[str, bool] = field(default_factory=dict)
    is_fully_compatible: bool = False

    def summary(self) -> str:
        """Resumo textual, uma linha por critério — útil para logs/testes."""
        lines = [
            f"registros: antigo={self.n_records_old} novo={self.n_records_new} "
            f"({'OK' if self.matches['records'] else 'DIVERGE'})",
            f"n_features: antigo={self.n_features_old} novo={self.n_features_new} "
            f"({'OK' if self.matches['n_features'] else 'DIVERGE'})",
            f"nomes_features: {'OK' if self.matches['feature_names'] else 'DIVERGE'} "
            f"(só_antigo={self.features_only_in_old}, só_novo={self.features_only_in_new})",
            f"distribuicao_classes: antigo={self.class_distribution_old} "
            f"novo={self.class_distribution_new} "
            f"({'OK' if self.matches['class_distribution_shape'] else 'DIVERGE'})",
            f"taxa_positiva: antigo={self.positive_rate_old:.4f} novo={self.positive_rate_new:.4f}",
            f"compatível_no_geral: {self.is_fully_compatible}",
        ]
        return "\n".join(lines)


def compare_preprocessing_outputs(
    X_old: pd.DataFrame,
    y_old: pd.Series,
    X_new: pd.DataFrame,
    y_new: pd.Series,
) -> ComparisonReport:
    """Compara dois resultados de pré-processamento (antigo vs. novo).

    Args:
        X_old, y_old: features e alvo produzidos pelo pipeline de
            referência (ex.: `data.preprocessamento.get_df_preprocessado()`).
        X_new, y_new: features e alvo produzidos por
            `src.data.pipeline.run_data_pipeline`.

    Returns:
        `ComparisonReport` com todas as métricas pedidas pela Tarefa 3.
    """
    feature_names_old = sorted(X_old.columns)
    feature_names_new = sorted(X_new.columns)

    only_old = sorted(set(feature_names_old) - set(feature_names_new))
    only_new = sorted(set(feature_names_new) - set(feature_names_old))

    dist_old = {int(k): int(v) for k, v in y_old.value_counts().sort_index().items()}
    dist_new = {int(k): int(v) for k, v in y_new.value_counts().sort_index().items()}

    matches = {
        "records": len(X_old) == len(X_new),
        "n_features": X_old.shape[1] == X_new.shape[1],
        "feature_names": feature_names_old == feature_names_new,
        "class_distribution_shape": set(dist_old) == set(dist_new),
    }

    report = ComparisonReport(
        n_records_old=len(X_old),
        n_records_new=len(X_new),
        n_features_old=X_old.shape[1],
        n_features_new=X_new.shape[1],
        feature_names_old=feature_names_old,
        feature_names_new=feature_names_new,
        features_only_in_old=only_old,
        features_only_in_new=only_new,
        target_name_old=y_old.name or "(sem nome)",
        target_name_new=y_new.name or "(sem nome)",
        class_distribution_old=dist_old,
        class_distribution_new=dist_new,
        positive_rate_old=float(y_old.mean()),
        positive_rate_new=float(y_new.mean()),
        matches=matches,
    )
    report.is_fully_compatible = all(matches.values())
    return report


def validate_against_legacy_naive_bayes_pipeline() -> ComparisonReport:
    """Compara `run_data_pipeline(target_strategy="median_split")` com o
    pipeline legado `data.preprocessamento.get_df_preprocessado()`.

    Esta é a comparação "natural" pedida pela Tarefa 3, já que o objetivo
    explícito é "transformar a lógica atualmente existente em
    `data/preprocessamento.py`" — ambos usam a mesma estratégia de alvo
    (mediana) e as mesmas regras de vazamento (`COLUNAS_VAZAMENTO`/
    `PREFIXOS_VAZAMENTO`, agora centralizadas em `src.data.leakage`).

    Requer o CSV do INEP disponível localmente (ver
    `src.configs.paths.DATASET_PATH_CANDIDATES`) e as dependências do
    projeto instaladas (pandas, numpy) — não é chamada pelos testes
    unitários, que usam dados sintéticos em memória.

    Returns:
        `ComparisonReport` comparando os dois pipelines.
    """
    import sys

    from src.configs.paths import PROJECT_ROOT
    from src.data.pipeline import run_data_pipeline

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from data.preprocessamento import get_df_preprocessado  # legado

    X_old, y_old = get_df_preprocessado()
    result = run_data_pipeline(target_strategy="median_split")

    return compare_preprocessing_outputs(X_old, y_old, result.X, result.y)


__all__ = [
    "ComparisonReport",
    "compare_preprocessing_outputs",
    "validate_against_legacy_naive_bayes_pipeline",
]
