"""Definição do estimador da Árvore de Decisão (Tarefa 5).

Responsabilidade única: construir um `DecisionTreeClassifier` **não
ajustado**, com os hiperparâmetros vencedores da busca original (ver
`src.configs.model_hyperparameters.DecisionTreeHyperparameters`). Não
carrega dados, não avalia, não plota, não salva nada.
"""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from src.configs.model_hyperparameters import DecisionTreeHyperparameters


def create_decision_tree(
    params: DecisionTreeHyperparameters | None = None,
) -> DecisionTreeClassifier:
    """Cria um `DecisionTreeClassifier` não ajustado.

    Args:
        params: hiperparâmetros a usar. Se ``None`` (padrão), usa
            `DecisionTreeHyperparameters()` — os valores vencedores da
            busca original (ver docstring do módulo de configuração).

    Returns:
        Estimador sklearn pronto para `.fit(X_train, y_train)`.
    """
    effective_params = params or DecisionTreeHyperparameters()
    return DecisionTreeClassifier(**effective_params.to_dict())


__all__ = ["create_decision_tree"]
