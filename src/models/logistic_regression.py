"""Definição do estimador de Regressão Logística (Tarefa 5).

Responsabilidade única: construir um `LogisticRegression` **não
ajustado**, com os hiperparâmetros vencedores da busca original (ver
`src.configs.model_hyperparameters.LogisticRegressionHyperparameters`).
Não carrega dados, não avalia, não plota, não salva nada.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from src.configs.model_hyperparameters import LogisticRegressionHyperparameters


def create_logistic_regression(
    params: LogisticRegressionHyperparameters | None = None,
) -> LogisticRegression:
    """Cria um `LogisticRegression` não ajustado.

    Args:
        params: hiperparâmetros a usar. Se ``None`` (padrão), usa
            `LogisticRegressionHyperparameters()` — os valores vencedores
            da busca original.

    Returns:
        Estimador sklearn pronto para `.fit(X_train, y_train)`.
    """
    effective_params = params or LogisticRegressionHyperparameters()
    return LogisticRegression(**effective_params.to_dict())


__all__ = ["create_logistic_regression"]
