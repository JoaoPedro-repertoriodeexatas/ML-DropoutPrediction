"""Definição do estimador SVM (Tarefa 5).

Responsabilidade única: construir um `SVC` **não ajustado**, com os
hiperparâmetros vencedores da busca original (ver
`src.configs.model_hyperparameters.SVMHyperparameters`). Não carrega
dados, não avalia, não plota, não salva nada.
"""

from __future__ import annotations

from sklearn.svm import SVC

from src.configs.model_hyperparameters import SVMHyperparameters


def create_svm(params: SVMHyperparameters | None = None) -> SVC:
    """Cria um `SVC` não ajustado.

    Args:
        params: hiperparâmetros a usar. Se ``None`` (padrão), usa
            `SVMHyperparameters()` — os valores vencedores da busca
            original (kernel RBF).

    Returns:
        Estimador sklearn pronto para `.fit(X_train, y_train)`.
    """
    effective_params = params or SVMHyperparameters()
    return SVC(**effective_params.to_dict())


__all__ = ["create_svm"]
