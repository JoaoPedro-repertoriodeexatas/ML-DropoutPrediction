"""Definição dos estimadores Naive Bayes — 3 variantes (Tarefa 5).

Reimplementação de `models/naive_bayes/model_factory.py::create_model`.
Responsabilidade única: construir o estimador Naive Bayes **não
ajustado** de uma das 3 variantes usadas no projeto (Gaussian, Bernoulli,
Complement). Não carrega dados, não avalia, não plota, não salva nada.

## Lógica de comparação preservada

`models/naive_bayes/train.py` compara as 3 variantes gerando ~30
configurações aleatórias de cada uma, avaliando todas por validação
cruzada e escolhendo a de melhor recall — a variante vencedora foi
`ComplementNB` (`alpha=0.16121212121212125`, `norm=True`, ver
`models/naive_bayes/artifacts/best_params.json`). Essa busca (a parte de
*treino/avaliação*) está fora do escopo desta tarefa — "não possuir
lógica de avaliação" é um requisito explícito da Tarefa 5. O que é
preservado aqui é a estrutura que torna essa comparação possível:
`create_naive_bayes` aceita qualquer uma das 3 variantes pelo mesmo
nome, e `create_all_naive_bayes_variants` constrói as 3 de uma vez —
exatamente o que um futuro `src/training/` precisaria para reproduzir a
comparação.
"""

from __future__ import annotations

from typing import Literal

from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB

from src.configs.model_hyperparameters import (
    BernoulliNBHyperparameters,
    ComplementNBHyperparameters,
    GaussianNBHyperparameters,
)

NaiveBayesVariant = Literal["gaussian", "bernoulli", "complement"]
NaiveBayesEstimator = GaussianNB | BernoulliNB | ComplementNB

_VALID_VARIANTS: tuple[NaiveBayesVariant, ...] = ("gaussian", "bernoulli", "complement")


def create_naive_bayes(
    variant: NaiveBayesVariant = "complement",
    *,
    gaussian_params: GaussianNBHyperparameters | None = None,
    bernoulli_params: BernoulliNBHyperparameters | None = None,
    complement_params: ComplementNBHyperparameters | None = None,
) -> NaiveBayesEstimator:
    """Cria o estimador Naive Bayes da variante pedida, não ajustado.

    Args:
        variant: ``"gaussian"``, ``"bernoulli"`` ou ``"complement"``
            (padrão — a variante vencedora da comparação original).
        gaussian_params: hiperparâmetros do GaussianNB (usados só se
            ``variant="gaussian"``); ``None`` usa os defaults do
            scikit-learn (ver `GaussianNBHyperparameters`).
        bernoulli_params: hiperparâmetros do BernoulliNB (usados só se
            ``variant="bernoulli"``); ``None`` usa os defaults do
            scikit-learn.
        complement_params: hiperparâmetros do ComplementNB (usados só se
            ``variant="complement"``); ``None`` usa os valores vencedores
            da comparação original.

    Returns:
        Estimador sklearn pronto para `.fit(X_train, y_train)`.

    Raises:
        ValueError: ``variant`` desconhecida.
    """
    if variant == "gaussian":
        params = gaussian_params or GaussianNBHyperparameters()
        return GaussianNB(**params.to_dict())

    if variant == "bernoulli":
        params = bernoulli_params or BernoulliNBHyperparameters()
        return BernoulliNB(**params.to_dict())

    if variant == "complement":
        params = complement_params or ComplementNBHyperparameters()
        return ComplementNB(**params.to_dict())

    raise ValueError(
        f"Variante de Naive Bayes inválida: {variant!r}. Use uma de {_VALID_VARIANTS}."
    )


def create_all_naive_bayes_variants() -> dict[NaiveBayesVariant, NaiveBayesEstimator]:
    """Cria as 3 variantes de uma vez, cada uma com seus hiperparâmetros padrão.

    Preserva a estrutura de comparação de `models/naive_bayes/train.py`
    sem reimplementar a busca de hiperparâmetros nem a validação cruzada.

    Returns:
        Dicionário ``{variante: estimador não ajustado}``.
    """
    return {variant: create_naive_bayes(variant) for variant in _VALID_VARIANTS}


__all__ = [
    "NaiveBayesVariant",
    "NaiveBayesEstimator",
    "create_naive_bayes",
    "create_all_naive_bayes_variants",
]
