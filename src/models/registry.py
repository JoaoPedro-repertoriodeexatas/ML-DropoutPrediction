"""Registro central de modelos e ponto único de criação (Tarefa 5).

```python
from src.models.registry import create_model

model = create_model("arvore_de_decisao")
model = create_model("rede_neural", n_features=21)
model = create_model("naive_bayes", variant="gaussian")
```

Cada entrada de `MODEL_REGISTRY` aponta para a factory real do modelo
(`src.models.<modelo>.create_<modelo>`), implementadas na Tarefa 5. Este
módulo não define nenhum hiperparâmetro — só encaminha para a factory
correta pelo nome canônico (`src.configs.constants.MODEL_NAMES`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.base import BaseEstimator

from src.configs.constants import MODEL_NAMES
from src.models.decision_tree import create_decision_tree
from src.models.logistic_regression import create_logistic_regression
from src.models.naive_bayes import create_naive_bayes
from src.models.neural_network import create_neural_network
from src.models.svm import create_svm


@dataclass(frozen=True)
class ModelSpec:
    """Especificação declarativa de um modelo registrado."""

    name: str
    factory: Callable[..., BaseEstimator]
    requires_n_features: bool = False
    has_variants: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "arvore_de_decisao": ModelSpec(name="arvore_de_decisao", factory=create_decision_tree),
    "regressao_logistica": ModelSpec(name="regressao_logistica", factory=create_logistic_regression),
    "svm": ModelSpec(name="svm", factory=create_svm),
    "rede_neural": ModelSpec(
        name="rede_neural", factory=create_neural_network, requires_n_features=True
    ),
    "naive_bayes": ModelSpec(name="naive_bayes", factory=create_naive_bayes, has_variants=True),
}

assert set(MODEL_REGISTRY) == set(MODEL_NAMES), "MODEL_REGISTRY divergiu de MODEL_NAMES"


def get_model_spec(name: str) -> ModelSpec:
    """Retorna a especificação registrada para ``name``.

    Raises:
        KeyError: ``name`` não é um dos nomes canônicos de modelo.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Modelo desconhecido: {name!r}. Use um de {tuple(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name]


def create_model(name: str, **kwargs: Any) -> BaseEstimator:
    """Cria um estimador não ajustado pelo nome canônico do modelo.

    Ponto único de criação para os 5 modelos — repassa ``kwargs`` para a
    factory específica registrada em `MODEL_REGISTRY` (ver o nome de cada
    parâmetro em `src.models.<modelo>.create_<modelo>`):

    - ``create_model("arvore_de_decisao", params=...)``
    - ``create_model("regressao_logistica", params=...)``
    - ``create_model("svm", params=...)``
    - ``create_model("rede_neural", n_features=..., params=...)`` —
      ``n_features`` é obrigatório.
    - ``create_model("naive_bayes", variant=..., complement_params=...)``

    Args:
        name: nome canônico do modelo (`src.configs.constants.MODEL_NAMES`).
        **kwargs: repassados diretamente à factory do modelo.

    Returns:
        Estimador scikit-learn não ajustado.

    Raises:
        KeyError: ``name`` desconhecido.
        TypeError: ``kwargs`` incompatíveis com a factory do modelo (ex.:
            ``n_features`` ausente para ``"rede_neural"``).
    """
    spec = get_model_spec(name)
    return spec.factory(**kwargs)


__all__ = ["ModelSpec", "MODEL_REGISTRY", "get_model_spec", "create_model"]
