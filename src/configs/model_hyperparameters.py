"""Hiperparâmetros "vencedores" de cada modelo, centralizados (Tarefa 5).

Nenhum valor aqui foi escolhido, ajustado ou otimizado por esta tarefa —
são exatamente os hiperparâmetros já em uso no projeto, hoje espalhados
entre arquivos `.txt`, um dicionário hardcoded em
`analysis/audit/evaluator.py`, e um `.json`. Esta é a única mudança: um
lugar único para lê-los.

Proveniência de cada valor:

- **Árvore de Decisão** e **Regressão Logística**: conferidos byte a byte
  contra `models/arvore_de_decisao/melhores_hiperparametros_arvore.txt` e
  `models/regrassao_logisticca/melhores_hiperparametros.txt` — os
  arquivos que os próprios notebooks salvaram ao final da busca
  (`RandomizedSearchCV`/`GridSearchCV`). Batem exatamente com
  `analysis/audit/evaluator.py::get_model_specs`.
- **Naive Bayes (ComplementNB)**: conferido contra
  `models/naive_bayes/artifacts/best_params.json`, salvo pelo próprio
  `models/naive_bayes/train.py`. Também bate com `evaluator.py`.
- **SVM** e **Rede Neural**: os notebooks correspondentes não salvam um
  arquivo de hiperparâmetros (só CSVs de métricas). Os valores abaixo
  vêm de `analysis/audit/evaluator.py::get_model_specs`, a única fonte
  disponível — confiável porque os 3 casos acima, onde existe um
  arquivo para conferir, batem exatamente com esse mesmo módulo.
- **Naive Bayes (GaussianNB, BernoulliNB)**: nenhum arquivo do projeto
  registra hiperparâmetros "vencedores" para essas duas variantes
  especificamente (só a variante `complement` venceu a comparação em
  `models/naive_bayes/train.py`). Os valores abaixo são os defaults do
  scikit-learn — documentado explicitamente, para não fingir uma
  precisão que a fonte original não tem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.configs.settings import RANDOM_STATE


@dataclass(frozen=True)
class DecisionTreeHyperparameters:
    """Hiperparâmetros da Árvore de Decisão pós-busca.

    Fonte: `models/arvore_de_decisao/melhores_hiperparametros_arvore.txt`.
    """

    criterion: str = "entropy"
    max_depth: int | None = 15
    min_samples_split: int = 46
    min_samples_leaf: int = 13
    max_features: str | float | None = None
    class_weight: dict[int, int] | str | None = field(default_factory=lambda: {0: 1, 1: 2})
    min_impurity_decrease: float = 0.0012123525181736484
    ccp_alpha: float = 0.0043528172066691975
    splitter: str = "best"
    random_state: int = RANDOM_STATE

    def to_dict(self) -> dict[str, Any]:
        """Kwargs prontos para ``DecisionTreeClassifier(**params.to_dict())``."""
        return asdict(self)


@dataclass(frozen=True)
class LogisticRegressionHyperparameters:
    """Hiperparâmetros da Regressão Logística pós-busca.

    Fonte: `models/regrassao_logisticca/melhores_hiperparametros.txt`.
    """

    C: float = 10.0
    penalty: str = "l1"
    solver: str = "liblinear"
    class_weight: str = "balanced"
    max_iter: int = 1000
    random_state: int = RANDOM_STATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SVMHyperparameters:
    """Hiperparâmetros do SVM pós-busca.

    Fonte: `analysis/audit/evaluator.py::get_model_specs` (nenhum arquivo
    de hiperparâmetros foi salvo por `SVM.ipynb`).
    """

    C: float = 3.14891164795686
    gamma: float = 5.669849511478847
    kernel: str = "rbf"
    class_weight: str = "balanced"
    probability: bool = True
    random_state: int = RANDOM_STATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NeuralNetworkHyperparameters:
    """Hiperparâmetros da Rede Neural (arquitetura "MLP 1 camada + Backprop/SGD").

    Fonte: `models/Redes_Neurais/RedesNeurais.ipynb` (função
    ``build_mlp_backprop`` — a única das 5 arquiteturas comparadas no
    notebook que chegou a ser avaliada com 5-fold CV completo, e por
    isso a "atualmente utilizada" para fins de produção) e
    `analysis/audit/evaluator.py::get_model_specs` para os parâmetros de
    treino (``max_iter``, ``early_stopping``) que a versão scikit-learn
    desta arquitetura precisa e que o notebook original, em Keras,
    controlava de outra forma (``epochs=30`` fixo, sem early stopping).
    Ver `docs/MODELS.md`, seção "Rede Neural — Keras vs. scikit-learn",
    para a justificativa completa dessa reimplementação.
    """

    hidden_layer_activation: str = "relu"
    solver: str = "sgd"
    learning_rate_init: float = 0.01
    momentum: float = 0.9
    max_iter: int = 300
    early_stopping: bool = True
    validation_fraction: float = 0.1
    random_state: int = RANDOM_STATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GaussianNBHyperparameters:
    """Hiperparâmetros do GaussianNB.

    Nenhum valor "vencedor" persistido para esta variante (só
    ``complement`` venceu a comparação original) — usa o default do
    scikit-learn.
    """

    var_smoothing: float = 1e-9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BernoulliNBHyperparameters:
    """Hiperparâmetros do BernoulliNB.

    Nenhum valor "vencedor" persistido para esta variante — usa os
    defaults do scikit-learn.
    """

    alpha: float = 1.0
    binarize: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComplementNBHyperparameters:
    """Hiperparâmetros do ComplementNB — a variante vencedora.

    Fonte: `models/naive_bayes/artifacts/best_params.json`.
    """

    alpha: float = 0.16121212121212125
    norm: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "DecisionTreeHyperparameters",
    "LogisticRegressionHyperparameters",
    "SVMHyperparameters",
    "NeuralNetworkHyperparameters",
    "GaussianNBHyperparameters",
    "BernoulliNBHyperparameters",
    "ComplementNBHyperparameters",
]
