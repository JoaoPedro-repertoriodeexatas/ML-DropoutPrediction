"""Testes de `src.models` (Tarefa 5).

Verificam que os 5 modelos podem ser instanciados sem Jupyter, que os
hiperparâmetros vêm da configuração centralizada (não de valores mágicos
espalhados) e que nenhuma factory ajusta (`fit`) o estimador — é
responsabilidade só de definição/configuração.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from src.configs.constants import MODEL_NAMES
from src.configs.model_hyperparameters import (
    BernoulliNBHyperparameters,
    ComplementNBHyperparameters,
    DecisionTreeHyperparameters,
    GaussianNBHyperparameters,
    LogisticRegressionHyperparameters,
    NeuralNetworkHyperparameters,
    SVMHyperparameters,
)
from src.models.decision_tree import create_decision_tree
from src.models.logistic_regression import create_logistic_regression
from src.models.naive_bayes import create_all_naive_bayes_variants, create_naive_bayes
from src.models.neural_network import compute_hidden_layer_size, create_neural_network
from src.models.registry import MODEL_REGISTRY, create_model, get_model_spec
from src.models.svm import create_svm


def _assert_not_fitted(model) -> None:
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


# --------------------------------------------------------------------------
# 1. Cada um dos 5 modelos pode ser instanciado
# --------------------------------------------------------------------------


def test_create_decision_tree_returns_unfitted_classifier() -> None:
    model = create_decision_tree()
    assert isinstance(model, DecisionTreeClassifier)
    _assert_not_fitted(model)


def test_create_logistic_regression_returns_unfitted_classifier() -> None:
    model = create_logistic_regression()
    assert isinstance(model, LogisticRegression)
    _assert_not_fitted(model)


def test_create_svm_returns_unfitted_classifier() -> None:
    model = create_svm()
    assert isinstance(model, SVC)
    _assert_not_fitted(model)


def test_create_neural_network_returns_unfitted_classifier() -> None:
    model = create_neural_network(n_features=21)
    assert isinstance(model, MLPClassifier)
    _assert_not_fitted(model)


@pytest.mark.parametrize(
    ("variant", "expected_class"),
    [("gaussian", GaussianNB), ("bernoulli", BernoulliNB), ("complement", ComplementNB)],
)
def test_create_naive_bayes_returns_unfitted_classifier_per_variant(variant, expected_class) -> None:
    model = create_naive_bayes(variant)
    assert isinstance(model, expected_class)
    _assert_not_fitted(model)


def test_create_all_naive_bayes_variants_returns_all_three() -> None:
    models = create_all_naive_bayes_variants()

    assert set(models) == {"gaussian", "bernoulli", "complement"}
    assert isinstance(models["gaussian"], GaussianNB)
    assert isinstance(models["bernoulli"], BernoulliNB)
    assert isinstance(models["complement"], ComplementNB)


def test_create_naive_bayes_unknown_variant_raises_value_error() -> None:
    with pytest.raises(ValueError):
        create_naive_bayes("nao_existe")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# create_model(...) — ponto único de criação pelo nome canônico
# --------------------------------------------------------------------------


def test_create_model_instantiates_all_five_registered_models() -> None:
    assert set(MODEL_REGISTRY) == set(MODEL_NAMES)

    for name in MODEL_NAMES:
        kwargs = {"n_features": 21} if name == "rede_neural" else {}
        model = create_model(name, **kwargs)
        _assert_not_fitted(model)


def test_create_model_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        create_model("modelo_que_nao_existe")


def test_create_model_neural_network_without_n_features_raises_type_error() -> None:
    with pytest.raises(TypeError):
        create_model("rede_neural")  # n_features é obrigatório


def test_get_model_spec_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_model_spec("modelo_que_nao_existe")


# --------------------------------------------------------------------------
# 2. Hiperparâmetros vêm da configuração centralizada, não de literais soltos
# --------------------------------------------------------------------------


def test_decision_tree_uses_confirmed_winning_hyperparameters() -> None:
    """Conferido contra `models/arvore_de_decisao/melhores_hiperparametros_arvore.txt`."""
    model = create_decision_tree()
    assert model.criterion == "entropy"
    assert model.max_depth == 15
    assert model.min_samples_split == 46
    assert model.min_samples_leaf == 13
    assert model.class_weight == {0: 1, 1: 2}
    assert model.ccp_alpha == pytest.approx(0.0043528172066691975)


def test_logistic_regression_uses_confirmed_winning_hyperparameters() -> None:
    """Conferido contra `models/regrassao_logisticca/melhores_hiperparametros.txt`."""
    model = create_logistic_regression()
    assert model.C == 10.0
    assert model.penalty == "l1"
    assert model.solver == "liblinear"
    assert model.class_weight == "balanced"


def test_complement_nb_uses_confirmed_winning_hyperparameters() -> None:
    """Conferido contra `models/naive_bayes/artifacts/best_params.json`."""
    model = create_naive_bayes("complement")
    assert model.alpha == pytest.approx(0.16121212121212125)
    assert model.norm is True


def test_custom_hyperparameters_override_defaults() -> None:
    """Configuração é separada da definição: passar `params` customizado
    deve mudar o estimador, não o valor hardcoded na factory."""
    custom = DecisionTreeHyperparameters(max_depth=3, criterion="gini")
    model = create_decision_tree(params=custom)

    assert model.max_depth == 3
    assert model.criterion == "gini"


def test_all_hyperparameter_dataclasses_are_immutable() -> None:
    """`frozen=True` garante que configuração não é mutada por acidente
    depois de criada — parte do requisito "configuração separada"."""
    for hyperparams in (
        DecisionTreeHyperparameters(),
        LogisticRegressionHyperparameters(),
        SVMHyperparameters(),
        NeuralNetworkHyperparameters(),
        GaussianNBHyperparameters(),
        BernoulliNBHyperparameters(),
        ComplementNBHyperparameters(),
    ):
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError é um subtipo de AttributeError
            hyperparams.alpha = -1  # type: ignore[attr-defined]  # qualquer atributo: frozen bloqueia todos


# --------------------------------------------------------------------------
# Rede Neural — arquitetura dinâmica preservada (não fixa em 64 neurônios)
# --------------------------------------------------------------------------


def test_compute_hidden_layer_size_matches_notebook_formula() -> None:
    # n_neurons = (n_features + 1) // 2, exatamente como no notebook original.
    assert compute_hidden_layer_size(21) == 11
    assert compute_hidden_layer_size(4) == 2
    assert compute_hidden_layer_size(1) == 1


def test_compute_hidden_layer_size_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        compute_hidden_layer_size(0)
    with pytest.raises(ValueError):
        compute_hidden_layer_size(-5)


def test_create_neural_network_hidden_layer_matches_n_features() -> None:
    model = create_neural_network(n_features=21)
    assert model.hidden_layer_sizes == (11,)

    model_small = create_neural_network(n_features=4)
    assert model_small.hidden_layer_sizes == (2,)


def test_create_neural_network_preserves_sgd_momentum_architecture() -> None:
    model = create_neural_network(n_features=21)
    assert model.solver == "sgd"
    assert model.activation == "relu"
    assert model.learning_rate_init == 0.01
    assert model.momentum == 0.9


# --------------------------------------------------------------------------
# Não possuir lógica de avaliação/dataset/gráficos/arquivos
# --------------------------------------------------------------------------


def test_model_modules_expose_no_evaluation_or_io_helpers() -> None:
    """Sanidade estrutural: os módulos de modelo só exportam factories de
    estimador — nada de `evaluate`, `load`, `plot`, `save`."""
    import src.models.decision_tree as dt_module
    import src.models.logistic_regression as lr_module
    import src.models.naive_bayes as nb_module
    import src.models.neural_network as nn_module
    import src.models.svm as svm_module

    forbidden_prefixes = ("evaluate", "load", "plot", "save", "fit_and", "train")
    for module in (dt_module, lr_module, svm_module, nn_module, nb_module):
        for exported_name in module.__all__:
            lowered = exported_name.lower()
            assert not any(lowered.startswith(prefix) for prefix in forbidden_prefixes), (
                f"{module.__name__}.{exported_name} parece lógica de avaliação/IO, "
                "fora do escopo da Tarefa 5"
            )


if __name__ == "__main__":
    test_functions = [
        obj
        for name, obj in list(globals().items())
        if name.startswith("test_") and not hasattr(obj, "pytestmark")
    ]
    failures = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"OK   - {test_fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL - {test_fn.__name__}: {exc}")
    total = len(test_functions)
    print(f"\n{total - failures}/{total} testes passaram (parametrizados exigem pytest).")
    sys.exit(1 if failures else 0)
