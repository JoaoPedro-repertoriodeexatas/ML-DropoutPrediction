"""Definição do estimador da Rede Neural (Tarefa 5).

Responsabilidade única: construir um `MLPClassifier` **não ajustado**,
preservando a arquitetura "MLP 1 camada oculta + Backprop (SGD)" do
notebook original (`RedesNeurais.ipynb`, função ``build_mlp_backprop``)
— a única das 5 arquiteturas comparadas ali que chegou a ser validada
com 5-fold CV completo, e por isso a arquitetura "atualmente utilizada"
para fins de produção. Não carrega dados, não avalia, não plota, não
salva nada.

## Keras vs. scikit-learn

O notebook original constrói a rede em Keras/TensorFlow:

```python
Sequential([
    Dense(n_neurons, input_dim=n_features, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
              loss='binary_crossentropy', metrics=['accuracy'])
```

Este módulo reimplementa a mesma arquitetura com
`sklearn.neural_network.MLPClassifier` em vez de Keras — pelo mesmo
motivo que `analysis/audit/evaluator.py` já havia feito essa mesma
substituição para reavaliar a Rede Neural junto dos outros 4 modelos:
manter todo o projeto em um único framework de ML (scikit-learn), sem
depender do TensorFlow (~500MB, e o próprio `README.md` já trata sua
instalação como opcional/manual, fora de `requirements.txt`). A
equivalência é direta, hiperparâmetro a hiperparâmetro:

| Notebook (Keras) | Este módulo (`MLPClassifier`) |
|---|---|
| `Dense(n_neurons, activation='relu')` | `hidden_layer_sizes=(n_neurons,)`, `activation='relu'` |
| `Dense(1, activation='sigmoid')` | (implícito — saída binária de `MLPClassifier` já usa sigmoid + log-loss) |
| `SGD(learning_rate=0.01, momentum=0.9)` | `solver='sgd'`, `learning_rate_init=0.01`, `momentum=0.9` |
| `loss='binary_crossentropy'` | (log-loss é a função de perda padrão de `MLPClassifier` para classificação) |

**Divergência documentada, não escondida:** o notebook treina por
`epochs=30` fixos, sem early stopping; a versão scikit-learn usa
`max_iter=300` com `early_stopping=True` (parâmetros de
`analysis/audit/evaluator.py`, necessários porque `MLPClassifier` não
tem o conceito de "épocas" do Keras da mesma forma). Isso é uma
diferença no *critério de parada do treino*, não na *arquitetura* da
rede (número de camadas, neurônios, ativação, otimizador) — que é
exatamente o que a Tarefa 5 pede para preservar.

**O número de neurônios da camada oculta não é fixo.** O notebook
calcula ``n_neurons = (n_features + 1) // 2`` dinamicamente a partir do
número de features de entrada — por isso `create_neural_network` exige
``n_features`` como argumento, em vez de embutir um valor fixo (ex.: os
64 neurônios hardcoded em `analysis/audit/evaluator.py`, que já é uma
simplificação daquele módulo específico, não a arquitetura original).
"""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier

from src.configs.model_hyperparameters import NeuralNetworkHyperparameters


def compute_hidden_layer_size(n_features: int) -> int:
    """Réplica exata de ``n_neurons = (n_features + 1) // 2`` do notebook original."""
    if n_features <= 0:
        raise ValueError(f"n_features deve ser positivo, recebido: {n_features}")
    return (n_features + 1) // 2


def create_neural_network(
    n_features: int,
    params: NeuralNetworkHyperparameters | None = None,
) -> MLPClassifier:
    """Cria um `MLPClassifier` não ajustado, com 1 camada oculta.

    Args:
        n_features: número de features de entrada — determina o número
            de neurônios da camada oculta via `compute_hidden_layer_size`,
            replicando o cálculo dinâmico do notebook original.
        params: demais hiperparâmetros. Se ``None`` (padrão), usa
            `NeuralNetworkHyperparameters()`.

    Returns:
        Estimador sklearn pronto para `.fit(X_train, y_train)`.

    Raises:
        ValueError: ``n_features`` não positivo.
    """
    effective_params = params or NeuralNetworkHyperparameters()
    hidden_layer_size = compute_hidden_layer_size(n_features)

    return MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation=effective_params.hidden_layer_activation,
        solver=effective_params.solver,
        learning_rate_init=effective_params.learning_rate_init,
        momentum=effective_params.momentum,
        max_iter=effective_params.max_iter,
        early_stopping=effective_params.early_stopping,
        validation_fraction=effective_params.validation_fraction,
        random_state=effective_params.random_state,
    )


__all__ = ["compute_hidden_layer_size", "create_neural_network"]
