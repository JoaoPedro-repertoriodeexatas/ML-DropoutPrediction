# Definição dos modelos em Python (Tarefa 5)

**Branch:** `audit/architecture-review`
**Baseado em:** [`ARCHITECTURE_AUDIT.md`](../ARCHITECTURE_AUDIT.md), [`docs/FEATURE_ENGINEERING.md`](FEATURE_ENGINEERING.md)

Migra a *definição* dos 5 estimadores para fora dos notebooks. Não migra
treino, avaliação, plots ou os notebooks em si — eles continuam existindo
e funcionando exatamente como antes.

---

## 1. Onde a solução vive

```text
src/
├── configs/
│   └── model_hyperparameters.py   # hiperparâmetros centralizados (dataclasses frozen)
└── models/
    ├── decision_tree.py            # create_decision_tree(...)
    ├── logistic_regression.py      # create_logistic_regression(...)
    ├── svm.py                      # create_svm(...)
    ├── neural_network.py           # create_neural_network(n_features, ...)
    ├── naive_bayes.py              # create_naive_bayes(variant, ...) / create_all_naive_bayes_variants()
    └── registry.py                 # MODEL_REGISTRY + create_model(name, **kwargs)
```

Cada `create_<modelo>` devolve um estimador scikit-learn **não
ajustado**. Nenhum módulo carrega dataset, gera gráfico, salva arquivo
ou avalia — só define/configura o estimador, como pedido.

```python
from src.models.registry import create_model

model = create_model("arvore_de_decisao")
model = create_model("rede_neural", n_features=21)
model = create_model("naive_bayes", variant="gaussian")
```

---

## 2. Hiperparâmetros centralizados

`src/configs/model_hyperparameters.py` — uma `@dataclass(frozen=True)`
por modelo/variante. Nenhum valor foi escolhido ou ajustado nesta
tarefa; cada um foi conferido contra a fonte real do projeto:

| Modelo | Fonte conferida | Como foi conferido |
|---|---|---|
| Árvore de Decisão | `models/arvore_de_decisao/melhores_hiperparametros_arvore.txt` | Comparação byte a byte — bate exatamente |
| Regressão Logística | `models/regrassao_logisticca/melhores_hiperparametros.txt` | Comparação byte a byte — bate exatamente |
| Naive Bayes (Complement) | `models/naive_bayes/artifacts/best_params.json` | Comparação byte a byte — bate exatamente |
| SVM | `analysis/audit/evaluator.py::get_model_specs` | Nenhum arquivo próprio salvo pelo notebook; fonte única disponível |
| Rede Neural | `models/Redes_Neurais/RedesNeurais.ipynb` (arquitetura) + `analysis/audit/evaluator.py` (parâmetros de treino) | Arquitetura do notebook; critério de parada do `evaluator.py` (ver seção 4) |
| Naive Bayes (Gaussian, Bernoulli) | — | Nenhum valor "vencedor" persistido para essas variantes; usa defaults do scikit-learn, documentado explicitamente como tal |

Os 3 casos com arquivo próprio batendo exatamente com
`analysis/audit/evaluator.py` dão confiança de que os outros 2 valores
sem arquivo (SVM, Rede Neural) também vêm de execuções reais, não de
invenção.

---

## 3. Tabela de hiperparâmetros

| Modelo | Hiperparâmetro | Valor | Fonte |
|---|---|---|---|
| Árvore de Decisão | `criterion` | `"entropy"` | notebook |
| | `max_depth` | `15` | notebook |
| | `min_samples_split` | `46` | notebook |
| | `min_samples_leaf` | `13` | notebook |
| | `class_weight` | `{0: 1, 1: 2}` | notebook |
| | `min_impurity_decrease` | `0.0012123525181736484` | notebook |
| | `ccp_alpha` | `0.0043528172066691975` | notebook (pós-poda) |
| Regressão Logística | `C` | `10.0` | notebook |
| | `penalty` | `"l1"` | notebook |
| | `solver` | `"liblinear"` | notebook |
| | `class_weight` | `"balanced"` | notebook |
| SVM | `C` | `3.14891164795686` | `evaluator.py` |
| | `gamma` | `5.669849511478847` | `evaluator.py` |
| | `kernel` | `"rbf"` | `evaluator.py` |
| | `class_weight` | `"balanced"` | `evaluator.py` |
| Rede Neural | `hidden_layer_sizes` | `((n_features+1)//2,)` — dinâmico | notebook |
| | `activation` | `"relu"` | notebook |
| | `solver` | `"sgd"` | notebook |
| | `learning_rate_init` | `0.01` | notebook |
| | `momentum` | `0.9` | notebook |
| | `max_iter` / `early_stopping` | `300` / `True` | `evaluator.py` (ver seção 4) |
| Naive Bayes — Complement | `alpha` | `0.16121212121212125` | `best_params.json` |
| | `norm` | `True` | `best_params.json` |
| Naive Bayes — Gaussian | `var_smoothing` | `1e-9` (default sklearn) | sem valor vencedor persistido |
| Naive Bayes — Bernoulli | `alpha` / `binarize` | `1.0` / `0.0` (defaults sklearn) | sem valor vencedor persistido |

Todos os estimadores recebem `random_state=42` (`src.configs.settings.RANDOM_STATE`).

---

## 4. Rede Neural — Keras vs. scikit-learn

O notebook original (`RedesNeurais.ipynb`) constrói a rede em
Keras/TensorFlow. `create_neural_network` reimplementa a mesma
arquitetura com `sklearn.neural_network.MLPClassifier` — a mesma
substituição que `analysis/audit/evaluator.py` já havia feito para
reavaliar a Rede Neural junto dos outros 4 modelos.

**Motivo:** manter o projeto inteiro em um único framework de ML
(scikit-learn), sem exigir TensorFlow (~500MB) como dependência — o
próprio `README.md` já trata a instalação do TensorFlow como manual e
separada, fora de `requirements.txt`.

**Equivalência preservada** (arquitetura, não framework):

| Notebook (Keras) | `create_neural_network` (`MLPClassifier`) |
|---|---|
| `Dense(n_neurons, activation='relu')` | `hidden_layer_sizes=(n_neurons,)`, `activation='relu'` |
| `Dense(1, activation='sigmoid')` | saída binária de `MLPClassifier` já usa sigmoid + log-loss |
| `SGD(learning_rate=0.01, momentum=0.9)` | `solver='sgd'`, `learning_rate_init=0.01`, `momentum=0.9` |
| `loss='binary_crossentropy'` | log-loss é o padrão de `MLPClassifier` para classificação |
| `n_neurons = (n_features + 1) // 2` (dinâmico) | `compute_hidden_layer_size(n_features)` — mesma fórmula |

**Divergência documentada, não escondida:** o notebook treina por
`epochs=30` fixos sem early stopping; esta reimplementação usa
`max_iter=300` com `early_stopping=True` (valores de
`analysis/audit/evaluator.py`), porque `MLPClassifier` não tem o
conceito de "épocas" do Keras da mesma forma. É uma diferença no
*critério de parada do treino*, não na *arquitetura* (camadas,
neurônios, ativação, otimizador) — que é o que a Tarefa 5 pede para
preservar. O número de neurônios da camada oculta **não é fixo**: seria
um erro embutir os 64 neurônios hardcoded em `evaluator.py` (uma
simplificação daquele módulo específico) em vez da fórmula dinâmica do
notebook original — por isso `n_features` é um parâmetro obrigatório de
`create_neural_network`, não uma constante.

---

## 5. Naive Bayes — 3 variantes e a lógica de comparação

`create_naive_bayes(variant, ...)` cria qualquer uma das 3 variantes
usadas no projeto (`gaussian`, `bernoulli`, `complement`), reimplementando
`models/naive_bayes/model_factory.py::create_model`.
`create_all_naive_bayes_variants()` cria as 3 de uma vez — preserva a
*estrutura* que torna a comparação de `models/naive_bayes/train.py`
possível (gerar candidatos de cada variante para depois avaliar), sem
reimplementar a busca de hiperparâmetros nem a validação cruzada em si:
isso é lógica de treino/avaliação, fora do escopo desta tarefa
("não possuir lógica de avaliação").

A variante vencedora da comparação original foi `ComplementNB`
(`models/naive_bayes/artifacts/best_params.json`) — por isso é o padrão
de `create_naive_bayes()` quando `variant` não é especificado.

---

## 6. Testes

`tests/test_models.py` — 23 testes:

- **Instanciação:** cada um dos 5 modelos (e as 3 variantes de Naive
  Bayes) pode ser criado sem Jupyter, e nenhum vem `fit`ado
  (`check_is_fitted` levanta `NotFittedError` em todos).
- **`create_model(...)` (registry):** cria os 5 modelos pelo nome
  canônico; nome desconhecido levanta `KeyError`; `rede_neural` sem
  `n_features` levanta `TypeError` (parâmetro obrigatório, não um valor
  mágico com default escondido).
- **Hiperparâmetros centralizados:** os 3 casos com arquivo de origem
  (Árvore, Regressão Logística, Naive Bayes Complement) são conferidos
  contra os valores exatos desses arquivos; hiperparâmetros customizados
  sobrescrevem os defaults corretamente; todas as dataclasses de
  configuração são imutáveis (`frozen=True`).
- **Rede Neural:** `compute_hidden_layer_size` replica a fórmula
  `(n_features+1)//2` do notebook para vários valores de entrada; a
  arquitetura (solver, ativação, learning rate, momentum) é preservada.
- **Escopo:** checagem estrutural de que nenhum módulo de modelo exporta
  algo parecido com avaliação/carregamento/plot/salvamento.

### Execução real

```text
python -m pytest tests/ -v
...
======================= 99 passed in 4.30s =======================
```

Os 23 testes novos mais os 76 já existentes das Tarefas 2–4 passam
juntos, sem regressão.

---

## 7. O que fica fora do escopo desta etapa

- **Treino** (`.fit(...)`) — nenhum modelo é ajustado aqui.
- **Busca de hiperparâmetros** — os valores centralizados são os já
  encontrados pelas buscas originais; refazer essa busca é
  responsabilidade de uma fase futura de `src/training/`.
- **Migração dos notebooks** — continuam existindo e funcionando.
- **Comparação/seleção de modelo** (ranking, métricas) — pertence a
  `src/evaluation/`, ainda não implementado.
