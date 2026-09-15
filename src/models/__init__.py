"""Definição dos 5 estimadores do projeto, sem depender de notebooks.

Implementado na Tarefa 5. Cada módulo expõe uma função
``create_<modelo>(...)`` que devolve um estimador scikit-learn **não
ajustado**, configurado com os hiperparâmetros centralizados em
`src.configs.model_hyperparameters`:

- ``decision_tree.py`` — `create_decision_tree`
- ``logistic_regression.py`` — `create_logistic_regression`
- ``svm.py`` — `create_svm`
- ``neural_network.py`` — `create_neural_network` (arquitetura MLP 1
  camada + SGD, reimplementada com `sklearn.neural_network.MLPClassifier`
  em vez do Keras original — ver docstring do módulo)
- ``naive_bayes.py`` — `create_naive_bayes` / `create_all_naive_bayes_variants`
  (3 variantes: Gaussian, Bernoulli, Complement)

``registry.py`` reúne as 5 factories em `MODEL_REGISTRY` e expõe um
ponto único de criação, `create_model(name, **kwargs)`. Ver
`docs/MODELS.md`.

Nenhum destes módulos carrega dados, gera gráficos, salva arquivos ou
avalia modelos — só definem/configuram o estimador.
"""
