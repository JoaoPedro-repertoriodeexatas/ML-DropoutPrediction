"""Engenharia de features e pré-processamento.

- ``base_preprocessing.py`` — transformações comuns aos 5 modelos
  (tratamento de infinito, imputação de ausentes, restrição a colunas
  numéricas). Implementado desde a Tarefa 3.
- ``feature_groups.py`` — conjuntos declarativos de features por modelo
  (implementado na Tarefa 2).
- ``feature_engineering.py`` — engenharia de features específica de cada
  modelo, implementada na Tarefa 4: `FeatureEngineer` expõe
  ``common_features()`` e um ``for_<modelo>()`` por modelo, cada um
  devolvendo um `sklearn.pipeline.Pipeline` não ajustado. Ver
  `docs/FEATURE_ENGINEERING.md`.

Scaling, seleção estatística de features e o universo de features do
Naive Bayes são responsabilidade de `feature_engineering.py`, não deste
pacote de forma genérica — ver `base_preprocessing.py` para a
justificativa de por que essas transformações ficam de fora do
pré-processamento comum. Balanceamento (SMOTE/`class_weight`) continua
fora do escopo deste pacote — é responsabilidade de `src/training/`
(ainda não implementado).
"""
