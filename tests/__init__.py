"""Testes automatizados da nova arquitetura (`src/`).

- `test_imports.py` — estrutura e configuração (Tarefa 2): importabilidade
  dos pacotes, ausência de features vazadas nas listas declarativas.
- `test_data_loading.py`, `test_target_construction.py`,
  `test_leakage_removal.py`, `test_feature_selection.py`,
  `test_invalid_values.py`, `test_pipeline_integration.py` — comportamento
  da camada de dados (Tarefa 3): carregamento, construção de alvo,
  remoção de vazamento, seleção de features, tratamento de valores
  inválidos e orquestração de ponta a ponta. Ver `docs/DATA_PIPELINE.md`.
- `test_feature_engineering.py` — engenharia de features específica de
  cada modelo (Tarefa 4): `FeatureEngineer.for_<modelo>()`, features
  proibidas nunca presentes na saída, ausência de NaN inesperado,
  contagem determinística de features e reprodutibilidade. Ver
  `docs/FEATURE_ENGINEERING.md`.

Testes de treino/threshold tuning serão adicionados junto com a lógica
correspondente, fase a fase — ver ARCHITECTURE_AUDIT.md, seção 8.
"""
