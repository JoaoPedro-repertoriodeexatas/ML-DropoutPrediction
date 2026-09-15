"""Camada de dados: carregamento, construção de alvo e remoção de vazamento.

Desde a Tarefa 3, este pacote implementa as três primeiras etapas do
pipeline de dados (ver `docs/DATA_PIPELINE.md`):

- ``loading.py`` — localiza e carrega o CSV bruto do INEP;
- ``target.py`` — constrói ``taxa_evasao`` e ``alto_risco_evasao``, com
  duas estratégias nomeadas (``median_split``, replicando
  `data/preprocessamento.py`; ``threshold_20pct``, replicando os 4
  notebooks);
- ``leakage.py`` — fonte única das regras de remoção de colunas com risco
  de vazamento, documentadas por grupo e motivo;
- ``pipeline.py`` — orquestra as três etapas acima mais o pré-processamento
  comum (`src.pipelines.base_preprocessing`);
- ``validation.py`` — compara o resultado do novo pipeline com o do
  pipeline legado (`data/preprocessamento.py`).

A engenharia de features específica de cada modelo (5ª etapa do
diagrama) ainda não está implementada — ver
`src.pipelines.feature_engineering`.
"""
