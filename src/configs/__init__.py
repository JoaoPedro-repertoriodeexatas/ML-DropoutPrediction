"""Camada de configuração centralizada do projeto.

Reúne em um só lugar o que hoje está espalhado (e duplicado) entre os
notebooks, `data/preprocessamento.py` e `analysis/audit/`:

- caminhos de arquivos e diretórios (``paths.py``);
- sementes aleatórias e parâmetros gerais de execução (``settings.py``);
- nomes de colunas, listas de features e nomes de arquivo de saída
  (``constants.py``).

Nenhum destes módulos executa lógica de dados — são apenas valores.
"""

from src.configs import constants, paths, settings

__all__ = ["constants", "paths", "settings"]
