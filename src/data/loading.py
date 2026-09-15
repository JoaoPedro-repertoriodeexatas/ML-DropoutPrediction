"""Carregamento do dataset bruto do Censo da Educação Superior (INEP).

Responsabilidade única deste módulo: localizar e ler
``MICRODADOS_CADASTRO_CURSOS_2024.CSV``. Nenhuma transformação de dados
acontece aqui — nem construção de alvo (`src.data.target`), nem remoção
de vazamento (`src.data.leakage`), nem pré-processamento
(`src.pipelines.base_preprocessing`).

Antes da Tarefa 3, a resolução do caminho do dataset e a leitura do CSV
estavam duplicadas em 3 lugares com pequenas divergências:
`data/preprocessamento.py::_carregar_dados_brutos`,
`analysis/audit/data.py::load_notebook_dataset`, e a célula "CARREGAMENTO
DOS DADOS" de cada um dos 4 notebooks. Todos os caminhos candidatos e o
nome do arquivo já foram centralizados em `src.configs.paths`
(Tarefa 2) — este módulo é o único ponto que efetivamente abre o arquivo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.configs.paths import resolve_dataset_path

# Mesmas colunas lidas pelos 4 notebooks (COLS_CURSOS) — carregar só essas
# evita o custo de ler as ~100+ colunas do CSV completo quando o alvo é a
# estratégia `threshold_20pct`. `data/preprocessamento.py` (estratégia
# `median_split`) lê o CSV completo, porque depende de colunas fora dessa
# lista (ex.: `QT_CONC*`, usadas só para o filtro de vazamento, não como
# feature). Por isso `load_raw_dataset` não restringe `usecols` por
# padrão — a restrição de colunas é responsabilidade da camada de
# features (`src/pipelines/`), não do carregamento.
CSV_READ_KWARGS: dict[str, object] = {
    "sep": ";",
    "encoding": "latin1",
    "low_memory": False,
}


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Carrega o CSV bruto do Censo da Educação Superior.

    Args:
        path: caminho explícito do CSV. Se ``None`` (padrão), resolve
            automaticamente via ``src.configs.paths.resolve_dataset_path``
            (procura na raiz do projeto, em ``data/`` e em
            ``~/Downloads/microdados_censo_da_educacao_superior_2024/dados/``).

    Returns:
        DataFrame com os microdados brutos, sem nenhuma transformação
        além da leitura (tipos ainda não convertidos, sem imputação).

    Raises:
        FileNotFoundError: nenhum caminho candidato existe.
        RuntimeError: o arquivo existe mas não pôde ser lido (formato
            inesperado, encoding incorreto, arquivo corrompido).
    """
    resolved_path = path if path is not None else resolve_dataset_path()

    try:
        return pd.read_csv(resolved_path, **CSV_READ_KWARGS)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Falha ao carregar o dataset em {resolved_path}.") from exc


__all__ = ["load_raw_dataset", "resolve_dataset_path", "CSV_READ_KWARGS"]
