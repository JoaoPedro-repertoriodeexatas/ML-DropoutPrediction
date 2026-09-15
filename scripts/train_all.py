"""Treina os 5 modelos via `src.training.trainer.Trainer` e imprime a
tabela comparativa (Tarefa 10).

Ponto de entrada único, fora de qualquer notebook — o script pedido
por `docs/VALIDATION_REPORT.md` para provar que "treinar todos os
modelos" não depende de Jupyter.

Uso:
    python scripts/train_all.py
    python scripts/train_all.py --output results/new_architecture/comparison.csv
    python scripts/train_all.py --n-splits 3          # CV mais rápido, para testes

Requer o dataset real do INEP disponível localmente (ver `README.md`,
seção "Dados") — sem ele, cada modelo falha com `FileNotFoundError`
explicando os caminhos verificados; o script reporta isso e continua
para os demais modelos em vez de abortar tudo por um caminho ausente.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.configs.constants import MODEL_NAMES
from src.data.pipeline import DataPipelineResult, run_data_pipeline
from src.evaluation.comparison import ModelComparisonRow, compare_models
from src.training.results import TrainingResult
from src.training.trainer import Trainer

# Cada modelo "notebook-style" (Árvore, LR, SVM, Rede Neural) usa o mesmo
# X; o Naive Bayes usa um universo de dados diferente — ver
# docs/FEATURE_ENGINEERING.md, seção 8.
_NOTEBOOK_STYLE_MODELS: tuple[str, ...] = (
    "arvore_de_decisao",
    "regressao_logistica",
    "svm",
    "rede_neural",
)


def _load_dataset_for(model_name: str) -> DataPipelineResult:
    """Carrega o `X`/`y` apropriado para ``model_name`` (ver docs/FEATURE_ENGINEERING.md, seção 8)."""
    if model_name == "naive_bayes":
        return run_data_pipeline(target_strategy="median_split")
    return run_data_pipeline(
        target_strategy="threshold_20pct",
        missing_strategy="mode",
        include_prefix_leakage_group=False,
    )


def train_all_models(*, n_splits: int, random_state: int) -> dict[str, TrainingResult]:
    """Treina os 5 modelos (Naive Bayes com sua variante vencedora, `complement`).

    Args:
        n_splits: número de folds do `Trainer` (padrão do projeto: 5).
        random_state: semente do `Trainer` (padrão do projeto: 42).

    Returns:
        ``{nome_canônico: TrainingResult}`` — só inclui modelos treinados
        com sucesso; modelos que falharem (ex.: dataset ausente) são
        reportados em stderr e omitidos do dicionário.
    """
    trainer = Trainer(n_splits=n_splits, random_state=random_state)
    results: dict[str, TrainingResult] = {}

    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 70}\nTreinando: {model_name}\n{'=' * 70}")
        try:
            data_result = _load_dataset_for(model_name)
        except FileNotFoundError as exc:
            print(f"AVISO: dataset indisponível para {model_name}: {exc}", file=sys.stderr)
            continue

        kwargs = {"variant": "complement"} if model_name == "naive_bayes" else {}
        try:
            result = trainer.train(model_name, data_result.X, data_result.y, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"ERRO ao treinar {model_name}: {exc}", file=sys.stderr)
            continue

        results[model_name] = result
        print(
            f"OK — F1={result.metrics['f1']:.4f} ± {result.metrics_std['f1']:.4f} | "
            f"Recall={result.metrics['recall']:.4f} | "
            f"estratégia de desbalanceamento: {result.imbalance_strategy}"
        )

    return results


def build_comparison_table(results: dict[str, TrainingResult]) -> pd.DataFrame:
    """Monta a tabela `Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Std`."""
    rows = [
        ModelComparisonRow(model_name=name, mean_metrics=result.metrics, std_metrics=result.metrics_std)
        for name, result in results.items()
    ]
    return compare_models(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5, help="Número de folds (padrão: 5).")
    parser.add_argument("--random-state", type=int, default=42, help="Semente aleatória (padrão: 42).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Se fornecido, salva a tabela comparativa nesse caminho CSV.",
    )
    args = parser.parse_args()

    results = train_all_models(n_splits=args.n_splits, random_state=args.random_state)

    if not results:
        print(
            "\nNenhum modelo foi treinado — dataset real não encontrado. "
            "Ver README.md, seção 'Dados', para baixar o CSV do INEP.",
            file=sys.stderr,
        )
        return 1

    table = build_comparison_table(results)
    print(f"\n{'=' * 70}\nComparação entre modelos\n{'=' * 70}")
    print(table.to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\nTabela salva em: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
