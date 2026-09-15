"""Camada centralizada de treinamento (Tarefa 6).

`Trainer.train(model_name="svm", X=X, y=y)` substitui a lógica de `fit`,
cross-validation, tratamento de desbalanceamento e threshold tuning hoje
espalhada e duplicada nos 4 notebooks e em `models/naive_bayes/train.py`.
Não substitui os notebooks ainda — eles continuam existindo.

Por fold de `StratifiedKFold` (5 folds, `random_state` fixo):

1. separa uma fatia de **calibração** dentro do treino do fold (nunca do
   fold de validação — ver `src.training.cross_validation.split_train_calibration`);
2. ajusta a engenharia de features (`FeatureEngineer.for_<modelo>()`,
   Tarefa 4) **só** na fatia de treino-menos-calibração;
3. aplica a estratégia de desbalanceamento **original** do modelo
   (`src.training.imbalance`) — nunca em calibração/validação;
4. ajusta o modelo (`src.models.registry.create_model`, Tarefa 5);
5. calibra o threshold de decisão na fatia de calibração
   (`src.training.threshold_tuning`);
6. avalia no fold de validação, com o threshold calibrado.

Nenhum gráfico é gerado. Nenhum arquivo é salvo. O resultado é só a
estrutura `TrainingResult` (`src.training.results`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.configs.settings import CALIBRATION_SPLIT_SIZE, CV_FOLDS, CV_SHUFFLE, RANDOM_STATE
from src.models.registry import create_model
from src.pipelines.feature_engineering import FeatureEngineer
from src.training.cross_validation import (
    aggregate_fold_metrics,
    compute_fold_metrics,
    make_stratified_kfold,
    split_train_calibration,
)
from src.training.imbalance import apply_smote, compute_balanced_sample_weight, get_imbalance_strategy
from src.training.results import FoldResult, TrainingResult
from src.training.threshold_tuning import find_best_threshold

_NAIVE_BAYES_VARIANT_PARAM_KWARG: dict[str, str] = {
    "gaussian": "gaussian_params",
    "bernoulli": "bernoulli_params",
    "complement": "complement_params",
}


def _feature_pipeline_for(
    model_name: str,
    engineer: FeatureEngineer,
    *,
    variant: str | None,
    neural_network_scale: bool,
):
    """Devolve um `Pipeline` de feature engineering **não ajustado** para
    ``model_name`` — a fábrica correta de `FeatureEngineer` (Tarefa 4)."""
    if model_name == "arvore_de_decisao":
        return engineer.for_decision_tree()
    if model_name == "regressao_logistica":
        return engineer.for_logistic_regression()
    if model_name == "svm":
        return engineer.for_svm()
    if model_name == "rede_neural":
        return engineer.for_neural_network(scale=neural_network_scale)
    if model_name == "naive_bayes":
        return engineer.for_naive_bayes(variant or "complement")
    raise KeyError(f"Modelo desconhecido: {model_name!r}.")


def _predict_positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Probabilidade da classe positiva — os 5 modelos deste projeto
    suportam `predict_proba` (o SVM é criado com `probability=True`, ver
    `SVMHyperparameters`), então não é preciso o caminho alternativo via
    `decision_function` que os notebooks originais usavam para o SVM."""
    return model.predict_proba(X)[:, 1]


def _build_model_creation_kwargs(
    model_name: str,
    *,
    n_features: int,
    variant: str | None,
    model_params: Any | None,
) -> dict[str, Any]:
    if model_name == "rede_neural":
        kwargs: dict[str, Any] = {"n_features": n_features}
        if model_params is not None:
            kwargs["params"] = model_params
        return kwargs

    if model_name == "naive_bayes":
        effective_variant = variant or "complement"
        kwargs = {"variant": effective_variant}
        if model_params is not None:
            kwargs[_NAIVE_BAYES_VARIANT_PARAM_KWARG[effective_variant]] = model_params
        return kwargs

    return {"params": model_params} if model_params is not None else {}


@dataclass
class Trainer:
    """Orquestra cross-validation, desbalanceamento e threshold tuning.

    Attributes:
        n_splits: número de folds (padrão: 5, `src.configs.settings.CV_FOLDS`).
        shuffle: embaralhar antes de dividir os folds (padrão: `True`).
        random_state: semente para o `StratifiedKFold`, a divisão de
            calibração e o SMOTE (padrão: `src.configs.settings.RANDOM_STATE`).
        calibration_split_size: fração do fold de treino reservada para
            calibrar o threshold (padrão: 0.2).
    """

    n_splits: int = CV_FOLDS
    shuffle: bool = CV_SHUFFLE
    random_state: int = RANDOM_STATE
    calibration_split_size: float = CALIBRATION_SPLIT_SIZE

    def train(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        variant: str | None = None,
        model_params: Any | None = None,
        feature_engineer: FeatureEngineer | None = None,
        neural_network_scale: bool = False,
    ) -> TrainingResult:
        """Executa `StratifiedKFold` de 5 folds para ``model_name``.

        Args:
            model_name: nome canônico do modelo
                (`src.configs.constants.MODEL_NAMES`).
            X: features de entrada (ex.: saída de
                `src.data.pipeline.run_data_pipeline`, ainda **sem**
                passar pela engenharia de features específica do modelo
                — isso acontece aqui, por fold).
            y: alvo binário, mesmo índice de ``X``.
            variant: variante de Naive Bayes (``"gaussian"``,
                ``"bernoulli"`` ou ``"complement"``) — ignorado para os
                outros modelos.
            model_params: hiperparâmetros customizados (ver
                `src.configs.model_hyperparameters`); ``None`` usa os
                valores vencedores centralizados na Tarefa 5.
            feature_engineer: instância de `FeatureEngineer` a usar;
                ``None`` cria uma com a configuração padrão.
            neural_network_scale: repassado a
                `FeatureEngineer.for_neural_network(scale=...)` — ``False``
                por padrão, replicando o notebook original (ver
                `docs/FEATURE_ENGINEERING.md`).

        Returns:
            `TrainingResult` com modelos, parâmetros, métricas,
            informação por fold, predições e probabilidades
            out-of-fold.

        Raises:
            KeyError: ``model_name`` desconhecido.
        """
        imbalance_strategy = get_imbalance_strategy(model_name)  # valida o nome cedo
        engineer = feature_engineer or FeatureEngineer()
        skf = make_stratified_kfold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )

        y_true_full = y.to_numpy()
        n_samples = len(y_true_full)
        out_of_fold_predictions = np.full(n_samples, fill_value=-1, dtype=int)
        out_of_fold_probabilities = np.full(n_samples, fill_value=np.nan, dtype=float)

        fold_results: list[FoldResult] = []
        fold_metric_dicts: list[dict[str, float]] = []
        fitted_feature_pipelines: list[Any] = []
        fitted_models: list[Any] = []

        for fold_index, (train_idx, val_idx) in enumerate(skf.split(X, y_true_full), start=1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            X_fit, X_calib, y_fit, y_calib = split_train_calibration(
                X_train,
                y_train,
                test_size=self.calibration_split_size,
                random_state=self.random_state,
            )

            feature_pipeline = _feature_pipeline_for(
                model_name, engineer, variant=variant, neural_network_scale=neural_network_scale
            )
            X_fit_transformed = feature_pipeline.fit_transform(X_fit, y_fit)
            X_calib_transformed = feature_pipeline.transform(X_calib)
            X_val_transformed = feature_pipeline.transform(X_val)

            sample_weight = None
            if imbalance_strategy == "smote":
                X_fit_transformed, y_fit = apply_smote(
                    X_fit_transformed, y_fit, random_state=self.random_state
                )
            elif imbalance_strategy == "sample_weight":
                sample_weight = compute_balanced_sample_weight(y_fit)
            # "class_weight": nada a fazer aqui — já embutido no hiperparâmetro do modelo.

            model_creation_kwargs = _build_model_creation_kwargs(
                model_name,
                n_features=X_fit_transformed.shape[1],
                variant=variant,
                model_params=model_params,
            )
            model = create_model(model_name, **model_creation_kwargs)

            if sample_weight is not None:
                model.fit(X_fit_transformed, y_fit, sample_weight=sample_weight)
            else:
                model.fit(X_fit_transformed, y_fit)

            calibration_proba = _predict_positive_class_probability(model, X_calib_transformed)
            threshold_selection = find_best_threshold(y_calib.to_numpy(), calibration_proba)

            validation_proba = _predict_positive_class_probability(model, X_val_transformed)
            validation_pred = (validation_proba >= threshold_selection.threshold).astype(int)

            metrics = compute_fold_metrics(y_val.to_numpy(), validation_pred, validation_proba)
            fold_metric_dicts.append(metrics)

            fold_results.append(
                FoldResult(
                    fold=fold_index,
                    threshold=threshold_selection.threshold,
                    threshold_criterion=threshold_selection.criterion,
                    train_size=len(X_fit),
                    calibration_size=len(X_calib),
                    validation_size=len(X_val),
                    **metrics,
                )
            )
            fitted_feature_pipelines.append(feature_pipeline)
            fitted_models.append(model)

            out_of_fold_predictions[val_idx] = validation_pred
            out_of_fold_probabilities[val_idx] = validation_proba

        aggregated_metrics, aggregated_std = aggregate_fold_metrics(fold_metric_dicts)
        effective_params = fitted_models[-1].get_params() if fitted_models else {}

        return TrainingResult(
            model_name=model_name,
            imbalance_strategy=imbalance_strategy,
            params=effective_params,
            fold_results=fold_results,
            metrics=aggregated_metrics,
            metrics_std=aggregated_std,
            predictions=out_of_fold_predictions,
            probabilities=out_of_fold_probabilities,
            fitted_feature_pipelines=fitted_feature_pipelines,
            fitted_models=fitted_models,
            y_true=y_true_full,
        )


__all__ = ["Trainer"]
