"""Engenharia de features específica de cada modelo (Tarefa 4).

Implementa o princípio pedido na Tarefa 4:

    Common Features → Model-specific Features → Model-specific preprocessing

Um único objeto `FeatureEngineer` expõe uma fábrica por modelo. Cada
método devolve um `sklearn.pipeline.Pipeline` **não ajustado** — nada é
`fit` neste módulo. Quem chama é responsável por `pipeline.fit(X_train,
y_train)` dentro de cada fold de validação cruzada, nunca no dataset
inteiro antes do split (ver "Leakage" em `docs/FEATURE_ENGINEERING.md`).

    engineer = FeatureEngineer()
    pipeline = engineer.for_decision_tree()   # não ajustado
    pipeline.fit(X_train, y_train)            # ajusta só no fold de treino
    X_val_transformed = pipeline.transform(X_val)

Este módulo reorganiza e reimplementa a lógica das células "ENGENHARIA
DE FEATURES" dos 4 notebooks e de `models/naive_bayes/preprocessing.py`
— não copia o código original. Ver `docs/FEATURE_ENGINEERING.md` para a
classificação completa de cada transformação e os desvios intencionais
em relação ao comportamento original (todos motivados por vazamento de
dados, nunca por performance).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.configs.constants import CATEGORICAL_FEATURES, DERIVED_FEATURES, RAW_NUMERIC_FEATURES
from src.configs.settings import (
    DECISION_TREE_SELECT_K_BEST,
    NAIVE_BAYES_CORRELATION_THRESHOLD,
    NAIVE_BAYES_VARIANCE_THRESHOLD,
    RANDOM_STATE,
    SVM_CORRELATION_THRESHOLD,
)

NaiveBayesVariant = Literal["gaussian", "bernoulli", "complement"]

# Colunas brutas do censo necessárias para computar as 9 features
# derivadas em `DERIVED_FEATURES` (grupo "common"). Correspondem
# exatamente às colunas que os 4 notebooks leem via `COLS_CURSOS`, menos
# as que já foram consumidas pela construção do alvo (`src.data.target`)
# e removidas como vazamento direto (`src.data.leakage`, grupo 2).
_DERIVED_FEATURE_SOURCE_COLUMNS: tuple[str, ...] = (
    "QT_CONC",
    "QT_VG_TOTAL_EAD",
    "QT_VG_TOTAL_NOTURNO",
    "QT_MAT_FINANC",
    "QT_ING_FIES",
    "QT_ING_PROUNIP",
    "QT_ING_18_24",
    "QT_ING_FEM",
)


def _safe_div(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divisão elemento a elemento que nunca produz ``inf``/``NaN`` por zero.

    Reimplementação de ``safe_div``, hoje duplicada nos 4 notebooks e em
    `analysis/audit/data.py`. Centralizada aqui porque só é usada pelas
    transformações de feature engineering — o restante do pipeline
    (`src.pipelines.base_preprocessing`) já recebe colunas sem essa
    necessidade.
    """
    denominator_numeric = pd.to_numeric(denominator, errors="coerce").fillna(0)
    numerator_numeric = pd.to_numeric(numerator, errors="coerce").fillna(0)
    return pd.Series(
        np.where(denominator_numeric > 0, numerator_numeric / denominator_numeric, fill),
        index=numerator.index,
    )


class CommonFeatureSelector(BaseEstimator, TransformerMixin):
    """Camada "Common Features": as 21 features usadas por Árvore,
    Regressão Logística, SVM e Rede Neural.

    Reproduz a célula "ENGENHARIA DE FEATURES" dos 4 notebooks: calcula
    as 9 razões de `DERIVED_FEATURES` a partir de colunas brutas do censo
    e aplica label encoding às 8 categóricas de `CATEGORICAL_FEATURES`.

    Desvio intencional em relação aos notebooks originais (ver
    `docs/FEATURE_ENGINEERING.md`, "Desvios intencionais"): o label
    encoding é ajustado **somente no fold de treino** (``fit``), com um
    valor sentinela (``"-1"``) sempre reservado no vocabulário para
    categorias nunca vistas no treino. Os notebooks originais ajustam o
    encoder sobre o dataset inteiro antes do split de CV — uma forma leve
    de vazamento que esta implementação corrige por exigência explícita
    da Tarefa 4 ("nunca faça fit(X) antes do cross-validation").

    Espera que ``X`` já tenha passado por `src.data.target.build_target`
    (estratégia ``threshold_20pct``) e por
    `src.data.leakage.remove_leakage_columns` com
    ``include_prefix_group=False`` — ver `docs/FEATURE_ENGINEERING.md`,
    seção "Como montar o X de entrada", para o porquê desse parâmetro.
    """

    UNSEEN_CATEGORY_TOKEN = "-1"

    def __init__(
        self,
        categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
        raw_numeric_features: tuple[str, ...] = RAW_NUMERIC_FEATURES,
        derived_features: tuple[str, ...] = DERIVED_FEATURES,
    ) -> None:
        self.categorical_features = categorical_features
        self.raw_numeric_features = raw_numeric_features
        self.derived_features = derived_features

    @property
    def output_feature_names_(self) -> list[str]:
        """Nomes e ordem das 21 colunas produzidas por `transform`."""
        return list(self.categorical_features) + list(self.raw_numeric_features) + list(self.derived_features)

    def _required_input_columns(self) -> list[str]:
        return list(self.categorical_features) + list(self.raw_numeric_features) + list(_DERIVED_FEATURE_SOURCE_COLUMNS)

    @staticmethod
    def _compute_derived_features(X: pd.DataFrame) -> pd.DataFrame:
        derived = pd.DataFrame(index=X.index)
        derived["TAXA_CONCLUSAO"] = _safe_div(X["QT_CONC"], X["QT_MAT"]) * 100
        derived["RAZAO_ING_MAT"] = _safe_div(X["QT_ING"], X["QT_MAT"])
        derived["PROPORCAO_EAD"] = _safe_div(X["QT_VG_TOTAL_EAD"], X["QT_VG_TOTAL"])
        derived["PROPORCAO_NOTURNO"] = _safe_div(X["QT_VG_TOTAL_NOTURNO"], X["QT_VG_TOTAL"])
        derived["INDICE_FINANCIAMENTO"] = _safe_div(X["QT_MAT_FINANC"], X["QT_MAT"])
        derived["PROPORCAO_FIES"] = _safe_div(X["QT_ING_FIES"], X["QT_ING"])
        derived["PROPORCAO_PROUNIP"] = _safe_div(X["QT_ING_PROUNIP"], X["QT_ING"])
        derived["PROPORCAO_18_24"] = _safe_div(X["QT_ING_18_24"], X["QT_ING"])
        derived["PROPORCAO_FEM"] = _safe_div(X["QT_ING_FEM"], X["QT_ING"])
        return derived

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CommonFeatureSelector":
        del y
        missing = [column for column in self._required_input_columns() if column not in X.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes para common_features: {missing}")

        self.encoders_: dict[str, LabelEncoder] = {}
        for column in self.categorical_features:
            values = X[column].fillna(-1).astype(str)
            vocabulary = pd.concat([values, pd.Series([self.UNSEEN_CATEGORY_TOKEN])], ignore_index=True)
            encoder = LabelEncoder().fit(vocabulary)
            self.encoders_[column] = encoder

        self.n_features_out_ = len(self.output_feature_names_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "encoders_")

        result = pd.DataFrame(index=X.index)
        for column, encoder in self.encoders_.items():
            values = X[column].fillna(-1).astype(str)
            known = set(encoder.classes_)
            values = values.where(values.isin(known), self.UNSEEN_CATEGORY_TOKEN)
            result[column] = encoder.transform(values)

        for column in self.raw_numeric_features:
            result[column] = X[column]

        derived = self._compute_derived_features(X)
        for column in self.derived_features:
            result[column] = derived[column]

        return result[self.output_feature_names_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        del input_features
        return np.asarray(self.output_feature_names_)


class TreeFeatureAugmenter(BaseEstimator, TransformerMixin):
    """Camada "Model-specific Features" da Árvore de Decisão.

    Adiciona as 9 features "avançadas" do notebook original, calculadas a
    partir da saída de `CommonFeatureSelector`. Todas são funções
    determinísticas linha a linha — nenhum parâmetro é aprendido dos
    dados, então ``fit`` só valida colunas.

    **Não inclui** ``RAZAO_CONCLUSAO_EVASAO`` — a auditoria identificou
    que essa feature é calculada a partir de `TAXA_EVASAO`, a mesma
    variável usada para construir o alvo (vazamento direto de alvo,
    ARCHITECTURE_AUDIT.md seção 5). Excluí-la é a única mudança de
    *conjunto* de features desta implementação em relação ao notebook
    original — todas as outras 9 features avançadas são preservadas
    exatamente como estavam.
    """

    _REQUIRED_COLUMNS: tuple[str, ...] = (
        "QT_ING",
        "QT_VG_TOTAL",
        "QT_INSCRITO_TOTAL",
        "PROPORCAO_EAD",
        "INDICE_FINANCIAMENTO",
        "PROPORCAO_FEM",
        "PROPORCAO_18_24",
    )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TreeFeatureAugmenter":
        del y
        missing = [column for column in self._REQUIRED_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes para tree_features: {missing}")
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fitted_")
        result = X.copy()

        result["EAD_PREDOMINANTE"] = (result["PROPORCAO_EAD"] > 0.5).astype(int)
        result["FINANC_ALTO"] = (result["INDICE_FINANCIAMENTO"] > 0.7).astype(int)
        result["CURSO_PEQUENO"] = (result["QT_ING"] < 50).astype(int)
        result["RAZAO_FEM_18_24"] = _safe_div(result["PROPORCAO_FEM"], result["PROPORCAO_18_24"] + 0.1)

        bins = pd.cut(result["QT_ING"], bins=[0, 30, 100, 300, np.inf], labels=[0, 1, 2, 3])
        result["TAMANHO_CATEGORIA"] = bins.cat.codes.astype(int)

        result["RAZAO_FINANC_EAD"] = _safe_div(result["INDICE_FINANCIAMENTO"], result["PROPORCAO_EAD"] + 0.1)
        result["TAXA_PREENCHIMENTO"] = _safe_div(result["QT_ING"], result["QT_VG_TOTAL"])
        result["RAZAO_INSCRITOS_VAGAS"] = _safe_div(result["QT_INSCRITO_TOTAL"], result["QT_VG_TOTAL"])
        result["CURSO_COMPETITIVO"] = (result["RAZAO_INSCRITOS_VAGAS"] > 2).astype(int)

        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = list(input_features) if input_features is not None else []
        extra = [
            "EAD_PREDOMINANTE", "FINANC_ALTO", "CURSO_PEQUENO", "RAZAO_FEM_18_24",
            "TAMANHO_CATEGORIA", "RAZAO_FINANC_EAD", "TAXA_PREENCHIMENTO",
            "RAZAO_INSCRITOS_VAGAS", "CURSO_COMPETITIVO",
        ]
        return np.asarray(base + extra)


class CorrelationThresholdSelector(BaseEstimator, TransformerMixin):
    """Camada "Model-specific Features" do SVM: seleção por correlação absoluta.

    Reimplementação de `SVM.ipynb` ("SELEÇÃO DE FEATURES POR CORRELAÇÃO"):
    mantém colunas com ``|corr(coluna, alvo)| >= threshold``; se nenhuma
    atingir o limiar, mantém a de maior correlação (mesmo *fallback* do
    notebook original).

    Desvio intencional: o notebook calcula essa correlação **uma vez,
    sobre o dataset inteiro**, antes de qualquer split — um caso do
    problema "sem nested cross-validation" descrito em
    ARCHITECTURE_AUDIT.md, seção 5. Aqui, por ser um `TransformerMixin`
    com `fit`/`transform` separados, a correlação só é calculada quando
    `fit` é chamado — responsabilidade de quem monta o pipeline garantir
    que isso aconteça só no fold de treino (nunca no dataset inteiro).
    """

    def __init__(self, threshold: float = SVM_CORRELATION_THRESHOLD) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CorrelationThresholdSelector":
        y_array = np.asarray(y, dtype=float)
        correlations: dict[str, float] = {}
        for column in X.columns:
            series = X[column].astype(float)
            if series.std(ddof=0) > 0:
                correlations[column] = abs(float(np.corrcoef(series, y_array)[0, 1]))
            else:
                correlations[column] = 0.0

        selected = [column for column, corr in correlations.items() if corr >= self.threshold]
        if not selected:
            selected = [max(correlations, key=correlations.get)]

        self.correlations_ = correlations
        self.selected_features_ = selected
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        del input_features
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_)


class HighCorrelationRemover(BaseEstimator, TransformerMixin):
    """Camada "Model-specific Features" do Naive Bayes Gaussian/Bernoulli.

    Remove, de cada par de colunas com ``|correlação| > threshold``, uma
    das duas (a que aparece depois, na ordem das colunas). Reimplementação
    de `models/naive_bayes/preprocessing.py::CorrelationRemover` mantendo
    a mesma semântica (limiar padrão 0.90) e devolvendo `DataFrame` (o
    original devolvia `numpy.ndarray`, perdendo os nomes das colunas).
    """

    def __init__(self, threshold: float = NAIVE_BAYES_CORRELATION_THRESHOLD) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HighCorrelationRemover":
        del y
        if X.shape[1] <= 1:
            self.columns_to_keep_ = list(X.columns)
            return self

        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        columns_to_drop = {
            column for column in upper_triangle.columns if (upper_triangle[column] > self.threshold).any()
        }
        self.columns_to_keep_ = [column for column in X.columns if column not in columns_to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "columns_to_keep_")
        return X[self.columns_to_keep_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        del input_features
        check_is_fitted(self, "columns_to_keep_")
        return np.asarray(self.columns_to_keep_)


class FeatureEngineer:
    """Fábrica central de pipelines de feature engineering, um por modelo.

    Cada método devolve um `sklearn.pipeline.Pipeline` **não ajustado**.
    `common_features()` sozinho também pode ser usado como um passo de
    pipeline independente (é um `TransformerMixin` válido).

    Não existe um `for_naive_bayes()` que compartilhe `common_features()`
    — o Naive Bayes usa um universo de features diferente por desenho
    (todas as colunas numéricas do censo, não as 21 features comuns),
    replicando `data/preprocessamento.py`. Ver
    `docs/FEATURE_ENGINEERING.md`, seção "Por que o Naive Bayes é
    diferente", para a justificativa completa.
    """

    def __init__(
        self,
        *,
        tree_select_k_best: int = DECISION_TREE_SELECT_K_BEST,
        svm_correlation_threshold: float = SVM_CORRELATION_THRESHOLD,
        naive_bayes_variance_threshold: float = NAIVE_BAYES_VARIANCE_THRESHOLD,
        naive_bayes_correlation_threshold: float = NAIVE_BAYES_CORRELATION_THRESHOLD,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.tree_select_k_best = tree_select_k_best
        self.svm_correlation_threshold = svm_correlation_threshold
        self.naive_bayes_variance_threshold = naive_bayes_variance_threshold
        self.naive_bayes_correlation_threshold = naive_bayes_correlation_threshold
        self.random_state = random_state

    def common_features(self) -> CommonFeatureSelector:
        """Camada "Common Features": as 21 features de Árvore/LR/SVM/Rede Neural."""
        return CommonFeatureSelector()

    def for_decision_tree(self) -> Pipeline:
        """Common Features → +9 features avançadas → SelectKBest(mutual_info, k).

        Sem scaling: árvores particionam o espaço por limiares em cada
        feature isoladamente, então são invariantes a transformações
        monotônicas de escala — normalizar não muda a árvore aprendida
        (regra explícita da Tarefa 4 para este modelo).
        """
        mutual_info_with_seed = _mutual_info_classif_with_seed(self.random_state)
        return Pipeline([
            ("common_features", self.common_features()),
            ("tree_features", TreeFeatureAugmenter()),
            (
                "select_k_best",
                SelectKBest(mutual_info_with_seed, k=self.tree_select_k_best).set_output(transform="pandas"),
            ),
        ])

    def for_logistic_regression(self) -> Pipeline:
        """Common Features → StandardScaler.

        A Regressão Logística estima coeficientes por gradiente sobre uma
        combinação linear das features; features em escalas muito
        diferentes fazem a otimização convergir de forma desigual entre
        coeficientes e distorcem a regularização L1/L2. `StandardScaler`
        (média 0, desvio 1) é a normalização padrão para esse caso — a
        mesma já usada no notebook original.
        """
        return Pipeline([
            ("common_features", self.common_features()),
            ("scaler", StandardScaler().set_output(transform="pandas")),
        ])

    def for_svm(self) -> Pipeline:
        """Common Features → seleção por correlação → RobustScaler.

        SVM com kernel RBF é sensível à escala das features (a distância
        euclidiana entra diretamente no kernel) — scaling é necessário.
        Uso `RobustScaler` (baseado em mediana/IQR) porque foi o scaler
        efetivamente usado na busca de hiperparâmetros final do notebook
        (`SVM.ipynb`, célula 8) — não o `StandardScaler` mencionado no
        nome da "melhor configuração" do benchmark da célula 6, que nunca
        chegou a ser aplicado de fato. Ver `docs/FEATURE_ENGINEERING.md`
        para essa divergência, já documentada em
        ARCHITECTURE_AUDIT.md, seção 3.3.
        """
        return Pipeline([
            ("common_features", self.common_features()),
            ("correlation_selection", CorrelationThresholdSelector(threshold=self.svm_correlation_threshold)),
            ("scaler", RobustScaler().set_output(transform="pandas")),
        ])

    def for_neural_network(self, *, scale: bool = False) -> Pipeline:
        """Common Features → (opcional) StandardScaler.

        ``scale=False`` por padrão, replicando fielmente o notebook
        original (`RedesNeurais.ipynb`), que treina a rede sobre as
        features sem nenhuma normalização — uma lacuna que a auditoria
        identificou (ARCHITECTURE_AUDIT.md, seção 3.4): redes treinadas
        com Adam/SGD tipicamente se beneficiam de entradas normalizadas.
        Por instrução explícita da Tarefa 4 ("a primeira implementação
        deve preservar o comportamento atual... não tente melhorar a
        performance"), essa lacuna **não é corrigida agora** — o
        parâmetro `scale` deixa a correção pronta para ser ligada em uma
        etapa futura e experimental, sem exigir mudança de arquitetura.
        """
        steps: list[tuple[str, BaseEstimator]] = [("common_features", self.common_features())]
        if scale:
            steps.append(("scaler", StandardScaler().set_output(transform="pandas")))
        return Pipeline(steps)

    def for_naive_bayes(
        self,
        variant: NaiveBayesVariant = "complement",
        *,
        binarize_threshold: float = 0.0,
    ) -> Pipeline:
        """Pipeline específico da variante de Naive Bayes.

        Não usa `common_features()` — opera sobre o universo mais amplo
        de colunas numéricas do censo (saída de
        ``src.data.pipeline.run_data_pipeline(target_strategy="median_split")``),
        replicando `models/naive_bayes/preprocessing.py::get_preprocessing_pipeline`:

        - ``gaussian``: `VarianceThreshold` → remoção de correlação alta → `StandardScaler`
          (features contínuas, GaussianNB assume distribuição normal por feature).
        - ``bernoulli``: `VarianceThreshold` → remoção de correlação alta → `Binarizer`
          (BernoulliNB espera entradas binárias 0/1).
        - ``complement``: apenas `MinMaxScaler`
          (ComplementNB exige valores não negativos).

        Raises:
            ValueError: ``variant`` desconhecida.
        """
        if variant == "gaussian":
            return Pipeline([
                ("variance_threshold", VarianceThreshold(
                    threshold=self.naive_bayes_variance_threshold
                ).set_output(transform="pandas")),
                ("correlation_removal", HighCorrelationRemover(threshold=self.naive_bayes_correlation_threshold)),
                ("scaler", StandardScaler().set_output(transform="pandas")),
            ])

        if variant == "bernoulli":
            return Pipeline([
                ("variance_threshold", VarianceThreshold(
                    threshold=self.naive_bayes_variance_threshold
                ).set_output(transform="pandas")),
                ("correlation_removal", HighCorrelationRemover(threshold=self.naive_bayes_correlation_threshold)),
                ("binarize", Binarizer(threshold=binarize_threshold).set_output(transform="pandas")),
            ])

        if variant == "complement":
            return Pipeline([("scaler", MinMaxScaler().set_output(transform="pandas"))])

        raise ValueError(
            f"Variante de Naive Bayes inválida: {variant!r}. Use 'gaussian', 'bernoulli' ou 'complement'."
        )


def _mutual_info_classif_with_seed(random_state: int):
    """Fixa ``random_state`` de `mutual_info_classif` para reprodutibilidade.

    `mutual_info_classif` usa um estimador baseado em k-NN com ruído
    aleatório para desempate — sem `random_state` fixo, `SelectKBest`
    poderia selecionar um conjunto de features levemente diferente a
    cada execução (viola o requisito de reprodutibilidade da Tarefa 4).
    """

    def scorer(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        return mutual_info_classif(X, y, random_state=random_state)

    return scorer


__all__ = [
    "CommonFeatureSelector",
    "TreeFeatureAugmenter",
    "CorrelationThresholdSelector",
    "HighCorrelationRemover",
    "FeatureEngineer",
    "NaiveBayesVariant",
]
