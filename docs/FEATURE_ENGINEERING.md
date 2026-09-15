# Engenharia de features específica de modelo (Tarefa 4)

**Branch:** `audit/architecture-review`
**Baseado em:** [`ARCHITECTURE_AUDIT.md`](../ARCHITECTURE_AUDIT.md), [`docs/DATA_PIPELINE.md`](DATA_PIPELINE.md)

Implementa o princípio central da Tarefa 4:

```text
Common Features → Model-specific Features → Model-specific preprocessing
```

Nenhum notebook foi migrado. Nenhum arquivo antigo foi removido. Nenhum
hiperparâmetro de modelo foi alterado. O objetivo desta etapa é
**reproduzir o comportamento atual de forma organizada** — as únicas
mudanças de comportamento em relação aos notebooks são correções de
vazamento de dados exigidas explicitamente pela própria Tarefa 4, todas
listadas na seção 7 ("Desvios intencionais").

---

## 1. Onde a solução vive

A implementação ficou em **`src/pipelines/feature_engineering.py`**, não
em `data/feature_engineering.py` como o enunciado sugere como exemplo.
Motivo (documentando a decisão, como pedido): `data/` na raiz do
repositório já é o pacote legado do Naive Bayes
(`data/preprocessamento.py`) — colocar um novo `feature_engineering.py`
ali colidiria com esse pacote e voltaria a misturar código novo com
código antigo, o mesmo problema de nomenclatura que
`ARCHITECTURE_AUDIT.md` (seção 2.1) já identificou. `src/pipelines/` é
onde a Tarefa 2 já havia reservado o lugar para isso (o arquivo existia
desde então como só-arquitetura, com um `Protocol` vazio) e onde
`docs/ARCHITECTURE.md` (seção 1) já documentou a mesma decisão de não
colidir com pastas legadas de mesmo nome.

---

## 2. Interface

Um único objeto `FeatureEngineer` (`src/pipelines/feature_engineering.py`)
expõe a fábrica pedida:

```python
from src.pipelines.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

engineer.common_features()             # -> CommonFeatureSelector (não ajustado)
engineer.for_decision_tree()            # -> Pipeline (não ajustado)
engineer.for_logistic_regression()      # -> Pipeline
engineer.for_svm()                      # -> Pipeline
engineer.for_neural_network(scale=...)  # -> Pipeline
engineer.for_naive_bayes(variant=...)   # -> Pipeline
```

Cada `for_<modelo>()` devolve um `sklearn.pipeline.Pipeline`
**não ajustado**. Nada é `fit` dentro deste módulo — só quem chama
decide quando ajustar, e isso deve acontecer sempre dentro de um fold de
treino (ver seção 6, "Leakage").

`for_naive_bayes` não usa `common_features()` — ver seção 5, "Por que o
Naive Bayes é diferente".

---

## 3. Classificação de cada transformação existente

Requisito explícito da Tarefa 4. Toda transformação hoje presente nos
notebooks e em `models/naive_bayes/preprocessing.py`, classificada em
uma das cinco categorias pedidas — mais uma sexta, "fora do escopo",
para o que pertence a treino/modelo, não a dados.

| Transformação | Onde aparece hoje | Classificação | Implementada em |
|---|---|---|---|
| Conversão `pd.to_numeric(...).fillna(0)` das colunas de contagem | 4 notebooks | Preprocessing comum | `src.pipelines.base_preprocessing` (Tarefa 3) |
| `QT_EVADIDOS`, `TAXA_EVASAO`/`taxa_evasao`, `ALTO_RISCO_EVASAO`/`alto_risco_evasao` | 4 notebooks + `data/preprocessamento.py` | Construção do alvo (não é feature) | `src.data.target` (Tarefa 3) |
| Remoção de identificadores, colunas de origem do alvo, prefixos `QT_MAT`/`QT_CONC`, `RAZAO_CONCLUSAO_EVASAO` | `data/preprocessamento.py` + achado da auditoria | Leakage removal (não é feature) | `src.data.leakage` (Tarefa 3) |
| `TAXA_CONCLUSAO`, `RAZAO_ING_MAT`, `PROPORCAO_EAD`, `PROPORCAO_NOTURNO`, `INDICE_FINANCIAMENTO`, `PROPORCAO_FIES`, `PROPORCAO_PROUNIP`, `PROPORCAO_18_24`, `PROPORCAO_FEM` | 4 notebooks (idêntico) | **Feature engineering** (razões derivadas — criam sinal novo) | `CommonFeatureSelector` |
| Label Encoding das 8 categóricas | 4 notebooks | **Preprocessing** (encoding — não cria sinal novo, só torna a coluna existente utilizável) | `CommonFeatureSelector` |
| Amostragem estratificada (30k/90k) | 4 notebooks | Amostragem de dataset (fora do escopo de feature engineering) | Não implementada — nota na seção 8 |
| `EAD_PREDOMINANTE`, `FINANC_ALTO`, `CURSO_PEQUENO`, `RAZAO_FEM_18_24`, `TAMANHO_CATEGORIA`, `RAZAO_FINANC_EAD`, `TAXA_PREENCHIMENTO`, `RAZAO_INSCRITOS_VAGAS`, `CURSO_COMPETITIVO` | Árvore de Decisão | **Feature engineering** (model-specific — interações/binning) | `TreeFeatureAugmenter` |
| `RAZAO_CONCLUSAO_EVASAO` | Árvore de Decisão | Feature engineering, mas **vazamento de alvo** — não implementada | Excluída deliberadamente (ver seção 7) |
| `SelectKBest(mutual_info_classif, k=21)` | Árvore de Decisão | **Seleção de features** | `SelectKBest` dentro de `for_decision_tree()` |
| SMOTE por fold | Árvore de Decisão, Naive Bayes | **Tratamento de imbalance** | Fora do escopo (ver seção 8 — pertence a `src/training/`) |
| `RandomizedSearchCV`/`GridSearchCV` de hiperparâmetros | Árvore, LR, SVM | Busca de hiperparâmetros (treino, não dado) | Fora do escopo |
| Pós-poda `ccp_alpha` | Árvore de Decisão | Configuração do modelo (treino) | Fora do escopo |
| `StandardScaler` | Regressão Logística | **Model-specific transformation** (scaling) | `StandardScaler` dentro de `for_logistic_regression()` |
| Seleção por correlação `\|corr\|>=0.05` | SVM | **Seleção de features** | `CorrelationThresholdSelector` |
| Benchmark de 5 combinações de pré-processamento | SVM | Exploratório — nunca vira a config efetivamente usada no treino final (ver seção 7) | Não implementada (mantida fora, deliberadamente) |
| `RobustScaler` | SVM (busca final, célula 8) | **Model-specific transformation** (scaling) | `RobustScaler` dentro de `for_svm()` |
| `RandomForestClassifier` + `mutual_info_classif` para ranking de importância | Rede Neural | Exploratório — não alimenta o modelo final treinado | Não implementada |
| `PCA(2)` | Rede Neural | Visualização — não alimenta o modelo final treinado | Não implementada (fora do escopo de feature engineering; pertence a `src.visualization`) |
| Ausência de scaling | Rede Neural | **Model-specific transformation** (lacuna documentada, não corrigida agora) | `for_neural_network(scale=False)` por padrão |
| `class_weight`/pesos de classe | Rede Neural | **Tratamento de imbalance** | Fora do escopo |
| `VarianceThreshold(threshold=0.01)` | Naive Bayes (Gaussian/Bernoulli) | **Seleção de features** | `VarianceThreshold` dentro de `for_naive_bayes()` |
| `CorrelationRemover(threshold=0.90)` | Naive Bayes (Gaussian/Bernoulli) | **Seleção de features** (redundância) | `HighCorrelationRemover` |
| `StandardScaler`/`Binarizer`/`MinMaxScaler` por variante | Naive Bayes | **Model-specific transformation** (requisito de domínio de cada variante) | Dentro de `for_naive_bayes()` |
| SMOTE (só treino) | Naive Bayes | **Tratamento de imbalance** | Fora do escopo |
| Busca aleatória de 90 configurações | Naive Bayes | Busca de hiperparâmetros (treino) | Fora do escopo |

---

## 4. Tabela por modelo

| Modelo | Features | Transformações | Justificativa |
|---|---|---|---|
| **Árvore de Decisão** | `common_features` (21) + 9 features avançadas (`EAD_PREDOMINANTE`, `FINANC_ALTO`, `CURSO_PEQUENO`, `RAZAO_FEM_18_24`, `TAMANHO_CATEGORIA`, `RAZAO_FINANC_EAD`, `TAXA_PREENCHIMENTO`, `RAZAO_INSCRITOS_VAGAS`, `CURSO_COMPETITIVO`) → `SelectKBest` seleciona 21 das 30 | Label encoding (fit só no treino) → derivação das 9 avançadas → `SelectKBest(mutual_info_classif, k=21)`. **Sem scaling.** | Árvores particionam o espaço por limiares em cada feature isoladamente — invariantes a transformações monotônicas de escala. Aplicar `StandardScaler`/`MinMaxScaler` não mudaria a árvore aprendida; incluí-lo seria complexidade sem efeito (regra explícita da Tarefa 4: "não aplique scaling sem justificativa"). |
| **Regressão Logística** | `common_features` (21) | Label encoding (fit só no treino) → `StandardScaler` | Regressão logística estima coeficientes por gradiente sobre combinação linear das features; escalas muito diferentes distorcem a convergência e a regularização L1 usada no notebook (penaliza magnitude do coeficiente, que depende da escala da feature). |
| **SVM** | `common_features` (21) → subconjunto selecionado por correlação (tamanho variável, nunca vazio) | Label encoding → `CorrelationThresholdSelector(threshold=0.05)` (fit só no treino) → `RobustScaler` | Kernel RBF usa distância euclidiana diretamente — sensível a escala. `RobustScaler` (mediana/IQR) foi o scaler efetivamente usado na busca de hiperparâmetros final do notebook, não o `StandardScaler` do nome do "melhor config" do benchmark (nunca aplicado de fato — ver seção 7). |
| **Rede Neural** | `common_features` (21) | Label encoding → **nenhum scaling por padrão** (`scale=False`); `scale=True` aplica `StandardScaler` | Replica fielmente o notebook original, que treina sem normalização — uma lacuna que a própria auditoria identificou (redes com Adam/SGD tipicamente se beneficiam de entradas normalizadas), mas que a Tarefa 4 pede para **não corrigir agora** ("a primeira implementação deve preservar o comportamento atual... não tente melhorar a performance"). O parâmetro `scale` deixa a correção pronta para uma etapa futura e experimental. |
| **Naive Bayes — Gaussian** | Todas as colunas numéricas do censo (universo próprio, não `common_features` — ver seção 5) → após filtros, um subconjunto sem baixa variância nem redundância | `VarianceThreshold(0.01)` → `HighCorrelationRemover(0.90)` → `StandardScaler` | GaussianNB assume, por feature, uma distribuição normal com média/variância estimadas por classe — colunas quase constantes ou redundantes distorcem essa estimativa sem agregar informação; `StandardScaler` deixa as features na mesma escala que a suposição gaussiana do modelo pressupõe implicitamente. |
| **Naive Bayes — Bernoulli** | Mesmo universo do Gaussian | `VarianceThreshold(0.01)` → `HighCorrelationRemover(0.90)` → `Binarizer(threshold)` | BernoulliNB modela cada feature como um evento binário (presente/ausente) — exige entrada 0/1; `Binarizer` implementa exatamente essa conversão. |
| **Naive Bayes — Complement** | Todas as colunas numéricas do censo, sem filtro adicional | `MinMaxScaler` apenas | ComplementNB exige valores não negativos (é uma variante pensada para contagens/TF-IDF); `MinMaxScaler` garante o domínio `[0, 1]` sem alterar a ordem relativa dos valores. Sem `VarianceThreshold`/`HighCorrelationRemover` porque o notebook/pipeline original também não os aplica a esta variante — replicado fielmente. |

---

## 5. Por que o Naive Bayes é diferente

`for_naive_bayes()` **não** chama `common_features()`. Isso é intencional
e já estava assim no projeto original: `data/preprocessamento.py` usa
**todas as colunas numéricas do censo** (menos vazamento), não as 21
features manualmente selecionadas pelos outros 4 notebooks — universos
de dados diferentes, com alvo e amostragem diferentes (ver
`docs/DATA_PIPELINE.md`, seção 3). Forçar o Naive Bayes a compartilhar
`common_features()` mudaria o comportamento atual do modelo, o que a
Tarefa 4 proíbe explicitamente. A arquitetura reflete essa diferença em
vez de escondê-la atrás de uma interface uniforme artificial.

---

## 6. Regras de scaling por modelo (como foram aplicadas)

| Regra da Tarefa 4 | Como foi seguida |
|---|---|
| Árvore de Decisão: "não aplique scaling sem justificativa" | Nenhum scaler no pipeline; justificativa na tabela da seção 4. |
| Regressão Logística: "garanta uma representação adequada" | `StandardScaler`, a normalização padrão para modelos lineares regularizados. |
| SVM: "avalie cuidadosamente a necessidade de scaling" | Scaling obrigatório (kernel RBF); `RobustScaler` escolhido por ser o que o notebook efetivamente usou na busca final, não o nome do "melhor config" do benchmark (divergência documentada, seção 7). |
| Rede Neural: "avalie scaling/normalização e demais transformações necessárias" | Avaliado e documentado como lacuna real (ARCHITECTURE_AUDIT.md, seção 3.4); **não corrigida agora**, por instrução explícita de preservar o comportamento atual primeiro — arquitetura pronta via `scale=True`, mas não ligada por padrão. |
| Naive Bayes: "preserve a natureza das transformações exigidas pela variante" | Cada variante recebe exatamente a transformação que seu domínio matemático exige (contínuo→`StandardScaler`, binário→`Binarizer`, não-negativo→`MinMaxScaler`), replicando `models/naive_bayes/preprocessing.py`. |

---

## 7. Desvios intencionais em relação ao comportamento original

A Tarefa 4 pede para preservar o comportamento atual e não "melhorar
performance" — mas também exige, em uma seção própria, nunca fazer
`fit` antes do cross-validation quando isso puder causar vazamento.
Estas duas instruções colidem em três pontos específicos dos notebooks
originais. Em todos os três, a regra de vazamento venceu — são
correções de corretude, não experimentos de performance, e estão todas
documentadas aqui:

1. **Label encoding ajustado só no treino.** Os 4 notebooks ajustam o
   `LabelEncoder` de cada categórica sobre o dataset inteiro, antes de
   qualquer split de CV. `CommonFeatureSelector.fit` agora ajusta o
   encoder apenas no fold de treino, com um valor sentinela (`"-1"`)
   sempre reservado no vocabulário para categorias nunca vistas — assim
   nunca falha ao transformar um fold de validação/teste com uma
   categoria rara ausente do treino.
2. **Seleção por correlação do SVM ajustada só no treino.**
   `SVM.ipynb` calcula `|corr(coluna, alvo)|` uma vez, sobre `X_raw`
   completo, antes de qualquer split — o mesmo padrão de "sem nested
   cross-validation" que `ARCHITECTURE_AUDIT.md` (seção 5) já apontava
   como risco em outros modelos. `CorrelationThresholdSelector` só
   calcula a correlação quando `fit` é chamado — cabe a quem monta o
   pipeline garantir que isso aconteça só no fold de treino (o próprio
   contrato de `Pipeline.fit(X_train, y_train)` já garante isso quando
   usado dentro de `sklearn.model_selection`).
3. **`RAZAO_CONCLUSAO_EVASAO` excluída.** Não é uma correção de "ajustar
   fit antes do CV" — é vazamento direto de alvo (a feature usa
   `TAXA_EVASAO`, a mesma variável binarizada no alvo). Já havia sido
   excluída desde a Tarefa 2 (`ARCHITECTURE_AUDIT.md`, seção 5) e
   permanece excluída aqui. É a única mudança de *conjunto* de features
   desta implementação — todas as outras 9 features avançadas da Árvore
   são preservadas.

Dois desvios adicionais, sem relação com vazamento, documentados por
transparência:

4. **SVM: `RobustScaler`, não `StandardScaler`.** O benchmark de
   pré-processamento do notebook (`SVM.ipynb`, célula 6) nomeia
   "Mode + Label + Standard" como melhor configuração, mas a busca de
   hiperparâmetros final (célula 8) usa `RobustScaler` na prática — o
   notebook nunca reconcilia essa diferença. `for_svm()` replica o que
   foi **efetivamente executado** para o modelo final (`RobustScaler`),
   não o texto do "melhor config" nunca aplicado. Já apontado em
   `ARCHITECTURE_AUDIT.md`, seção 3.3.
5. **Rede Neural: robustez de `pd.cut`/`safe_div` idêntica à original.**
   Nenhuma mudança de comportamento aqui — mencionado só para registrar
   que `_safe_div` e o binning de `TAMANHO_CATEGORIA` (Árvore) foram
   reimplementados (não copiados literalmente), mas produzem os mesmos
   valores para os mesmos dados de entrada.

**Nada mais foi alterado.** Em particular, o vazamento indireto
"moderado" já documentado em `ARCHITECTURE_AUDIT.md` (seção 5) —
`QT_MAT` como feature bruta, e `TAXA_CONCLUSAO`/`RAZAO_ING_MAT`/
`INDICE_FINANCIAMENTO` dependendo dele — **não foi corrigido nesta
etapa**. Corrigi-lo exigiria remover essas 3 das 21 `common_features`,
uma mudança de *conjunto* de features que a Tarefa 4 proíbe explicitamente
("não altere arbitrariamente as features atuais... a mudança inicial é
arquitetural, não experimental"). Fica registrado como candidato a uma
fase futura e experimental, não como algo esquecido.

---

## 8. Como montar o X de entrada (composição com `src.data.pipeline`)

`CommonFeatureSelector` espera um `X` que já passou por
`src.data.target.build_target` (estratégia `threshold_20pct`) e por
`src.data.leakage.remove_leakage_columns` com
**`include_prefix_group=False`** — porque 3 das 9 `DERIVED_FEATURES`
comuns (`TAXA_CONCLUSAO`, `INDICE_FINANCIAMENTO`, e indiretamente
`RAZAO_ING_MAT` via `QT_MAT`) dependem de colunas com prefixo
`QT_MAT`/`QT_CONC`, removidas por padrão pelo grupo 3 de
`src.data.leakage` (ver seção 7, último parágrafo, sobre por que esse
risco moderado é preservado nesta etapa). Uso recomendado:

```python
from src.data.pipeline import run_data_pipeline
from src.pipelines.feature_engineering import FeatureEngineer

# Para Árvore, Regressão Logística, SVM, Rede Neural:
result = run_data_pipeline(
    target_strategy="threshold_20pct",
    missing_strategy="mode",              # replica handle_missing_mode dos notebooks
    include_prefix_leakage_group=False,    # preserva QT_MAT/QT_CONC p/ as derivadas comuns
)
engineer = FeatureEngineer()
pipeline = engineer.for_decision_tree()    # ou for_logistic_regression/for_svm/for_neural_network

# DENTRO de cada fold de cross-validation (nunca fora):
pipeline.fit(result.X.iloc[train_idx], result.y.iloc[train_idx])
X_train_transformed = pipeline.transform(result.X.iloc[train_idx])
X_val_transformed = pipeline.transform(result.X.iloc[val_idx])

# Para Naive Bayes, a composição usa os padrões (não precisa dos parâmetros acima):
result_nb = run_data_pipeline(target_strategy="median_split")
nb_pipeline = FeatureEngineer().for_naive_bayes("complement")
nb_pipeline.fit(result_nb.X.iloc[train_idx], result_nb.y.iloc[train_idx])
```

Esta composição ainda não está automatizada em um único ponto de entrada
— `experiments/run_model.py` (mencionado em `docs/ARCHITECTURE.md`) é
onde ela deveria ser encapsulada, junto com o loop de cross-validation
real (`src/training/`, ainda não implementado). Nesta etapa, a
composição fica documentada e testável em partes, não automatizada de
ponta a ponta — consistente com "não migre os notebooks ainda".

---

## 9. Leakage

Resumo das garantias desta implementação (além dos desvios da seção 7):

- **Nenhum `fit` acontece dentro deste módulo.** `FeatureEngineer.for_*()`
  só constrói e devolve `Pipeline`s não ajustados.
- **Todo transformador com estado é um `TransformerMixin` com `fit`/
  `transform` separados** (`CommonFeatureSelector`,
  `TreeFeatureAugmenter`, `CorrelationThresholdSelector`,
  `HighCorrelationRemover`), nunca um `fit_transform` monolítico que
  esconderia a fronteira treino/validação.
- **`SelectKBest`, `VarianceThreshold`, `StandardScaler`, `RobustScaler`,
  `MinMaxScaler`, `Binarizer` são nativos do scikit-learn**, com
  contrato `fit`/`transform` já correto — usados via `.set_output(
  transform="pandas")` só para preservar nomes de coluna, sem alterar
  sua semântica de ajuste.
- **`mutual_info_classif` tem `random_state` fixo** (via
  `FeatureEngineer(random_state=...)`, padrão 42), porque seu estimador
  interno usa ruído aleatório — sem isso, a seleção de `SelectKBest`
  poderia variar entre execuções com os mesmos dados, violando o
  requisito de reprodutibilidade.
- **`Pipeline` do scikit-learn é o mecanismo de composição recomendado**
  em todos os `for_<modelo>()` — quando usado dentro de
  `sklearn.model_selection.cross_validate`/`GridSearchCV`, o próprio
  scikit-learn garante que `fit` roda só no fold de treino de cada
  iteração.

---

## 10. Testes

`tests/test_feature_engineering.py` — 28 testes, cobrindo os 5 requisitos
da Tarefa 4:

| Requisito | Testes (exemplos) |
|---|---|
| 1. Cada modelo recebe as features esperadas | `test_common_features_output_matches_declared_common_features`, `test_for_decision_tree_selects_exactly_k_best_features`, `test_for_svm_selects_subset_of_common_features`, `test_for_naive_bayes_*` (3 variantes) |
| 2. Nenhuma feature proibida entra | `test_known_leaky_feature_never_appears_in_decision_tree_output`, `test_target_source_columns_never_appear_in_any_pipeline_output` (parametrizado nos 4 modelos alinhados) |
| 3. Não existem NaNs inesperados | `test_common_features_never_produce_nan_even_with_zero_denominators` (denominadores zerados deliberadamente), `test_all_pipelines_produce_no_nan`, `test_naive_bayes_pipelines_produce_no_nan` |
| 4. Número de features determinístico | `test_decision_tree_feature_count_is_deterministic_across_calls`, `test_common_feature_count_is_always_21_regardless_of_input_size` |
| 5. Transformações reproduzíveis | `test_common_feature_selector_is_reproducible`, `test_decision_tree_pipeline_is_reproducible_with_fixed_random_state`, `test_svm_pipeline_is_reproducible`, `test_naive_bayes_pipeline_is_reproducible`, `test_fit_on_train_transform_on_validation_does_not_refit`, `test_high_correlation_remover_is_reproducible_and_deterministic_count` |

Dados sintéticos apenas — nenhum teste depende do CSV real do INEP.
`tests/test_feature_selection.py` (Tarefa 3) foi ajustado: os dois testes
que verificavam `get_feature_engineer(...)` levantando
`NotImplementedError` foram removidos, porque essa engenharia de
features **agora está implementada** (o comportamento que eles
verificavam deixou de existir por design, não por regressão).

### Execução real

Ambiente com Python 3.14.7 e as dependências de `requirements.txt` +
`pytest` instaladas (mesmo ambiente configurado durante a Tarefa 3):

```text
python -m pytest tests/ -v
...
======================= 76 passed in 4.24s =======================
```

Os 28 testes novos de `test_feature_engineering.py` mais os 48 já
existentes das Tarefas 2 e 3 passam juntos, sem regressão.

---

## 11. O que fica fora do escopo desta etapa

Por instrução explícita da Tarefa 4 ("não faça"):

- **SMOTE e `class_weight`** — classificados como "tratamento de
  imbalance" na tabela da seção 3, permanecem fora deste módulo (já era
  a fronteira estabelecida na Tarefa 3, `base_preprocessing.py`).
  Pertencem a `src/training/`.
- **Hiperparâmetros de modelo** (profundidade da árvore, `C` da
  regressão, kernel do SVM, arquitetura da rede, `var_smoothing` do
  Naive Bayes) — não tocados; nem sequer referenciados aqui.
- **Threshold tuning** — pertence a `src/training/`, ainda não
  implementado.
- **Migração dos notebooks** — os 4 notebooks e
  `models/naive_bayes/*.py` continuam existindo e funcionando como
  antes; nada em `src/` é importado por eles.
- **Correção da lacuna de scaling da Rede Neural e do vazamento indireto
  de `QT_MAT`** — identificadas, documentadas, deliberadamente não
  corrigidas agora (seção 7).

## 12. Próximos passos (fora do escopo desta etapa)

Ver `ARCHITECTURE_AUDIT.md`, seção 8: `src/training/` (cross-validation
real usando estes pipelines, dentro de cada fold) e `src/evaluation/`
são as próximas camadas a implementar.
