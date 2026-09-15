# Pipeline de dados — `src/data` e `src/pipelines` (Tarefa 3)

**Branch:** `audit/architecture-review`
**Baseado em:** [`ARCHITECTURE_AUDIT.md`](../ARCHITECTURE_AUDIT.md), [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)

Esta etapa refatora a lógica de `data/preprocessamento.py` (o único
pipeline de dados do projeto que já era um módulo Python, não um
notebook) em componentes pequenos e testáveis, seguindo o diagrama
pedido:

```text
Data Loading → Target Construction → Leakage Removal → Base Preprocessing → Feature Engineering
```

`data/preprocessamento.py` **não foi alterado**. Nenhum notebook foi
migrado. Nenhum modelo foi tocado.

---

## 1. Onde cada etapa vive

| Etapa | Módulo | Status |
|---|---|---|
| Data Loading | [`src/data/loading.py`](../src/data/loading.py) | Implementado |
| Target Construction | [`src/data/target.py`](../src/data/target.py) | Implementado |
| Leakage Removal | [`src/data/leakage.py`](../src/data/leakage.py) | Implementado |
| Base Preprocessing | [`src/pipelines/base_preprocessing.py`](../src/pipelines/base_preprocessing.py) | Implementado |
| Feature Engineering (específica por modelo) | [`src/pipelines/feature_engineering.py`](../src/pipelines/feature_engineering.py) | **Arquitetura apenas** (interface + registry vazio) — pedido explícito da Tarefa 3 |
| Orquestração das 4 primeiras etapas | [`src/data/pipeline.py`](../src/data/pipeline.py) | Implementado |
| Validação de compatibilidade | [`src/data/validation.py`](../src/data/validation.py) | Implementado |

---

## 2. Data Loading (`src/data/loading.py`)

Responsabilidade única: localizar e ler `MICRODADOS_CADASTRO_CURSOS_2024.CSV`.

- `load_raw_dataset(path=None)` — se `path` não for passado, resolve o
  arquivo via `src.configs.paths.resolve_dataset_path()` (Tarefa 2), que
  já centraliza os 3 caminhos candidatos hoje duplicados em
  `data/preprocessamento.py` e `analysis/audit/data.py`.
- Nenhum caminho é escrito diretamente neste módulo — o único literal de
  caminho no projeto novo é a lista `DATASET_PATH_CANDIDATES` em
  `src/configs/paths.py`, atendendo ao requisito "não deixe caminhos
  hardcoded espalhados pelo projeto".
- Não restringe `usecols`: diferente dos notebooks (que só leem 22
  colunas fixas), a estratégia `median_split` de `src.data.target`
  precisa de outras colunas do censo. A restrição de colunas é
  responsabilidade da camada de features, não do carregamento — assim
  o mesmo `load_raw_dataset` serve às duas estratégias de alvo.
- Erros diferenciados: `FileNotFoundError` (nenhum caminho existe) vs.
  `RuntimeError` (arquivo existe mas não pôde ser lido — encoding,
  parsing).

---

## 3. Target Construction (`src/data/target.py`)

Centraliza a criação de `taxa_evasao` e `alto_risco_evasao` — as duas
colunas nomeadas explicitamente na Tarefa 3 (em `snake_case`, a
convenção de `data/preprocessamento.py`).

O projeto tem **duas definições incompatíveis** de alvo hoje (achado
central da auditoria, `ARCHITECTURE_AUDIT.md` seção 3 e 5). Em vez de
escolher uma silenciosamente, `build_target(df, strategy=...)` expõe as
duas como estratégias nomeadas:

| Estratégia | Fórmula de `taxa_evasao` | Corte do alvo | Filtros | Onde era usada antes |
|---|---|---|---|---|
| `median_split` (**padrão**) | `QT_SIT_DESVINCULADO / (QT_MAT + QT_SIT_DESVINCULADO)` | `>= mediana(taxa_evasao)` | `QT_ING >= 10`, `QT_MAT > 0` | `data/preprocessamento.py` (Naive Bayes) |
| `threshold_20pct` | `(QT_SIT_DESVINCULADO + QT_SIT_TRANCADA) / QT_ING * 100` | `>= 20.0` | `QT_ING > 0` | 4 notebooks + `analysis/audit/data.py` |

`median_split` é o padrão porque a Tarefa 3 pede explicitamente para
"transformar a lógica atualmente existente em `data/preprocessamento.py`".
Os dois limiares (`20.0` e o filtro mínimo de `QT_ING`) vêm de
`src.configs.settings` (Tarefa 2), não são literais soltos neste módulo.

`build_target` retorna um `TargetResult` com o DataFrame já filtrado, a
estratégia usada, o limiar efetivo (a mediana calculada, no caso de
`median_split`) e a taxa de positivos — dados que a rotina de validação
(seção 6) usa diretamente.

---

## 4. Leakage Removal (`src/data/leakage.py`)

Fonte única de verdade para "o que nunca pode virar feature". Antes da
Tarefa 3, essas regras existiam só parcialmente, dentro de
`data/preprocessamento.py`, e não cobriam o vazamento específico
encontrado nos notebooks (`ARCHITECTURE_AUDIT.md`, seção 5).

Quatro grupos, cada um com o motivo documentado no próprio código-fonte:

| Grupo | Exemplos | Motivo (resumo) |
|---|---|---|
| 1. Identificadores | `NO_CURSO`, `CO_IES`, `NO_MUNICIPIO` | Identidade do registro, não característica do curso — risco de "decorar" em vez de generalizar. |
| 2. Origem do alvo | `QT_SIT_DESVINCULADO`, `QT_SIT_TRANCADA`, `taxa_evasao`/`TAXA_EVASAO`, `alto_risco_evasao`/`ALTO_RISCO_EVASAO` | Matéria-prima do alvo — mantê-las como feature permite reconstruir o alvo diretamente (vazamento direto). Inclui o próprio nome do alvo, como rede de segurança para que ele nunca vaze para `X`. |
| 3. Prefixos de matrícula/conclusão | `QT_MAT*`, `QT_CONC*` | Estruturalmente correlacionados com quem não evadiu — vazamento indireto (a auditoria mostrou que os 4 notebooks mantêm `QT_MAT` como feature bruta; este pipeline não repete esse erro). |
| 4. Features derivadas vazadas | `RAZAO_CONCLUSAO_EVASAO` | Calculada a partir de `TAXA_EVASAO` — a mesma variável usada para construir o alvo. Achado crítico da auditoria (seção 5), corrigido aqui na origem. |

- `leakage_columns_for(columns)` — calcula o conjunto de colunas a
  remover, restrito ao que existe em `columns` (nunca falha por coluna
  ausente).
- `remove_leakage_columns(df)` — aplica a remoção, sem alterar `df` in-place.
- `explain_removed_columns(columns)` — mesma coisa, mas retorna um
  dicionário por grupo/motivo, para logging e para a rotina de
  validação mostrar *por que* cada coluna foi removida.

O grupo 3 (prefixos) pode ser desligado via `include_prefix_group=False`
— só para inspeção manual; `src.data.pipeline.run_data_pipeline` nunca
desliga isso por padrão.

---

## 5. Base Preprocessing (`src/pipelines/base_preprocessing.py`)

Contém **somente** transformações comuns aos 5 modelos:

1. `replace_infinite_with_nan` — necessário porque `taxa_evasao` e outras
   razões usam divisão.
2. Imputação de ausentes, duas estratégias nomeadas:
   - `impute_missing_with_zero` (**padrão**, replica
     `data/preprocessamento.py::_selecionar_features`);
   - `impute_missing_with_mode` (replica `handle_missing_mode` dos 4
     notebooks).
3. `select_numeric_columns` — restringe a colunas numéricas (não faz
   encoding categórico, porque essa etapa já diverge entre modelos hoje).

**Deliberadamente fora deste módulo** (fica para a engenharia de
features específica de cada modelo, seção 6, ou para `src/training/`):
`StandardScaler`/`RobustScaler`/`MinMaxScaler`, `SelectKBest`/seleção
por correlação, SMOTE e `class_weight`. Colocar qualquer um desses aqui
violaria o requisito explícito da Tarefa 3 ("não coloque scaling
específico de SVM ou rede neural nesta camada").

---

## 6. Feature Engineering — arquitetura apenas (`src/pipelines/feature_engineering.py`)

Pedido explícito da Tarefa 3: **não implementar** a engenharia de
features específica de cada modelo ainda. O que existe hoje:

- `FeatureEngineer` — um `Protocol` com `fit_transform(X, y)` e
  `transform(X)`, para que qualquer implementação futura respeite a
  regra de nunca ajustar (`fit`) em dados de validação/teste.
- `FEATURE_ENGINEERING_REGISTRY` — mapeamento nome do modelo →
  implementação, hoje todas `None`.
- `get_feature_engineer(model_name)` — levanta `NotImplementedError`
  explicando onde está a lógica original (no notebook/script do modelo),
  em vez de retornar algo incorreto silenciosamente.

Os *conjuntos* de colunas por modelo (quais features cada um usa) já
existem de forma declarativa desde a Tarefa 2, em
`src/pipelines/feature_groups.py` — esse módulo passou a importar
`KNOWN_LEAKY_DERIVED_FEATURES` de `src.data.leakage` (Tarefa 3) em vez de
duplicar a lista, agora que `src.data.leakage` é a fonte única.

---

## 7. Orquestração (`src/data/pipeline.py`)

`run_data_pipeline(...)` encadeia as 4 primeiras etapas e devolve um
`DataPipelineResult` com `X`, `y` e metadados de auditoria
(`n_records_raw`, `n_records_final`, `removed_leakage_columns`,
`target_threshold`, `positive_rate`).

```python
from src.data.pipeline import run_data_pipeline

result = run_data_pipeline(
    target_strategy="median_split",   # ou "threshold_20pct"
    missing_strategy="zero",          # ou "mode"
)
result.X            # features prontas para receber feature engineering
result.y             # alto_risco_evasao
result.positive_rate # proporção da classe positiva
```

Ordem interna: `load_raw_dataset` → `build_target` → (extrai `y` do
DataFrame **antes** de remover colunas de vazamento) →
`remove_leakage_columns` (que também remove a própria coluna de alvo de
`X`, como rede de segurança) → `run_base_preprocessing`. O parâmetro
`feature_engineer` (opcional, `None` por padrão) é o ponto de extensão
para a 5ª etapa, quando ela existir.

---

## 8. Validação de compatibilidade (`src/data/validation.py`)

Requisito explícito da Tarefa 3. `compare_preprocessing_outputs(X_old,
y_old, X_new, y_new)` retorna um `ComparisonReport` com:

- número de registros (`n_records_old`/`n_records_new`);
- número de features (`n_features_old`/`n_features_new`);
- nomes das features, com a diferença simétrica já calculada
  (`features_only_in_old`/`features_only_in_new`);
- distribuição das classes do alvo (`class_distribution_old`/`_new`) e
  taxa de positivos;
- `matches` (um booleano por critério) e `is_fully_compatible` (E lógico
  de todos os critérios).

A rotina **não decide** se uma divergência é aceitável — só a torna
visível. Por exemplo: comparar `median_split` com `threshold_20pct`
mostrará `records` e `class_distribution_shape` divergentes, o que é
esperado (são estratégias diferentes por definição) e não indica um bug.

`validate_against_legacy_naive_bayes_pipeline()` é a comparação natural
para esta etapa: roda `run_data_pipeline(target_strategy="median_split")`
e `data.preprocessamento.get_df_preprocessado()` (o pipeline legado) lado
a lado. Requer o CSV do INEP disponível localmente — por isso não é
chamada pelos testes automatizados (que usam dados sintéticos em
memória), mas está pronta para ser rodada manualmente assim que o
dataset e um interpretador Python estiverem disponíveis (ver seção 10).

---

## 9. Testes

| Arquivo | Cobre |
|---|---|
| `tests/test_data_loading.py` | Carregamento: kwargs do CSV, leitura de um CSV sintético, erro em caminho inexistente, resolução automática via config. |
| `tests/test_target_construction.py` | Target: fórmulas das duas estratégias, filtros (`QT_ING`, `QT_MAT`), limiar = mediana, estratégia desconhecida, colunas obrigatórias ausentes, resultado vazio após filtros. |
| `tests/test_leakage_removal.py` | Leakage: os 4 grupos de remoção, preservação de features legítimas, não-mutação do DataFrame de entrada, desligar o grupo de prefixos, colunas ausentes ignoradas silenciosamente, `explain_removed_columns` por grupo. |
| `tests/test_feature_selection.py` | Seleção: `select_numeric_columns`, `run_base_preprocessing` com/sem restrição numérica, registry de feature engineering (todos `NotImplementedError` nesta etapa), modelo desconhecido. |
| `tests/test_invalid_values.py` | Valores inválidos: substituição de `inf`/`-inf`, imputação por moda (incluindo fallback quando tudo é `NaN`), imputação por zero, pipeline completo de ponta a ponta. |
| `tests/test_pipeline_integration.py` (bônus, não pedido explicitamente mas cobre a seção "Compatibilidade" de ponta a ponta) | `run_data_pipeline` completo com as duas estratégias, garantia de que o alvo nunca vaza para `X`, `compare_preprocessing_outputs` com pipelines idênticos e divergentes. |

Todos os testes usam dados sintéticos pequenos (DataFrames construídos
em memória ou CSVs temporários de poucas linhas) — nenhum depende do
CSV real do INEP (>100MB, não versionado).

---

## 10. Execução dos testes — limitação deste ambiente

**Esta máquina não tem nenhum interpretador Python funcional instalado**
(`python`/`python3`/`py` resolvem apenas para os aliases quebrados da
Microsoft Store; busca extensiva por instalações em `C:\Python*`,
Program Files, `AppData\Local\Programs\Python`, Miniconda/Anaconda e no
registro do Windows não encontrou nada). Não foi possível executar
`pytest` nem rodar um `python -c "..."` simples.

Em vez disso, cada arquivo novo foi **revisado estaticamente por completo**
após a escrita: sintaxe, parênteses/chaves balanceados, e cada `import`
entre módulos conferido manualmente contra o nome exportado no módulo de
origem (ex.: todo nome importado de `src.data.target` em `pipeline.py`
existe em `target.py`; `KNOWN_LEAKY_DERIVED_FEATURES` importado por
`feature_groups.py` existe em `leakage.py`). Dois problemas foram
encontrados e corrigidos nessa revisão antes de considerar a etapa
concluída:

1. `src/data/pipeline.py` usava `__import__("src.data.target", ...)`
   para acessar `TARGET_COLUMN` em vez de um import direto — corrigido
   para `from src.data.target import TARGET_COLUMN, ...`.
2. `src/data/validation.py` importava `pathlib.Path` sem usar — import
   morto removido.

Todos os arquivos de teste seguem o mesmo padrão de `tests/test_imports.py`
(Tarefa 2): funções `test_*` compatíveis com `pytest` **e** um bloco
`if __name__ == "__main__"` que roda todos os testes do arquivo e reporta
OK/FAIL sem exigir `pytest` instalado.

**Ação recomendada para quem retomar este trabalho**, com Python 3.10+ e
as dependências de `requirements.txt` instaladas:

```bash
python -m pytest tests/ -v
```

ou, arquivo a arquivo, sem pytest:

```bash
python tests/test_data_loading.py
python tests/test_target_construction.py
python tests/test_leakage_removal.py
python tests/test_feature_selection.py
python tests/test_invalid_values.py
python tests/test_pipeline_integration.py
```

Depois, para validar contra o pipeline legado de verdade (requer o CSV
do INEP baixado localmente — ver `README.md`, seção "Dados"):

```python
from src.data.validation import validate_against_legacy_naive_bayes_pipeline

report = validate_against_legacy_naive_bayes_pipeline()
print(report.summary())
assert report.is_fully_compatible
```

---

## 11. Compatibilidade com a arquitetura antiga

- `data/preprocessamento.py`, os 4 notebooks e `analysis/` não foram
  modificados.
- `git status` após esta etapa mostra apenas arquivos novos/modificados
  dentro de `src/`, `tests/` e `docs/` — nenhuma mudança fora dessas
  pastas.
- `src.data.leakage` agora é a fonte única de verdade para regras de
  vazamento; `src.configs.constants` (Tarefa 2) foi ajustado para não
  duplicar mais essas listas, apontando para `src.data.leakage` em vez
  disso — evitando o mesmo tipo de duplicação que a auditoria original
  identificou no projeto legado.

## 12. Próximos passos (fora do escopo desta etapa)

Ver `ARCHITECTURE_AUDIT.md`, seção 8, fase 2 em diante: implementar a
engenharia de features específica de cada modelo em
`src/pipelines/feature_engineering.py` (uma implementação de
`FeatureEngineer` por modelo), depois `src/training/` e `src/evaluation/`.
