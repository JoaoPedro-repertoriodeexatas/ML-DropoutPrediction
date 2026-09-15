# Arquitetura do projeto — estrutura nova (Tarefa 2)

**Branch:** `audit/architecture-review`
**Baseado em:** [`ARCHITECTURE_AUDIT.md`](../ARCHITECTURE_AUDIT.md)

Este documento descreve a estrutura de diretórios criada nesta etapa.
**Nenhuma lógica foi migrada.** Nenhum notebook, script ou resultado
existente foi alterado, movido ou removido — apenas novos diretórios e
módulos "esqueleto" foram adicionados.

---

## 1. Por que a estrutura ficou sob `src/`

A auditoria (seção 7 de `ARCHITECTURE_AUDIT.md`) propôs uma árvore nova
com `configs/`, `data/`, `models/`, `pipelines/`, `training/`,
`evaluation/`, `visualization/`, `experiments/`, `scripts/`, `results/`
e `tests/`. Ao tentar criar isso literalmente na raiz do repositório,
três desses nomes **colidem com diretórios já existentes e usados pela
arquitetura antiga**:

| Nome desejado | Já existe na raiz? | Conteúdo atual |
|---|---|---|
| `data/` | Sim | `data/preprocessamento.py` — pipeline de dados do Naive Bayes |
| `models/` | Sim | notebooks + artefatos (CSV/PNG/PDF) de cada um dos 5 modelos |
| `results/` | Sim | resultados pré-auditoria, pós-auditoria e de validação final |

Como a Tarefa 2 exige explicitamente **coexistência temporária** ("não
apague os notebooks existentes", "não apague os arquivos existentes",
"a prioridade é criar a estrutura sem quebrar o projeto atual"), a
arquitetura foi adaptada assim:

- **Todo o código novo** (`configs`, `data`, `pipelines`, `models`,
  `training`, `evaluation`, `visualization`) vive dentro de um único
  pacote Python, **`src/`**, evitando qualquer colisão de nome com as
  pastas antigas.
- **`experiments/`, `scripts/` e `tests/`** não colidiam com nada e
  foram criados diretamente na raiz, como pedido.
- **`results/`** não foi recriado: a pasta antiga já existe e continuará
  sendo o destino dos artefatos até a migração de fato acontecer (fase
  6–7 do plano de migração da auditoria). `src/configs/paths.py` já
  define `NEW_ARTIFACTS_DIR` como referência para um futuro diretório
  `artifacts/` (proposto na seção 7 da auditoria), mas esse diretório
  **não foi criado** nesta etapa — não há nada para escrever nele ainda.

Essa é a única adaptação estrutural em relação ao pedido original; tudo
o mais segue a lista de diretórios exatamente como especificada.

---

## 2. Árvore criada

```text
Projeto-Final-Machine-Learning/          (raiz do repositório)
│
├── src/                          # NOVO — todo o código da arquitetura nova
│   ├── __init__.py
│   ├── configs/                  # camada de configuração centralizada
│   │   ├── __init__.py
│   │   ├── paths.py              # PROJECT_ROOT e todos os caminhos (pathlib)
│   │   ├── settings.py           # random_state, folds, thresholds, amostragem
│   │   └── constants.py          # nomes de colunas, features, arquivos de saída
│   ├── data/                     # carregamento de dados (interface, sem lógica)
│   │   ├── __init__.py
│   │   └── loading.py            # load_raw_dataset() — stub, NotImplementedError
│   ├── pipelines/                # engenharia de features / pré-processamento
│   │   ├── __init__.py
│   │   └── feature_groups.py     # grupos de features por modelo (declarativo)
│   ├── models/                   # definição/registro de modelos
│   │   ├── __init__.py
│   │   └── registry.py           # MODEL_REGISTRY (esqueleto, sem factories)
│   ├── training/                 # treinamento e validação cruzada
│   │   └── __init__.py
│   ├── evaluation/                # métricas, sanity checks, ranking
│   │   └── __init__.py
│   └── visualization/            # gráficos comparativos
│       └── __init__.py
│
├── experiments/                  # NOVO — pontos de entrada de linha de comando
│   └── __init__.py
├── scripts/                      # NOVO — utilitários avulsos (migração, manutenção)
│   └── __init__.py
├── tests/                        # NOVO — testes automatizados
│   ├── __init__.py
│   └── test_imports.py           # smoke test: importabilidade + consistência
├── docs/                         # NOVO
│   └── ARCHITECTURE.md           # este arquivo
│
├── ARCHITECTURE_AUDIT.md         # (Tarefa 1, já existente)
│
├── data/                         # pipeline legado do Naive Bayes (conteúdo inalterado)
├── models/                       # notebooks e artefatos dos 5 modelos (conteúdo inalterado)
├── analysis/                     # auditoria/validação legadas (inalterado)
├── results/                      # resultados pré/pós-auditoria (conteúdo inalterado)
├── slides/                       # inalterado
├── README.md, requirements.txt, .gitignore   # inalterados
```

> `report/` e `EDA/` existiam nesta etapa (Tarefa 2) mas foram removidos
> na Tarefa 9 (`docs/FILE_CLEANUP.md`) — `EDA/` estava vazia desde a
> criação do projeto, e `report/final_report.pdf` era uma duplicata
> byte a byte de `results/final_report.pdf`. `.gitkeep` redundantes em
> `data/`, `models/`, `results/` (pastas já com conteúdo real) também
> foram removidos na mesma etapa, sem afetar o conteúdo dessas pastas.
>
> **Atualização posterior (limpeza final, ver `docs/FILE_CLEANUP.md`
> seção 3):** com todas as camadas já migradas para `src/`/`experiments/`
> (Tarefas 3-8) e validadas cientificamente (Tarefa 10), `data/` inteiro
> foi removido (só continha `preprocessamento.py`, já substituído por
> `src/data/`), os 4 notebooks originais e os scripts `.py` originais do
> Naive Bayes foram removidos de `models/<modelo>/` (mantendo os
> artefatos de resultado/hiperparâmetro que eles geraram), e `slides/`
> foi removido junto com os entregáveis pré-auditoria em `results/`
> (`final_report.pdf/.tex`, `ranking.csv` e demais CSVs superados por
> `results/audit/`). `analysis/` foi mantido integralmente por ser a
> origem dos resultados oficiais — note que isso deixa
> `analysis/final_validation/*.py`, `analysis/compare_models.py` e
> `analysis/audit/final_report_pdf.py` não executáveis como estão, pois
> leem como entrada arquivos de `results/` que não existem mais; isso é
> aceitável porque eles geravam justamente o relatório/ranking
> pré-auditoria (já substituído), não os números oficiais
> (`analysis/audit/run_audit.py`, que não depende deles).

---

## 3. Responsabilidade de cada diretório

### `src/configs/`
Camada de configuração centralizada pedida na Tarefa 2. Três módulos:

- **`paths.py`** — `PROJECT_ROOT` e todos os caminhos derivados dele via
  `pathlib.Path`, incluindo os caminhos `LEGACY_*` (somente leitura, para
  referenciar a estrutura antiga) e `LEGACY_MODEL_DIRS` (mapeia o nome
  canônico de cada modelo à sua pasta em `models/`). Também centraliza
  `DATASET_PATH_CANDIDATES` e `resolve_dataset_path()`, hoje duplicados
  em `data/preprocessamento.py` e `analysis/audit/data.py`.
- **`settings.py`** — `RANDOM_STATE`, número de folds, tamanho de
  amostra, parâmetros de threshold tuning com guarda-corpo. Onde os
  pipelines legados divergem entre si (ex.: amostra de 30.000 vs. 90.000
  da Árvore), os dois valores são registrados explicitamente em vez de
  se escolher um silenciosamente — ver comentários no próprio arquivo.
- **`constants.py`** — nomes de colunas (`TARGET_COLUMN`), grupos de
  features (`COMMON_FEATURES`, `ADVANCED_TREE_FEATURES`), a lista
  `KNOWN_LEAKY_FEATURES` (hoje contendo `RAZAO_CONCLUSAO_EVASAO`, a
  feature de vazamento identificada na auditoria) e nomes de arquivo de
  saída padronizados.

### `src/data/`
Vai concentrar o carregamento do dataset bruto do INEP. Hoje contém só a
assinatura de `load_raw_dataset()`, que levanta `NotImplementedError`
explicando onde está a lógica real por enquanto (nos notebooks e em
`data/preprocessamento.py`).

### `src/pipelines/`
Vai concentrar a engenharia de features e o pré-processamento. Hoje
contém `feature_groups.py`, que apenas organiza — de forma declarativa —
quais dos grupos de features definidos em `constants.py` cada modelo usa
(`TREE_FEATURE_SET`, `SVM_FEATURE_SET` etc.), e um utilitário
`assert_no_known_leakage()` que qualquer pipeline futuro poderá chamar
antes de treinar.

### `src/models/`
Vai concentrar a definição e o registro dos modelos. Hoje contém
`registry.py` com um `MODEL_REGISTRY` cujas chaves são os 5 nomes
canônicos de modelo (`arvore_de_decisao`, `regressao_logistica`, `svm`,
`rede_neural`, `naive_bayes`), sem nenhuma factory de estimador
implementada ainda.

### `src/training/`
Vai concentrar o protocolo único de cross-validation, a busca de
hiperparâmetros corrigida para nested CV e a única implementação de
threshold tuning com guarda-corpo (hoje em `analysis/audit/threshold.py`).
Nesta etapa contém só o `__init__.py` com a responsabilidade documentada.

### `src/evaluation/`
Vai concentrar métricas, sanity checks e a fórmula única de ranking
(substituindo os três sistemas de ranking hoje concorrentes). Nesta
etapa contém só o `__init__.py`.

### `src/visualization/`
Vai concentrar os gráficos comparativos e por modelo, com um destino de
saída único. Nesta etapa contém só o `__init__.py`.

### `experiments/`
Vai concentrar os pontos de entrada de linha de comando (`run_model.py`,
`run_all.py`) que permitirão treinar qualquer modelo via Python, sem
depender de notebook — hoje só o Naive Bayes tem essa capacidade
(`models/naive_bayes/train.py`).

### `scripts/`
Utilitários avulsos que não são "rodar um experimento completo" — por
exemplo, o futuro script de migração incremental de cada notebook para
`src/`.

### `tests/`
Testes automatizados. Nesta etapa, `test_imports.py` verifica apenas
estrutura: importabilidade de todos os pacotes novos, integridade da
configuração central e ausência da feature de vazamento conhecida nos
grupos de features declarativos. Testes de comportamento (carregamento
real de dados, treino, threshold tuning) serão adicionados junto com a
lógica correspondente, fase a fase.

---

## 4. O que foi centralizado na configuração (pedido explícito da Tarefa 2)

| Categoria | Onde | Exemplos |
|---|---|---|
| Paths | `src/configs/paths.py` | `PROJECT_ROOT`, `LEGACY_MODEL_DIRS`, `resolve_dataset_path()` |
| Random seeds | `src/configs/settings.py` | `RANDOM_STATE = 42`, `SMOTE_RANDOM_STATE` |
| Constantes | `src/configs/constants.py`, `src/configs/settings.py` | `TARGET_COLUMN`, `EVASION_RATE_THRESHOLD_PCT`, `CV_FOLDS` |
| Nomes de arquivo | `src/configs/constants.py` | `METRICS_SUMMARY_FILENAME`, `BEST_PARAMS_FILENAME`, `RANKING_FILENAME` |
| Configurações gerais | `src/configs/settings.py` | tamanho de amostra, faixa de busca de threshold, split de calibração |

---

## 5. Verificação de importabilidade

**Limitação do ambiente:** esta máquina não tem um interpretador Python
funcional instalado (`python`/`python3`/`py` resolvem apenas para os
aliases quebrados da Microsoft Store, e nenhuma instalação real —
sistema, venv ou conda — foi encontrada). Não foi possível executar
`pytest` ou `python -c "import ..."` para uma verificação dinâmica real.

Em vez disso, a verificação foi feita **estaticamente**:

- todos os arquivos foram revisados linha a linha após a criação (via
  leitura completa do conteúdo escrito) para confirmar sintaxe válida,
  parênteses/chaves balanceados e imports consistentes entre módulos;
- cada `import` entre módulos novos foi conferido manualmente contra o
  nome real exportado no módulo de origem (ex.: todos os nomes
  importados em `feature_groups.py` a partir de `constants.py` existem
  em `constants.py`);
- `tests/test_imports.py` foi escrito para também funcionar sem
  `pytest` instalado (bloco `if __name__ == "__main__"` que roda todas
  as funções `test_*` e reporta OK/FAIL), para que baste rodar
  `python tests/test_imports.py` em uma máquina com Python assim que
  disponível.

**Ação recomendada para quem retomar este trabalho:** rodar, com Python
3.10+ instalado,

```bash
python -m pytest tests/test_imports.py -v
```

ou, sem pytest:

```bash
python tests/test_imports.py
```

Ambos devem reportar sucesso em todos os testes antes de qualquer
lógica adicional ser migrada para `src/`.

---

## 6. Compatibilidade com a arquitetura antiga

- Nenhum arquivo de `data/`, `models/`, `analysis/`, `results/`,
  `report/`, `slides/`, `EDA/` foi criado, movido, editado ou removido.
- `git status` após esta etapa mostra apenas arquivos novos e
  não-rastreados (`src/`, `experiments/`, `scripts/`, `tests/`,
  `docs/`) — nenhuma modificação (`M`) em arquivo pré-existente.
- Os notebooks continuam funcionando exatamente como antes; nada em
  `src/` é importado por eles nesta etapa.
- `src/configs/paths.py::LEGACY_MODEL_DIRS` referencia as pastas de
  modelo existentes só para leitura (usado hoje apenas pelo teste de
  smoke `test_legacy_model_dirs_registered`).

## 7. Próximos passos (fora do escopo desta etapa)

Ver `ARCHITECTURE_AUDIT.md`, seção 8 ("Plano de migração"), fases 1 a 8 —
em particular a Fase 1 (extrair `src/data/` e `src/pipelines/` a partir
do código hoje triplicado nos notebooks, com testes de regressão
bit-a-bit antes de tocar em qualquer notebook) e a Fase 2 (remover
`RAZAO_CONCLUSAO_EVASAO` do conjunto de features da Árvore de Decisão).
