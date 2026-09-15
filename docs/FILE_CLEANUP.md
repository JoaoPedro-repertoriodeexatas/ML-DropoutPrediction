# Auditoria final de arquivos (Tarefa 9 + Rodada 2, ver seção 6)

**Branch:** `audit/architecture-review`
**Baseado em:** `ARCHITECTURE_AUDIT.md`, seção 6 ("Análise de arquivos")

Para cada candidato considerado, verifiquei, nesta ordem (conforme
pedido pela tarefa): referências no código, imports, chamadas,
referências no `README.md`, referências na documentação (`docs/*.md`,
`ARCHITECTURE_AUDIT.md`), dependências em notebooks, necessidade para
reprodução, e necessidade para a entrega acadêmica. As evidências de
cada verificação estão registradas abaixo, por arquivo.

**Resultado:** dois arquivos e duas pastas-placeholder removidos, todos
com evidência de que são seguros. **Nenhum resultado científico,
notebook, PDF de entrega ou dataset foi removido.** A reorganização de
`models/<modelo>/` para `results/` pedida pela tarefa foi **investigada
e adiada, não executada** — a razão está detalhada na seção 3, com
evidência concreta de que executá-la agora quebraria a reprodução de
dois artefatos já entregues.

---

## 1. Tabela de ações

| Arquivo | Ação | Justificativa |
|---|---|---|
| `EDA/.gitkeep` | **Removido** | Pasta `EDA/` estava completamente vazia (só o `.gitkeep`) desde a criação do projeto — nenhum notebook/script de EDA foi versionado. Sem referência em código, `README.md` não a menciona na árvore de diretórios real do projeto (só a documentação desta auditoria a cita, como achado). Não é artefato de reprodução nem de entrega — é um placeholder nunca preenchido. |
| `report/final_report.pdf` | **Removido** | Duplicata **byte a byte** confirmada (mesmo MD5, 1.197.887 bytes) de `results/final_report.pdf`. `results/final_report.pdf` é a cópia ativamente referenciada — citada em `results/audit/audit_report.md` e `results/project_inventory.md`, e é o caminho de saída hardcoded de `analysis/compare_models.py` (`REPORT_PATH = RESULTS_DIR / "final_report.pdf"`). `report/final_report.pdf` não é citado em nenhum código, `README.md` ou documento — só aparecia na árvore de diretórios do `README.md` como pasta genérica, sem menção ao arquivo em si. |
| `report/.gitkeep` | **Removido** | Consequência de remover o único outro arquivo da pasta (acima) — a pasta `report/` fica vazia e sem propósito documentado. |
| `data/.gitkeep`, `models/.gitkeep`, `results/.gitkeep` | **Removidos** | Redundantes: as três pastas já têm conteúdo real versionado (`data/preprocessamento.py`; os 5 subdiretórios de modelo; dezenas de arquivos de resultado) — o `.gitkeep` existe para manter uma pasta vazia rastreada pelo Git, o que deixou de ser necessário assim que cada pasta ganhou conteúdo. Nenhum código ou documentação depende da presença do arquivo (`grep -rl gitkeep` só retorna esta própria auditoria). As pastas `data/`, `models/`, `results/` **continuam existindo normalmente**, com todo o conteúdo real intacto — só o marcador vestigial foi removido. |
| Todo o restante do repositório | **Mantido** | Ver seções 2 e 3 para os candidatos que foram investigados e propositalmente **não** removidos, com a evidência de por que não. |

---

## 2. Candidatos investigados e **mantidos** (com evidência)

### 2.1 `results/visualizations/*.png` vs. `results/audit/visualizations/*.png`

Mesmos 4 nomes de arquivo (`metric_heatmap.png`, `metrics_bar_chart.png`,
`radar_chart.png`, `ranking_heatmap.png`) em dois diretórios —
pareciam duplicatas suspeitas na Tarefa 1.

**Verificação:** comparação byte a byte (`cmp`) dos 4 pares — **todos
diferentes** (tamanhos de arquivo distintos em cada par: ex.
`metric_heatmap.png` tem 83.191 bytes em `results/visualizations/` e
77.439 bytes em `results/audit/visualizations/`). São gráficos gerados
por pipelines diferentes (`analysis/visualizations.py`, pré-auditoria,
vs. a etapa de auditoria) e representam estágios científicos distintos
do projeto (antes/depois da correção de threshold). **Mantidos** — são
resultados científicos importantes, não duplicatas.

### 2.2 `models/naive_bayes/report/project_naive_bayes_architecture.html`

Sinalizado na Tarefa 1 como "propósito não documentado".

**Verificação:** é um slide-deck HTML autocontido (`<title>Naive Bayes —
Arquitetura do Projeto</title>`, com MathJax e estilo de apresentação),
do mesmo gênero de `slides/audit_results_presentation.html`. Não é
código, não é duplicata de nada, e não há evidência de que tenha sido
substituído por outra coisa. É material de apresentação/documentação da
arquitetura original do Naive Bayes — potencialmente parte da entrega
acadêmica (apresentação do projeto). **Mantido** — nenhuma evidência de
obsolescência, e a regra "não remova documentação relevante" se aplica
diretamente.

### 2.3 `results/ranking.csv`, `results/final_comparison.csv`, `results/extracted_metrics.csv` (métricas pré-auditoria)

Já avaliados na Tarefa 1 como parte do rastro de auditoria.

**Verificação:** citados explicitamente em
`results/inconsistencies_report.md` como a "fonte pré-auditoria" que
demonstra o problema de threshold corrigido depois — são a evidência
documental de um achado científico do projeto (não um resultado
"antigo e substituído sem valor", mas a prova do "antes" que justifica o
"depois"). **Mantidos.**

### 2.4 Os quatro notebooks originais (`models/*/​*.ipynb`) e `models/naive_bayes/*.py`

**Verificação:** são a **única fonte de resultados reais** (produzidos
com o dataset verdadeiro do INEP) que o projeto tem hoje. Os novos
notebooks de `experiments/*.ipynb` (Tarefa 8) foram construídos sobre a
arquitetura migrada, mas só foram executados contra um dataset
**sintético** de verificação (ver `docs/EXPERIMENTS.md`, seção 5) — não
produziram nenhum resultado real ainda. Remover os notebooks/scripts
originais agora apagaria a única evidência de resultado real do projeto
para 4 dos 5 modelos, além dos PDFs de relatório e artefatos
(`best_model.pkl`, `best_params.json` etc.) que dependem deles.
**Mantidos** — são "artefatos necessários para reprodução" e
"resultados científicos importantes" por definição, e ainda não têm um
substituto com resultados reais equivalentes.

### 2.5 `analysis/` (pipeline de auditoria/validação em Python)

**Verificação:** embora `src/` (Tarefas 2-7) reimplemente boa parte da
mesma lógica de forma mais organizada, `analysis/` é o código que
efetivamente **gerou** `results/audit/*`, `results/final_validated_ranking.csv`,
`results/inconsistencies_report.md` e outros artefatos já entregues e
citados. Removê-lo quebraria a capacidade de explicar/reproduzir como
esses resultados foram gerados — o mesmo motivo que impede a
reorganização de `models/` descrita na seção 3. **Mantido.**

### 2.6 `slides/apresentacao.pdf`, `slides/audit_results_presentation.html`

**Verificação:** materiais de apresentação, não referenciados por
código, mas coerentes com uma entrega acadêmica que normalmente inclui
uma apresentação. Nenhuma evidência de que sejam versões antigas
substituídas por algo mais novo. **Mantidos** — regra explícita "não
remova... PDFs necessários para a entrega".

---

## 3. Reorganização de `models/<modelo>/` para `results/` — investigada e **adiada**

A tarefa pede: "se houver arquivos de resultado espalhados dentro de
`models/<modelo>/`, mova-os para uma estrutura centralizada em
`results/`, mantendo a rastreabilidade". Há, de fato, bastante material
disperso (CSVs de métrica, PNGs, PDFs de relatório, `.txt` de
hiperparâmetros, e os artefatos do Naive Bayes) espalhado em
`models/arvore_de_decisao/`, `models/regrassao_logisticca/`,
`models/svm/`, `models/Redes_Neurais/` e `models/naive_bayes/`.

**Por que não executei a movimentação agora**, apesar de ser o pedido
explícito da tarefa: antes de mover, segui o próprio checklist da
tarefa ("verifique se é artefato necessário para reprodução") e
encontrei **duas dependências de caminho hardcoded** que quebrariam se
os arquivos fossem movidos:

1. **`analysis/metrics_discovery.py`** (linha 14): `MODELS_DIR =
   PROJECT_ROOT / "models"`. A função `discover_model_metrics()`
   percorre `models/<subpasta>/` inteiro (`model_dir.rglob("*.csv")`)
   para descobrir e consolidar as métricas de cada modelo — é assim que
   `results/ranking.csv` e `results/final_comparison.csv` são
   (re)gerados (`analysis/compare_models.py`, etapa 1/5: "descoberta
   automática de métricas em `models/`"). Mover os CSVs para fora de
   `models/` faria essa descoberta automática não encontrar mais nada.
2. **`analysis/final_validation/consistency.py`** (linhas 155 e 240):
   caminho hardcoded `MODELS / "Redes_Neurais" /
   "resultados_rede_neural_5fold.csv"`, lido diretamente para comparar
   os números pré- e pós-auditoria da Rede Neural — é assim que
   `results/inconsistencies_report.md` (um documento já citado
   extensivamente nesta auditoria) é gerado. Mover esse arquivo
   quebraria essa comparação sem qualquer aviso — o script simplesmente
   deixaria de encontrar o arquivo.

Executar a movimentação sem também reescrever esses dois módulos
quebraria, silenciosamente, a reprodução de `results/ranking.csv`,
`results/final_comparison.csv` e `results/inconsistencies_report.md` —
exatamente o que a regra "não remova artefatos necessários para
reprodução" da própria Tarefa 9 proíbe. Reescrever os dois módulos
para apontar para uma nova estrutura em `results/` é uma mudança de
**código** (não uma limpeza de arquivo), em módulos que produzem
artefatos científicos já entregues — o tipo de alteração que todas as
Tarefas 2-8 trataram com o máximo de cautela, sempre preservando o
comportamento observável do código legado.

**Recomendação registrada para uma etapa futura dedicada** (não
executada aqui): mover os arquivos de resultado de `models/<modelo>/`
para `results/<modelo>/`, atualizar `MODELS_DIR` em
`metrics_discovery.py` e o caminho hardcoded em
`consistency.py` para a nova localização, e então validar rodando
`analysis.metrics_discovery.discover_model_metrics()` antes/depois da
mudança para confirmar que o resultado consolidado não muda — essa
validação **não precisa do dataset real do INEP** (só lê CSVs já
existentes), então é totalmente executável numa etapa dedicada, com o
devido cuidado que a mudança de dois módulos de código merece.

---

## 4. O que a Tarefa 9 não encontrou

Nenhum dos seguintes apareceu no repositório, então não há nada a
remover nessas categorias: código morto/duplicado dentro de `src/`
(cada módulo criado nas Tarefas 2-8 é usado por outro módulo ou por
testes, verificado nesta auditoria por leitura); scripts temporários ou
arquivos de scratch (nenhum ficou para trás — cada tarefa anterior já
limpou seus próprios artefatos de verificação, ex.: o CSV sintético e os
scripts que geraram/executaram os notebooks de `experiments/` na Tarefa
8 nunca foram commitados); `.ipynb_checkpoints`/`__pycache__` versionados
(já cobertos por `.gitignore`, e a checagem confirma que nenhum está
rastreado); arquivos de sistema operacional (`Thumbs.db`, `.DS_Store`).

---

## 5. Verificação pós-limpeza

Executada de verdade, depois da remoção (`git rm`, staged — não
commitado):

1. **Busca por referências quebradas:** `grep` por `EDA/` e
   `report/final_report` em todo o repositório (código, markdown,
   notebooks). Únicas ocorrências restantes: as desta própria auditoria
   e uma em `docs/ARCHITECTURE.md` (Tarefa 2), que dizia que `report/` e
   `EDA/` estavam "inalterados" — **corrigida** nesta etapa, já que
   deixou de ser verdade.
2. **Suíte de testes completa:**

   ```text
   python -m pytest tests/ -v
   ...
   ===================== 190 passed in 34.99s ======================
   ```

   Nenhuma regressão — os 190 testes das Tarefas 2-8 continuam
   passando, incluindo `test_imports.py` (que confere
   `LEGACY_DATA_DIR.is_dir()`, `LEGACY_MODELS_DIR.is_dir()`,
   `LEGACY_RESULTS_DIR.is_dir()`) e os 34 testes de
   `test_experiment_notebooks.py`/`test_imports.py` rodados
   isoladamente para confirmar a estrutura dos notebooks.
3. **Scripts/imports principais:** `python -c "from src.data.pipeline
   import run_data_pipeline; from src.training.trainer import Trainer;
   ..."` — todos os módulos centrais de `src/` continuam importáveis;
   `LEGACY_DATA_DIR`/`LEGACY_MODELS_DIR`/`LEGACY_RESULTS_DIR` (Tarefa 2)
   continuam apontando para diretórios existentes.
4. **Notebooks:** os 5 notebooks de `experiments/` (Tarefa 8) não
   referenciam nenhum dos arquivos removidos — `test_experiment_notebooks.py`
   passou integralmente. Os 4 notebooks originais e
   `models/naive_bayes/*.py` não foram tocados por esta tarefa.
5. **Caminhos de resultado:** `results/final_report.pdf` (o arquivo
   canônico, mantido) confirmado presente e íntegro
   (1.197.887 bytes, mesmo tamanho de antes); `results/`, `models/`,
   `data/` confirmados com todo o conteúdo real intacto.

`git status` final, resumido:

```text
Changes to be committed:
	deleted:    EDA/.gitkeep
	deleted:    data/.gitkeep
	deleted:    models/.gitkeep
	deleted:    report/.gitkeep
	deleted:    report/final_report.pdf
	deleted:    results/.gitkeep
```

Nenhuma outra mudança — a remoção ficou restrita exatamente aos 6
arquivos justificados na seção 1, staged via `git rm` (reversível pelo
histórico do Git; nada foi commitado nesta etapa).

---

## 6. Rodada 2 — limpeza final pós-refatoração completa (a pedido do usuário)

**Contexto:** as seções 1-5 acima são da Tarefa 9, executada **antes**
da migração de lógica estar terminada e validada (Tarefas 3-10 ainda
não existiam). Por isso, as seções 2.3, 2.4 e 2.6 mantiveram
propositalmente os notebooks originais, os CSVs pré-auditoria e os
materiais de apresentação — a justificativa era que ainda não havia
substituto com resultados equivalentes.

Com a arquitetura em `src/`/`experiments/` **totalmente migrada e
validada cientificamente** (Tarefa 10, `docs/VALIDATION_REPORT.md`) e
com o `README.md` já reescrito citando `results/audit/` como única
fonte de resultados oficiais (Tarefa 11), o usuário pediu explicitamente
para eliminar a duplicação entre a versão legada e a versão atual que
coexistiam na raiz do projeto. As decisões abaixo foram tomadas
respondendo a perguntas diretas feitas ao usuário (não por iniciativa
unilateral), item a item:

| Arquivo/pasta | Ação | Justificativa |
|---|---|---|
| `models/*/​*.ipynb` (4 notebooks originais) | **Removidos** | Totalmente substituídos por `experiments/*.ipynb` (Tarefa 8), já validados contra os originais (Tarefa 10). Os artefatos que geraram (CSV de métricas, PNG, PDF, `.txt` de hiperparâmetros) foram **mantidos** em `models/<modelo>/` — só o código-fonte do notebook saiu. |
| `models/naive_bayes/{evaluate,model_factory,preprocessing,report_generator,train}.py` | **Removidos** | Mesma lógica do item acima: substituídos por `src/models/naive_bayes.py`, `src/pipelines/feature_engineering.py::for_naive_bayes` e `src/training/trainer.py`. Artefatos (`artifacts/best_model.pkl`, `best_params.json`, `metrics_*.csv`, `report/*.pdf`) **mantidos**. |
| `data/__init__.py`, `data/preprocessamento.py` | **Removidos (pasta `data/` inteira)** | Substituídos por `src/data/` (Tarefa 3), já validado byte a byte (`docs/DATA_PIPELINE.md`). `data/__init__.py` só existia para expor `preprocessamento.py`; sem ele, o pacote `data/` ficava vazio. Confirmado por grep que nenhum outro módulo importa o pacote `data` (só `models/naive_bayes/preprocessing.py`, removido no item acima). `src/configs/paths.py::LEGACY_DATA_DIR` continua definido como candidato de localização do CSV bruto (mesma convenção usada em `analysis/audit/data.py`), sem assumir que o diretório exista — `tests/test_imports.py` e o comentário do módulo foram atualizados de acordo. |
| `results/final_report.pdf`, `results/final_report.tex`, `results/IEEEtran.cls` | **Removidos** | Relatório final pré-auditoria (o mesmo cuja duplicata em `report/final_report.pdf` já havia sido removida na Tarefa 9). Substituído, como fonte de resultados, por `results/audit/audit_report.md` + `README.md` seção 2. |
| `results/ranking.csv`, `results/extracted_metrics.csv`, `results/final_comparison.csv`, `results/validated_metrics.csv`, `results/final_validated_ranking.csv`, `results/final_validation_manifest.json`, `results/ranking_analysis.md`, `results/scientific_discussion.md`, `results/inconsistencies_report.md`, `results/project_inventory.md` | **Removidos** | Toda a cadeia de saída do pipeline "pré-auditoria"/"validação final" legado (`analysis/compare_models.py`, `analysis/metrics_discovery.py`, `analysis/final_validation/*`), hoje redundante com `results/audit/*` (fonte oficial) e com `docs/VALIDATION_REPORT.md` (validação científica atual). O achado que `inconsistencies_report.md` documentava (bug de threshold pré-auditoria) continua registrado em prosa em `ARCHITECTURE_AUDIT.md` e `docs/DATA_PIPELINE.md` — só o CSV/relatório bruto saiu, não a explicação do achado. |
| `results/decision_tree/*` (4 arquivos), `results/visualizations/*.png` (4 arquivos) | **Removidos** | Saída do mesmo pipeline "final_validation" legado (`analysis/final_validation/decision_tree_audit.py`, `analysis/compare_models.py`). `results/visualizations/*.png` era a versão **pré-auditoria** confirmada distinta de `results/audit/visualizations/*.png` (seção 2.1 acima) — a versão pós-auditoria (oficial, mantida) já cobre o mesmo conteúdo. |
| `slides/apresentacao.pdf`, `slides/audit_results_presentation.html` | **Removidos** | Material de apresentação da versão pré-auditoria/pré-refatoração do projeto. Removido a pedido explícito do usuário, ciente de que a seção 2.6 acima os havia mantido por serem potencial entregável acadêmico — decisão consciente de que a entrega atual é o código+README, não a apresentação antiga. |

**Efeito colateral aceito, não corrigido nesta rodada:**
`analysis/final_validation/*.py`, `analysis/compare_models.py` e
`analysis/audit/final_report_pdf.py` deixam de rodar de ponta a ponta,
porque leem como entrada arquivos de `results/` removidos acima
(`ranking.csv`, `validated_metrics.csv` etc.). Isso é aceitável porque
esses três módulos geravam exatamente o relatório/ranking
pré-auditoria e o `final_report.pdf` legado — os artefatos que esta
rodada removeu por serem redundantes. **`analysis/audit/run_audit.py`
não foi afetado** — não lê nenhum dos arquivos removidos, e continua
sendo a fonte executável dos resultados oficiais em `results/audit/`.
Reescrever os três módulos legados para apontar para uma nova fonte
está fora do escopo desta limpeza (seria alterar código de
metodologia/resultado, não remover arquivo) — se o projeto ainda
precisar rodá-los, isso deve ser uma tarefa dedicada.

**Verificação executada:** `py -3.14 -m pytest tests/ -q` →
**190 passed**, nenhuma regressão. `README.md` e `docs/ARCHITECTURE.md`
foram atualizados para não citar mais os arquivos removidos como se
ainda existissem (árvore de diretórios, comando
`python models/naive_bayes/train.py`, citações a
`results/inconsistencies_report.md`/`results/ranking.csv`).
