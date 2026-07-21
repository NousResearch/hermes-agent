---
sidebar_position: 12
title: "Kanban (Board Multi-Agente)"
description: "Board de tarefas durável com SQLite para coordenar múltiplos profiles Hermes"
---

# Kanban — Colaboração Multi-Profile {#kanban--multi-agent-profile-collaboration}

> **Quer um walkthrough?** Leia o [Tutorial Kanban](./kanban-tutorial) — quatro histórias de uso (dev solo, fleet farming, pipeline de papéis com retry, circuit breaker) com screenshots do dashboard de cada uma. Esta página é a referência; o tutorial é a narrativa.

O Hermes Kanban é um board de tarefas durável, compartilhado entre todos os seus profiles Hermes, que deixa múltiplos agentes nomeados colaborarem em trabalho sem swarms frágeis de subagent in-process. Cada tarefa é uma linha em `~/.hermes/kanban.db`; cada handoff é uma linha que qualquer um pode ler e escrever; cada worker é um processo OS completo com identidade própria.

### Duas superfícies: o model fala via ferramentas, você fala via CLI {#two-surfaces-the-model-talks-through-tools-you-talk-through-the-cli}

O board tem duas portas, ambas respaldadas pelo mesmo `~/.hermes/kanban.db`:

- **Agents dirigem o board por um toolset `kanban_*` dedicado** — `kanban_show`, `kanban_list`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`, `kanban_unblock`. O dispatcher spawna cada worker com essas ferramentas já no schema; profiles orchestrator também podem habilitar o toolset `kanban` explicitamente. O model lê e roteia tarefas chamando ferramentas diretamente, *não* fazendo shell out para `hermes kanban`. Veja [Como workers interagem com o board](#how-workers-interact-with-the-board) abaixo.
- **Você (e scripts, e cron) dirigem o board via `hermes kanban …`** na CLI, `/kanban …` como slash command, ou o dashboard. São para humanos e automação — os lugares sem tool-calling de model por trás.

Ambas as superfícies passam pela mesma camada `kanban_db`, então leituras veem vista consistente e escritas não drift. O resto desta página mostra exemplos CLI porque são fáceis de copiar e colar, mas todo verbo CLI tem equivalente em tool-call que o model usa.

Este é o formato que cobre workloads que `delegate_task` não cobre:

- **Research triage** — researchers paralelos + analyst + writer, human-in-the-loop.
- **Scheduled ops** — briefs diários recorrentes que constroem journal ao longo de semanas.
- **Digital twins** — assistentes nomeados persistentes (`inbox-triage`, `ops-review`) que acumulam memória ao longo do tempo.
- **Engineering pipelines** — decompose → implement em worktrees paralelos → review → iterate → PR.
- **Fleet work** — um specialist gerenciando N subjects (50 contas sociais, 12 serviços monitorados).

Para rationale de design completo, análise comparativa contra Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterprise, e os oito padrões canônicos de colaboração, veja `docs/hermes-kanban-v1-spec.pdf` no repositório.

## Kanban vs. `delegate_task` {#kanban-vs-delegate_task}

Parecem similares; não são a mesma primitiva.

| | `delegate_task` | Kanban |
|---|---|---|
| Forma | Chamada RPC (fork → join) | Fila de mensagens durável + máquina de estados |
| Parent | Bloqueia até o filho retornar | Fire-and-forget após `create` |
| Identidade do filho | Subagent anônimo | Profile nomeado com memória persistente |
| Retomabilidade | Nenhuma — falhou = falhou | Block → unblock → re-run; crash → reclaim |
| Human-in-the-loop | Não suportado | Comment / unblock a qualquer momento |
| Agents por tarefa | Uma chamada = um subagent | N agents ao longo da vida da tarefa (retry, review, follow-up) |
| Trilha de auditoria | Perdida na compressão de contexto | Linhas duráveis no SQLite para sempre |
| Coordenação | Hierárquica (caller → callee) | Peer — qualquer profile lê/escreve qualquer tarefa |

**Distinção em uma frase:** `delegate_task` é uma chamada de função; Kanban é uma fila de trabalho onde cada handoff é uma linha que qualquer profile (ou humano) pode ver e editar.

**Use `delegate_task` quando** o agent parent precisa de uma resposta curta de reasoning antes de continuar, sem humanos envolvidos, e o resultado volta para o contexto do parent.

**Use Kanban quando** o trabalho cruza fronteiras de agent, precisa sobreviver a restarts, pode precisar de input humano, pode ser assumido por um papel diferente, ou precisa ser descobível depois.

Coexistem: um worker kanban pode chamar `delegate_task` internamente durante sua run.

## Conceitos centrais {#core-concepts}

- **Board** — fila standalone de tarefas com SQLite DB próprio, diretório
  `workspaces/` e loop de dispatcher. Uma instalação pode ter vários boards
  (ex.: um por projeto, repo ou domínio); veja [Boards (multi-project)](#boards-multi-project)
  abaixo. Usuários de projeto único ficam no board `default` e nunca veem a
  palavra "board" fora desta seção de docs.
- **Task** — linha com title, body opcional, um assignee (nome de profile), status (`triage | todo | ready | running | blocked | done | archived`), tenant namespace opcional, idempotency key opcional (dedup para automação retentada).
- **Link** — linha `task_links` registrando dependência parent → child. O dispatcher promove `todo → ready` quando todos os parents estão `done`.
- **Comment** — protocolo inter-agent. Agents e humanos anexam comments; quando um worker é (re-)spawnado, lê a thread completa de comments como parte do contexto.
- **Workspace** — diretório em que um worker opera. Três tipos:
  - `scratch` (padrão) — tmp dir fresh em `~/.hermes/kanban/workspaces/<id>/` (ou `~/.hermes/kanban/boards/<slug>/workspaces/<id>/` em boards não-default). **Deletado quando a tarefa completa** — scratch é efêmero por design. Arquivos declarados explicitamente via `kanban_complete(artifacts=[...])` são copiados para storage durável de attachments por tarefa antes da limpeza; paths de deliverable em summaries de conclusão legacy recebem o mesmo tratamento. Outros arquivos scratch são removidos. Um artefato scratch declarado ausente mantém a tarefa in-flight para o worker corrigir o path e retentar. Use `worktree:` ou `dir:<path>` quando o workspace inteiro deve permanecer disponível. Na primeira vez que um workspace scratch é criado numa instalação, o dispatcher loga warning e emite evento `tip_scratch_workspace` na tarefa (visível via `hermes kanban show <id>`).
  - `dir:<path>` — diretório compartilhado existente (vault Obsidian, dir de mail ops, pasta por conta). **Deve ser path absoluto.** Paths relativos como `dir:../tenants/foo/` são rejeitados no dispatch porque resolveriam contra o CWD que o dispatcher tiver, o que é ambíguo e vetor de escape confused-deputy. O path é trusted — é sua máquina, seu filesystem, o worker roda com seu uid. Modelo de ameaça trusted-local-user; kanban é single-host por design. **Preservado na conclusão.**
  - `worktree` — git worktree em `.worktrees/<id>/` para tarefas de código. Use `worktree:<path>` para fixar o path alvo exato. `git worktree add` no lado worker cria, usando `--branch` quando fornecido. **Preservado na conclusão.**
- **Dispatcher** — loop long-lived que, a cada N segundos (padrão 60): reclaima claims stale, reclaima workers crashed (PID sumiu mas TTL ainda não expirou), promove tarefas ready, faz claim atômico, spawna profiles atribuídos. Roda **dentro do gateway** por padrão (`kanban.dispatch_in_gateway: true`). Um dispatcher varre todos os boards por tick; workers são spawnados com `HERMES_KANBAN_BOARD` fixado para não verem outros boards. Após `kanban.failure_limit` falhas consecutivas de spawn na mesma tarefa (padrão: 2) o dispatcher auto-block com o último erro como razão — evita thrashing em tarefas cujo profile não existe, workspace não monta, etc.
- **Tenant** — namespace string opcional *dentro* de um board. Uma frota specialist pode servir vários negócios (`--tenant business-a`) com isolamento de dados por workspace path e prefix de memory key. Tenants são filtro soft; boards são o limite hard de isolamento.

## Boards (multi-project) {#boards-multi-project}

Boards separam fluxos de trabalho não relacionados — um por projeto, repo
ou domínio — em filas isoladas. Uma instalação nova tem exatamente um board
chamado `default` (DB em `~/.hermes/kanban.db` por back-compat). Usuários que
só querem um fluxo de trabalho nunca precisam saber de boards; o recurso
é opt-in.

Isolamento por board é absoluto:

- SQLite DB separado por board (`~/.hermes/kanban/boards/<slug>/kanban.db`).
- Diretórios `workspaces/` e `logs/` separados.
- Workers spawnados para uma tarefa veem **apenas** as tarefas do board deles — o
  dispatcher seta `HERMES_KANBAN_BOARD` no env do filho e toda
  ferramenta `kanban_*` que o worker tem acesso lê isso.
- Linkar tarefas entre boards não é permitido (mantém o schema simples; se
  realmente precisar de refs cross-project, use menções em texto livre e busque
  por id manualmente).

### Gerenciando boards pela CLI {#managing-boards-from-the-cli}

```bash
# See what's on disk. Fresh installs show only "default".
hermes kanban boards list

# Create a new board.
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch                   # optional: make it the active board

# Operate on a specific board without switching.
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# Change which board is "current" for subsequent calls.
hermes kanban boards switch atm10-server
hermes kanban boards show             # who's active right now?

# Rename the display name (the slug is immutable — it's the directory name).
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# Archive (default) — moves the board's dir to boards/_archived/<slug>-<ts>/.
# Recoverable by moving the dir back.
hermes kanban boards rm atm10-server

# Hard delete — `rm -rf` the board dir. No recovery.
hermes kanban boards rm atm10-server --delete
```

Ordem de resolução de board (maior precedência primeiro):

1. `--board <slug>` explícito na chamada CLI.
2. Env var `HERMES_KANBAN_BOARD` (setada pelo dispatcher ao spawnar um
   worker, para workers não verem outros boards).
3. `~/.hermes/kanban/current` — slug persistido por `hermes kanban
   boards switch`.
4. `default`.

Slugs são validados: alfanuméricos minúsculos + hífens + underscores, 1-64
chars, devem começar com alfanumérico. Input maiúsculo é auto-downcased.
Qualquer outra coisa (slashes, espaços, dots, `..`) é rejeitada na camada CLI
para path-traversal não poder nomear um board.

### Gerenciando boards pelo dashboard {#managing-boards-from-the-dashboard}

`hermes dashboard` → aba Kanban mostra um board switcher no topo assim que
existe mais de um board (ou qualquer board tem tarefas). Usuários de board único
veem só um botão pequeno `+ New board`; o switcher fica oculto até
importar.

- **Board dropdown** — escolha o board ativo. Sua seleção é salva no
  `localStorage` do browser para persistir entre reloads sem
  mudar o ponteiro `current` da CLI debaixo de um terminal que você deixou
  aberto.
- **+ New board** — abre modal pedindo slug, display name,
  description e icon. Opção de auto-switch para o novo board.
- **Settings** — abre modal para editar display name,
  description e **project directory** (`default_workdir`) do board atual. O
  project directory é o default de workspace em nível de board que toda tarefa nova
  herda (git repo → worktree preservado, dir plain → diretório
  preservado); cada tarefa ainda pode override na criação. Limpar
  o campo reverte tarefas novas para workspaces scratch descartáveis.
- **Archive** — só em boards não-`default`. Confirma, depois move
  o dir do board para `boards/_archived/`.

Todos os endpoints de API do dashboard aceitam `?board=<slug>` para escopo de board. O
WebSocket de events é fixado a um board na conexão; trocar na
UI abre WS fresh contra o novo board.


## Anexos de arquivo {#file-attachments}

Tarefas podem carregar anexos de arquivo — PDFs, imagens, documentos fonte — para um
worker ter o material fonte sem você colar paths no
body e torcer para ele achar.

- **Upload** — abra uma tarefa no drawer do dashboard e use o botão *Upload file*
  da seção **Attachments** (vários arquivos de uma vez
  são ok). Cada upload tem teto de 25 MB.
- **Storage** — arquivos vão para
  `<hermes-home>/kanban/attachments/<task_id>/` no board default, ou
  `<hermes-home>/kanban/boards/<slug>/attachments/<task_id>/` num board
  nomeado. Set `HERMES_KANBAN_ATTACHMENTS_ROOT` para fixar local customizado.
- **O que o worker vê** — quando o dispatcher entrega uma tarefa a um worker,
  o contexto do worker inclui seção **Attachments** listando nome de cada
  arquivo e seu **path absoluto**. O worker tem acesso completo a file/terminal
  tools, então lê attachments diretamente (`read_file`, ou shell
  tools como `pdftotext`).
- **Download / remove** — o drawer lista cada attachment com link de download
  e controle remove (×). Remover um attachment deleta a linha de metadata
  e o arquivo no disco.

:::note Backends de terminal remotos
Paths de attachment resolvem diretamente no backend de terminal **local**, que
é o padrão para workers Kanban. Se você roda workers num backend remoto
(Docker, Modal), monte o diretório `attachments/` do board no
sandbox para os paths absolutos no contexto do worker serem alcançáveis.
:::


## Início rápido {#quick-start}

Os comandos abaixo são **você** (o humano) configurando o board e criando tarefas. Quando uma tarefa é atribuída, o dispatcher spawna o profile atribuído como worker, e daí **o model dirige a tarefa por tool calls `kanban_*`, não comandos CLI** — veja [Como workers interagem com o board](#how-workers-interact-with-the-board).


```bash
# 1. Create the board (you)
hermes kanban init

# 2. Start the gateway (hosts the embedded dispatcher)
hermes gateway start

# 3. Create a task (you — or an orchestrator agent via kanban_create)
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. Watch activity live (you)
hermes kanban watch

# 5. See the board (you)
hermes kanban list
hermes kanban stats
```

Quando o dispatcher pega `t_abcd` e spawna o profile `researcher`, a primeira coisa que o model desse worker faz é chamar `kanban_show()` para ler sua tarefa. Ele não roda `hermes kanban show t_abcd`.

### Dispatcher embedded no gateway (padrão) {#gateway-embedded-dispatcher-default}

O dispatcher roda dentro do processo gateway. Nada para instalar, nenhum
serviço separado para gerenciar — se o gateway está up, tarefas ready são pegas
no próximo tick (60s por padrão).

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true        # default
  dispatch_interval_seconds: 60    # default
```

Override da flag de config em runtime via `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0`
para debug. Supervisão padrão de gateway se aplica: rode `hermes gateway
start` diretamente, ou configure o gateway como systemd user unit (veja docs do
gateway). Sem gateway rodando, tarefas `ready` ficam onde estão
até um subir — `hermes kanban create` avisa disso na criação.

Rodar `hermes kanban daemon` como processo separado é **deprecated**;
use o gateway. Se realmente não puder rodar o gateway (host headless cuja
política proíbe serviços long-lived, etc.) um escape hatch `--force` mantém
o daemon standalone antigo vivo por um ciclo de release, mas rodar dispatcher
embedded no gateway E daemon standalone contra o mesmo
`kanban.db` causa claim races e não é suportado.

### Create idempotente (para automação / webhooks) {#idempotent-create-for-automation--webhooks}

```bash
# First call creates the task. Any subsequent call with the same key
# returns the existing task id instead of duplicating.
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

### Verbos CLI em bulk {#bulk-cli-verbs}

Todos os verbos de lifecycle aceitam vários ids para limpar um batch
num comando:

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

:::note Onde uma tarefa desbloqueada cai
`unblock` em si só move uma tarefa para **`ready`** (todos os parents `done`) ou
**`todo`** (um parent ainda aberto — a tarefa está dependency-gated e o
dispatcher auto-promove quando o parent termina). Nunca roteia para
`triage`.

Se você desbloqueia uma tarefa e ela depois aparece em **`triage`**, o unblock não
foi o que a colocou lá. Um *re-block subsequente pela mesma razão* foi: depois que uma tarefa
é blocked → unblocked → re-blocked pela mesma causa `BLOCK_RECURRENCE_LIMIT`
vezes (padrão `2`), o breaker de loop unblock para de mandá-la de volta para `blocked`
— onde um cron só desbloquearia de novo — e roteia para `triage` para
decisão humana. Guarda determinística de DB, não julgamento LLM, e
o body da tarefa não pode opt out: o contador de recorrência deliberadamente
sobrevive a cada unblock (reseta só num `complete` bem-sucedido). Para manter uma
tarefa desbloqueada no pool de trabalho, resolva *por que ela continua re-blocking* (parent
inacabado, input faltando, capability não atendida) antes de desbloquear, ou suba
`BLOCK_RECURRENCE_LIMIT` se o loop for esperado.
:::

## Como workers interagem com o board {#how-workers-interact-with-the-board}

**Workers não fazem shell out para `hermes kanban`.** Quando o dispatcher spawna um worker, seta `HERMES_KANBAN_TASK=t_abcd` no env do filho, e essa env var liga um **toolset kanban** dedicado no schema do model. O mesmo toolset também está disponível a profiles orchestrator que habilitam `kanban` na config de toolsets. Essas ferramentas leem e mutam o board diretamente via camada Python `kanban_db`, igual à CLI. Um worker em execução chama essas como qualquer outra tool; nunca vê nem precisa da CLI `hermes kanban`.

| Tool | Propósito | Params obrigatórios |
|---|---|---|
| `kanban_show` | Lê a tarefa atual (title, body, tentativas anteriores, handoffs de parent, comments, `worker_context` pré-formatado completo). Default para o task id do env. | — |
| `kanban_list` | Lista summaries de tarefas com filtros de `assignee`, `status`, `tenant`, visibilidade archived e limit. Destinado a orchestrators descobrindo trabalho no board. | — |
| `kanban_complete` | Termina com handoff estruturado `summary` + `metadata`. | pelo menos um de `summary` / `result` |
| `kanban_block` | Para trabalho e roteia pelo porquê: `kind=dependency` (espera em `todo`, auto-resume), `needs_input`/`capability`/`transient` (superfície para humano). Re-blocks repetidos do mesmo kind auto-escalam para `triage`. | `reason` |
| `kanban_heartbeat` | Sinaliza vivacidade durante operações longas. Side-effect puro. | — |
| `kanban_comment` | Anexa nota durável à thread da tarefa. | `task_id`, `body` |
| `kanban_create` | (Orchestrators) fan-out em tarefas filhas com `assignee`, `parents`, `skills` opcionais, etc. | `title`, `assignee` |
| `kanban_link` | (Orchestrators) adiciona aresta de dependência `parent_id → child_id` depois do fato. | `parent_id`, `child_id` |
| `kanban_unblock` | (Orchestrators) move tarefa blocked para `ready` quando todos os parents estão done, ou `todo` enquanto algum parent permanece aberto. | `task_id` |

Um turn típico de worker parece:

```
# Model's tool calls, in order:
kanban_show()                                     # no args — uses HERMES_KANBAN_TASK
# (model reads the returned worker_context, does the work via terminal/file tools)
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# (more work)
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

Um worker **orchestrator** faz fan-out em vez disso:

```
kanban_show()
kanban_create(
    title="research ICP funding 2024-2026",
    assignee="researcher-a",
    body="focus on seed + series A, North America, AI-adjacent",
)
# → returns {"task_id": "t_r1", ...}
kanban_create(title="research ICP funding — EU angle", assignee="researcher-b", body="…")
# → returns {"task_id": "t_r2", ...}
kanban_create(
    title="synthesize findings into launch brief",
    assignee="writer",
    parents=["t_r1", "t_r2"],                     # promotes to ready when both complete
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

As ferramentas "(Orchestrators)" — `kanban_list`, `kanban_create`, `kanban_link`, `kanban_unblock`, e `kanban_comment` em tarefas estrangeiras — estão disponíveis pelo mesmo toolset; a convenção (codificada na orientação kanban auto-injetada) é que profiles worker não fazem fan-out nem roteiam trabalho não relacionado, e profiles orchestrator não executam trabalho de implementação. Workers spawnados pelo dispatcher ainda são task-scoped para operações de lifecycle destrutivas e não podem mutar tarefas não relacionadas.

### Por que ferramentas em vez de shell para `hermes kanban` {#why-tools-instead-of-shelling-to-hermes-kanban}

Três razões:

1. **Portabilidade de backend.** Workers cujo terminal tool aponta para backend remoto (Docker / Modal / Singularity / SSH) rodariam `hermes kanban complete` *dentro* do container, onde `hermes` não está instalado e `~/.hermes/kanban.db` não está montado. As ferramentas kanban rodam no processo Python do agent e sempre alcançam `~/.hermes/kanban.db` independente do backend de terminal.
2. **Sem fragilidade de shell-quoting.** Passar `--metadata '{"files": [...]}'` por shlex + argparse é footgun latente. Args estruturados de tool evitam isso.
3. **Erros melhores.** Resultados de tool são JSON estruturado que o model pode raciocinar, não strings stderr para parsear.

**Zero footprint de schema em sessões normais.** Uma sessão regular `hermes chat` tem zero ferramentas `kanban_*` no schema a menos que o profile ativo habilite explicitamente o toolset `kanban` para trabalho orchestrator. Workers de tarefa spawnados pelo dispatcher recebem ferramentas task-scoped porque `HERMES_KANBAN_TASK` está setado; profiles orchestrator recebem a superfície de roteamento mais ampla via config. Sem tool bloat para usuários que nunca tocam kanban.

A orientação kanban auto-injetada ensina o model qual ferramenta chamar quando e em que ordem.

### Evidência de handoff recomendada {#recommended-handoff-evidence}

`kanban_complete(summary=..., metadata={...})` é intencionalmente flexível:
o summary é o closeout legível por humanos, e `metadata` é o
handoff legível por máquina que agents downstream, reviewers ou dashboards podem
reusar sem raspar prose.

Para tarefas de engenharia e review, prefira esta forma opcional de metadata:

```json
{
  "changed_files": ["path/to/file.py"],
  "verification": ["pytest tests/hermes_cli/test_kanban_db.py -q"],
  "dependencies": ["parent task id or external issue, if any"],
  "blocked_reason": null,
  "retry_notes": "what failed before, if this was a retry",
  "residual_risk": ["what was not tested or still needs human review"]
}
```

Essas keys são convenção, não requisito de schema. A propriedade útil é
que todo worker deixa evidência suficiente para o próximo leitor responder quatro
perguntas rapidamente:

1. O que mudou?
2. Como foi verificado?
3. O que pode desbloquear ou retentar se falhar?
4. Que risco ainda fica deliberadamente aberto?

Mantenha secrets, logs raw, tokens, material OAuth e transcripts não relacionados fora de
`metadata`. Armazene ponteiros e summaries. Se uma tarefa não tem arquivos ou
tests, diga explicitamente no `summary` e use `metadata` para a evidência que
existir, como URLs fonte, ids de issue ou passos de review manual.

### O ciclo de vida do worker {#the-worker-lifecycle}

Todo profile que trabalha tarefas kanban recebe automaticamente o ciclo de vida worker — é injetado no system prompt do worker no spawn (bloco `KANBAN_GUIDANCE`), então **nada para instalar ou configurar**. Ensina o worker o ciclo completo em **tool calls**, não comandos CLI:

1. No spawn, chame `kanban_show()` para ler title + body + handoffs de parent + tentativas anteriores + thread completa de comments.
2. `cd $HERMES_KANBAN_WORKSPACE` (via terminal tool) e faça o trabalho ali.
3. Chame `kanban_heartbeat(note="...")` a cada poucos minutos durante operações longas. **Se seu trabalho pode rodar mais de 1 hora, chame `kanban_heartbeat` pelo menos uma vez por hora** — o dispatcher reclaima tarefas que rodaram além de `kanban.dispatch_stale_timeout_seconds` (padrão 4 h) sem heartbeat na última hora, assumindo que o worker crashed sem cleanup. Um reclaim é benigno (a tarefa volta para `ready` para re-dispatch sem tick no contador de falha) mas você perde o progresso da run atual.
4. Complete com `kanban_complete(summary="...", metadata={...})`, ou `kanban_block(reason="...")` se travado.

Essa chamada final `kanban_complete` / `kanban_block` faz parte do protocolo worker.
Se o processo worker sai com status 0 enquanto a tarefa ainda está
`running`, o dispatcher trata como violação de protocolo e emite evento
`protocol_violation`.

**Prevenção no lado agent:** Antes do worker sair, Hermes injeta até dois
nudges sintéticos quando detecta que o model está prestes a parar sem chamada terminal
de board tool. Isso pega o caso comum em que o model narra o próximo
passo ("Let me write the report") e para com `finish_reason=stop`. O nudge
lembra o model de chamar `kanban_complete` ou `kanban_block` imediatamente. Essa
guarda só está ativa para workers spawnados pelo dispatcher (`HERMES_KANBAN_TASK`
setado) e pode ser desabilitada com `HERMES_KANBAN_STOP_NUDGE=0`.

**Recuperação no lado dispatcher:** Se os nudges se esgotam ou o worker crash
antes de alcançar o nudge, o dispatcher dá à violação um **retry limitado**
(até `_PROTOCOL_VIOLATION_FAILURE_LIMIT` violações consecutivas, padrão 3)
antes de auto-block a tarefa em vez de respawná-la no mesmo loop. O
orçamento conta só violações consecutivas de protocolo de saída limpa — requeues
intercalados rate-limited são neutros, e qualquer outro tipo de falha reseta a
sequência — e um `max_retries` por tarefa override o limite. Isso geralmente significa
que o model escreveu resposta plain-text e saiu sem usar a superfície de tool
Kanban.

O ciclo de vida mais os detalhes de referência load-bearing (tipos de workspace, `artifacts` deliverable, claiming de cards criados) vêm nesse bloco de system prompt, então todo worker os tem independente do profile — sem setup de skill por profile.

### Fixando skills extras numa tarefa específica {#pinning-extra-skills-to-a-specific-task}

Às vezes uma tarefa precisa de contexto specialist que o profile assignee não carrega por padrão — job de tradução que precisa da skill `translation`, tarefa de review que precisa `github-code-review`, audit de segurança que precisa `security-pr-audit`. Em vez de editar o profile assignee toda vez, anexe as skills diretamente à tarefa.

**De um agent orchestrator** (caso usual — um agent roteando trabalho para outro), use o array `skills` da tool `kanban_create`:

```
kanban_create(
    title="translate README to Japanese",
    assignee="linguist",
    skills=["translation"],
)

kanban_create(
    title="audit auth flow",
    assignee="reviewer",
    skills=["security-pr-audit", "github-code-review"],
)
```

**De um humano (CLI / slash command)**, repita `--skill` para cada uma:

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

**Pelo dashboard**, digite as skills separadas por vírgula no campo **skills** do diálogo create-task.

O dispatcher emite uma flag `--skills <name>` por skill listada, então o worker spawna com todas carregadas além da orientação kanban auto-injetada. Os nomes de skill devem corresponder a skills realmente instaladas no profile assignee (rode `hermes skills list` para ver o disponível); não há install em runtime.

### Cards goal-mode (`--goal`) {#goal-mode-cards-goal}

Por padrão cada worker tem **uma chance** no card — faz o trabalho, chama `kanban_complete`/`kanban_block`, sai. Passe `--goal` (CLI) ou `goal_mode=True` (tool `kanban_create` / dashboard) para rodar esse worker num **goal loop**, o mesmo engine estilo Ralph por trás do slash command `/goal`: após cada turn um judge auxiliar checa a saída do worker contra title + body do card (tratados como critérios de aceitação), e se o trabalho não está done — e o orçamento de turns resta — o worker continua **na mesma sessão** até o judge concordar, o worker terminar a tarefa, ou o orçamento acabar (o que **block** o card para review humana em vez de sair silenciosamente).

```bash
hermes kanban create "Translate the docs site to French" \
    --body "Acceptance: every page translated, no English left, links intact." \
    --assignee linguist \
    --goal \
    --goal-max-turns 15      # optional; default 20
```

Use para cards open-ended, multi-step ou "continue até X ser verdade". Pule para trabalho one-shot barato — o overhead do judge por turn não vale, e o retry/circuit-breaker existente do dispatcher já trata falhas transientes de worker. O judge só é tão bom quanto seu texto de goal, então escreva o body como **critérios de aceitação explícitos**.

### Como o orchestrator se comporta {#how-the-orchestrator-behaves}

Um **orchestrator bem-comportado não faz o trabalho.** Decompõe o goal do usuário em tarefas, linka, atribui cada uma a um dos profiles que você configurou, e recua. A orientação orchestrator — regras anti-tentação, prompt Step-0 de descoberta de profiles (o dispatcher falha silenciosamente em nomes assignee desconhecidos, então o orchestrator deve fundamentar todo card em profiles que realmente existem na sua máquina), e playbook de decomposição keyed em `kanban_create` / `kanban_link` / `kanban_comment` — é injetada no system prompt do worker automaticamente; nada para instalar.

Um turn canônico de orchestrator (dois researchers paralelos handoff para writer):

```
# Goal from user: "draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # promoted to 'ready' when both researchers complete
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# Optional: add cross-cutting deps discovered later without re-creating tasks
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

A orientação orchestrator vem no system prompt do worker automaticamente — nada para instalar ou sincronizar por profile.

Para melhores resultados, combine com profile cujos toolsets são restritos a operações de board (`kanban`, `gateway`, `memory`) para o orchestrator literalmente não poder executar tarefas de implementação mesmo se tentar.

## Dashboard (GUI) {#dashboard-gui}

A CLI `/kanban` e o slash command bastam para rodar o board headless, mas um board visual costuma ser a interface certa para human-in-the-loop: triage, supervisão cross-profile, ler threads de comments e arrastar cards entre colunas. Hermes shipa isso como **plugin de dashboard bundled** em `plugins/kanban/` — não feature core, não serviço separado — seguindo o modelo em [Estendendo o Dashboard](./extending-the-dashboard).

Abra com:

```bash
hermes kanban init      # one-time: create kanban.db if not already present
hermes dashboard        # "Kanban" tab appears in the nav, after "Skills"
```

### O que o plugin oferece {#what-the-plugin-gives-you}

- Aba **Kanban** com uma coluna por status: `triage`, `todo`, `ready`, `running`, `blocked`, `done` (mais `archived` quando o toggle está ligado).
  - `triage` é a coluna de estacionamento para ideias brutas. Por padrão (`kanban.auto_decompose: true`), o dispatcher auto-roda o **decomposer** em tarefas que caem aqui. O decomposer built-in usa o path de model `auxiliary.kanban_decomposer`, lê seu roster de profiles (com descriptions) e faz fan-out da tarefa num pequeno grafo de tarefas filhas roteadas aos specialists best-fit. A tarefa original permanece viva como parent de todo filho para seu assignee (`kanban.orchestrator_profile`, ou o profile default ativo quando unset) acordar de volta para julgar conclusão quando tudo terminar. Alterne a pill **Orchestration: Auto/Manual** no topo da página (emerald = Auto, muted gray = Manual), ou editando `config.yaml` diretamente. Ambos os modos coexistem com `hermes kanban specify` — ainda disponível como rewrite de spec de tarefa única quando você não quer fan-out.
- Cards mostram task id, title, badge de priority, tag tenant, profile atribuído, contagens de comment/link, **progress pill** (`N/M` filhos done quando a tarefa tem dependentes) e "created N ago". Checkbox por card habilita multi-select.
- **Lanes por profile dentro de Running** — checkbox na toolbar alterna sub-agrupamento da coluna Running por assignee.
- **Updates live via WebSocket** — o plugin taila a tabela append-only `task_events` num intervalo curto de poll; o board reflete mudanças no instante em que qualquer profile (CLI, gateway ou outra aba do dashboard) age. Reloads são debounced para burst de events disparar um único refetch.
- **Drag-drop** de cards entre colunas para mudar status. O drop envia `PATCH /api/plugins/kanban/tasks/:id` que roteia pela mesma code `kanban_db` que a CLI usa — as três superfícies nunca drift. Moves para statuses destrutivos (`done`, `archived`, `blocked`) pedem confirmação. Dispositivos touch usam fallback pointer-based para o board ser usável num tablet.
- **Diálogo create-task** — clique `+` em qualquer header de coluna para abrir modal com campos rotulados: title, assignee, priority, skills, kind/path de workspace (seeded do project directory do board; override por tarefa), goal mode e (opcionalmente) tarefa parent num dropdown sobre toda tarefa existente. Enter cria a tarefa, Shift+Enter insere newline no campo title, Escape cancela. Criar da coluna Triage estaciona automaticamente a nova tarefa em triage.
- **Multi-select com bulk actions** — shift/ctrl-click num card ou marque checkbox para adicionar à seleção. Barra de bulk action no topo com transições de status em batch, archive e reassign (por dropdown de profile, ou "(unassign)"). Batches destrutivos confirmam primeiro. Falhas parciais por id são reportadas sem abortar o resto.
- **Clique num card** (sem shift/ctrl) abre drawer lateral (Escape ou click-outside fecha) com:
  - **Title editável** — clique no heading para renomear.
  - **Assignee / priority editáveis** — clique na meta row para reescrever.
  - **Description editável** — markdown-rendered por padrão (headings, bold, italic, inline code, fenced code, links `http(s)` / `mailto:`, bullet lists), com botão "edit" que troca por textarea. Renderização markdown é renderer minúsculo e XSS-safe — toda substituição roda em input HTML-escaped, só links `http(s)` / `mailto:` passam, e `target="_blank"` + `rel="noopener noreferrer"` sempre setados.
  - **Editor de dependências** — chip list de parents e children, cada um com `×` para unlink, mais dropdowns sobre toda outra tarefa para adicionar parent ou child. Tentativas de ciclo rejeitadas server-side com mensagem clara.
  - **Linha de ações de status** (→ triage / → ready / → running / block / unblock / complete / archive) com prompts de confirmação para transições destrutivas. Para cards na coluna **Triage** a linha também expõe duas ações LLM-driven: **⚗ Decompose** faz fan-out da tarefa num grafo de tarefas filhas roteadas a profiles specialist por description, e **✨ Specify** faz rewrite de spec de tarefa única. Decompose cai para promoção estilo specify quando o LLM decide que a tarefa não se beneficia de fan-out, então é superset estrito. Ambas alcançáveis pela CLI (`hermes kanban decompose <id>` / `specify <id>` / `--all`), de qualquer plataforma gateway (`/kanban decompose <id>`), e programaticamente via `POST /api/plugins/kanban/tasks/:id/decompose` e `…/specify`. Configure os models sob `auxiliary.kanban_decomposer` e `auxiliary.triage_specifier` em `config.yaml`.
  - Seção Result (também markdown-rendered), thread de comments com Enter-to-submit, os últimos 20 events.
- **Filtros da toolbar** — busca free-text, dropdown tenant (default `dashboard.kanban.default_tenant` de `config.yaml`), dropdown assignee, toggle "show archived", toggle "lanes by profile", e botão **Nudge dispatcher** para não esperar o próximo tick de 60 s.

Visualmente o alvo é o layout familiar Linear / Fusion: dark theme, headers de coluna com contagens, status dots coloridos, pill chips para priority e tenant. O plugin lê só CSS vars de theme (`--color-*`, `--radius`, `--font-mono`, ...), então reskin automaticamente com qualquer theme de dashboard ativo.

### Orquestração Auto vs Manual {#auto-vs-manual-orchestration}

O board kanban tem duas formas de tratar uma tarefa que você joga na coluna Triage:

**Auto (padrão)** — `kanban.auto_decompose: true`. O dispatcher embedded no gateway roda o **decomposer** a cada tick, limitado por `kanban.auto_decompose_per_tick` (padrão 3 tarefas por tick) para bulk-load de tarefas triage não estourar gasto do LLM auxiliar. O decomposer usa o prompt de decomposição built-in mais o path de model `auxiliary.kanban_decomposer`, lê seus profiles instalados + descriptions, e pede ao LLM um grafo JSON de tarefas: quais spawnar, para quem vão, e quais dependem de quais. A tarefa triage original vira parent de toda folha no grafo, então permanece viva até o grafo inteiro completar — e depois promove de volta para `ready` para seu assignee (`kanban.orchestrator_profile`, ou o profile default ativo quando unset) julgar conclusão e adicionar mais tarefas se o trabalho não estiver done. Este é o fluxo "drop a one-liner, walk away".

**Manual** — `kanban.auto_decompose: false`. Tarefas triage ficam em triage até você agir. Clique **⚗ Decompose** num card, rode `hermes kanban decompose <id>` (ou `--all`), ou use `/kanban decompose <id>` de um chat. Corresponde ao comportamento pré-decomposer do board, útil quando você quer controle total sobre o que roda quando.

Alterne entre os dois modos pela pill **Orchestration: Auto/Manual** no topo da página kanban (emerald = Auto, muted gray = Manual), ou editando `config.yaml` diretamente. Ambos coexistem com `hermes kanban specify` — ainda disponível como rewrite de spec de tarefa única quando você não quer fan-out.

As decisões de roteamento do decomposer dependem de profile descriptions, primitivo de labeling por profile que você seta com `hermes profile create --description "..."`, `hermes profile describe <name> --text "..."`, `hermes profile describe <name> --auto` (LLM gera a partir das skills instaladas + model do profile), ou o editor por profile do dashboard no painel expandido **Orchestration settings**. Profiles sem description ainda aparecem no roster — routable por nome, só menos precisamente. O decomposer NUNCA deixa tarefa filha com `assignee=None`: quando o LLM escolhe profile desconhecido, a filha vai para `kanban.default_assignee` (ou o profile default ativo se unset).

`kanban.orchestrator_profile` não carrega prompt, skills ou lógica customizada desse profile na chamada de decomposição. Controla quem possui a tarefa root/orchestration após fan-out. Para mudar model/provider do decomposer, configure `auxiliary.kanban_decomposer`. Para usar lógica customizada de split de tarefas de um profile em vez do decomposer built-in, mude para Manual mode e faça esse profile criar ou decompor tarefas explicitamente.

Knobs de config (todos sob `kanban:` em `~/.hermes/config.yaml`):

| Key | Padrão | Propósito |
|---|---|---|
| `auto_decompose` | `true` | Dispatcher auto-roda o decomposer a cada tick. |
| `auto_decompose_per_tick` | `3` | Teto de decomposições por tick do dispatcher. Excesso adia para o próximo tick. |
| `orchestrator_profile` | `""` | Profile atribuído à tarefa root/orchestration após decomposição. Vazio = fallback para profile default ativo. |
| `default_assignee` | `""` | Onde uma tarefa filha cai quando o LLM escolhe profile desconhecido. Vazio = fallback para default ativo. |
| `auto_subscribe_on_create` | `true` | Quando um worker chama `kanban_create` de dentro de sessão com canal de delivery persistente (messaging gateway ou TUI), a sessão originadora é auto-inscrita nos eventos completion/block da nova tarefa. O dispatcher ainda dirige a delivery — isso só muda se o chat/key do caller aparece na tabela notify-sub. Set `false` para exigir chamadas explícitas `kanban_notify-subscribe` por tarefa. |

E os dois slots LLM auxiliares:

| Key | Propósito |
|---|---|
| `auxiliary.kanban_decomposer` | Model que produz o grafo de tarefas (chamado por Decompose). Set `provider`/`model` para override do model de chat principal. |
| `auxiliary.profile_describer` | Model que auto-gera profile descriptions (chamado por `hermes profile describe --auto`). |

### Arquitetura {#architecture}

A GUI é estritamente camada **read-through-the-DB + write-through-kanban_db** sem lógica de domínio própria:

<!-- ascii-guard-ignore -->
```
┌────────────────────────┐      WebSocket (tails task_events)
│   React SPA (plugin)   │ ◀──────────────────────────────────┐
│   HTML5 drag-and-drop  │                                    │
└──────────┬─────────────┘                                    │
           │ REST over fetchJSON                              │
           ▼                                                  │
┌────────────────────────┐     writes call kanban_db.*        │
│  FastAPI router        │     directly — same code path      │
│  plugins/kanban/       │     the CLI /kanban verbs use      │
│  dashboard/plugin_api.py                                    │
└──────────┬─────────────┘                                    │
           │                                                  │
           ▼                                                  │
┌────────────────────────┐                                    │
│  ~/.hermes/kanban.db   │ ───── append task_events ──────────┘
│  (WAL, shared)         │
└────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

### Superfície REST {#rest-surface}

Todas as rotas são montadas sob `/api/plugins/kanban/` e protegidas pelo session token efêmero do dashboard:

| Method | Path | Propósito |
|---|---|---|
| `GET` | `/board?tenant=<name>&include_archived=…` | Board completo agrupado por coluna de status, mais tenants + assignees para dropdowns de filtro |
| `GET` | `/tasks/:id` | Task + comments + events + links |
| `POST` | `/tasks` | Create (wraps `kanban_db.create_task`, aceita `triage: bool` e `parents: [id, …]`) |
| `PATCH` | `/tasks/:id` | Status / assignee / priority / title / body / result |
| `POST` | `/tasks/bulk` | Aplica o mesmo patch (status / archive / assignee / priority) a todo id em `ids`. Falhas por id reportadas sem abortar irmãos |
| `POST` | `/tasks/:id/comments` | Anexa comment |
| `POST` | `/tasks/:id/specify` | Roda o triage specifier — LLM auxiliar flesha o body da tarefa e promove de `triage` para `todo`. Retorna `{ok, task_id, reason, new_title}`; `ok=false` com razão legível em "not in triage" / sem aux client / erro LLM é 200, não 4xx |
| `POST` | `/tasks/:id/decompose` | Roda o kanban decomposer — LLM auxiliar produz grafo de tarefas e o helper cria atomicamente os filhos + linka a root + flip `triage → todo`. Retorna `{ok, task_id, reason, fanout, child_ids, new_title}`. Mesma convenção 200-on-LLM-error que `/specify`. |
| `GET` | `/profiles` | Lista profiles instalados com descriptions (consumido pelo editor de profile-description do dashboard e pelo picker orchestrator). |
| `PATCH` | `/profiles/:name` | Set ou clear description de profile (autorado pelo usuário — `description_auto: false`). Retorna `{ok, profile, description}`. |
| `POST` | `/profiles/:name/describe-auto` | Gera description para profile via `auxiliary.profile_describer`. Persiste com `description_auto: true` para o dashboard mostrar badge "review". |
| `GET` | `/orchestration` | Lê settings de orquestração kanban (`orchestrator_profile`, `default_assignee`, `auto_decompose`) mais valores efetivos *resolved* após fallbacks. |
| `PUT` | `/orchestration` | Atualiza uma ou mais das três keys de orquestração em `config.yaml`. Valida que nomes de profile não vazios existem de fato. |
| `POST` | `/links` | Adiciona dependência (`parent_id` → `child_id`) |
| `DELETE` | `/links?parent_id=…&child_id=…` | Remove dependência |
| `POST` | `/dispatch?max=…&dry_run=…` | Nudge no dispatcher — pula a espera de 60 s |
| `GET` | `/config` | Lê preferências `dashboard.kanban` de `config.yaml` — `default_tenant`, `lane_by_profile`, `include_archived_by_default`, `render_markdown` |
| `WS` | `/events?since=<event_id>` | Stream live de linhas `task_events` |

Todo handler é wrapper fino — o plugin tem ~700 linhas de Python (router + WebSocket tail + bulk batcher + config reader) e não adiciona lógica de negócio nova. Um helper `_conn()` minúsculo auto-inicializa `kanban.db` em todo read e write, então instalação fresh funciona se o usuário abriu o dashboard primeiro, bateu na REST API diretamente, ou rodou `hermes kanban init`.

### Config do dashboard {#dashboard-config}

Qualquer dessas keys sob `dashboard.kanban` em `~/.hermes/config.yaml` muda os defaults da aba — o plugin lê no load time via `GET /config`:

```yaml
dashboard:
  kanban:
    default_tenant: acme              # preselects the tenant filter
    lane_by_profile: true             # default for the "lanes by profile" toggle
    include_archived_by_default: false
    render_markdown: true             # set false for plain <pre> rendering
```

Cada key é opcional e faz fallback para o default mostrado.

### Modelo de segurança {#security-model}

O middleware de auth HTTP do dashboard [pula explicitamente `/api/plugins/`](./extending-the-dashboard#backend-api-routes) — rotas de plugin são unauthenticated por design porque o dashboard faz bind em localhost por padrão. Isso significa que a superfície REST kanban é alcançável por qualquer processo no host.

O WebSocket dá um passo adicional: exige o session token efêmero do dashboard como query param `?token=…` (browsers não podem setar `Authorization` num upgrade request), seguindo o padrão usado pela PTY bridge in-browser.

Se você roda `hermes dashboard --host 0.0.0.0`, toda rota de plugin — kanban incluído — fica alcançável pela rede. **Não faça isso num host compartilhado.** O board contém bodies de tarefas, comments e workspace paths; um atacante alcançando essas rotas obtém read access à sua superfície de colaboração inteira e também pode create / reassign / archive tarefas.

Tarefas em `~/.hermes/kanban.db` são profile-agnostic de propósito (esse é o primitivo de coordenação). Se você abre o dashboard com `hermes -p <profile> dashboard`, o board ainda mostra tarefas criadas por qualquer outro profile no host. O mesmo usuário possui todos os profiles, mas vale saber se várias personas coexistem.

### Updates live {#live-updates}

`task_events` é tabela SQLite append-only com `id` monotônico. O endpoint WebSocket guarda o último event id visto de cada client e empurra linhas novas conforme chegam. Quando burst de events chega, o frontend recarrega o endpoint de board (muito barato) — mais simples e correto que tentar patch local state de todo kind de event. Modo WAL significa que o loop de read nunca bloqueia as transações `BEGIN IMMEDIATE` de claim do dispatcher.

### Estendendo {#extending-it}

O plugin usa o contrato padrão de plugin de dashboard Hermes — veja [Estendendo o Dashboard](./extending-the-dashboard) para referência completa de manifest, shell slots, page-scoped slots e Plugin SDK. Colunas extras, chrome customizado de card, layouts filtrados por tenant, ou replacements completos `tab.override` são expressíveis sem fork deste plugin.

Para desabilitar sem remover: adicione `dashboard.plugins.kanban.enabled: false` em `config.yaml` (ou delete `plugins/kanban/dashboard/manifest.json`).

### Limite de escopo {#scope-boundary}

A GUI é deliberadamente fina. Tudo que o plugin faz é alcançável pela CLI; o plugin só torna confortável para humanos. Auto-assignment, budgets, governance gates e views org-chart permanecem user-space — profile router, outro plugin, ou reuse de `tools/approval.py` — exatamente como listado na seção out-of-scope da spec de design.

## Referência de comandos CLI {#cli-command-reference}

Esta é a superfície que **você** (ou scripts, cron, o dashboard) usa para dirigir o board. Workers rodando dentro do dispatcher usam a [superfície de tool](#how-workers-interact-with-the-board) `kanban_*` para as mesmas operações — a CLI aqui e as ferramentas ali roteiam por `kanban_db`, então as duas superfícies concordam por construção.

```
hermes kanban init                                     # create kanban.db + print daemon hint
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|worktree:<path>|dir:<path>]
                                [--branch <name>]
                                [--priority N] [--triage] [--idempotency-key KEY]
                                [--max-runtime 30m|2h|1d|<seconds>]
                                [--max-retries N]
                                [--goal] [--goal-max-turns N]
                                [--skill <name>]...
                                [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived]
        [--workflow-template-id <id>] [--current-step-key <key>]
        [--sort created|created-desc|priority|priority-desc|status|assignee|title|updated]
        [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>                    # or 'none' to unassign
hermes kanban reassign <id>... <profile>               # bulk re-assign tasks to a profile
hermes kanban edit <id> [--title ...] [--body ...]     # edit task title / body / priority in place
        [--priority N]
hermes kanban promote <id>...                          # move todo/blocked tasks to ready (recovery)
hermes kanban schedule <id> --at <ISO8601>             # set/clear a task's scheduled_at start time
hermes kanban diagnostics [--json]                     # board health snapshot (alias: diag)
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]

# Bulk verbs — accept multiple ids:
hermes kanban complete <id>... [--result "..."]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...

hermes kanban tail <id>                                # follow a single task's event stream
hermes kanban watch [--assignee P] [--tenant T]        # live stream ALL events to the terminal
        [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]            # worker liveness signal for long ops
hermes kanban runs <id> [--json]                       # attempt history (one row per run)
hermes kanban assignees [--json]                       # profiles on disk + per-assignee task counts
hermes kanban dispatch [--dry-run] [--max N]           # one-shot pass
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # DEPRECATED — standalone dispatcher (use `hermes gateway start` instead)
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # per-status + per-assignee counts
hermes kanban log <id> [--tail BYTES]                  # worker log from ~/.hermes/kanban/logs/
hermes kanban notify-subscribe <id>                    # gateway bridge hook (used by /kanban in the gateway)
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # what a worker sees
hermes kanban specify [<id> | --all] [--tenant T]      # flesh out a triage-column idea
        [--author NAME] [--json]                       #   into a full spec and promote to todo
hermes kanban gc [--event-retention-days N]            # workspaces + old events + old logs
        [--log-retention-days N]
```

Todos os comandos também estão disponíveis como slash command na CLI interativa e no messaging gateway (veja [comando slash `/kanban`](#kanban-slash-command) abaixo).

`--max-retries` é override de circuit-breaker por tarefa para o dispatcher. `--max-retries 1` block a tarefa na primeira tentativa não bem-sucedida, enquanto `--max-retries 3` permite dois retries e block na terceira falha. Omita para usar `kanban.failure_limit` de `config.yaml`, depois o default built-in.

### Config de concorrência, agendamento e promoção de filhos {#concurrency-scheduling-and-child-promotion-config}

| Config key | Padrão | O que faz |
|------------|---------|--------------|
| `kanban.max_in_progress` | unset (unlimited) | Limita o número de tarefas rodando simultaneamente. Quando o board já tem N running, o dispatcher pula spawn de mais — útil para workers lentos (LLMs locais, hosts com recursos limitados) para terminarem o que têm antes de mais acumularem e timeout. Valores inválidos ou abaixo de 1 logam warning e comportam como unlimited. |
| `kanban.max_in_progress_per_profile` | unset (unlimited) | Variante por profile de `max_in_progress` — limita quantas tarefas qualquer profile assignee pode rodar concorrentemente. Útil quando um profile é lento ou rate-limited mas outros devem continuar fluindo. Aplica junto com `max_in_progress` board-wide; ambos devem permitir spawn para proceder. |
| `kanban.auto_promote_children` | `true` | Depois que `decompose_triage_task()` produz filhos sem dependências parent-blocker, são automaticamente promovidos para `ready` para o dispatcher pegá-los. Set `false` para exigir review manual — filhos ficam em `todo` até você promovê-los. |
| `kanban.default_workdir` | unset | Diretório de trabalho default em nível de board aplicado a tarefas novas quando nem `--workspace` nem a tarefa override. `workspace:` por tarefa ainda vence. |

```yaml
kanban:
  max_in_progress: 2
  auto_promote_children: false
  default_workdir: ~/work/active-project
```

### Inícios de tarefa agendados (`scheduled_at`) {#scheduled-task-starts-scheduled_at}

Set `scheduled_at` numa tarefa para atrasar dispatch até horário específico. O dispatcher pula tarefas ready cujo `scheduled_at` está no futuro e as pega no primeiro tick após esse timestamp.

```bash
hermes kanban create "nightly backup audit" \
  --assignee ops --scheduled-at "2026-06-01T03:00:00Z"
```

### Respawn guard {#respawn-guard}

O dispatcher recusa re-spawn de tarefa ready quando ela teve erro quota/auth/429 na run anterior (`blocker_auth`), ou completou run com sucesso dentro da janela de guard (`recent_success`), ou comment recente da tarefa linka para GitHub PR (`active_pr`). Isso evita worker storms repetidos no mesmo bug ou tarefa enquanto um humano alcança. Veja a linha `respawn_guarded` na [referência de events](#event-reference).

### Drag-to-delete e bulk delete (dashboard) {#drag-to-delete-and-bulk-delete-dashboard}

O dashboard expõe **zona de drop lixeira** na página kanban — arraste qualquer card para deletar a tarefa (cascades por `task_events`, child links e subscriptions). Prompt de confirmação protege contra acidentes. Bulk delete também alcançável via `DELETE /api/plugins/kanban/tasks` com body JSON `{"ids": ["t_abc", "t_def", ...]}`.

### Endpoints de visibilidade de worker {#worker-visibility-endpoints}

A API do plugin dashboard agora expõe estes endpoints read-only (mais um verbo run-control) para monitores externos:

| Endpoint | Retorna |
|----------|---------|
| `GET /api/plugins/kanban/workers/active` | Workers spawnados atualmente com PID, profile, task id, started-at, last heartbeat |
| `GET /api/plugins/kanban/runs/{id}` | Detalhe de run única — task id, status, started/ended, exit code, log path |
| `POST /api/plugins/kanban/runs/{run_id}/terminate` | Termina run reclaimable — para o worker e libera a tarefa para re-dispatch |
| `GET /api/plugins/kanban/inspect` | Snapshot combinado do dispatcher — backlog, contagem in-progress vs. `max_in_progress`, events recentes |

Todos gated pela mesma auth de plugin dashboard que o resto da API kanban plugin.

### Helper de topologia Kanban Swarm {#kanban-swarm-topology-helper}

`hermes kanban swarm` cria grafo durável **Kanban Swarm v1** num tiro: card root/blackboard completed, N cards worker paralelos, card verifier gated em todos os workers, e card synthesizer gated no verifier. Contexto swarm compartilhado (o "blackboard") é armazenado como comments JSON estruturados no card root para qualquer worker ler.

```bash
hermes kanban swarm "Design a multi-region failover plan" \
  --workers researcher,architect,sre \
  --verifier reviewer --synthesizer writer
```

O grafo resultante dispatch normalmente — workers rodam em paralelo, o verifier acorda depois que todos terminam, o synthesizer acorda depois que o verifier marca o trabalho limpo.

## Comando slash `/kanban` {#kanban-slash-command}

Todo verbo `hermes kanban <action>` também é alcançável como `/kanban <action>` — de dentro de sessão interativa `hermes chat` **e** de qualquer plataforma gateway (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost, email, SMS). Ambas as superfícies chamam exatamente o mesmo entry point `hermes_cli.kanban.run_slash()` que reusa a árvore argparse `hermes kanban`, então superfície de argumentos, flags e formato de saída são idênticos entre CLI, `/kanban` e `hermes kanban`. Você não precisa sair do chat para dirigir o board.

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # flesh out a triage one-liner into a real spec
/kanban specify --all --tenant engineering  # sweep every triage task in one tenant
```

Cite argumentos multi-palavra como faria num shell — `run_slash` parseia o resto da linha com `shlex.split`, então `"..."` e `'...'` funcionam.

### Uso mid-run: `/kanban` bypassa o running-agent guard {#mid-run-usage-kanban-bypasses-the-running-agent-guard}

O gateway normalmente enfileira slash commands e mensagens de usuário enquanto um agent ainda está pensando — isso impede você de acidentalmente iniciar segundo turn enquanto o primeiro está in flight. **`/kanban` é explicitamente isento deste guard.** O board vive em `~/.hermes/kanban.db`, não no estado do agent rodando, então reads (`list`, `show`, `context`, `tail`, `watch`, `stats`, `runs`) e writes (`comment`, `unblock`, `block`, `assign`, `archive`, `create`, `link`, …) passam imediatamente, mesmo mid-turn.

Este é o ponto da separação:

- Um worker block esperando peer → você manda `/kanban unblock t_abcd` do celular e o dispatcher pega o peer no próximo tick. O worker blocked não é interrompido — só deixa de estar blocked.
- Você vê card que precisa de contexto humano → `/kanban comment t_xyz "use the 2026 schema, not 2025"` cai na thread da tarefa e a *próxima* run dessa tarefa lerá em `kanban_show()`.
- Você quer saber o que sua frota faz sem parar o orchestrator → `/kanban list --mine` ou `/kanban stats` inspeciona o board sem tocar sua conversa principal.

### Auto-subscribe em `/kanban create` (só gateway) {#auto-subscribe-on-kanban-create-gateway-only}

Quando você cria tarefa do gateway com `/kanban create "…"`, o chat originador (platform + chat id + thread id) é automaticamente inscrito nos eventos terminais dessa tarefa (`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`). Você recebe uma mensagem por evento terminal — incluindo a primeira linha do result summary do worker em `completed` — sem precisar poll ou lembrar o task id.

```
you> /kanban create "transcribe today's podcast" --assignee transcriber
bot> Created t_9fc1a3  (ready, assignee=transcriber)
     (subscribed — you'll be notified when t_9fc1a3 completes or blocks)

… ~8 minutes later …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

Subscriptions auto-remove quando a tarefa atinge `done` ou `archived`. Se você scripta create com `--json` (saída machine) o auto-subscribe é pulado — assume-se que callers scriptados querem gerenciar subscriptions explicitamente via `/kanban notify-subscribe`.

### Truncamento de saída em messaging {#output-truncation-in-messaging}

Plataformas gateway têm caps práticos de tamanho de mensagem. Se `/kanban list`, `/kanban show` ou `/kanban tail` produzem mais de ~3800 caracteres de saída, a resposta é truncada com footer `… (truncated; use \`hermes kanban …\` in your terminal for full output)`. A superfície CLI não tem esse cap.

### Autocomplete {#autocomplete}

Na CLI interativa, digitar `/kanban ` e Tab alterna pela lista built-in de subcomandos (`list`, `ls`, `show`, `create`, `assign`, `link`, `unlink`, `claim`, `comment`, `complete`, `block`, `unblock`, `archive`, `tail`, `dispatch`, `context`, `init`, `gc`). Os verbos restantes listados na referência CLI acima (`watch`, `stats`, `runs`, `log`, `assignees`, `heartbeat`, `notify-subscribe`, `notify-list`, `notify-unsubscribe`, `daemon`) também funcionam — só não estão na lista de hint de autocomplete ainda.

## Padrões de colaboração {#collaboration-patterns}

O board suporta estes oito padrões sem novos primitivos:

| Padrão | Forma | Exemplo |
|---|---|---|
| **P1 Fan-out** | N irmãos, mesmo papel | "research 5 angles in parallel" |
| **P2 Pipeline** | cadeia de papéis: scout → editor → writer | daily brief assembly |
| **P3 Voting / quorum** | N irmãos + 1 agregador | 3 researchers → 1 reviewer picks |
| **P4 Long-running journal** | mesmo profile + dir compartilhado + cron | Obsidian vault |
| **P5 Human-in-the-loop** | worker blocks → user comments → unblock | ambiguous decisions |
| **P6 `@mention`** | roteamento inline da prose | `@reviewer look at this` |
| **P7 Thread-scoped workspace** | `/kanban here` num thread | per-project gateway threads |
| **P8 Fleet farming** | um profile, N subjects | 50 social accounts |
| **P9 Triage specifier** | ideia bruta → `triage` → `hermes kanban specify` expande body → `todo` | "turn this one-liner into a spec'd task" |

Para exemplos trabalhados de cada um, veja `docs/hermes-kanban-v1-spec.pdf`.

## Uso multi-tenant {#multi-tenant-usage}

Quando uma frota specialist serve vários negócios, tag cada tarefa com tenant:

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

Workers recebem `$HERMES_TENANT` e namespaciam writes de memória por prefix. O board, o dispatcher e as definições de profile são compartilhados; só os dados são scoped.

## Notificações gateway {#gateway-notifications}

Quando você roda `/kanban create …` do gateway (Telegram, Discord, Slack, etc.), o chat originador é automaticamente inscrito na nova tarefa. O notifier em background do gateway poll `task_events` a cada poucos segundos e entrega uma mensagem por evento terminal (`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`) àquele chat. Tarefas completed também enviam a primeira linha do `--result` do worker para você ver o outcome sem `/kanban show`.

Você pode gerenciar subscriptions explicitamente pela CLI — útil quando script / cron job quer notificar chat que não originou:

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

Uma subscription se remove automaticamente quando a tarefa atinge `done` ou `archived`; sem cleanup necessário.

## Runs — uma linha por tentativa {#runs-one-row-per-attempt}

Uma tarefa é unidade lógica de trabalho; uma **run** é uma tentativa de executá-la. Quando o dispatcher claim uma tarefa ready cria linha em `task_runs` e aponta `tasks.current_run_id` para ela. Quando essa tentativa termina — completed, blocked, crashed, timed out, spawn-failed, reclaimed — a linha de run fecha com `outcome` e o ponteiro da tarefa limpa. Tarefa tentada três vezes tem três linhas `task_runs`.

Por que duas tabelas em vez de só mutar a tarefa: você precisa de **histórico completo de tentativas** para postmortems reais ("a segunda tentativa de reviewer chegou a approve, a terceira merged"), e precisa de lugar limpo para metadata por tentativa — quais arquivos mudaram, quais tests rodaram, quais achados um reviewer anotou. Esses são fatos de run, não fatos de task.

Runs também são onde vive **handoff estruturado**. Quando um worker completa tarefa (via `kanban_complete(...)`) pode passar:

- `summary` (tool param) / `--summary` (CLI) — handoff humano; vai na run; filhos downstream veem em `build_worker_context`.
- `metadata` (tool param) / `--metadata` (CLI) — dict JSON free-form na run; filhos veem serializado junto ao summary.
- `result` (tool param) / `--result` (CLI) — linha de log curta que vai na linha da tarefa (campo legacy, mantido por back-compat).

Filhos downstream leem summary + metadata da run completed mais recente de cada parent. Workers retentando leem tentativas anteriores na própria tarefa (outcome, summary, error) para não repetir caminho que já falhou.

```
# What a worker actually does — a tool call, from inside the agent loop:
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

O mesmo handoff é alcançável pela CLI quando você (humano) precisa fechar tarefa que worker não pode — ex.: tarefa abandonada, ou que você marcou done manualmente no dashboard:

```bash
hermes kanban complete t_abcd \
    --result "rate limiter shipped" \
    --summary "implemented token bucket, keys on user_id with IP fallback, all tests pass" \
    --metadata '{"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14}'

# Review the attempt history on a retried task:
hermes kanban runs t_abcd
#   #  OUTCOME       PROFILE           ELAPSED  STARTED
#   1  blocked       worker               12s  2026-04-27 14:02
#        → BLOCKED: need decision on rate-limit key
#   2  completed     worker                8m   2026-04-27 15:18
#        → implemented token bucket, keys on user_id with IP fallback
```

Runs são expostos no dashboard (seção Run History no drawer, uma linha colorida por tentativa) e na REST API (`GET /api/plugins/kanban/tasks/:id` retorna array `runs[]`). `PATCH /api/plugins/kanban/tasks/:id` com `{status: "done", summary, metadata}` encaminha ambos ao kernel, então o botão "mark done" do dashboard é equivalente à CLI. Linhas `task_events` carregam o `run_id` a que pertencem para a UI agrupar por tentativa, e o event `completed` embute summary de primeira linha no payload (cap 400 chars) para notifiers gateway renderizarem handoffs estruturados sem segunda round-trip SQL.

**Caveat de bulk close.** `hermes kanban complete a b c --summary X` é recusado — handoff estruturado é por-run, então copy-paste do mesmo summary para N tarefas quase sempre está errado. Bulk close *sem* `--summary` / `--metadata` ainda funciona para o caso comum "terminei pilha de tarefas admin".

**Runs reclaimed por mudanças de status.** Se você arrasta tarefa running fora de `running` no dashboard (de volta para `ready`, ou direto para `todo`), ou archive tarefa que ainda estava running, a run in-flight fecha com `outcome='reclaimed'` em vez de ficar órfã. A linha `task_runs` está sempre em estado terminal quando `tasks.current_run_id` é `NULL`, e vice versa — esse invariante vale em CLI, dashboard, dispatcher e notifier.

**Runs sintéticas para conclusões never-claimed.** Completar ou block tarefa que nunca foi claimed (ex.: humano fecha tarefa `ready` no dashboard com summary, ou usuário CLI roda `hermes kanban complete <ready-task> --summary X`) senão droparia o handoff. Em vez disso o kernel insere linha de run zero-duration (`started_at == ended_at`) carregando summary / metadata / reason para histórico de tentativas ficar completo. O `run_id` do event `completed` / `blocked` aponta para essa linha.

**Refresh live do drawer.** Quando o stream WebSocket de events do dashboard reporta events novos para a tarefa que o usuário está vendo, o drawer recarrega (via contador de events por tarefa no dependency list do `useEffect`). Fechar e reabrir não é mais necessário para ver nova linha de run ou outcome atualizado.

### Compatibilidade forward {#forward-compatibility}

Duas colunas nullable em `tasks` estão reservadas para roteamento workflow v2: `workflow_template_id` (qual template esta tarefa pertence) e `current_step_key` (qual step nesse template está ativo). O kernel v1 ignora para roteamento mas deixa clients escrevê-las, então release v2 pode adicionar maquinaria de roteamento sem outra migration de schema.

## Referência de events {#event-reference}

Toda transição anexa linha em `task_events`. Cada linha carrega `run_id` opcional para UIs agruparem events por tentativa. Kinds agrupam em três clusters para filtragem fácil (`hermes kanban watch --kinds completed,gave_up,timed_out`):

**Lifecycle** (o que mudou sobre a tarefa como unidade lógica):

| Kind | Payload | Quando |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | Task inserida. `run_id` é `NULL`. |
| `promoted` | — | `todo → ready` porque todos os parents atingiram `done`. `run_id` é `NULL`. |
| `claimed` | `{lock, expires, run_id}` | Dispatcher claim atômico de tarefa `ready` para spawn. |
| `completed` | `{result_len, summary?}` | Worker escreveu `--result` / `--summary` e tarefa atingiu `done`. `summary` é handoff de primeira linha (cap 400 chars); versão completa vive na linha de run. Se `complete_task` é chamado em tarefa never-claimed com campos de handoff, run zero-duration é sintetizada para `run_id` ainda apontar algo. |
| `blocked` | `{reason, kind, recurrences}` | Worker ou humano flipou tarefa para `blocked`. `kind` é razão typed de block (`needs_input`, `capability`, `transient`, ou `null` para block genérico); `recurrences` é contador de loop unblock. Sintetiza run zero-duration quando chamado em tarefa never-claimed com `--reason`. |
| `dependency_wait` | `{reason, kind}` | Worker blocked com `kind=dependency` — tarefa só espera outra tarefa, então roteia para `todo` (parent-gated, auto-promoted) em vez de `blocked`. Sem humano necessário. |
| `block_loop_detected` | `{reason, kind, recurrences, limit}` | Tarefa foi unblocked e re-blocked pela mesma razão `BLOCK_RECURRENCE_LIMIT` vezes (padrão 2). Em vez de cair em `blocked` de novo — onde cron só desbloquearia — roteia para `triage` para decisão humana, quebrando loop unblock↔re-block. |
| `unblocked` | — | `blocked → ready` (ou `todo` se parents ainda abertos), manualmente ou via `/unblock`. Reseta `consecutive_failures` do dispatcher mas preserva deliberadamente `block_recurrences` para o loop breaker manter memória. `run_id` é `NULL`. |
| `archived` | — | Oculto do board default. Se tarefa ainda estava running, carrega `run_id` da run que foi reclaimed como side effect. |

**Edits** (mudanças driven por humano que não são transições):

| Kind | Payload | Quando |
|---|---|---|
| `assigned` | `{assignee}` | Assignee mudou (incluindo unassignment). |
| `edited` | `{fields}` | Title ou body atualizado. |
| `reprioritized` | `{priority}` | Priority mudou. |
| `status` | `{status}` | Drag-drop do dashboard escreveu status diretamente (ex.: `todo → ready`). Carrega `run_id` da run reclaimed ao arrastar off `running`; senão `run_id` é NULL. |

**Telemetria worker** (sobre o processo de execução, não a tarefa lógica):

| Kind | Payload | Quando |
|---|---|---|
| `spawned` | `{pid}` | Dispatcher iniciou processo worker com sucesso. |
| `heartbeat` | `{note?}` | Worker chamou `hermes kanban heartbeat $TASK` para sinalizar vivacidade durante operações longas. |
| `reclaimed` | `{stale_lock}` | TTL de claim expirou sem completion; tarefa volta para `ready`. |
| `crashed` | `{pid, claimer}` | PID do worker não mais vivo mas TTL ainda não expirou. |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | `max_runtime_seconds` excedido; dispatcher SIGTERM (depois SIGKILL após 5 s grace) e re-queued. |
| `stale` | `{elapsed_seconds, last_heartbeat_at, heartbeat_age_seconds, timeout_seconds, pid, terminated}` | Tarefa rodou mais que `kanban.dispatch_stale_timeout_seconds` (padrão 4 h) E nenhum `kanban_heartbeat` chegou na última hora. Dispatcher SIGTERM no worker host-local (se houver), reset tarefa para `ready` para re-dispatch. NÃO ticka contador de falha (stale é detecção de ausência no dispatcher, não falha do worker). Workers em operações longas devem chamar `kanban_heartbeat` pelo menos uma vez por hora para evitar isso. |
| `respawn_guarded` | `{reason}` | Dispatcher recusou re-spawn desta tarefa ready neste tick. Razões: `blocker_auth` (última falha foi erro quota/auth/429 — espere janela de rate reset), `recent_success` (run completed na última hora — espere review antes de re-run), `active_pr` (URL GitHub PR aparece em comment recente — worker anterior já abriu PR). Tarefa fica em `ready`; próximo tick tem outra chance de spawn. Se condição subjacente persistir, circuit breaker normal `consecutive_failures` auto-block via `gave_up` após `failure_limit` falhas. |
| `spawn_failed` | `{error, failures}` | Uma tentativa de spawn falhou (PATH faltando, workspace não montável, …). Contador incrementa; tarefa volta para `ready` para retry. |
| `protocol_violation` | `{pid, claimer, exit_code, protocol_violation}` | Worker saiu com sucesso enquanto tarefa ainda estava `running`, geralmente porque respondeu sem chamar `kanban_complete` ou `kanban_block`. Emitido em toda violação (marker `protocol_violation: true` do payload é copiado para metadata da run e alimenta orçamento de retry só de violação). Abaixo do orçamento — até `_PROTOCOL_VIOLATION_FAILURE_LIMIT` (padrão 3) violações *consecutivas*, `max_retries` por tarefa override — tarefa simplesmente volta para `ready` para outra tentativa; quando streak atinge bound dispatcher também emite `gave_up` e auto-block. |
| `gave_up` | `{failures, effective_limit, limit_source, error}` | Circuit breaker disparou após N tentativas consecutivas não bem-sucedidas. Tarefa auto-block com último erro. Limite efetivo resolve como `max_retries` da tarefa, depois `failure_limit` do dispatcher / `kanban.failure_limit`, depois default built-in. |

`hermes kanban tail <id>` mostra estes para tarefa única. `hermes kanban watch` stream board-wide.

## Fora de escopo {#out-of-scope}

Kanban é deliberadamente single-host. `~/.hermes/kanban.db` é arquivo SQLite local e o dispatcher spawna workers na mesma máquina. Rodar board compartilhado entre dois hosts não é suportado — não há primitivo de coordenação para "worker X no host A, worker Y no host B," e o path de crash-detection assume PIDs host-local. Se precisar multi-host, rode board independente por host e use `delegate_task` / fila de mensagens para fazer ponte.

## Spec de design {#design-spec}

O design completo — arquitetura, correção de concorrência, comparação com outros sistemas, plano de implementação, riscos, questões abertas — vive em `docs/hermes-kanban-v1-spec.pdf`. Leia antes de abrir PR de mudança de comportamento.
