# Tutorial Kanban {#kanban-tutorial}

Um walkthrough dos quatro casos de uso para os quais o sistema Hermes Kanban foi projetado, com o dashboard aberto no browser. Se ainda não leu a [visão geral Kanban](./kanban), comece por lá — este tutorial assume que você sabe o que são task, run, assignee e dispatcher.

## Setup {#setup}

```bash
hermes kanban init           # optional; first `hermes kanban <anything>` auto-inits
hermes dashboard             # opens http://127.0.0.1:9119 in your browser
# click Kanban in the left nav
```

O dashboard é a interface mais confortável para **você** observar o sistema. Workers agent que o dispatcher spawna nunca veem o dashboard nem a CLI — eles dirigem o board por um [toolset](./kanban#how-workers-interact-with-the-board) `kanban_*` dedicado (`kanban_show`, `kanban_list`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`, `kanban_unblock`). As três superfícies — dashboard, CLI, ferramentas worker — passam pelo mesmo SQLite DB por board (`~/.hermes/kanban.db` para o board default, `~/.hermes/kanban/boards/<slug>/kanban.db` para qualquer board que você criar depois), então cada board fica consistente não importa de qual lado da cerca veio a mudança.

Este tutorial usa o board `default` em todo lugar. Se quiser múltiplas filas isoladas (uma por projeto / repo / domínio), veja [Boards (multi-project)](./kanban#boards-multi-project) na visão geral — os mesmos fluxos CLI / dashboard / worker se aplicam por board, e workers fisicamente não podem ver tarefas em outros boards.

Ao longo do tutorial, **blocos de código rotulados `bash` são comandos que *você* roda.** Blocos rotulados `# worker tool calls` são o que o model do worker spawnado emite como tool calls — mostrados aqui para você ver o loop end-to-end, não porque rodaria isso você mesmo.

## O board num relance {#the-board-at-a-glance}

![Visão geral do board Kanban](/img/kanban-tutorial/01-board-overview.png)

Seis colunas, da esquerda para a direita:

- **Triage** — ideias brutas. Por padrão o dispatcher auto-roda o **decomposer** em tarefas aqui: o decomposer built-in usa `auxiliary.kanban_decomposer`, lê seu roster de profiles + descriptions, e produz um grafo de tarefas filhas roteadas para os specialists best-fit. A tarefa original é mantida viva como parent de toda folha no grafo para seu assignee (`kanban.orchestrator_profile`, ou o profile default ativo quando unset) acordar de volta para julgar conclusão quando tudo terminar. Alterne a pill **Orchestration: Auto/Manual** no topo da página kanban para trocar modos. Em Manual mode clique **⚗ Decompose** num card, ou rode `hermes kanban decompose <id>` / `/kanban decompose <id>`. Para tarefas únicas que não precisam fan-out, **✨ Specify** faz rewrite de spec one-shot (goal, approach, acceptance criteria) e promove para `todo`. Configure os models sob `auxiliary.kanban_decomposer` e `auxiliary.triage_specifier` em `config.yaml`. Veja [Auto vs Manual orchestration](./kanban#auto-vs-manual-orchestration) no guia Kanban principal.
- **Todo** — criada mas aguardando dependências, ou ainda não atribuída.
- **Ready** — atribuída e aguardando o dispatcher claim.
- **In progress** — um worker está executando a tarefa ativamente. Com "Lanes by profile" ligado (o padrão), esta coluna sub-agrupa por assignee para você ver num relance o que cada worker faz.
- **Blocked** — um worker pediu input humano, ou o circuit breaker disparou.
- **Done** — concluída.

A barra superior tem filtros para search, tenant e assignee, plus botão `Nudge dispatcher` que roda um tick de dispatch agora em vez de esperar o intervalo do daemon. Clicar qualquer card abre seu drawer à direita.

### Vista flat {#flat-view}

Se as profile lanes são barulhentas, desligue "Lanes by profile" e a coluna In Progress colapsa para lista flat única ordenada por claim time:

![Board com lanes by profile desligadas](/img/kanban-tutorial/02-board-flat.png)

## História 1 — Dev solo entregando uma feature {#story-1--solo-dev-shipping-a-feature}

Você está construindo uma feature. Fluxo clássico: desenhar schema, implementar API, escrever testes. Três tarefas com dependências parent→child.

```bash
SCHEMA=$(hermes kanban create "Design auth schema" \
    --assignee backend-dev --tenant auth-project --priority 2 \
    --body "Design the user/session/token schema for the auth module." \
    --json | jq -r .id)

API=$(hermes kanban create "Implement auth API endpoints" \
    --assignee backend-dev --tenant auth-project --priority 2 \
    --parent $SCHEMA \
    --body "POST /register, POST /login, POST /refresh, POST /logout." \
    --json | jq -r .id)

hermes kanban create "Write auth integration tests" \
    --assignee qa-dev --tenant auth-project --priority 2 \
    --parent $API \
    --body "Cover happy path, wrong password, expired token, concurrent refresh."
```

Como `API` tem `SCHEMA` como parent, e `tests` tem `API` como parent, só `SCHEMA` começa em `ready`. As outras duas ficam em `todo` até seus parents completarem. Este é o engine de promoção de dependência fazendo seu trabalho — nenhum outro worker pegará test-writing até haver API para testar.

No próximo tick do dispatcher (60s por padrão, ou imediatamente se você apertar **Nudge dispatcher**) o profile `backend-dev` spawna como worker com `HERMES_KANBAN_TASK=$SCHEMA` no env. Assim fica o loop de tool-call do worker de dentro do agent:

```python
# worker tool calls — NOT commands you run
kanban_show()
# → returns title, body, worker_context, parents, prior attempts, comments

# (worker reads worker_context, uses terminal/file tools to design the schema,
#  write migrations, run its own checks, commit — the real work happens here)

kanban_heartbeat(note="schema drafted, writing migrations now")

kanban_complete(
    summary="users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); "
            "refresh tokens stored as sessions with type='refresh'",
    metadata={
        "changed_files": ["migrations/001_users.sql", "migrations/002_sessions.sql"],
        "decisions": ["bcrypt for hashing", "JWT for session tokens",
                      "7-day refresh, 15-min access"],
    },
)
```

`kanban_show` default `task_id` para `$HERMES_KANBAN_TASK`, então o worker não precisa saber seu próprio id. `kanban_complete` grava summary + metadata na linha `task_runs` atual, fecha essa run, e transiciona a tarefa para `done` — tudo num hop atômico via `kanban_db`.

Quando `SCHEMA` atinge `done`, o engine de dependência promove `API` para `ready` automaticamente. O worker API, quando pegar, chamará `kanban_show()` e verá summary e metadata de `SCHEMA` anexados ao parent handoff — então sabe as decisões de schema sem reler um design doc longo.

Clique a tarefa schema completada no board e o drawer mostra tudo:

![Solo dev — drawer da tarefa schema completada](/img/kanban-tutorial/03-drawer-schema-task.png)

A seção Run History na parte inferior é a adição-chave. Uma tentativa: outcome `completed`, worker `@backend-dev`, duração, timestamp, e o handoff summary completo. O blob metadata (`changed_files`, `decisions`) também fica na run e é exposto a qualquer worker downstream que leia este parent.

Você pode inspecionar os mesmos dados do terminal a qualquer momento — estes comandos são **você** espiando o board, não o worker:

```bash
hermes kanban show $SCHEMA
hermes kanban runs $SCHEMA
# #  OUTCOME       PROFILE       ELAPSED  STARTED
# 1  completed     backend-dev        0s  2026-04-27 19:34
#     → users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); refresh tokens ...
```

## História 2 — Fleet farming {#story-2--fleet-farming}

Você tem três workers (translator, transcriber, copywriter) e uma pilha de tarefas independentes. Quer os três puxando em paralelo com progresso visível. Este é o caso de uso kanban mais simples e o que o design original otimizou.

Crie o trabalho:

```bash
for lang in Spanish French German; do
    hermes kanban create "Translate homepage to $lang" \
        --assignee translator --tenant content-ops
done
for i in 1 2 3 4 5; do
    hermes kanban create "Transcribe Q3 customer call #$i" \
        --assignee transcriber --tenant content-ops
done
for sku in 1001 1002 1003 1004; do
    hermes kanban create "Generate product description: SKU-$sku" \
        --assignee copywriter --tenant content-ops
done
```

Inicie o gateway e vá embora — ele hospeda o dispatcher embedded
que pega tarefas dos três specialist profiles no mesmo
kanban.db:

```bash
hermes gateway start
```

Agora filtre o board para `content-ops` (ou só busque "Transcribe") e você vê isto:

![Vista fleet filtrada para tarefas transcribe](/img/kanban-tutorial/07-fleet-transcribes.png)

Duas transcribes done, uma running, duas ready aguardando o próximo tick do dispatcher. A coluna In Progress é agrupada por profile (default "Lanes by profile") para você ver a tarefa ativa de cada worker sem varrer lista mista. O dispatcher promoverá a próxima ready task para running assim que a atual completar. Com três daemons trabalhando em três assignee pools em paralelo, toda a fila content drena sem mais input humano.

**Tudo que a História 1 disse sobre structured handoff ainda se aplica aqui.** Um worker translator completando uma call emite `kanban_complete(summary="translated 4 pages, style matched existing marketing voice", metadata={"duration_seconds": 720, "tokens_used": 2100})` — útil para analytics e para qualquer tarefa downstream que dependa desta.

## História 3 — Pipeline de papéis com retry {#story-3--role-pipeline-with-retry}

É aqui que Kanban ganha sobre uma TODO list flat. Um PM escreve spec. Um engineer implementa. Um reviewer rejeita a primeira tentativa. O engineer tenta de novo com mudanças. O reviewer aprova.

A vista dashboard, filtrada por `auth-project`:

![Vista pipeline para feature multi-papel](/img/kanban-tutorial/08-pipeline-auth.png)

O screenshot usa o modelo de **card downstream pré-criado**: o card de implementação tem um child reviewer dedicado. Nesse modelo o engineer deve chamar `kanban_complete` quando a implementação estiver pronta para o child reviewer sair de `todo`. Nunca bloqueie o parent de implementação só para pedir review.

Para workflows em que o mesmo card dono da implementação e da review, use o lifecycle de review first-class. A coreografia completa implement → review → changes → re-review é:

```python
# --- Engineer: first implementation attempt ---
kanban_show()
# (write code, run tests, prepare the candidate)
kanban_request_review(
    summary="implemented reset flow; candidate is ready for review",
    metadata={"changed_files": ["auth/reset.py"], "tests_run": 8},
    reviewer="reviewer",
)
# → the same card enters review; the implementation run closes as
#   outcome='review_requested'

# --- Reviewer: request concrete changes ---
kanban_show()
# (inspect the handoff and candidate)
kanban_request_changes(
    reason="Add password-strength validation and make reset tokens single-use."
)
# → the review run closes as outcome='changes_requested'; the card returns
#   to backend-dev in ready/todo without touching block-loop accounting

# --- Engineer: second implementation attempt ---
kanban_show()  # prior review evidence is in worker_context
# (apply feedback and re-run tests)
kanban_request_review(
    summary="added zxcvbn validation and single-use reset tokens",
    metadata={
        "changed_files": [
            "auth/reset.py",
            "auth/tests/test_reset.py",
            "migrations/003_single_use_reset_tokens.sql",
        ],
        "tests_run": 11,
        "review_iteration": 2,
    },
    reviewer="reviewer",
)

# --- Reviewer: approve ---
kanban_complete(summary="review passed; acceptance criteria verified")
# → done
```

O histórico de runs da tarefa agora registra `review_requested → changes_requested → review_requested → completed`. Cada tentativa tem seu próprio actor, summary, metadata e outcome, então o segundo engineer vê exatamente o que o reviewer rejeitou e a aprovação final permanece auditável. `kanban_block` fica reservado para uma escalação externa real (acesso faltando, uma decisão de produto, infraestrutura indisponível), não feedback normal de review.

Se você usa de propósito o modelo de card downstream mostrado no screenshot, o reviewer abre `Review password reset PR` depois que o parent de implementação completa:

![Vista drawer do reviewer no pipeline](/img/kanban-tutorial/09-drawer-pipeline-review.png)

O `worker_context` do card reviewer inclui o handoff da implementação completada. Isso é um workflow de cards separados; não combine com `kanban_request_review` no mesmo card ou você duplicará a lane de review.

## História 4 — Circuit breaker e crash recovery {#story-4--circuit-breaker-and-crash-recovery}

Workers reais falham. Credenciais faltando, kills OOM, erros de rede transientes. O dispatcher tem duas linhas de defesa: um **circuit breaker** que auto-block após N falhas consecutivas para o board não thrash forever, e **crash detection** que reclaima tarefa cujo worker PID sumiu antes do TTL expirar.

### Circuit breaker — falha com aparência permanente {#circuit-breaker--permanent-looking-failure}

Uma tarefa deploy que não consegue spawnar worker porque `AWS_ACCESS_KEY_ID` não está set no environment do profile:

```bash
hermes kanban create "Deploy to staging (missing creds)" \
    --assignee deploy-bot --tenant ops \
    --max-retries 3
```

O dispatcher tenta spawnar o worker. Spawn falha (`RuntimeError: AWS_ACCESS_KEY_ID not set`). O dispatcher libera o claim, incrementa contador de falha, e tenta de novo no próximo tick. Como este exemplo seta `--max-retries 3`, o circuit dispara após três falhas consecutivas: a tarefa vai para `blocked` com outcome `gave_up`. Se omitir a flag, Hermes usa `kanban.failure_limit` (padrão: 2). Sem mais retries até humano desbloquear.

Clique a tarefa blocked:

![Circuit breaker — 2 spawn_failed + 1 gave_up](/img/kanban-tutorial/11-drawer-gave-up.png)

Três runs, todas com o mesmo erro no campo `error`. As duas primeiras são `spawn_failed` (retryable), a terceira é `gave_up` (terminal). O event log acima mostra a sequência completa: `created → claimed → spawn_failed → claimed → spawn_failed → claimed → gave_up`.

No terminal:

```bash
hermes kanban runs t_ef5d
# #   OUTCOME        PROFILE        ELAPSED  STARTED
# 1   spawn_failed   deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
# 2   spawn_failed   deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
# 3   gave_up        deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
```

Se Telegram / Discord / Slack estiver wired, uma notificação gateway dispara no evento `gave_up` para você ouvir sobre a outage sem checar o board.

### Crash recovery — worker morre mid-flight {#crash-recovery--worker-dies-mid-flight}

Às vezes o spawn sucede mas o processo worker morre depois — segfault, OOM, `systemctl stop`. O dispatcher poll `kill(pid, 0)` e detecta o pid morto; o claim libera, a tarefa volta para `ready`, e o próximo tick dá a um worker fresh.

O exemplo nos seed data é uma migration que estava running out of memory:

```bash
# Worker claims, starts scanning 2.4M rows, OOM kills it at ~2.3M
# Dispatcher detects dead pid, releases claim, increments attempt counter
# Retry with a chunked strategy succeeds
```

O drawer mostra o histórico completo de duas tentativas:

![Crash and recovery — 1 crashed + 1 completed](/img/kanban-tutorial/06-drawer-crash-recovery.png)

Run 1 — `crashed`, com erro `OOM kill at row 2.3M (process 99999 gone)`. Run 2 — `completed`, com `"strategy": "chunked with LIMIT + WHERE id > last_id"` em metadata. O worker retrying viu o crash da run 1 em seu contexto e escolheu estratégia mais segura; metadata deixa óbvio para observador futuro (ou postmortem writer) o que mudou.

## Structured handoff — por que `summary` e `metadata` importam {#structured-handoff--why-summary-and-metadata-matter}

Em toda história acima, workers chamaram `kanban_complete(summary=..., metadata=...)` no fim. Isso não é decoração — é o canal primário de handoff entre estágios de um workflow.

Quando um worker na tarefa B spawna e chama `kanban_show()`, o `worker_context` que recebe inclui:

- **Tentativas anteriores** de B (runs anteriores: outcome, summary, error, metadata) para worker retrying não repetir caminho falho.
- **Resultados de tarefas parent** — para cada parent, summary e metadata da run completed mais recente — para workers downstream verem por quê e como o trabalho upstream foi feito.

Isso substitui a dança "cavar comentários e output do trabalho" que assola sistemas kanban flat. Um PM escreve acceptance criteria no metadata da spec, e o worker engineer vê estruturalmente no parent handoff. Um engineer registra quais testes rodou e quantos passaram, e o worker reviewer tem essa lista em mãos antes de abrir diff.

O bulk-close guard existe porque estes dados são per-run. `hermes kanban complete a b c --summary X` (você, da CLI) é recusado — copy-paste do mesmo summary para três tarefas quase sempre está errado. Bulk close sem flags de handoff ainda funciona para o caso comum "terminei uma pilha de tarefas admin". A tool surface não expõe variante bulk; `kanban_complete` é sempre single-task-at-a-time pela mesma razão.

## Follow-up em um card done — remediação de CI via o link parent {#follow-up-on-a-done-card--ci-remediation-via-the-parent-link}

O card de implementação da História 1 está `done`. Duas horas depois o CI falha no branch mergeado. Não reabra o card done — cards completed são história, e o handoff flui para frente. Crie um card de remediação com o card done como **parent**:

```bash
hermes kanban create "Fix CI: test_backoff_jitter flakes on 3.11" \
    --assignee backend-dev \
    --parent t_impl \
    --workspace worktree --branch wt/ci-fix-backoff \
    --body "CI run #4812 failed after t_impl completed.
FAILED tests/test_retry.py::test_backoff_jitter - TimeoutError
Acceptance: tests/test_retry.py green on 3.11 and 3.12."
```

Três coisas fazem isso funcionar:

- **Dispatch imediato.** Como o parent já está `done`, o child é criado direto em `ready` — o dispatcher pode reivindicá-lo no próximo tick. (Um child de um parent ainda aberto esperaria em `todo`.)
- **Contexto herdado.** O contexto do worker de remediação inclui uma seção *Parent task results* carregando o summary e metadata de completion de `t_impl` — os arquivos mudados e decisões que o worker original registrou — então ele sabe por que o código está moldado assim antes de ler uma linha.
- **Evidência fresh no body.** O log de CI não existia quando `t_impl` completou, então não pode estar no handoff do parent — vai no body do card novo, junto com critérios de aceitação explícitos.

Prefira um worktree/branch fresh para o card de remediação. Checkout do branch original dá ao worker *estado* do repo mas nenhuma da *razão* — o handoff do parent carrega isso. O mesmo profile assignee geralmente está certo: o profile que escreveu o código tem as skills para consertá-lo.

## Inspecionando tarefa currently running {#inspecting-a-task-currently-running}

Para completude — aqui está o drawer de tarefa ainda in flight (implementação API da História 1, claimed por `backend-dev` mas ainda não complete):

![Tarefa claimed, in-flight](/img/kanban-tutorial/10-drawer-in-flight.png)

Status é `Running`. A run ativa aparece na seção Run History com outcome `active` e sem `ended_at`. Se este worker morrer ou timeout, o dispatcher fecha esta run com outcome apropriado e abre nova no próximo claim — a linha de tentativa nunca desaparece.

## Próximos passos {#next-steps}

- [Visão geral Kanban](./kanban) — modelo de dados completo, vocabulário de eventos e referência CLI.
- `hermes kanban --help` — todo subcomando, toda flag.
- `hermes kanban watch --kinds completed,gave_up,timed_out` — stream live de eventos terminais no board inteiro.
- `hermes kanban notify-subscribe <task> --platform telegram --chat-id <id>` — receba ping gateway quando tarefa específica terminar.
