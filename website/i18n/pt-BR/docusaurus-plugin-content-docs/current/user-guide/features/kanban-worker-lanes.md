# Lanes de worker Kanban {#kanban-worker-lanes}

Uma **worker lane** é uma classe de processo para a qual o dispatcher kanban pode rotear tarefas. Cada lane tem uma identidade (a string assignee), um mecanismo de spawn e um contrato sobre o que deve fazer com a tarefa depois de spawnada.

Esta página é o contrato. Ela existe para dois públicos:

- **Operadores** escolhendo quais lanes conectar a um board (quais profiles criar, quais assignees usar).
- **Autores de plugins / integrações** que querem adicionar uma nova forma de lane (um worker CLI que envolve Codex / Claude Code / OpenCode, um worker de review containerizado, um serviço não-Hermes que puxa tarefas via API).

Se você está escrevendo o código do worker em si — o agente que roda *dentro* de uma lane — o ciclo de vida kanban e os detalhes de referência são injetados automaticamente no system prompt do worker (o bloco `KANBAN_GUIDANCE` em [`agent/prompt_builder.py`](https://github.com/NousResearch/hermes-agent/blob/main/agent/prompt_builder.py)).

## A hierarquia {#the-hierarchy}

```text
Hermes Kanban  =  ciclo de vida canônico de tarefa + trilha de auditoria
Worker lane    =  executor de implementação para um card atribuído
Reviewer       =  humano ou proxy humano que controla "done"
GitHub PR      =  artefato upstreamável (opcional, para lanes de código)
```

O Hermes Kanban possui a verdade do ciclo de vida — `ready` → `running` → `review` / `blocked` / `done` / `archived`. Worker lanes executam o trabalho, mas nunca possuem essa verdade; tudo que fazem volta pelo kernel kanban via ferramentas `kanban_*` (ou, para workers externos não-Hermes, via API). Reviewers controlam a transição de "mudança de código escrita" para "tarefa done."

## O que uma lane fornece {#what-a-lane-provides}

Para ser uma worker lane kanban, uma integração deve fornecer três coisas:

### 1. Uma string assignee {#1-an-assignee-string}

O dispatcher combina `task.assignee` com um nome de profile Hermes (a forma de lane padrão) ou um identificador registrado não spawnável (a forma de lane plugin — veja [Adicionando uma worker lane CLI externa](#adding-an-external-cli-worker-lane) abaixo). Tarefas cujo assignee não resolve ficam em `ready` com um evento `skipped_nonspawnable` para que um operador do board possa corrigi-las; não são silenciosamente descartadas nem executadas por um fallback arbitrário.

### 2. Um mecanismo de spawn {#2-a-spawn-mechanism}

Para lanes de profile Hermes, o `_default_spawn` do dispatcher roda `hermes -p <assignee> chat -q <prompt>` (ou a forma equivalente em módulo quando o shim `hermes` não está no `$PATH`) dentro do workspace fixado da tarefa, com estas variáveis de ambiente definidas:

| Variável | Carrega |
|---|---|
| `HERMES_KANBAN_TASK` | o id da tarefa em que o worker opera |
| `HERMES_KANBAN_DB` | path absoluto para o arquivo SQLite por board |
| `HERMES_KANBAN_BOARD` | slug do board |
| `HERMES_KANBAN_WORKSPACES_ROOT` | raiz da árvore de workspaces do board |
| `HERMES_KANBAN_WORKSPACE` | path absoluto para o workspace *desta* tarefa |
| `HERMES_KANBAN_RUN_ID` | o id da run atual (para o lifecycle gate) |
| `HERMES_KANBAN_CLAIM_LOCK` | a string de lock de claim (`<host>:<pid>:<uuid>`) |
| `HERMES_PROFILE` | o nome do profile do worker (para atribuição de autor em `kanban_comment`) |
| `HERMES_TENANT` | namespace tenant, se a tarefa tiver um |

Para lanes não-Hermes (registradas via plugin), o plugin fornece seu próprio callable `spawn_fn` que recebe `task`, `workspace` e `board` e retorna um pid opcional para detecção de crash.

### 3. Um terminador de ciclo de vida {#3-a-lifecycle-terminator}

Todo claim deve terminar em exatamente um de:

- `kanban_complete(summary=..., metadata=...)` — tarefa bem-sucedida, status vira `done`.
- `kanban_request_review(summary=..., metadata=..., reviewer=...)` — a implementação same-card está completa e entra em review de primeira classe; o status vira `review`. O dispatcher carrega a skill bundled `sdlc-review` a menos que `kanban.review_dispatch` esteja desabilitado. Um reviewer aprova com `kanban_complete`, devolve rework acionável com `kanban_request_changes`, ou escala um blocker externo genuíno com `kanban_block`.
- `kanban_block(reason=...)` — tarefa aguarda input humano, status vira `blocked`. O dispatcher respawna quando `kanban_unblock` roda.
- O processo worker sai sem chamada de ferramenta. O kernel o colhe e emite `crashed` (PID morreu) ou `gave_up` (circuit breaker de falhas consecutivas disparou) ou `timed_out` (max_runtime excedido). Este é o caminho de falha; workers saudáveis não terminam aqui.

O kernel kanban garante que exatamente um destes termina cada run. Um worker que não chama nenhum e sai normalmente é tratado como crashed.

## Saídas e o handoff de review {#outputs-and-the-review-handoff}

Para tarefas que alteram código, escolha o modelo de review encodado pelo grafo de tarefas:

- **Review same-card:** chame `kanban_request_review(summary=..., metadata=..., reviewer=...)`. A tarefa entra em `review` sem tocar a contabilidade de recorrência de block. O dispatcher faz claim com a skill bundled `sdlc-review` por padrão. O reviewer aprova com `kanban_complete`, chama `kanban_request_changes(reason=...)` para fechar a run de review e devolver a tarefa ao implementer original, ou bloqueia só para uma escalação externa genuína.
- **Card downstream de review/QA/release pré-criado:** `kanban_show` lista IDs filhos; inspecione esses cards com `kanban_show(task_id=...)` antes de escolher a ação terminal. Quando um filho é a fase downstream de review/QA/release, chame `kanban_complete` na fase de implementação. Ele não pode promover até este parent estar `done`/`archived`. Não peça review same-card adicional e nunca faça sticky-block do parent com `review-required:` — qualquer uma das duas escolhas trava ou duplica a lane downstream.
- **Boards só-humano:** defina `kanban.review_dispatch: false`. Uma tarefa pode então permanecer em `review` até um humano aprová-la ou usar `reopen-review`/o dashboard para devolvê-la a `ready`/`todo`.

Ambos os modelos de review carregam o handoff estruturado na própria transição de ciclo de vida. Não coloque secrets, tokens ou PII bruto em `summary` ou `metadata`; linhas de run são duráveis.

O `KANBAN_GUIDANCE` injetado cobre ambas as formas de grafo, `kanban_complete`, o loop de review same-card, e `kanban_block` para blockers genuínos.

## Logs e trilha de auditoria {#logs-and-audit-trail}

O dispatcher grava stdout/stderr do worker por tarefa em `<board-root>/logs/<task_id>.log`. Logs são auditáveis a partir de metadata kanban:

- Linhas `task_runs` carregam `log_path`, código de saída (quando disponível), summary e metadata.
- Linhas `task_events` carregam toda transição de estado (`promoted`, `claimed`, `heartbeat`, `completed`, `blocked`, `review_requested`, `changes_requested`, `review_reopened`, `gave_up`, `crashed`, `timed_out`, `reclaimed`, `claim_extended`).
- `kanban_show` retorna ambos, então um revisor (ou um worker de follow-up) lendo a tarefa obtém o histórico completo sem precisar de acesso ao dashboard.

O dashboard renderiza histórico de runs com summaries, blocos de metadata e badges de status de saída. Usuários CLI podem rodar `hermes kanban tail <task_id>` para acompanhar ao vivo, ou `hermes kanban runs <task_id>` para a lista histórica de tentativas.

## Formas de lane existentes {#existing-lane-shapes}

### Lane de profile Hermes (padrão) {#hermes-profile-lane-default}

A forma que todo worker kanban assume hoje: o assignee é um nome de profile, o dispatcher spawna `hermes -p <profile>`, o worker recebe o bloco `KANBAN_GUIDANCE` no system prompt injetado automaticamente, e usa as ferramentas `kanban_*` para terminar a run. Nenhuma configuração além de definir o profile.

Quando você cria profiles para sua frota, escolha nomes que correspondam ao *papel* para o qual quer que o orchestrator roteie. O orchestrator (quando existe) descobre seus nomes de profile via `hermes profile list` — não há roster fixo que o sistema assume (o lado orchestrator do contrato faz parte do `KANBAN_GUIDANCE` injetado).

### Lane de profile orchestrator {#orchestrator-profile-lane}

Uma especialização da lane de profile: um orchestrator é um profile Hermes cujo toolset inclui `kanban` mas exclui `terminal` / `file` / `code` / `web` para implementação. Seu trabalho é decompor um objetivo de alto nível em tarefas filhas via `kanban_create` + `kanban_link` e recuar. A skill orchestrator codifica as regras anti-tentação.

## Adicionando uma worker lane CLI externa {#adding-an-external-cli-worker-lane}

Conectar uma ferramenta CLI não-Hermes (Codex CLI, Claude Code CLI, OpenCode CLI, um runner local de modelo de código, etc.) como worker lane kanban *ainda não é um caminho pavimentado*. A função de spawn do dispatcher é plugável (`spawn_fn` é um parâmetro em `dispatch_once`), e um plugin poderia registrar seu próprio `spawn_fn` para um assignee não-Hermes, mas o trabalho de integração ao redor — envolver o código de saída da CLI em chamadas `kanban_complete` / `kanban_block`, mapear convenções de workspace/sandbox da CLI no `HERMES_KANBAN_WORKSPACE` env do dispatcher, lidar com auth e política por CLI — ainda é trabalho de design por integração.

Se você está considerando adicionar uma lane CLI, abra uma issue descrevendo a CLI específica e o fluxo que quer habilitar. O contrato acima são as restrições que qualquer lane desse tipo deve satisfazer; a forma de implementação (um plugin por CLI vs um plugin genérico runner-CLI parametrizado por config) está em aberto.

A issue histórica para isso é [#19931](https://github.com/NousResearch/hermes-agent/issues/19931) e o PR Codex-específico fechado-não-mergeado [#19924](https://github.com/NousResearch/hermes-agent/pull/19924) — descrevem a proposta de arquitetura original, mas não entregaram um runner.

## Modos de falha que o dispatcher trata {#failure-modes-the-dispatcher-handles}

Para que autores de lane não precisem reimplementá-los:

- **TTL de claim stale** — um worker que faz claim e nunca heartbeat / complete / block é reclaimado após `DEFAULT_CLAIM_TTL_SECONDS` (15 min padrão) — mas só se o processo worker realmente morreu. Um worker vivo (modelo lento gastando 20+ min em uma chamada LLM sem ferramentas) tem o claim *estendido* em vez de morto; só um PID morto é reclaimado.
- **Worker crashed** — um worker cujo PID host-local desapareceu é detectado por `detect_crashed_workers` e colhido; a tarefa incrementa `consecutive_failures` e pode auto-block quando o breaker dispara.
- **Retry em nível de run** — quando uma tarefa é retentada (pós-block, pós-crash, pós-reclaim), o worker pode usar o parâmetro `expected_run_id` em ferramentas terminadoras para falhar rápido se sua própria run já foi substituída.
- **Max runtime por tarefa** — `task.max_runtime_seconds` limita duro o tempo de parede por run, independente da vivacidade do PID. Pega workers genuinamente deadlocked que a extensão de PID vivo manteria rodando.
- **Detecção de tarefa stranded** — uma tarefa ready cujo assignee nunca produz claim dentro de `kanban.stranded_threshold_seconds` (padrão 30 min) aparece em `hermes kanban diagnostics` como aviso `stranded_in_ready`. A severidade escala para error em 2× o threshold e critical em 6×. Pega assignees com typo, profiles deletados e pools de worker externo down em um sinal — agnóstico de identidade, sem allowlist por board para curar.
- **Deadlock legado de dependência de review** — um parent sticky-blocked com `review-required:` enquanto um ou mais filhos diretos continuam gated por dependência em `todo` produz um error imediato `review_dependency_deadlock`. O diagnóstico é read-only: sugere completar a fase concluída ou unsinkar a aresta incorreta, mas nunca remove um block do usuário automaticamente.

## Relacionados {#related}

- [Visão geral Kanban](./kanban) — a introdução voltada ao usuário.
- [Tutorial Kanban](./kanban-tutorial) — walkthrough com o dashboard aberto.
- [`KANBAN_GUIDANCE`](https://github.com/NousResearch/hermes-agent/blob/main/agent/prompt_builder.py) — o ciclo de vida worker + orchestrator injetado no system prompt de todo worker kanban.
