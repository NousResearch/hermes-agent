---
sidebar_position: 16
title: "Metas persistentes"
description: "Defina uma meta contínua e deixe o Hermes continuar trabalhando entre turnos até concluir. Nossa versão do Ralph loop."
---

# Metas persistentes (`/goal`)

`/goal` dá ao Hermes um objetivo contínuo que sobrevive entre turnos. Após cada turno, um model judge leve verifica se a meta foi satisfeita pela última resposta do assistente. Se não, o Hermes alimenta automaticamente um prompt de continuação de volta na mesma sessão e continua trabalhando — até a meta ser alcançada, você pausar ou limpar, ou o orçamento de turnos acabar.

É nossa versão do **Ralph loop**, inspirada diretamente no [`/goal` do Codex CLI 0.128.0](https://github.com/openai/codex) de Eric Traut (OpenAI). A ideia central — manter uma meta viva entre turnos e não parar até alcançá-la — é deles. A implementação aqui é independente e adaptada à arquitetura do Hermes.

## Quando usar {#when-to-use-it}

Use `/goal` para tarefas em que você quer que o Hermes itere sozinho sem você re-promptar a cada turno:

- "Corrija todo erro de lint em `src/` e verifique que `ruff check` passa"
- "Porte a feature X do repo Y, incluindo testes, e deixe o CI verde"
- "Investigue por que session IDs às vezes driftam na compressão mid-run e escreva um relatório"
- "Construa um CLI pequeno para renomear arquivos pelas datas EXIF, depois teste contra a pasta photos/"

Tarefas em que o agente faz um turno e para não precisam de `/goal`. Tarefas em que *você teria que dizer "continua" três vezes* são onde isso brilha.

## Metas vs Kanban: qual eu quero? {#goals-vs-kanban-which-one-do-i-want}

`/goal` e [Kanban](./kanban) mantêm o Hermes trabalhando sem você re-promptar, então é tentador assumir que um flui para o outro. Não flui — a fronteira é nítida:

- **`/goal` é single-session.** O loop alimenta prompts de continuação de volta a *esta* conversa até o judge dizer pronto. Definir uma meta nunca cria um card kanban, nunca atribui trabalho a outro profile e nunca faz fan-out. Não há handoff para o board, implícito ou não.
- **Kanban é um board de muitas tarefas.** Cada card é despachado para seu próprio processo worker com sua própria sessão. Cards, dependências, assignees e handoffs vivem no board — não em `/goal`.
- **A sobreposição é deliberada, e pequena.** Um card kanban criado com `--goal` roda o mesmo motor de continuação estilo Ralph que `/goal` — mas *dentro da sessão worker daquele card*. Ele empresta o motor, não o board. Veja [Cards em modo goal](./kanban#goal-mode-cards---goal).

| Você quer | Use |
|---|---|
| Continuar iterando em uma tarefa neste chat até terminar | `/goal <texto>` |
| Muitas tarefas independentes, com dependências, handoffs ou múltiplos profiles | [Kanban](./kanban) — `hermes kanban create …` |
| Um card no board que deve continuar iterando até os critérios de aceitação | Um card kanban com `--goal` |

:::note
Se você quer trabalho no board, coloque lá você mesmo (`hermes kanban create …`) — `/goal` não faz isso por você. O inverso também é verdade: pausar, retomar ou limpar uma meta neste chat nunca cria, reivindica ou move um card kanban.
:::

## Início rápido {#quick-start}

```
/goal Fix every failing test in tests/hermes_cli/ and make sure scripts/run_tests.sh passes for that directory
```

O que você verá:

1. **Meta aceita** — `⊙ Goal set (20-turn budget): <sua meta>`
2. **Turno 1 roda** — o Hermes começa a trabalhar como se você tivesse enviado a meta como mensagem normal.
3. **Judge roda** — após o turno, o model judge decide `done` ou `continue`.
4. **Loop dispara se necessário** — se `continue`, você verá `↻ Continuing toward goal (1/20): <motivo do judge>` e o Hermes dá o próximo passo automaticamente.
5. **Termina** — eventualmente você vê `✓ Goal achieved: <motivo>` ou `⏸ Goal paused — N/20 turns used`.

## Comandos {#commands}

| Comando | O que faz |
|---|---|
| `/goal <texto>` | Define (ou substitui) a meta contínua. Dispara o primeiro turno imediatamente para você não precisar enviar mensagem separada. |
| `/goal draft <texto>` | Rascunha um contrato de conclusão estruturado a partir de um objetivo em linguagem natural, depois o define. Veja [Contratos de conclusão](#completion-contracts). |
| `/goal show` | Imprime o contrato de conclusão da meta ativa. |
| `/goal` ou `/goal status` | Mostra a meta atual, status e turnos usados. |
| `/goal pause` | Para o loop de auto-continuação sem limpar a meta. |
| `/goal resume` | Retoma o loop (reseta o contador de turnos para zero). |
| `/goal clear` | Remove a meta por completo. |
| `/goal wait <pid> [motivo]` | Estaciona o loop em um processo em background — para de cutucar o agente a cada turno enquanto o processo roda, e retoma automaticamente quando ele termina. |
| `/goal unwait` | Remove a barreira de wait e retoma o loop imediatamente. |
| `/goal gate add <comando>` | Adiciona um **quality gate**: um comando shell que deve passar antes da meta poder ser julgada como concluída. Veja [Quality gates](#quality-gates). |
| `/goal gate` ou `/goal gate list` | Lista os gates da meta e seu estado pass/fail. |
| `/goal gate remove <N>` | Remove o N-ésimo gate (base 1). |
| `/goal gate clear` | Remove todos os gates. |

Funciona de forma idêntica no CLI e em toda plataforma gateway (Telegram, Discord, Slack, Matrix, Signal, WhatsApp, SMS, iMessage, Webhook, API server e o dashboard web).

## Contratos de conclusão {#completion-contracts}

Um `/goal <texto>` simples funciona bem, mas uma meta *vaga* gera julgamento vago — o judge só pode verificar o que você disse que quer. A orientação do `/goal` do Codex faz o mesmo ponto: um objetivo durável funciona melhor quando nomeia **o que "pronto" significa, como provar, o que não quebrar, o que está no escopo e quando parar**. O Hermes adapta isso como um **contrato de conclusão** opcional sobre o loop de meta existente.

Um contrato tem cinco campos, todos opcionais:

| Campo | Significado |
|---|---|
| `outcome` | O único estado final que deve ser verdadeiro quando concluído. |
| `verification` | O teste / comando / artefato específico que *prova* o outcome. |
| `constraints` | O que não deve mudar ou regredir. |
| `boundaries` | Quais arquivos, dirs, ferramentas ou sistemas estão no escopo. |
| `stop_when` | A condição em que o Hermes deve parar e pedir input. |

Quando um contrato está definido, ambos os prompts mudam: o **prompt de continuação** diz ao agente para mirar na superfície de verificação e respeitar as constraints, e o **prompt do judge** decide `done` *somente quando o critério de verificação é atendido com evidência concreta* (resultado de comando, trecho de arquivo, saída de teste) — não uma afirmação solta de "parece pronto". Isso aperta diretamente o failure mode mais comum de `/goal` (conclusão prematura ou over-continuação infinita em objetivo mal especificado).

### Duas formas de definir um contrato {#two-ways-to-set-a-contract}

**1. Deixe o Hermes rascunhar** (recomendado — adaptado da dica do Codex "deixe o agente rascunhar a meta"):

```
/goal draft Migrate the auth service from session cookies to JWT
```

O Hermes expande seu one-liner em um contrato completo via model auxiliar `goal_judge`, define e mostra o resultado para você revisar ou apertar qualquer campo. Se o model aux estiver indisponível, cai para meta free-form simples — rascunhar nunca bloqueia definir uma meta.

**2. Escreva inline** com linhas `campo: valor`:

```
/goal Migrate auth to JWT
verify: pytest tests/auth passes
constraints: keep the /login response shape unchanged
boundaries: only touch services/auth and its tests
stop when: a DB schema migration is required
```

A(s) primeira(s) linha(s) que não são campo são o headline da meta; prefixos de campo reconhecidos (`verify:`, `verified by:`, `constraints:`, `preserve:`, `boundaries:`, `scope:`, `stop when:`, `blocked:`, …) populam o contrato. Uma meta simples com dois-pontos incidental (`Fix bug: the parser drops commas`) **não** é estragada — só prefixos de campo conhecidos são extraídos.

Use `/goal show` para revisar o contrato ativo. Contratos persistem em `SessionDB.state_meta` junto com a meta, então sobrevivem a `/resume`. Metas antigas de antes desta feature carregam inalteradas (sem contrato). Contratos e critérios de `/subgoal` compõem: subgoals entram no contrato como critérios extras que o judge também deve satisfazer.

## Adicionar critérios mid-goal: `/subgoal` {#adding-criteria-mid-goal-subgoal}

Com uma meta ativa, você pode anexar critérios de aceitação extras com `/subgoal <texto>` sem resetar o loop. Cada chamada adiciona um item numerado à lista de subgoals da meta; o **prompt de continuação** que o agente vê no próximo turno inclui a meta original mais um bloco "Additional criteria the user added mid-loop", e o **prompt do judge** é reescrito para o veredicto considerar todo subgoal — a meta não é marcada como done até o objetivo original **e** todo subgoal serem atendidos.

| Comando | O que faz |
|---|---|
| `/subgoal <texto>` | Anexa um novo critério à meta ativa. Requer um `/goal` ativo. |
| `/subgoal` (sem args) | Mostra a lista numerada de subgoals atual. |
| `/subgoal remove <N>` | Remove o N-ésimo subgoal (base 1). |
| `/subgoal clear` | Remove todo subgoal mas mantém a meta original intacta. |

Subgoals persistem junto com a meta em `SessionDB.state_meta`, então sobrevivem a `/resume`. Definir um novo `/goal <texto>` substitui a meta e limpa a lista de subgoals; `/goal clear` faz o mesmo.

Use quando você inicia um loop ("corrija os testes falhando") e percebe no meio que também quer "e adicione um teste de regressão para o bug que acabou de corrigir" — `/subgoal add a regression test` aperta os critérios de sucesso sem quebrar o loop em execução.

## Quality gates {#quality-gates}

Um contrato de conclusão deixa o judge mais rigoroso, mas o judge ainda é um LLM lendo prosa. Um **quality gate** é mais forte: um comando shell determinístico que deve sair com código 0 antes da meta poder concluir de fato. Inspirado no modo autônomo limitado do Prime-Agent (`--autonomous-gate`).

```
/goal Fix the flaky session tests
/goal gate add scripts/run_tests.sh tests/hermes_cli/test_goals.py
```

Como funciona, a cada turno:

1. **Gates rodam antes do judge.** Se algum gate falhar, o judge *não* é chamado — um gate vermelho é evidência determinística de que a meta não está pronta. O exit code e a cauda de saída do gate (últimos ~3 KB) viram o prompt de continuação, então o agente itera contra a falha real em vez de um "feeling".
2. **Todos os gates passam → julgamento normal.** O judge LLM então decide done/continue/wait exatamente como antes.
3. **Workspace inalterado → sem re-run.** Se um gate falhou e nada mudou no workspace desde então (rastreado via fingerprint git de HEAD + status da working tree), o gate não é reexecutado — a falha registrada é reproduzida e o contador de tentativas avança. Um agente preso não pode queimar wall-clock reexecutando a mesma suite vermelha idêntica. Fora de um repo git, gates simplesmente sempre reexecutam.
4. **Retries são limitados.** Cada gate tem por padrão 3 retries e timeout de 5 minutos. Quando um gate esgota retries, a meta auto-pausa (como o orçamento de turnos) com mensagem dizendo para corrigir manualmente, remover o gate ou `/goal resume`.

Gates persistem com a meta em `SessionDB.state_meta` (sobrevivem a `/resume` e compressão de contexto), e gerenciamento de gates (`/goal gate …`) é seguro mid-run no gateway — gates só rodam no limite de turno.

Gates e contratos compõem: use um contrato para moldar *no que o agente mira*, e gates para tornar *"pronto"* mecanicamente verificável. Quando ambos estão definidos, gates rodam primeiro.

## Estacionar em processo em background: automático, com override manual {#parking-on-a-background-process-automatic-with-a-manual-override}

Algumas metas dependem de algo que leva minutos e roda sozinho — CI em um PR pushed, build longo, matriz de testes, deploy, cooldown de rate limit. Sem ajuda, o loop de meta cutucaria o agente a cada turno com busy-work de "já terminou?" enquanto espera.

**Isso é tratado automaticamente.** A cada turno, o judge vê os processos em background vivos do agente (o registro de `terminal(background=true)` — pid, session id, comando, uptime, saída recente e qualquer trigger `watch_patterns` / `notify_on_complete`) junto com a meta e a resposta do agente. Quando o progresso do agente está genuinamente condicionado a um deles, o judge retorna veredicto **`wait`** em vez de `continue`, e o loop **estaciona**: os próximos turnos são pulados (sem chamada ao judge, sem continuação, sem turno consumido) até o wait ser satisfeito — depois retoma normalmente com o resultado em mãos. O judge também pode estacionar por **tempo** (`wait_for_seconds`) para waits de backoff/cooldown. `/goal status` mostra `⏳ Goal (parked …)` enquanto estacionado.

O judge escolhe o tipo certo de wait a partir do sinal do processo:

- **`wait_on_session <id>`** — libera quando o *próprio trigger* do processo dispara: ele termina, **ou** (se foi iniciado com `watch_patterns`) seu pattern combina. Este é para um watcher/servidor/poller de longa duração que sinaliza **mid-run** (ex.: processo de build que imprime `BUILD SUCCESSFUL` e continua rodando, ou watcher `notify_on_complete`) e pode nunca terminar sozinho.
- **`wait_on_pid <pid>`** — libera só na saída do processo.
- **`wait_for_seconds <n>`** — libera após atraso fixo.

Você não digita nada para isso — é decisão do judge, feita a partir do contexto de processo que o loop entrega. Os comandos manuais existem como override:

| Comando | O que faz |
|---|---|
| `/goal wait <pid> [motivo]` | Estaciona manualmente o loop até o processo com aquele PID terminar. |
| `/goal unwait` | Limpa qualquer barreira de wait (do judge ou manual) e retoma imediatamente. |

A barreira (por pid ou tempo) persiste com a meta em `SessionDB.state_meta`, então sobrevive a `/resume`. `/goal pause`, `/goal resume` e `/goal clear` a removem. Se o PID já estiver morto quando a barreira é definida (ou morrer enquanto estacionado), ou o deadline de tempo passar, a barreira limpa na próxima checagem — uma barreira stale nunca pode travar o loop.

Fluxo típico: o agente faz push de um PR, inicia um watcher de CI com `terminal(background=true, notify_on_complete=true)` e reporta "watching CI." O judge vê o watcher ainda rodando, retorna `wait` no pid dele, e o loop fica quieto — depois retoma no instante em que o CI termina e julga a meta contra o resultado real.

## Detalhes de comportamento {#behavior-details}

### O judge {#the-judge}

Após cada turno, o Hermes chama um model auxiliar com:

- O texto da meta contínua
- A resposta final mais recente do agente (últimos ~4 KB de texto)
- Um system prompt dizendo ao judge para responder com JSON estrito de uma linha: `{"verdict": "done" | "continue" | "wait", "reason": "<racional em uma frase>"}` (veredictos wait adicionam `wait_on_session` / `wait_on_pid` / `wait_for_seconds`; a forma legada `{"done": <bool>, "reason": "..."}` ainda é aceita)

O judge é deliberadamente conservador: marca uma meta como `done` só quando a resposta **explicitamente** confirma que a meta está completa, quando o deliverable final está claramente produzido, ou quando a meta é inalcançável/bloqueada (tratada como DONE com motivo de bloqueio para não queimar orçamento em tarefas impossíveis).

### Semântica fail-open {#fail-open-semantics}

Se o judge der erro (falha de rede, resposta malformada, client aux indisponível), o Hermes trata o veredicto como `continue` — um judge quebrado nunca trava o progresso. O **orçamento de turnos** é o backstop real.

### Orçamento de turnos {#turn-budget}

O padrão é 20 turnos de continuação (`goals.max_turns` em `config.yaml`). Quando o orçamento acaba, o Hermes auto-pausa e diz exatamente como proceder:

```
⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going, or /goal clear to stop.
```

`/goal resume` reseta o contador para zero, então você pode continuar em blocos medidos.

### Mensagens do usuário sempre preemptam {#user-messages-always-preempt}

Qualquer mensagem real que você enviar com uma meta ativa tem prioridade sobre o loop de continuação. No CLI sua mensagem cai em `_pending_input` à frente da continuação enfileirada; no gateway passa pelo FIFO do adapter da mesma forma. O judge roda de novo após seu turno — então se sua mensagem completar a meta por acaso, o judge captura e para.

### Segurança mid-run (gateway) {#mid-run-safety-gateway}

Enquanto um agente já está rodando, `/goal status`, `/goal pause`, `/goal clear`, `/goal wait` e `/goal unwait` são seguros — só tocam estado do plano de controle e não interrompem o turno atual. Definir uma **nova** meta mid-run (`/goal <novo texto>`) é rejeitado com mensagem pedindo `/stop` primeiro, para a continuação antiga não correr contra a nova.

### Persistência {#persistence}

O estado da meta vive em `SessionDB.state_meta` keyed por `goal:<session_id>`. Isso significa que `/resume` retoma exatamente de onde parou — defina uma meta, feche o laptop, volte amanhã, `/resume`, e a meta ainda está de pé exatamente como você deixou (ativa, pausada ou concluída).

### Prompt cache {#prompt-cache}

O prompt de continuação é uma mensagem plain user-role anexada ao histórico. Ele **não** muta o system prompt, troca toolsets ou toca a conversa de forma que invalide o prompt cache do Hermes. Rodar uma meta de 20 turnos custa o mesmo em cache que 20 turnos de conversa normal.

## Configuração {#configuration}

Adicione em `~/.hermes/config.yaml`:

```yaml
goals:
  # Máximo de turnos de continuação antes do Hermes auto-pausar e pedir
  # /goal resume. Padrão 20. Diminua para loops mais apertados;
  # aumente para refactors longos.
  max_turns: 20
```

### Escolhendo o model judge {#choosing-the-judge-model}

O judge usa a tarefa auxiliar `goal_judge`. Por padrão resolve para seu model principal (veja [Models auxiliares](/user-guide/configuration#auxiliary-models)). Se quiser rotear o judge para um model barato e rápido para reduzir custo, adicione um override:

```yaml
auxiliary:
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

A chamada do judge é pequena (~200 tokens de saída) e roda uma vez por turno, então um model barato e rápido costuma ser a escolha certa.

## Walkthrough de exemplo {#example-walkthrough}

```
You: /goal Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

  ⊙ Goal set (20-turn budget): Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

Hermes: Creating /tmp/note_1.txt now.
  💻 echo "1" > /tmp/note_1.txt   (0.1s)
  I've created /tmp/note_1.txt with the content "1". I'll continue with the remaining files on the next turn as you specified.

  ↻ Continuing toward goal (1/20): Only 1 of 4 files has been created; 3 files remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "2" > /tmp/note_2.txt   (0.1s)
  Created /tmp/note_2.txt. Two more to go.

  ↻ Continuing toward goal (2/20): 2 of 4 files created; 2 remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "3" > /tmp/note_3.txt   (0.1s)
  Created /tmp/note_3.txt.

  ↻ Continuing toward goal (3/20): 3 of 4 files created; 1 remains.

Hermes: [Continuing toward your standing goal]
  💻 echo "4" > /tmp/note_4.txt   (0.1s)
  All four files have been created: /tmp/note_1.txt through /tmp/note_4.txt, each containing its number.

  ✓ Goal achieved: All four files were created with the specified content, completing the goal.

You: _
```

Quatro turnos, uma invocação de `/goal`, zero prompts "continua" seus.

## Quando o judge erra {#when-the-judge-gets-it-wrong}

Nenhum judge é perfeito. Dois failure modes para observar:

**Falso negativo — judge diz continue quando a meta já está pronta.** O orçamento de turnos captura isso. Você verá `⏸ Goal paused` e pode `/goal clear` ou só enviar uma nova mensagem.

**Falso positivo — judge diz done quando ainda falta trabalho.** Você verá `✓ Goal achieved` mas sabe que não. Envie mensagem de follow-up para continuar, ou redefina a meta com mais precisão: `/goal <texto mais específico>`. O system prompt do judge é deliberadamente conservador para falsos positivos serem mais raros que falsos negativos.

Se achar um veredicto do judge pouco convincente, o texto de motivo na linha `↻ Continuing toward goal` ou `✓ Goal achieved` diz exatamente o que o judge viu. Isso costuma bastar para diagnosticar se o texto da meta era ambíguo ou a resposta do model.

## Atribuição {#attribution}

`/goal` é a versão do Hermes do padrão **Ralph loop**. O design voltado ao usuário — manter uma meta viva entre turnos, não parar até alcançá-la, com controles create/pause/resume/clear — foi popularizado e lançado no [Codex CLI 0.128.0](https://github.com/openai/codex) por Eric Traut no time Codex da OpenAI. Nossa implementação é independente (registro central `CommandDef`, persistência `SessionDB.state_meta`, judge via auxiliary client, continuação adapter-FIFO no lado gateway), mas a ideia é deles. Crédito onde é devido.
