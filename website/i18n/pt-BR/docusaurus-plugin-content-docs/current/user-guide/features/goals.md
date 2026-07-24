---
sidebar_position: 16
title: "Objetivos persistentes"
description: "Defina um objetivo permanente e deixe o Hermes continuar trabalhando entre turns até concluir. Nossa versão do loop Ralph."
---

# Objetivos persistentes (`/goal`) {#persistent-goals-goal}

`/goal` dá ao Hermes um objetivo permanente que sobrevive entre turns. Após cada turn, um model judge leve verifica se o objetivo foi satisfeito pela última resposta do assistant. Se não, o Hermes alimenta automaticamente um prompt de continuação de volta na mesma sessão e continua trabalhando — até o objetivo ser alcançado, você pausar ou limpar, ou o orçamento de turns acabar.

É nossa versão do **loop Ralph**, inspirada diretamente no [`/goal` do Codex CLI 0.128.0](https://github.com/openai/codex) de Eric Traut (OpenAI). A ideia central — manter um objetivo vivo entre turns e não parar até alcançá-lo — é deles. A implementação aqui é independente e adaptada à arquitetura do Hermes.

## Quando usar {#when-to-use-it}

Use `/goal` para tarefas em que quer que o Hermes itere por conta própria sem você re-promptar a cada turn:

- "Corrija todo erro de lint em `src/` e verifique que `ruff check` passa"
- "Porte a feature X do repo Y, incluindo testes, e deixe o CI verde"
- "Investigue por que IDs de sessão às vezes driftam na compressão mid-run e escreva um relatório"
- "Construa um CLI pequeno para renomear arquivos pelas datas EXIF, depois teste contra a pasta photos/"

Tarefas em que o agente faz um turn e para não precisam de `/goal`. Tarefas em que *você teria que dizer "continue" três vezes* são onde isso brilha.

## Início rápido {#quick-start}

```
/goal Fix every failing test in tests/hermes_cli/ and make sure scripts/run_tests.sh passes for that directory
```

O que você verá:

1. **Objetivo aceito** — `⊙ Goal set (20-turn budget): <seu objetivo>`
2. **Turn 1 roda** — Hermes começa a trabalhar como se você tivesse enviado o objetivo como mensagem normal.
3. **Judge roda** — após o turn, o model judge decide `done` ou `continue`.
4. **Loop dispara se necessário** — se `continue`, você verá `↻ Continuing toward goal (1/20): <razão do judge>` e o Hermes dá o próximo passo automaticamente.
5. **Termina** — eventualmente você vê `✓ Goal achieved: <reason>` ou `⏸ Goal paused — N/20 turns used`.

## Comandos {#commands}

| Comando | O que faz |
|---|---|
| `/goal <text>` | Define (ou substitui) o objetivo permanente. Dispara o primeiro turn imediatamente para você não precisar enviar mensagem separada. |
| `/goal draft <text>` | Rascunha um contrato de conclusão estruturado a partir de um objetivo em linguagem natural, depois o define. Veja [Contratos de conclusão](#completion-contracts). |
| `/goal show` | Imprime o contrato de conclusão do objetivo ativo. |
| `/goal` ou `/goal status` | Mostra o objetivo atual, status e turns usados. |
| `/goal pause` | Para o loop de auto-continuação sem limpar o objetivo. |
| `/goal resume` | Retoma o loop (reseta o contador de turns para zero). |
| `/goal clear` | Descarta o objetivo completamente. |
| `/goal wait <pid> [reason]` | Estaciona o loop em um processo em background — para de re-cutucar o agente a cada turn enquanto o processo roda, e retoma automaticamente quando ele sai. |
| `/goal unwait` | Remove a barreira de wait e retoma o loop imediatamente. |

Funciona de forma idêntica na CLI e em toda plataforma gateway (Telegram, Discord, Slack, Matrix, Signal, WhatsApp, SMS, iMessage, Webhook, API server e web dashboard).

## Contratos de conclusão {#completion-contracts}

Um `/goal <text>` simples funciona, mas um objetivo *vago* produz julgamento vago — o judge só pode verificar o que você disse que quer. A orientação `/goal` do Codex faz o mesmo ponto: um objetivo durável funciona melhor quando nomeia **o que done significa, como provar, o que não quebrar, o que está no escopo e quando parar**. O Hermes adapta isso como um **contrato de conclusão** opcional em camadas sobre o loop de objetivo existente.

Um contrato tem cinco campos, todos opcionais:

| Campo | Significado |
|---|---|
| `outcome` | O único estado final que deve ser verdadeiro quando done. |
| `verification` | O teste / comando / artefato específico que *prova* o outcome. |
| `constraints` | O que não deve mudar ou regredir. |
| `boundaries` | Quais arquivos, dirs, ferramentas ou sistemas estão no escopo. |
| `stop_when` | A condição sob a qual o Hermes deve parar e pedir input. |

Quando um contrato está definido, ambos os prompts mudam: o **prompt de continuação** diz ao agente para mirar na superfície de verificação e respeitar as constraints, e o **prompt do judge** decide `done` *somente quando o critério de verificação é atendido com evidência concreta* (resultado de comando, trecho de arquivo, saída de teste) — não uma afirmação solta de "parece done". Isso aperta diretamente o modo de falha `/goal` mais comum (conclusão prematura ou over-continuação sem fim em objetivo subespecificado).

### Duas formas de definir um contrato {#two-ways-to-set-a-contract}

**1. Deixe o Hermes rascunhar** (recomendado — adaptado da dica Codex "deixe o agente rascunhar o objetivo"):

```
/goal draft Migrate the auth service from session cookies to JWT
```

O Hermes expande seu one-liner em um contrato completo via model auxiliar `goal_judge`, define-o e mostra o resultado para você revisar ou apertar qualquer campo. Se o model aux estiver indisponível, cai para um objetivo free-form simples — rascunhar nunca bloqueia definir um objetivo.

**2. Escreva inline** com linhas `field: value`:

```
/goal Migrate auth to JWT
verify: pytest tests/auth passes
constraints: keep the /login response shape unchanged
boundaries: only touch services/auth and its tests
stop when: a DB schema migration is required
```

A(s) primeira(s) linha(s) não-field são o headline do objetivo; prefixos de campo reconhecidos (`verify:`, `verified by:`, `constraints:`, `preserve:`, `boundaries:`, `scope:`, `stop when:`, `blocked:`, …) populam o contrato. Um objetivo simples com dois-pontos incidentais (`Fix bug: the parser drops commas`) **não** é mangled — só prefixos de campo conhecidos são extraídos.

Use `/goal show` para revisar o contrato ativo. Contratos persistem em `SessionDB.state_meta` junto com o objetivo, então sobrevivem a `/resume`. Objetivos antigos de antes desta feature carregam inalterados (sem contrato). Contratos e critérios `/subgoal` compõem: subgoals entram no contrato como critérios extras que o judge também deve satisfazer.

## Adicionando critérios mid-goal: `/subgoal` {#adding-criteria-mid-goal-subgoal}

Enquanto um objetivo está ativo você pode acrescentar critérios de aceitação extras com `/subgoal <text>` sem resetar o loop. Cada chamada adiciona um item numerado à lista de subgoals do objetivo; o **prompt de continuação** que o agente vê no próximo turn inclui o objetivo original mais um bloco "Additional criteria the user added mid-loop", e o **prompt do judge** é reescrito para o veredito considerar todo subgoal — o objetivo não é marcado done até o objetivo original **e** todo subgoal serem atendidos.

| Comando | O que faz |
|---|---|
| `/subgoal <text>` | Acrescenta um novo critério ao objetivo ativo. Requer um `/goal` ativo. |
| `/subgoal` (sem args) | Mostra a lista numerada de subgoals atual. |
| `/subgoal remove <N>` | Remove o N-ésimo subgoal (base 1). |
| `/subgoal clear` | Descarta todo subgoal mas mantém o objetivo original intacto. |

Subgoals são persistidos junto com o objetivo em `SessionDB.state_meta`, então sobrevivem a `/resume`. Definir um novo `/goal <text>` substitui o objetivo e limpa a lista de subgoals; `/goal clear` faz o mesmo.

Use quando você inicia um loop ("fix the failing tests") e percebe no meio que também quer "and add a regression test for the bug you just patched" — `/subgoal add a regression test` aperta os critérios de sucesso sem quebrar o loop em execução.

## Estacionando em processo em background: automático, com override manual {#parking-on-a-background-process-automatic-with-a-manual-override}

Alguns objetivos dependem de algo que leva minutos e roda por conta própria — CI em um PR pushed, build longo, matriz de testes, deploy, cooldown de rate limit. Sem ajuda, o loop de objetivo re-cutucaria o agente a cada turn em busy-work "já terminou?" enquanto espera.

**Isso é tratado automaticamente.** A cada turn, o judge vê os processos em background vivos do agente (o registry `terminal(background=true)` — pid, session id, command, uptime, saída recente e qualquer trigger `watch_patterns` / `notify_on_complete`) junto com o objetivo e a resposta do agente. Quando o progresso do agente está genuinamente gated em um deles, o judge retorna veredito **`wait`** em vez de `continue`, e o loop **estaciona**: os próximos turns são pulados (sem chamada ao judge, sem continuação, sem turn consumido) até o wait ser satisfeito — depois retoma normalmente com o resultado em mãos. O judge também pode estacionar em base de **tempo** (`wait_for_seconds`) para waits de backoff/cooldown. `/goal status` mostra `⏳ Goal (parked …)` enquanto estacionado.

O judge escolhe o tipo certo de wait a partir do sinal do processo:

- **`wait_on_session <id>`** — libera quando o trigger *próprio* do processo dispara: ele sai, **ou** (se foi iniciado com `watch_patterns`) seu pattern combina. Este é o certo para watcher / server / poller long-lived que sinaliza **mid-run** (ex.: processo de build que imprime `BUILD SUCCESSFUL` e continua rodando, ou watcher `notify_on_complete`) e pode nunca sair por conta própria.
- **`wait_on_pid <pid>`** — libera só na saída do processo.
- **`wait_for_seconds <n>`** — libera após delay fixo.

Você não digita nada para isso — é decisão do judge, feita a partir do contexto de processo que o loop entrega. Os comandos manuais existem como override:

| Comando | O que faz |
|---|---|
| `/goal wait <pid> [reason]` | Estaciona manualmente o loop até o processo com esse PID sair. |
| `/goal unwait` | Limpa qualquer barreira de wait (judge- ou manualmente definida) e retoma imediatamente. |

A barreira (pid- ou time-based) é persistida com o objetivo em `SessionDB.state_meta`, então sobrevive a `/resume`. `/goal pause`, `/goal resume` e `/goal clear` a descartam. Se o PID já estiver morto quando a barreira é definida (ou morrer enquanto estacionado), ou o deadline de tempo passar, a barreira limpa na próxima checagem — uma barreira stale nunca pode travar o loop.

Fluxo típico: o agente faz push de um PR, inicia um watcher CI com `terminal(background=true, notify_on_complete=true)` e reporta "watching CI." O judge vê o processo watcher ainda rodando, retorna `wait` no pid, e o loop fica quieto — depois retoma no instante em que CI termina e julga o objetivo contra o resultado real.

## Detalhes de comportamento {#behavior-details}

### O judge {#the-judge}

Após cada turn, o Hermes chama um model auxiliar com:

- O texto do objetivo permanente
- A resposta final mais recente do agente (últimos ~4 KB de texto)
- Um system prompt dizendo ao judge para responder com JSON estrito: `{"done": <bool>, "reason": "<one-sentence rationale>"}`

O judge é deliberadamente conservador: marca um objetivo `done` só quando a resposta **explicitamente** confirma que o objetivo está completo, quando o entregável final está claramente produzido, ou quando o objetivo é inalcançável/blocked (tratado como DONE com razão de block para não queimar orçamento em tarefas impossíveis).

### Semântica fail-open {#fail-open-semantics}

Se o judge errar (blip de rede, resposta malformada, client aux indisponível), o Hermes trata o veredito como `continue` — um judge quebrado nunca trava progresso. O **orçamento de turns** é o backstop real.

### Orçamento de turns {#turn-budget}

O padrão é 20 turns de continuação (`goals.max_turns` em `config.yaml`). Quando o orçamento é atingido, o Hermes auto-pausa e diz exatamente como proceder:

```
⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going, or /goal clear to stop.
```

`/goal resume` reseta o contador para zero, então você pode continuar em blocos medidos.

### Mensagens do usuário sempre preemptam {#user-messages-always-preempt}

Qualquer mensagem real que você envia com um objetivo ativo tem prioridade sobre o loop de continuação. Na CLI sua mensagem cai em `_pending_input` à frente da continuação enfileirada; no gateway passa pelo FIFO do adapter da mesma forma. O judge roda de novo após seu turn — então se sua mensagem completar o objetivo por acidente, o judge pega e para.

### Segurança mid-run (gateway) {#mid-run-safety-gateway}

Enquanto um agente já está rodando, `/goal status`, `/goal pause`, `/goal clear`, `/goal wait` e `/goal unwait` são seguros — só tocam estado do plano de controle e não interrompem o turn atual. Definir um **novo** objetivo mid-run (`/goal <new text>`) é rejeitado com mensagem dizendo para `/stop` primeiro, para a continuação antiga não correr contra a nova.

### Persistência {#persistence}

Estado de objetivo vive em `SessionDB.state_meta` keyed por `goal:<session_id>`. Isso significa que `/resume` retoma exatamente de onde parou — defina um objetivo, feche o laptop, volte amanhã, `/resume`, e o objetivo ainda está de pé exatamente como você deixou (active, paused ou done).

### Prompt cache {#prompt-cache}

O prompt de continuação é uma mensagem plain user-role anexada ao histórico. Ele **não** muta o system prompt, troca toolsets ou toca a conversa de forma que invalide o prompt cache do Hermes. Rodar um objetivo de 20 turns custa o mesmo cache-wise que 20 turns de conversa normal.

## Configuração {#configuration}

Adicione em `~/.hermes/config.yaml`:

```yaml
goals:
  # Max continuation turns before Hermes auto-pauses and asks you to
  # /goal resume. Default 20. Lower this if you want tighter loops;
  # raise it for long-running refactors.
  max_turns: 20
```

### Escolhendo o model judge {#choosing-the-judge-model}

O judge usa a tarefa auxiliar `goal_judge`. Por padrão resolve para seu model principal (veja [Models auxiliares](/user-guide/configuration#auxiliary-models)). Se quiser rotear o judge para um model barato e rápido para manter custos baixos, adicione override:

```yaml
auxiliary:
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

A chamada ao judge é pequena (~200 output tokens) e roda uma vez por turn, então um model barato e rápido costuma ser a escolha certa.

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

Quatro turns, uma invocação `/goal`, zero prompts "continue" seus.

## Quando o judge erra {#when-the-judge-gets-it-wrong}

Nenhum judge é perfeito. Dois modos de falha para observar:

**Falso negativo — judge diz continue quando o objetivo está done.** O orçamento de turns pega isso. Você verá `⏸ Goal paused` e pode `/goal clear` ou só enviar nova mensagem.

**Falso positivo — judge diz done quando trabalho resta.** Você verá `✓ Goal achieved` mas sabe melhor. Envie mensagem de follow-up para continuar, ou redefina o objetivo com mais precisão: `/goal <more specific text>`. O system prompt do judge é deliberadamente conservador para falsos positivos serem mais raros que falsos negativos.

Se achar um veredito do judge pouco convincente, o texto reason na linha `↻ Continuing toward goal` ou `✓ Goal achieved` diz exatamente o que o judge viu. Isso costuma bastar para diagnosticar se o texto do objetivo era ambíguo ou a resposta do model era.

## Atribuição {#attribution}

`/goal` é a versão Hermes do padrão **loop Ralph**. O design voltado ao usuário — manter um objetivo vivo entre turns, não parar até alcançá-lo, com controles create/pause/resume/clear — foi popularizado e lançado no [Codex CLI 0.128.0](https://github.com/openai/codex) por Eric Traut no time Codex da OpenAI. Nossa implementação é independente (registry central `CommandDef`, persistência `SessionDB.state_meta`, judge auxiliary-client, continuação adapter-FIFO no lado gateway) mas a ideia é deles. Crédito onde crédito é devido.
