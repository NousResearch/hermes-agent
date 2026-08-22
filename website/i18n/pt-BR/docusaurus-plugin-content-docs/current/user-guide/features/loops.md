---
sidebar_position: 17
title: "Loops recorrentes"
description: "Reexecuta um prompt em um intervalo recorrente dentro da sua sessão — a versão do Hermes do /loop do Claude Code."
---

# Loops recorrentes (`/loop`) {#recurring-loops}

`/loop` reexecuta um prompt (ou um slash command) em uma cadência recorrente **dentro da sessão atual**. Cada wakeup é um turno real do agente: o Hermes lê o estado atual de novo — o último resultado de CI, a profundidade mais recente da fila, o arquivo como está agora — faz o trabalho, reporta e fica quieto até o próximo tick.

É a versão do Hermes do **`/loop` do Claude Code** (e o alias `/proactive`, que também funciona aqui). Onde [`/goal`](./goals.md) é guiado por um judge — "continue trabalhando até este objetivo ser alcançado" — `/loop` é guiado por timer: "faça isso de novo a cada N minutos (ou quando fizer sentido) até algo dizer para parar."

## Quando usar {#when-to-use-it}

- **Polling de estado externo.** "Observe o deploy / o CI / a fila e me avise quando mudar." O caso de uso canônico.
- **Iterar até ficar verde.** "Rode os testes, corrija o que falhar, repita até passarem."
- **Monitorar durante uma sessão de trabalho.** Acompanhe taxas de erro ou o progresso de um job longo enquanto você faz outra coisa na mesma conversa.
- **Manutenção periódica.** Reexecute um lint ou um resumo de status a cada N minutos durante uma sessão longa.

Quando o trabalho deve rodar **sem supervisão** — de madrugada, em um schedule real, sobrevivendo a restarts do terminal — use um [cron job](./cron.md). `/loop` vive dentro de uma sessão; cron vive fora de todas elas. E quando a tarefa é um único objetivo com definição de pronto, [`/goal`](./goals.md) costuma ser o melhor fit.

## Início rápido {#quick-start}

```
/loop 5m check the deploy status and tell me if it's live yet
```

O que você vai ver:

1. **Loop aceito** — `↻ Loop set (every 5m): check the deploy status…`
2. **Primeiro wakeup em 5m** — enquanto a sessão está idle, o Hermes injeta o wakeup e roda um turno normal contra o estado atual.
3. **Repetir** — a cada 5 minutos, até uma condição de parada disparar ou você parar.

Loopar um slash command é igualmente simples:

```
/loop 10m /recap
```

## Os dois modos de cadência {#the-two-cadence-modes}

**Intervalo fixo — você define o relógio.** Passe um intervalo (`30s`, `5m`, `2h`, `1h30m`) e o loop dispara nesse schedule. Use quando o que você observa muda no próprio tempo dele:

```
/loop 2m poll the build at ci.example.com/job/42 and ping me the moment it finishes
```

**Self-paced — o Hermes define o relógio.** Omita o intervalo e o loop se regula: começa no piso (1 minuto por padrão) e, enquanto as respostas do agente param de mudar, recua exponencialmente — 2m, 4m, 8m, até o teto (15 minutos por padrão). No momento em que uma resposta difere da anterior, a cadência volta ao piso. A detecção de mudança é uma comparação local de digest (timestamps são ignorados), então esperas idle não custam extra:

```
/loop keep an eye on the migration and summarize progress
```

Regra prática: **intervalo fixo quando um relógio externo dirige o trabalho; self-paced quando o trabalho dirige o ritmo.**

## Condições de parada {#stop-conditions}

Um loop termina quando qualquer uma destas dispara:

| Condição | Como |
|---|---|
| O agente decide que terminou | O prompt de wakeup ensina o agente a encerrar a resposta com `LOOP_COMPLETE` numa linha própria quando a tarefa está pronta ou sem objeto. |
| Um cap de execuções | `--times N` — para depois de N wakeups. |
| Uma condição baseada em evidência | `--until <condition>` — após cada wakeup, o mesmo auxiliary judge que alimenta `/goal` checa a resposta contra a sua condição (fail-open: um judge quebrado nunca trava o loop). |
| Você | `/loop stop` (ou `/loop pause` para mantê-lo). |
| O orçamento de backstop | `loops.max_ticks` (padrão 100) pausa o loop para uma sessão sem supervisão não queimar tokens para sempre. `0` = ilimitado. |

Exemplos:

```
/loop 2m poll CI --times 30
/loop 5m watch the queue --until queue depth reaches zero
```

## Comandos {#commands}

| Comando | O que faz |
|---|---|
| `/loop [interval] <prompt> [--times N] [--until <cond>]` | Inicia (ou substitui) o loop desta sessão. |
| `/loop` ou `/loop status` | Mostra cadência, ticks disparados e tempo até o próximo wakeup. |
| `/loop pause` | Para de disparar sem perder o loop. |
| `/loop resume` | Retoma. |
| `/loop stop` | Encerra o loop. |
| `/proactive …` | Alias de `/loop` (paridade com o Claude Code). |

Funciona na CLI, na TUI (`hermes --tui`), no chat do web dashboard, no app desktop e em cada plataforma do gateway (Telegram, Discord, Slack, WhatsApp, …). Em plataformas de mensageria o gateway dispara wakeups mesmo entre as suas mensagens — o loop pertence à sessão do chat, e os resultados chegam como respostas comuns.

## Misturando com `/goal` {#mixing-with-goal}

Os dois recursos injetam turnos sintéticos nos limites de idle, então seguem uma regra: **uma meta ativa é dona da sessão.** Enquanto um `/goal` está dirigindo de fato (judge dizendo "continue"), os wakeups do loop adiám. O loop retoma o tempo idle assim que a meta termina, pausa ou se estaciona numa barreira de espera (`/goal wait`, ou o veredito WAIT automático do judge). Uma meta estacionada mais um `/loop` é uma combinação natural: a meta espera o grande async enquanto o loop mantém um heartbeat em outra coisa.

Uma mensagem real do usuário sempre vence as duas — wakeups só disparam enquanto a sessão está idle e nada seu está na fila.

## Detalhes de comportamento {#behavior-details}

- **Um wakeup é um turno normal com role de usuário.** Sem mutação do system prompt, sem troca de toolset — o prompt caching permanece intacto.
- **Sobrevive a `/resume` e compressão.** O estado do loop persiste por sessão e migra pelas fronteiras de compressão de contexto, igual a `/goal`.
- **Um loop por sessão.** Definir um novo `/loop` substitui o anterior. Rode vários loops rodando várias sessões (ou use cron para uma frota de schedules).
- **Interromper um turno de wakeup (Ctrl+C) pausa o loop** — recuperável com `/loop resume`, então cancelar de fato cancela.
- **O custo de tokens escala com a cadência.** Cada tick é um turno completo do agente. Ajuste o intervalo à frequência com que o estado realmente muda; prefira self-pacing para esperas idle.

## Configuração {#configuration}

```yaml
# ~/.hermes/config.yaml
loops:
  min_interval_seconds: 30       # floor for fixed intervals
  max_ticks: 100                 # backstop budget (0 = unlimited)
  self_paced_floor_seconds: 60   # self-paced starting cadence
  self_paced_ceiling_seconds: 900  # self-paced max backoff
```

O judge de `--until` roteia pela tarefa auxiliar `goal_judge`, então os overrides de `auxiliary.goal_judge.*` (provider, model, max_tokens) também se aplicam às condições do loop.

## `/loop` vs `/goal` vs cron {#loop-vs-goal-vs-cron}

| | `/loop` | `/goal` | cron |
|---|---|---|---|
| **Gatilho** | Timer (ou self-paced) | Veredito do judge após cada turno | Schedule, fora de qualquer sessão |
| **Vive em** | A sessão atual | A sessão atual | A própria sessão por execução |
| **Termina quando** | Condição de parada / caps / você | Meta alcançada / orçamento / você | Você remove o job |
| **Melhor para** | Polling, monitoramento, reexecuções periódicas | Um objetivo, iterar até concluir | Schedules sem supervisão e de longo prazo |
