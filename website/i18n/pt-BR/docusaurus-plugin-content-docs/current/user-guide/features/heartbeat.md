---
sidebar_position: 17
title: "Heartbeats de sessão"
description: "Um prompt recorrente que reentra na sessão atual sempre que ela está ociosa — /heartbeat every 10m Check the deployment."
---

# Heartbeats de sessão (`/heartbeat`) {#session-heartbeats-heartbeat}

`/heartbeat` dá à **sessão atual** uma instrução recorrente. Sempre que a sessão está ociosa e o intervalo passou, o prompt dispara como um turno normal de usuário — mesma conversa, mesmo contexto, mesmo prompt cache.

```
/heartbeat every 10m Check the deployment and report meaningful changes
```

Inspirado no `/heartbeat` do Prime-Agent. A adaptação do Hermes mantém os invariantes estritos de fluxo de mensagens: o heartbeat é injetado apenas entre turnos (nunca no meio de uma execução), como uma mensagem comum de role user.

## Heartbeat vs cron: qual usar? {#heartbeat-vs-cron-which-one-do-i-want}

Parecem similares, mas servem a trabalhos diferentes:

| | `/heartbeat` | [`hermes cron`](./cron) |
|---|---|---|
| Roda em | **Esta conversa** — contexto completo, memória da discussão | Uma sessão isolada nova a cada tick |
| Sobrevive a reinício do processo | Estado sobrevive (SessionDB); disparo retoma quando a sessão for acionada | Sim — agendador totalmente durável |
| Quantidade | Um por sessão | Jobs ilimitados |
| Melhor para | "Fique de olho em X *nesta thread* enquanto trabalhamos" | Jobs permanentes, relatórios, watchdogs, entregas |

Regra prática: se o prompt recorrente precisa do contexto da conversa, use `/heartbeat`. Se for um job autocontido, use cron.

## Comandos {#commands}

| Comando | O que faz |
|---|---|
| `/heartbeat every <interval> <prompt>` | Define (ou substitui) o heartbeat da sessão. Intervalos: `90s`, `10m`, `2h`, `1d` (mínimo 60s). |
| `/heartbeat` ou `/heartbeat status` | Mostra o heartbeat, seu intervalo e tempo até o próximo disparo. |
| `/heartbeat pause` | Para de disparar sem limpar. |
| `/heartbeat resume` | Retoma (reancora o timer — sem disparo stale instantâneo). |
| `/heartbeat clear` | Remove o heartbeat. |

`/hb` é um alias. Funciona no CLI e em plataformas de gateway (no Slack, use `/hermes heartbeat …`).

## Detalhes de comportamento {#behavior-details}

- **Somente quando ocioso.** Um heartbeat nunca interrompe um turno em execução. Se o agente estiver ocupado quando o tick vencer, dispara no próximo poll ocioso.
- **Ticks perdidos coalescem.** Se a sessão ficou ocupada (ou o processo não estava rodando) por vários intervalos, você recebe **um** turno de heartbeat, não um backlog. O timer reancora a cada disparo.
- **Mensagens do usuário vencem.** Uma mensagem do usuário na fila sempre tem prioridade; o heartbeat espera a fila de input esvaziar.
- **Seguro para cache.** O prompt injetado é uma mensagem user comum. Sem mutação de system prompt, sem mudança de toolset.
- **Persistência.** O estado fica em `SessionDB.state_meta` com chave `heartbeat:<session_id>` — sobrevive a `/resume` e atravessa rotações de sessão por compressão de contexto. Disparar exige o processo dono (sessão CLI ou gateway) em execução; para agendamentos que devem sobreviver a qualquer coisa, use cron.
- **Guarda contra inventar trabalho.** O prompt injetado diz ao agente para responder brevemente e parar quando nada significativo mudou, para um heartbeat ocioso não gerar trabalho fictício.

## Exemplo {#example}

```
You: /heartbeat every 15m Check whether the CI run for PR #1234 finished; summarize the result when it does

  ♥ Heartbeat set (every 15m): Check whether the CI run for PR #1234 finished; ...

[15 minutes of you working on other things in the same session]

Hermes: [Heartbeat — recurring instruction, fires every 15m]
  💻 gh pr checks 1234   (1.2s)
  CI is still running (14/37 checks complete). Nothing to report yet.
```

Quando a resposta parar de mudar, use `/heartbeat clear` — ou deixe continuar de olho.
