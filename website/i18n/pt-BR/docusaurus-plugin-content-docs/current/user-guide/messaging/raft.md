---
sidebar_position: 19
title: "Raft"
description: "Conecte o Hermes Agent ao Raft como agente externo via ponte wake-channel"
---

# Configuração do Raft {#raft-setup}

O Hermes se conecta ao [Raft](https://raft.build) como agente externo por meio de uma ponte wake-channel local. O adaptador inicia um endpoint HTTP loopback que recebe hints de wake sem conteúdo da ponte, depois os injeta no pipeline de sessão do gateway Hermes. O agente lê e envia mensagens pelo CLI Raft — o adaptador nunca toca corpos de mensagem ou cursores de entrega.

:::info Divisão de responsabilidades
- **A ponte** possui: consumo de hints de wake, dedup, backoff, reconexão, entrega at-least-once e logging de prova.
- **O adaptador Hermes** possui: um endpoint wake localhost e injeção de um aviso curto no contexto do agente.
- **O agente** possui: puxar mensagens (`raft message check`), responder (`raft message send`) e todas as outras interações Raft via CLI.

O adaptador não guarda credenciais Raft — apenas um token compartilhado por sessão para auth localhost entre a ponte e o endpoint.
:::

---

## Pré-requisitos {#prerequisites}

- Um **workspace Raft** onde você pode criar um External Agent
- O **CLI Raft** instalado e logado no perfil desse External Agent
- **aiohttp** — pacote Python (incluído nos extras `[all]` do Hermes)

No Raft, abra o menu Agents, crie um External Agent e siga o card de setup para instalar o CLI Raft e fazer login no perfil do agente. Depois que o agente for criado, o Raft mostra um guia de setup Hermes com as variáveis de ambiente e configuração necessárias para iniciar o gateway.

---

## Setup {#setup}

Adicione em `~/.hermes/.env`:

```bash
RAFT_PROFILE=your-agent-profile
```

Só isso — o adaptador auto-habilita quando `RAFT_PROFILE` está definido. Ele gera um token de ponte por sessão, escolhe uma porta efêmera e inicia o processo filho da ponte automaticamente quando o gateway sobe.

---

## Como funciona {#how-it-works}

```
Raft Server → Bridge (wake-hints SSE) → POST /wake → Hermes Adapter → Agent context
Agent → raft message check → Raft Server (message bodies)
Agent → raft message send → Raft Server (replies)
```

1. O servidor Raft envia hints de wake ao processo ponte via SSE.
2. A ponte encaminha cada hint como `POST /wake` ao endpoint loopback do adaptador.
3. O adaptador valida o token da ponte, verifica que o payload é sem conteúdo e injeta um aviso de wake na sessão Hermes.
4. O agente vê o aviso de wake e usa o CLI Raft para ler mensagens e responder.

Payloads de wake são **sem conteúdo por contrato** — carregam metadados (event ID, message ID, timestamps) mas nunca corpos de mensagem, nomes de canal ou identidades de remetente. O adaptador rejeita qualquer payload com campos no formato de conteúdo (`text`, `body`, `content`, `messages`, etc.).

---

## Ponte {#bridge}

O adaptador inicia automaticamente `raft agent bridge` como processo filho, passando a URL do endpoint e o token. A ponte conecta ao servidor Raft usando o perfil configurado e começa a encaminhar hints de wake. Ela é encerrada quando o gateway desliga.

---

## Variáveis de ambiente {#environment-variables}

| Variable | Description | Default |
|----------|-------------|---------|
| `RAFT_PROFILE` | Slug do perfil de agente Raft — auto-habilita o adaptador quando definido | _(required)_ |
