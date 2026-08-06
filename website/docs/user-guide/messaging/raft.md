---
sidebar_position: 19
title: "Raft"
description: "Connect Hermes Agent to Raft as an external agent via wake-channel bridge"
---

# Raft Setup

Hermes connects to [Raft](https://raft.build) as an external agent through a local wake-channel bridge. The adapter starts a loopback HTTP endpoint that receives content-free wake hints from the bridge, then injects them into the Hermes gateway session pipeline. The agent reads and sends messages through the Raft CLI — the adapter never touches message bodies or delivery cursors.

:::info Division of Labor
- **The bridge** owns: wake-hint consumption, dedup, backoff, reconnection, at-least-once delivery, and proof logging.
- **The Hermes adapter** owns: a localhost wake endpoint and injecting a short notice into the agent's context.
- **The agent** owns: pulling messages (`raft message check`), replying (`raft message send`), and all other Raft interactions via the CLI.

The adapter holds no Raft credentials — only a per-session shared token for localhost auth between the bridge and the endpoint.
:::

---

## Prerequisites

- A **Raft workspace** where you can create an External Agent
- The **Raft CLI** installed and logged in to that External Agent profile
- **aiohttp** — Python package (included in Hermes `[all]` extras)

In Raft, open the Agents menu, create an External Agent, and follow the setup card to install the Raft CLI and log in the agent profile. Once the agent is created, Raft shows a Hermes setup guide with the environment variables and configuration needed to start the gateway.

---

## Setup

Add to `~/.hermes/.env`:

```bash
RAFT_PROFILE=your-agent-profile
```

That's it — the adapter auto-enables when `RAFT_PROFILE` is set. It generates a per-session bridge token, picks an ephemeral port, and spawns the bridge child process automatically when the gateway starts.

---

## How It Works

```
Raft Server → Bridge (wake-hints SSE) → POST /wake → Hermes Adapter → Agent context
Agent → raft message check → Raft Server (message bodies)
Agent → raft message send → Raft Server (replies)
```

1. The Raft server sends wake hints to the bridge process via SSE.
2. The bridge forwards each hint as a `POST /wake` to the adapter's loopback endpoint.
3. The adapter validates the bridge token, verifies the payload is content-free, and injects a wake notice into the Hermes session.
4. The agent sees the wake notice and uses the Raft CLI to read messages and reply.

Wake payloads are **content-free by contract** — they carry metadata (event ID, message ID, timestamps) but never message bodies, channel names, or sender identities. The adapter rejects any payload containing content-shaped fields (`text`, `body`, `content`, `messages`, etc.).

---

## Bridge

The adapter automatically spawns `raft agent bridge` as a child process, passing the endpoint URL and token. The bridge connects to the Raft server using the configured profile and begins forwarding wake hints. It is terminated when the gateway shuts down.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RAFT_PROFILE` | Raft agent profile slug — auto-enables the adapter when set | _(required)_ |
