---
sidebar_position: 12
title: "A2A (Agent2Agent) Server"
description: "Run Hermes Agent as an A2A server so other agents can discover it and delegate tasks over JSON-RPC + SSE"
---

# A2A (Agent2Agent) Server

Hermes Agent can run as an [A2A](https://a2a-protocol.org) server, letting any
A2A-compatible client or peer agent discover it and delegate tasks over
HTTP(S). Where MCP connects an agent to _tools_, A2A connects an agent to
_other agents_ — so A2A turns Hermes into a callable worker for orchestrators
like LangGraph, CrewAI, Google ADK, the `a2a-inspector`, or another Hermes.

It is the sibling of [ACP](./acp.md) (editor integration over stdio) and the
MCP server (tools over MCP): A2A is **Hermes as a remote agent for other
agents**.

## What Hermes exposes in A2A mode

- An **Agent Card** at `/.well-known/agent-card.json` describing Hermes' name,
  version, capabilities (streaming), and skills.
- `message/send` — synchronous request/response (returns a completed task).
- `message/stream` — Server-Sent Events streaming of task status updates and
  artifacts.
- `tasks/get` and `tasks/cancel`.

Each turn composes Hermes' existing coding/research toolsets (shell,
filesystem, web/browser, memory, todo, skills, `execute_code`, and
`delegate_task`) without adding an A2A-specific core toolset or interactive
messaging/audio surfaces.

## Installation

Install Hermes normally, then add the A2A extra:

```bash
pip install -e '.[a2a]'
```

This installs the `a2a-sdk[http-server]` dependency and enables:

- `hermes-a2a`
- `python -m plugins.platforms.a2a`
- the bundled A2A gateway platform plugin

## Launching the A2A server

```bash
hermes-a2a
```

```bash
python -m plugins.platforms.a2a
```

By default the server binds `127.0.0.1:9100`. The Agent Card is served at
`/.well-known/agent-card.json` and the JSON-RPC endpoint at `/`.

```bash
hermes-a2a --host 127.0.0.1 --port 9100
hermes-a2a --public-url https://agents.example.com/hermes/   # URL advertised in the card
```

For non-interactive checks:

```bash
hermes-a2a --version
hermes-a2a --check
```

:::warning Exposing to the network
The A2A endpoint is **unauthenticated**. Binding `--host 0.0.0.0` exposes
Hermes — and its shell/filesystem tools — to anything that can reach the port.
Only do so behind a reverse proxy or auth layer you control. The server logs a
warning when started on `0.0.0.0`.
:::

## Talking to the server

Fetch the card, then send a message. With `curl`:

```bash
curl http://127.0.0.1:9100/.well-known/agent-card.json

curl http://127.0.0.1:9100/ -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": "1", "method": "message/send",
  "params": {"message": {"role": "user", "kind": "message", "messageId": "m1",
    "parts": [{"kind": "text", "text": "Summarize what this repo does."}]}}
}'
```

`message/stream` uses the same body with `"method": "message/stream"` (the
method name is what selects streaming; an `Accept: text/event-stream` header is
the conventional client courtesy). The response is an SSE stream of
`TaskStatusUpdateEvent` (working) and `TaskArtifactUpdateEvent` (the result),
ending in a `completed` status.

## Conversation continuity

A2A `contextId` maps to a persistent Hermes session: one `AIAgent` plus its
rolling history per context. Follow-up messages that reuse the same `contextId`
continue the same conversation. Each `taskId` is one turn within a context.
Sessions are held in memory for the lifetime of the server process.

## Configuration and credentials

A2A mode uses the same Hermes profile configuration as the CLI. Behavioral
settings live in `~/.hermes/config.yaml`:

```yaml
a2a:
  enabled: false # true starts A2A with `hermes gateway`
  host: 127.0.0.1
  port: 9100
  public_url: null # externally advertised base URL
  max_concurrency: 16 # simultaneous blocking agent turns
  max_sessions: 512 # in-memory context LRU cap
  max_tasks: 2048 # retained protocol task cap
  max_task_history: 100 # retained status messages per task
  tool_io: preview # preview | none | full

platform_toolsets:
  a2a: [web, terminal, file, vision, skills, browser, todo, memory,
        session_search, code_execution, delegation]
```

`tool_io: preview` bounds peer-visible tool arguments/results, `none` sends
tool names only, and `full` sends unbounded values. Command-line host, port,
and public-URL flags override the file for standalone launches.

`platform_toolsets.a2a` uses the same tool configuration surface as every
gateway platform. `agent.disabled_toolsets` remains authoritative, so an
operator can globally remove sensitive capabilities such as `terminal` or
`file`. Because A2A has no interactive approval round trip, dangerous commands
that require confirmation are denied instead of reading from server stdin.

The plugin can also be managed by the existing gateway lifecycle:

```bash
hermes config set a2a.enabled true
hermes gateway run
```

Credentials remain in the normal secret store:

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`

Provider resolution uses Hermes' normal runtime resolver, so A2A inherits the
currently configured provider and credentials. Configure credentials with
`hermes model` or by editing `~/.hermes/.env`.

## Troubleshooting

### Server starts but tasks fail immediately

Verify dependencies and provider setup:

```bash
hermes-a2a --check
hermes model
hermes doctor
```

### A client cannot discover the agent

Confirm the card is reachable and the client points at the base URL (not the
card URL):

```bash
curl -fsS http://127.0.0.1:9100/.well-known/agent-card.json
```

## See also

- [A2A Internals](../../developer-guide/a2a-internals.md)
- [ACP Editor Integration](./acp.md)
- [Provider Runtime Resolution](../../developer-guide/provider-runtime.md)
