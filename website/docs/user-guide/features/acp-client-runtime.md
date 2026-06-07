---
title: ACP Client Runtime (optional)
sidebar_label: ACP Client Runtime
---

# ACP Client Runtime

Hermes can optionally route turns to any external agent that speaks the [Agent Client Protocol (ACP)](https://github.com/NousResearch/hermes-agent/blob/main/acp_adapter/README.md) instead of running its own tool loop. When enabled, Hermes sends each user turn to the external agent via JSON-RPC 2.0 over stdio (`session/new` + `session/prompt`), streams the agent's response back, and projects the result into Hermes' session model so memory and skill review keep working.

This is **opt-in only**. Default Hermes behavior is unchanged unless you flip the flag. Hermes never auto-routes you onto this runtime.

## Why

- Run any ACP-compliant agent (e.g. `@agentclientprotocol/claude-agent-acp`, or your own ACP-compliant implementation) from within a Hermes session without leaving the gateway, TUI, or CLI surface.
- Keep Hermes' session management — sessions DB, slash commands, cron, skill review, memory nudges, gateway platforms — while delegating the turn itself to a specialized external agent.
- Bridge agent types: point Hermes at a Claude Code ACP bridge and your Slack/Discord/Telegram users get Claude Code's full tool surface through the Hermes gateway.

## How it works

```
User message
    │
    ▼
AIAgent.run_conversation()
    if api_mode == "acp_client":
        │
        ▼
    ACPClientSession (acp_client_session.py)
        session/new ────► ACP agent (subprocess)
        session/prompt ──► ACP agent
        ◄── session/update (streaming chunks)
        ◄── PromptResponse
        │
        ▼
    Assembled text + projected_messages
        │
        ▼
Memory / skill review (background)
    │
    ▼
User sees response
```

The external agent subprocess is spawned on the first turn and kept alive across turns (one session per Hermes session). Hermes declines `fs/*` and `terminal/*` server-initiated requests — those surfaces are Hermes' domain, not the agent's.

## What Hermes capabilities are preserved

| Capability | Status |
|---|---|
| All gateway platforms (Slack, Discord, Telegram, WhatsApp, …) | yes |
| CLI / TUI | yes |
| Session persistence (sessions DB, resume, history) | yes |
| Memory nudges (every 10 turns) | yes |
| Skill nudges (every 10 tool iterations) | yes |
| `/goal` (Ralph loop) | yes |
| Cron jobs | yes |
| Slash commands (except those requiring a live agent loop) | yes |

## Prerequisites

You need an ACP-compliant agent binary on PATH or at a known path. Examples:

- **`@agentclientprotocol/claude-agent-acp`** — the reference ACP implementation from Zed Industries:
  ```bash
  npm install -g @agentclientprotocol/claude-agent-acp
  claude-agent-acp --version
  ```
- **Any ACP-compliant server** implementing the `session/new` + `session/prompt` JSON-RPC surface can be used.

## Enabling

In a Hermes session:

```
/acp-client-runtime on claude-agent-acp
```

With extra arguments:

```
/acp-client-runtime on claude-agent-acp --model claude-opus-4-5
```

That command:
- Verifies that `claude-agent-acp` is reachable (blocks with an error if not found on PATH).
- Persists `model.provider: "acp-client"` + `model.acp_command` + `model.acp_args` to your config.yaml.
- Takes effect on the **next** session — the current cached agent keeps the prior runtime to preserve prompt cache.

Synonyms: `acp_client`, `enable`, `acp`.

To check current state without changing anything:

```
/acp-client-runtime
```

To disable:

```
/acp-client-runtime off
```

## Manual config

You can also set it directly in `~/.hermes/config.yaml`:

```yaml
model:
  provider: "acp-client"
  acp_command: "claude-agent-acp"
  acp_args:
    - "--model"
    - "claude-opus-4-5"
```

To revert to the Hermes default runtime, remove those three keys (or set `provider` to your normal provider value and remove `acp_command`/`acp_args`).

## How this differs from copilot-acp

Hermes already has a `copilot-acp` provider that speaks ACP to GitHub Copilot's ACP endpoint. The `acp-client` runtime differs in scope:

| | `copilot-acp` | `acp-client` runtime |
|---|---|---|
| Target | GitHub Copilot ACP endpoint (HTTP) | Any ACP-compliant **subprocess** (stdio) |
| Transport | HTTP / SSE | JSON-RPC 2.0 over stdin/stdout |
| Subprocess lifecycle | N/A — remote service | Hermes spawns and manages the process |
| Session model | Copilot manages sessions | Hermes owns `session/new` + `session/prompt` |
| Use case | Copilot as inference backend | Bridging to local agents (Claude Code, custom) |

## Architecture detail

The wire protocol is JSON-RPC 2.0 newline-delimited over stdio:

1. **Spawn** — Hermes spawns `<acp_command> [<acp_args>]` as a subprocess.
2. **Initialize** — `initialize` handshake. Hermes sends `clientCapabilities: {fs: {readTextFile: false, writeTextFile: false}, terminal: false}` to signal it will not proxy file-system or terminal calls.
3. **Session** — `session/new` with the current working directory.
4. **Turn** — `session/prompt` with the user message. Hermes streams `session/update` chunks to the terminal while waiting for `PromptResponse`.
5. **Close** — `session/close` when the session ends.

Server-initiated requests (`fs/*`, `terminal/*`, `session/request_permission`) are declined by Hermes with a `-32601` error. The ACP agent should not depend on those surfaces.

## Self-improvement loop (memory + skill nudges)

Memory and skill nudges work identically to the default runtime. Streamed `session/update` chunks are projected into `projected_messages`, and the background review fork sees a normal-looking transcript with `{role: "assistant", content: "..."}` messages. Turn and tool-iteration counters increment as expected.

## Limitations

This runtime is **opt-in beta**:

- **One subprocess per session** — the ACP agent is spawned lazily on the first turn and kept alive. If the subprocess crashes, `should_retire=True` is set and the next turn starts a fresh session.
- **No streaming interruption** — mid-stream Ctrl+C is not forwarded to the ACP agent. The turn runs to completion. Cancellation support may be added in a future version.
- **fs/terminal proxying declined** — Hermes does not bridge file-system or terminal calls from the ACP agent. Agents that require those surfaces should implement them internally.
- **Single provider** — ACP client mode replaces the entire inference path. You cannot mix ACP and non-ACP providers in the same session.

If you find a bug, [open an issue](https://github.com/NousResearch/hermes-agent/issues) with the output of `hermes logs --since 5m` and mention `acp-client-runtime` in the title.

## For implementors

If you are building an ACP-compliant agent and want to test against Hermes as the client:

1. Implement `initialize`, `session/new`, `session/prompt`, `session/close` from the ACP spec.
2. Stream `session/update` notifications with `{"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "..."}}` for incremental output.
3. Install your binary and enable via `/acp-client-runtime on <your-binary>`.
4. Run the E2E smoke test:
   ```bash
   HERMES_E2E_ACP=1 HERMES_E2E_ACP_COMMAND=/path/to/your/binary \
     pytest tests/e2e/test_acp_client_e2e.py -v
   ```
