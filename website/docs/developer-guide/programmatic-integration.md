---
sidebar_position: 8
title: "Programmatic Integration"
description: "Three protocols for driving hermes-agent from external programs: ACP, the TUI gateway JSON-RPC, and the OpenAI-compatible HTTP API"
---

# Programmatic Integration

Hermes ships three protocols for driving the agent from external programs — IDE plugins, custom UIs, CI pipelines, embedded sub-agents. Pick the one that matches your transport and consumer.

| Protocol | Transport | Best for | Defined by |
|----------|-----------|----------|------------|
| **ACP** | JSON-RPC over stdio | IDE clients (VS Code, Zed, JetBrains) that already speak the [Agent Client Protocol](https://github.com/zed-industries/agent-client-protocol) | `acp_adapter/` |
| **TUI gateway** | JSON-RPC over stdio (or WebSocket) | Custom hosts that want fine-grained control of sessions, slash commands, approvals, and streaming events | `tui_gateway/server.py` |
| **API server** | HTTP + Server-Sent Events | OpenAI-compatible frontends (Open WebUI, LobeChat, LibreChat…) and language-agnostic web clients | `gateway/platforms/api_server.py` |

All three drive the same `AIAgent` core. They differ only in wire format and which set of features they expose.

---

## ACP (Agent Client Protocol)

`hermes acp` starts a stdio JSON-RPC server speaking ACP. Used in production by VS Code (Zed Industries' ACP extension), Zed, and any JetBrains IDE with an ACP plugin.

Capabilities exposed: session creation, prompt submission, streaming agent message chunks, tool-call events, permission requests, session fork, cancel, and authentication. Tool output is rendered into ACP `Diff`/`ToolCall` content blocks the IDE understands.

Full lifecycle, event bridge, and approval flow: [ACP Internals](./acp-internals).

```bash
hermes acp                  # serve ACP on stdio
hermes acp --check          # verify ACP dependencies and adapter imports
hermes acp --setup          # interactive provider/model setup for ACP terminal auth
```

---

## TUI Gateway JSON-RPC

`tui_gateway/server.py` is the method and event catalog behind the Ink TUI (`hermes --tui`), the Desktop app and the embedded dashboard PTY bridge. The Ink TUI reaches it on the profile's shared gateway over WebSocket (`/api/ws`, protocol `hermes-gateway-v1`, a single-use ticket per connection); any external host can speak the same frames over stdio (`tui_gateway/entry.py`) or a WebSocket (`tui_gateway/ws.py`).

### A Node client: `@hermes/shared/gateway-client`

The Ink TUI's gateway client lives in `apps/shared` (`@hermes/shared`), so another Node frontend can use it without compiling Ink's source. It dials the owner with a ticket, reads `runtime.describe` before `session.create`, translates requests and results for `hermes-gateway-v1` (`@hermes/shared/gateway-canonical`), reconnects with backoff, re-ensures a gone owner a bounded number of times per minute, and emits typed events (`@hermes/shared/gateway-types`).

The client does not know where its connection comes from. The caller passes `connect(start, recover)`, which returns `{url, protocols, profile_id, instance_id}` and runs again for every reconnect (tickets are single-use). Ink passes its `gateway_bootstrap.py` helper; any other local client can run [`hermes gateway attach --json`](./gateway-internals.md#hermes-gateway-attach), with `--start` on the first call and `--recover` when asked to recover, and pass its `url` and `protocols`:

```ts
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

import { GatewayClient } from '@hermes/shared/gateway-client'
import type { AnyGatewayEvent } from '@hermes/shared/gateway-types'

const gw = new GatewayClient({
  connect: async (start, recover) => {
    const flags = start ? ['--start'] : recover ? ['--recover'] : []
    const { stdout } = await promisify(execFile)('hermes', ['gateway', 'attach', '--json', ...flags])
    const { url, protocols, endpoint } = JSON.parse(stdout)

    return { url, protocols, profile_id: endpoint.profile_id, instance_id: endpoint.instance_id }
  }
})

gw.on('event', (event: AnyGatewayEvent) => { /* … */ })
gw.start()
gw.drain()
const session = await gw.request('session.create', {})
```

`hermes gateway attach` exits non-zero with the verdict on stdout when there is no owner to attach, so `execFile` rejects and the client retries on its backoff. Sessions are created with `source: "tui"`, the gateway's terminal-client policy, and `session.create` refuses an owner whose `runtime.describe` does not offer it.

Two options say who is launching. `source` picks another policy label from `runtime.describe`'s `session_create.sources` (`session.create` is refused if the owner does not list it). `client` is your client's own name, sent on `session.create` only when the owner lists `client` in `session_create.parameters`; an older owner gets the request without it. `name` is 1-64 characters of `a-z`, `0-9`, `.`, `_` and `-`, starting with a letter or digit (`my-client`, not `My Client` or `@acme/client`), and the optional `version` is 1-64 printable ASCII characters without spaces, with no other keys: an owner that takes `client` refuses `session.create` for anything else. The name is a label for attribution: Hermes never grants anything because of it.

```ts
const gw = new GatewayClient({
  connect,                                           // as above
  source: 'tui',                                     // the default: the terminal-client policy
  client: { name: 'my-client', version: '1.4.0' }    // version is optional
})
```

Other options (`terminalConnectFailure`, `attachUrl`, `sidecarUrl`, timeouts, `toolProgress`, `onLifecycle`, `text`, `fallbackWebSocket`) are documented on `GatewayClientOptions`. Browser-safe modules stay in `apps/shared/src/`; Node-only ones (the client, `prompt-history`) are in `apps/shared/src/node/`, and `apps/shared/src/node/boundary.test.ts` fails if the package's program ever includes a file from another workspace.

### Which methods the shared gateway answers

On the shared gateway's `/api/ws` socket (the one `hermes gateway attach --json` opens), the socket answers the heartbeat `gateway.ping` itself, and every other frame goes to the session authority first. A method the authority does not serve can still reach an older handler in `tui_gateway/server.py`, but only for a connection whose ticket carries the full interactive grant (an attach ticket does). The older handler is never used for a method that acts on the whole process, or for a `session.*`, `prompt.*` or `message.*` method other than a few store-wide reads and the foreign-history import (`session.most_recent`, `session.active_list`, `session.events.stats`, `session.foreign.*`; `tui_gateway/ws_legacy_fallback.py`). Everything else gets `-32601`.

The OpenRPC document's `x-canonical-methods` (53 names) is not the list of methods a client may call. It lists the methods with a `hermes-gateway-v1` wire model of their own (`tui_gateway/contracts/canonical*.py`). The authority also serves methods whose wire is the same as the older catalog's (`RpcMethods` in the generated TypeScript), and validates those against that model.

Build on the methods the gateway serves. The methods answered by an older handler have no owner decision yet (`tests/hermes_cli/serve_handler_ledger.json`, `decision: undecided`), so treat `-32601` from one of them as "not available here". The lists below are generated from the dispatch tables:

<!-- gateway-methods:start -->
<!-- Generated by scripts/gen_serve_handler_ledger.py from the /api/ws dispatch tables. Do not edit. -->

**Method families served by the gateway (one name per pattern)** (1): `plugin.{plugin_id}.{name}`

**Served for a call about a session the client holds; an account-wide call uses the older handler** (4): `connection.respond`, `connectors.connect`, `connectors.operation.status`, `connectors.operation.wake`

**Refused (`-32601`)** (16): `agents.list`, `session.activate`, `session.archive`, `session.branch`, `session.branch_stored`, `session.branch_whole`, `session.close`, `session.compress`, `session.cwd.set`, `session.delete`, `session.set_hidden`, `session.status`, `session.title`, `session.undo`, `session.usage`, `session.workspace.move`

**Served by the gateway** (110): `a2a.forward`, `approval.pending`, `approval.received`, `approval.respond`, `bot_relay.cancel`, `bot_relay.deliver`, `bot_relay.outbox.drain`, `bot_relay.reply`, `bot_relay.roster.sync`, `bot_relay.settle`, `clarify.lock`, `clarify.respond`, `command.dispatch`, `command.observe`, `commands.catalog`, `complete.path`, `complete.slash`, `config.get`, `config.set`, `cron.cancel`, `cron.recover`, `cron.status`, `cron.submit`, `delegation.pause`, `file.attach`, `groups.approve`, `groups.attachment.download`, `groups.attachment.upload`, `groups.capabilities`, `groups.context`, `groups.continue`, `groups.create`, `groups.disband`, `groups.discard`, `groups.list`, `groups.log`, `groups.peer.invite`, `groups.peer.register`, `groups.peer.revoke`, `groups.rename`, `groups.retry`, `groups.send`, `groups.state`, `groups.stop`, `handoff.fail`, `handoff.request`, `handoff.state`, `image.attach_bytes`, `kanban.run`, `llm.oneshot`, `message.react`, `model.options`, `onboarding.skip`, `onboarding.start`, `ping`, `plugins.list`, `preview.restart`, `process.kill`, `process.list`, `process.stop`, `profiles.list`, `prompt.background`, `prompt.btw`, `prompt.cancel`, `prompt.receipt`, `prompt.replace_queued`, `prompt.resolve_unknown`, `prompt.steer_queued`, `prompt.submit`, `reload.env`, `reload.mcp`, `rollback.diff`, `rollback.list`, `rollback.restore`, `rollback.session_diff`, `runtime.describe`, `session.active_list`, `session.automation`, `session.context_breakdown`, `session.control`, `session.control.read`, `session.create`, `session.detach`, `session.events.since`, `session.export`, `session.history`, `session.info`, `session.interrupt`, `session.list`, `session.mutate`, `session.prune`, `session.redirect`, `session.resume`, `session.save`, `session.stats`, `session.steer`, `session.transcript`, `setup.runtime_check`, `setup.status`, `shell.exec`, `skills.reload`, `slash.exec`, `subagent.interrupt`, `subagent.list`, `subagent.steer`, `subagent.tail`, `tools.show`, `worker.adopt`, `worker.persist`, `worker.register`

**Answered by an older handler** (160): `billing.auto_reload`, `billing.charge`, `billing.charge_status`, `billing.keep_going_on_credits`, `billing.state`, `billing.step_up`, `browser.manage`, `cli.exec`, `client.capabilities`, `clipboard.paste`, `command.resolve`, `config.show`, `connectors.accounts`, `connectors.accounts.remove`, `connectors.accounts.rename`, `connectors.catalog`, `connectors.list`, `connectors.policy.get`, `connectors.policy.set`, `connectors.tools`, `cron.manage`, `delegation.status`, `diagnostics.share_nous`, `display.install`, `display.lease.acquire`, `display.lease.release`, `display.observe`, `display.start`, `display.status`, `display.stop`, `display.switchSandboxImage`, `display.thumbnail`, `free_tier.ack_notice`, `free_tier.challenge_result`, `free_tier.claim_nudge`, `free_tier.provision`, `free_tier.status`, `gateway.capabilities`, `groups.demote`, `groups.promote`, `groups.replica_state`, `groups.replicate`, `i18n.catalog`, `i18n.languages`, `image.attach`, `image.detach`, `image.generate`, `image.render`, `input.detect_drop`, `insights.get`, `learning.delete`, `learning.detail`, `learning.edit`, `learning.frames`, `machine.facts`, `mcp.catalog`, `mcp.servers.add`, `mcp.servers.list`, `mcp.servers.oauth.callback`, `mcp.servers.oauth.cancel`, `mcp.servers.oauth.poll`, `mcp.servers.oauth.start`, `mcp.servers.remove`, `mcp.servers.set_api_key`, `mcp.servers.status`, `mcp.servers.test`, `model.disconnect`, `model.save_key`, `onboarding.set_run`, `onboarding.state`, `paste.collapse`, `pdf.attach`, `pet.cancel`, `pet.cells`, `pet.disable`, `pet.export`, `pet.gallery`, `pet.generate`, `pet.generate.status`, `pet.hatch`, `pet.info`, `pet.info.meta`, `pet.remove`, `pet.rename`, `pet.scale`, `pet.select`, `pet.thumb`, `plugins.catalog.lookup`, `plugins.manage`, `profiles.configure`, `profiles.create`, `profiles.describe`, `profiles.get_asset`, `profiles.set_asset`, `project.facts`, `projects.add_folder`, `projects.archive`, `projects.create`, `projects.delete`, `projects.discover_repos`, `projects.for_cwd`, `projects.get`, `projects.list`, `projects.project_sessions`, `projects.record_repos`, `projects.remove_folder`, `projects.set_active`, `projects.set_primary`, `projects.tree`, `projects.update`, `request.answer`, `session.events.stats`, `session.foreign.import`, `session.foreign.list`, `session.foreign.preview`, `session.most_recent`, `shared_metrics.desktop_daily`, `shared_metrics.desktop_dislike`, `shared_metrics.desktop_feature_use`, `shared_metrics.desktop_friction`, `shared_metrics.desktop_onboarding`, `shared_metrics.set`, `shared_metrics.slash_command`, `shared_metrics.startup_latency`, `shared_metrics.status`, `shared_metrics.update_run`, `skills.manage`, `spawn_tree.list`, `spawn_tree.load`, `spawn_tree.save`, `subagent.roster`, `subscription.change`, `subscription.preview`, `subscription.resume`, `subscription.state`, `subscription.upgrade`, `system.battery`, `terminal.resize`, `tools.configure`, `tools.list`, `toolsets.list`, `usage.bars`, `vault.add`, `vault.list`, `vault.lock`, `vault.preview.fill`, `vault.remove`, `vault.source.set`, `vault.sources`, `vault.unlock`, `verification.status`, `voice.record`, `voice.toggle`, `voice.tts`, `wake.feed`, `wake.pause`, `wake.resume`, `wake.start`, `wake.status`, `wake.stop`

<!-- gateway-methods:end -->

Plugins add `plugin.<plugin_id>.<name>` methods (`ctx.register_rpc`) and session events of type `plugin.<plugin_id>.<event>` (`ctx.emit_to_session`); an owner serving them lists `plugin-rpc-v1` in `runtime.describe`'s `capabilities`. See [plugin RPC methods and session events](./plugins/index.md#talk-to-your-own-frontend-plugin-rpc-methods-and-session-events).

`session.active_list`, `session.activate`, and `session.close` are the live-session controls of a standalone `tui_gateway` process (stdio or its own WebSocket). On the shared gateway, `session.activate` and `session.close` are refused: use `session.resume` and `session.detach`. Use `session.list` for saved conversations.

On the shared gateway's `hermes-gateway-v1` socket, a session picker reads stored conversations without attaching: `session.list` takes a `workspace` folder filter and, with `picker: true`, returns each row's `preview`, `last_reply`, `cwd`, `git_branch`, `workspace`, `last_active_at` and `pinned`; with `scope: "stored"` it also takes a `query` (title or id text) and pages with `cursor` / `next_cursor` (capability `session-list-cursor-v1`); `session.transcript({session_id, subagent_id?, cursor?, limit?, order?})` pages a conversation's display rows across its compression lineage from the oldest or the newest, each compaction handoff a `row_kind: "compaction"` row at its position (`{rows, next_cursor, reset}`, opaque cursor, at most 2 MiB a page); an oldest-order cursor also polls for rows written since (`reset: true` when the rows it covered changed shape), and `subagent_id` reads a delegated child of the session. None of them subscribes the caller or starts an agent. Details: [Gateway local sessions](./gateway-local-sessions.md).

Within one authenticated gateway, resuming or activating a live session attaches another event subscriber rather than replacing the previous connection. Streaming and terminal events go to all attached clients; disconnecting one client does not end a session another client is viewing. Existing submit exclusivity and configured busy-input policy remain in force. Attached clients can steer the session's subagents. This does not enable independent gateway processes to write the same session, nor does it imply durable prompt admission across an owner restart.

### Model overrides on `session.create`

`session.create` accepts per-session `model` / `provider` overrides. A pair the provider cannot serve (`model: gpt-5.5` with `provider: anthropic`, or with no `provider` when the profile's configured provider is Anthropic) is refused up front with JSON-RPC code `-32602` instead of minting a session whose first turn fails at the provider; `error.data` carries `model`, `provider` and up to five `suggestions` from that provider's catalog, and `error.message` repeats them. The check is offline and only refuses names Hermes knows belong elsewhere: custom endpoints (`custom`, `custom:<name>`), aggregators (OpenRouter, Nous, …), models in the provider's own family that the curated list has not caught up with, and names no catalog lists are all accepted as before.

### Rewinding history on `prompt.submit`

A rewind / edit / regenerate is a `prompt.submit` that drops part of the stored transcript before running the new turn. Because that write is a destructive rewrite of the session's durable rows, the gateway honors it only when the client states its intent:

| Parameter | Meaning |
|-----------|---------|
| `truncate_before_user_ordinal` | Zero-based index of the user turn to cut at. Everything from that turn onward is dropped. Display-only timeline rows (`display_kind`) are not counted. Must be a real integer — a JSON boolean is refused with code `4004`. |
| `truncate_before_row_id` | Integer SQLite row ID (`messages.id` / `row_id`) of the target user turn to cut at. Preferred durable address. When both ordinal and row ID are provided, gateway verifies they match (returning `4030` on mismatch). An unknown/stale row ID is refused with `4018` — it does **not** fall back to the ordinal. |
| `confirm_truncate` | Required whenever an ordinal, message ID, or row ID is sent. Declares that this submit really is a rewind, not an ordinary send that happens to carry leftover parameters. Sending it without a target is refused with code `4004`. |
| `confirm_empty_truncate` | Additionally required when the cut would leave the transcript empty (ordinal `0`). |
| `confirm_deep_truncate` | Send it when the cut would archive more than one user turn (anything deeper than regenerating or editing the last turn), after asking the user. This release logs an unconfirmed deep cut and still runs it, so older clients keep working. The next release will refuse it with code `4033` (`error.data` carries `archived_messages` and `archived_user_turns`, and nothing is written). |

A truncation parameter without `confirm_truncate` is refused with code `4004` or `4029` and nothing is written. Hosts that implement rewind must set the flag at the moment the user asks for it, and must never keep truncation parameters in state across ordinary submits. Prefer `truncate_before_row_id` (from resume `row_id` / `_row_id`) over ordinals; keep the ordinal as a back-compat / optimistic-row path only when no durable id is available yet.

A truncating submit is never absorbed by the busy-input policy. While a turn is still running, an ordinary `prompt.submit` is steered, redirected, or queued (`display.busy_input_mode`), but a rewind / edit / regenerate refuses with code `4009` (`session busy`) instead — queueing it would drop the history cut and run the edit as a plain follow-up after the un-edited turn. Hosts call `session.interrupt` and retry the same submit until it lands; the Desktop app does this automatically, so editing a message while Hermes is still thinking stops the live turn and reruns from the edited prompt.

On a successful truncating submit against a durable session, the `prompt.submit` result additionally carries `survivor_user_row_ids` — the fresh post-rewrite row IDs of the surviving user turns, in visible-user-ordinal order. The rewrite re-inserts the kept prefix as new rows, so every row ID the host cached before the rewind is stale afterward; rebind cached IDs from this list (a `null` entry means that turn has no durable ID — drop the cached one) or the next rewind targeting an older surviving turn will be refused with `4018`.

### Events streamed back

`message.delta`, `message.complete`, `tool.start`, `tool.generating`, `tool.complete`, `gateway.ready`, `request.cancel`, plus session lifecycle and error events.

### Server→client requests (questions the agent asks you)

Approvals, clarify questions, sudo/secret prompts, vault unlock, MCP setup and the desktop read/act bridges are **JSON-RPC requests from the gateway to the client**, not events. The frame carries a string id, and the client answers with a normal JSON-RPC response bearing the same id:

```
← {"jsonrpc":"2.0","id":"srq-7","method":"approval","params":{"session_id":"…","request_id":"…","command":"rm -rf build","description":"…"}}
→ {"jsonrpc":"2.0","id":"srq-7","result":{"choice":"once"}}
```

Methods: `approval` → `{choice}`; `clarify` → `{answers}` keyed by question id, or `{}` to cancel (`clarify.lock` locks one answer early; `null` skips that question); `sudo`, `secret`, `vault.code`, `vault.unlock_prompt` → `{value}`; `connection` → `{settled_by, targets}` (the `manage_connections` card: one outcome per target); `terminal.read`, `window.read`, `preview.act`, `tour` → `{value}` (JSON text). Respond with a JSON-RPC error (`-32601`) for a method your host does not implement so the agent fails fast instead of waiting out the timeout.

**Advertise that you answer them (breaking for existing WebSocket integrations).** Once per connection, after `gateway.ready`, call `client.capabilities` with `{"server_requests": true}`; the result lists the request methods this backend may send. A WebSocket client that never does is treated as a build that predates server→client requests: the gateway fails every such request for it immediately (the agent sees the same "no answer" an error response produces; an approval is withdrawn, not denied) instead of stalling for the full deadline. There is no grace path — a third-party WebSocket client that answered `clarify`/`approval`/`sudo`/… before this change but never sends `client.capabilities` now has every such request refused until it adds the one call. A session with no client attached is not affected — its open questions wait in `open_requests` for the reconnect replay. The stdio TUI, the desktop app and the dashboard advertise through the shared `JsonRpcRequestChannel`.

When the gateway withdraws a question (timeout, interrupt, answered from another surface) it emits `request.cancel` `{ id, method, reason }`; clear only the matching prompt. `session.resume` / `session.activate` results and `session.events.since` carry `open_requests` — the still-open frames — so a reconnecting client re-renders (and can still answer) them.

### Rebuilding the in-flight turn on reconnect

`session.resume` / `session.activate` results carry `inflight` — the turn still running (or the retained failed one) that history does not hold yet: `user`, `assistant` streamed so far, `streaming`, mid-turn `corrections`, and error fields. When the turn was started by the gateway rather than typed by a person (a background-process completion, an async delegation result, a hidden scaffolding prompt) `inflight` also carries the same `display_kind` / `display_metadata` the persisted `messages` row will get, so a client renders the live prompt exactly as it will render history after the turn lands — a `process_complete` timeline marker with `display_metadata.display_text`, nothing at all for `hidden`. Both fields are absent for genuine user input; never infer origin from the prompt text (a user quoting a marker string is still a user).

### Pi-style RPC mapping

Every command in the Pi-mono RPC spec ([issue #360](https://github.com/NousResearch/hermes-agent/issues/360)) has a TUI-gateway equivalent:

| Pi command | Hermes equivalent |
|------------|-------------------|
| `prompt` | `prompt.submit` (or ACP `session/prompt`) |
| `steer` | `session.steer` |
| `follow_up` | `prompt.submit` queued after current turn |
| `abort` | `session.interrupt` |
| `set_model` | `command.dispatch` for `/model <provider:model>` (mid-session, persistent) |
| `compact` | `session.compress` |
| `get_state` | `session.status` |
| `get_messages` | `session.history` |
| `switch_session` | `session.resume` |
| `fork` | `session.branch` |
| `ui_request` / `ui_response` | server→client requests `clarify` / `sudo` / `secret` / `approval` answered by JSON-RPC response frames |

---

## OpenAI-Compatible API Server

`gateway/platforms/api_server.py` exposes hermes over HTTP for any client that already speaks the OpenAI format. Useful when you want a web frontend, a curl-driven CI runner, or a non-Python consumer.

Endpoints:

```
POST /v1/chat/completions        OpenAI Chat Completions (streaming via SSE)
POST /v1/responses               OpenAI Responses API (stateful)
POST /v1/runs                    Start a run, returns run_id (202)
GET  /v1/runs/{id}               Run status
GET  /v1/runs/{id}/events        SSE stream of lifecycle events
POST /v1/runs/{id}/approval      Resolve a pending approval
POST /v1/runs/{id}/steer         Inject mid-run guidance at the next tool boundary
POST /v1/runs/{id}/resolve-unknown Acknowledge an exact restart-unknown admission
POST /v1/runs/{id}/stop          Interrupt the run
GET  /v1/capabilities            Machine-readable feature flags
GET  /v1/models                  Lists hermes-agent
GET  /api/model/options          Provider-aware picker inventory
GET  /health, /health/detailed
```

Setup, headers (`X-Hermes-Session-Id`, `X-Hermes-Session-Key`), and frontend wiring: [API Server](../user-guide/features/api-server).

The browser-extension controller protocol has been removed; to let Hermes drive a browser you
already run, use `browser.cdp_url` or `/browser connect` — see [Browser-extension control (removed)](../user-guide/features/api-server#browser-extension-control-removed).

### Model catalog surfaces

The OpenAI-compatible API intentionally keeps `GET /v1/models` minimal: it is
the compatibility endpoint frontends expect, not the full Hermes provider/model
picker catalog.

If an external control plane needs Hermes' curated provider rows, per-model
pricing, or capability hints, use one of the authenticated picker surfaces:

- API server REST: `GET /api/model/options` with the API-server bearer key
- Dashboard backend REST: `GET /api/model/options` with `X-Hermes-Session-Token`
- TUI gateway RPC: `model.options`

Those surfaces share the same payload builder and the same custom-provider
probe policy:

- Normal open: probe only the current custom provider so offline saved
  endpoints do not stall the picker.
- Explicit refresh (`refresh=1` or `refresh: true`): bust the provider-model
  cache and probe all saved custom providers so live catalogs repopulate fully.

Use `/v1/models` for OpenAI-client compatibility. Use `/api/model/options` or
`model.options` when you are building a Hermes-aware model picker.

`POST /v1/runs/{id}/steer` is the HTTP equivalent of Hermes `/steer`: it does not create a new user turn or immediately rewrite the assistant output already in flight. Instead, the text is appended to the live run and becomes visible to the agent after the next tool boundary, so it can course-correct without discarding the current tool-calling loop.

`/v1/runs/{id}/steer` is only accepted while the run status is `running`. Queued, approval-paused, stopping, cancelled, failed, and completed runs return `409 run_not_accepting_steer`, even if the server still retains internal agent references during cooperative shutdown.

A `200` (and the `run.steered` event) means the text was **queued**, not that the agent consumed it. If a steer lands after the agent's final response — with no later tool boundary to deliver it at — the undelivered text is returned as `pending_steer` on the terminal event (`run.completed`, `run.failed`, or `run.cancelled`) and run status, so the client can replay it as the next user turn instead of losing it.

#### Terminal run status

The terminal status of a run is derived from how the agent's turn actually ended, and the terminal event name always matches it (`run.<status>`):

| Turn outcome | Status | Terminal event | Flags on the event / status |
|---|---|---|---|
| Final answer produced | `completed` | `run.completed` | `completed: true` |
| Interrupted (`/stop`, or an interrupt inside the agent) | `cancelled` | `run.cancelled` | `completed: false`, `interrupted: true`, `turn_exit_reason` naming the issuer — `interrupted_by_user` for a human stop, `interrupted_by_system(<issuer>)` / `interrupted_during_api_call(<issuer>)` when a watchdog (e.g. `cron_inactivity_watchdog`, `turn_liveness_watchdog`, `gateway_inactivity_watchdog`, `session_turn_lease_lost`) ended the turn |
| Provider/agent failure | `failed` | `run.failed` | `completed: false`, `error` |
| Gateway shut down while the run was active (`/v1/runs` only) | `interrupted` | `run.interrupted` | `error: "Gateway shutdown interrupted the run."` — recorded before the agent is interrupted and never overwritten by the turn's late result; a client `/stop` still settles as `cancelled` |
| Ended without finishing (iteration budget, truncated or partial reply) | `failed` | `run.failed` | `completed: false`, `partial` when applicable, `turn_exit_reason` (e.g. `max_iterations_reached(60/60)`), `output` with any fallback text |

A run is never reported as `completed` with `completed: false` or `partial: true` in the same payload. The same rule applies to `/api/sessions/{id}/chat/stream`, whose `assistant.completed` payload carries the real `completed` / `partial` / `interrupted` flags and whose terminal event is `run.completed`, `run.failed`, or `run.cancelled`.

`POST /v1/runs/{id}/resolve-unknown` is a canonical-authority recovery
operation, not a generic status mutation. Send exactly the `admission_id` and
integer `execution_generation` projected by the run status. The server binds the
authenticated owner, selected profile, path run, API principal, target session,
admission, and generation before the authority performs its compare-and-set.
Resolution settles the lost head as interrupted and releases the next FIFO item;
it never requeues the head. Feature-detect
`features.run_unknown_resolution` because legacy execution mode does not
advertise this endpoint.

New canonical API admissions retain an opaque owner scope in the admission row,
so keyed and non-keyed runs can be authorized after an adapter restart for as
long as that canonical admission remains available. This does not change
non-keyed request semantics: identical submissions still create distinct runs.
Pre-upgrade non-keyed admissions contain no owner scope and remain deliberately
unresolvable after restart; there is no safe credential ownership to backfill.
Existing keyed records retain their replay-ledger recovery path. Explicit
session retirement removes the canonical payload and therefore ends this owner
recovery path. Replay reservation and canonical admission still use separate
databases, so this endpoint does not provide a global exactly-once guarantee.

---

## Which one should I use?

- **You're writing an IDE plugin and the IDE already speaks ACP** → ACP. Zero protocol work on the IDE side.
- **You're writing a custom desktop / web / TUI host and want every Hermes feature** (slash commands, approvals, clarify, multi-agent, session branching) → TUI gateway JSON-RPC.
- **You want any OpenAI-compatible frontend, a language-agnostic HTTP client, or curl-driven automation** → API server.
- **You want a Python in-process embed without a subprocess** → import `run_agent.AIAgent` directly. See [Agent Loop](./agent-loop).
- **Your client needs the user's clipboard image** (image paste) → run `hermes clipboard image --out PATH --json` on the user's machine. It is device-local, never a gateway call, because the gateway may be remote. See [`hermes clipboard`](../reference/cli-commands.md#hermes-clipboard).

---

## Model hot-swapping

Mid-session model switching works on every surface — it's the `/model` slash command under the hood.

- **CLI / TUI:** `/model claude-sonnet-4` or `/model openrouter:anthropic/claude-sonnet-4.6`
- **TUI gateway RPC:** `command.dispatch` with `{"command": "/model claude-sonnet-4"}`
- **ACP:** the IDE sends the slash command as a prompt; the agent dispatches it
- **API server:** include a `model` field in the request body

Provider-aware resolution (the same model name picks the right format for whatever provider you're on) is built in. See `hermes_cli/model_switch.py`.

---

## A note on `--mode rpc`

Hermes does not have a `--mode rpc` flag. The three protocols above already cover the use cases — ACP for IDE-protocol clients, the TUI gateway for stdio JSON-RPC hosts, and the API server for HTTP. If you find a real gap that none of them fill, open an issue with the concrete consumer you're building.
