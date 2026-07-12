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
hermes acp --bootstrap      # print install snippet for an ACP-capable IDE
```

---

## TUI Gateway JSON-RPC

`tui_gateway/server.py` is the protocol the Ink TUI (`hermes --tui`) and the embedded dashboard PTY bridge talk to. Any external host can speak the same protocol over stdio (or WebSocket via `tui_gateway/ws.py`).

The generic gateway has a broad, trusted-host surface. Native/mobile clients should use the capability-negotiated, fail-closed WebSocket grant below instead of assuming that every generic gateway method is remotely available.

### Native/mobile WebSocket contract

The Mobile Client Contract is an additive profile of the existing `/api/ws` JSON-RPC transport. It does not add a mobile backend or another source of conversation state. Hermes remains authoritative for history, live execution, approvals, and durable mutation outcomes.

#### Authenticate and mint one connection ticket

Complete the deployment's configured dashboard login flow (beginning at `/auth/login?provider=<provider>`) so the HTTP client owns an authenticated dashboard session. Then request the minimum scopes needed by one mobile WebSocket:

```http
POST /api/auth/ws-ticket
Content-Type: application/json

{
  "audience": "hermes.mobile",
  "scopes": [
    "conversation.read",
    "conversation.write",
    "conversation.control"
  ]
}
```

The response is:

```json
{
  "ticket": "<opaque bearer value>",
  "ttl_seconds": 30,
  "audience": "hermes.mobile",
  "granted_scopes": [
    "conversation.read",
    "conversation.write",
    "conversation.control"
  ]
}
```

Immediately connect to `wss://<deployment>/api/ws?ticket=<ticket>`. The ticket is a single-use, 30-second bearer credential for one WebSocket. Mint a new one for every reconnect and after every failed upgrade; credential validation consumes a valid ticket before later Host/Origin checks. A mobile-audience ticket is rejected outside `/api/ws`.

Because the ticket appears in the URL, use `https://` and `wss://` on non-loopback or otherwise untrusted links. Cleartext HTTP/WebSocket is appropriate only for an operator-explicit trusted local deployment. WebSocket close code `4401` means the credential was missing or rejected. Code `4403` means the gateway was unavailable or a request-boundary check rejected the upgrade.

Do not use a bodyless ticket-mint request from a mobile client. Bodyless `POST /api/auth/ws-ticket` deliberately preserves the legacy dashboard's full authority rather than minting a scoped mobile grant.

#### Validate the server hello and effective grant

The first accepted WebSocket frame is the existing `gateway.ready` event. It is the server hello; there is no client-hello method:

```json
{
  "jsonrpc": "2.0",
  "method": "event",
  "params": {
    "type": "gateway.ready",
    "payload": {
      "skin": "<name>",
      "server": {
        "version": "<release>",
        "release_date": "<date>",
        "instance_id": "<process identity>"
      },
      "protocol": {
        "name": "hermes.tui.jsonrpc",
        "major": 1
      },
      "contract": {
        "name": "hermes.mobile",
        "major": 1
      },
      "schemas": {
        "gateway.ready": 1,
        "authorization.grant": 1,
        "authorization.error": 1,
        "session.synchronization": 1,
        "session.event": 1,
        "mutation.receipt": 1,
        "approval.lifecycle": 1
      },
      "capabilities": {
        "auth.ws_scopes": {
          "version": 1
        },
        "conversation.sync": {
          "version": 1,
          "delta_offsets": {
            "unit": "utf8_bytes"
          },
          "replay": {
            "max_events": 512,
            "max_bytes": 1048576
          }
        },
        "mutation.idempotency": {
          "version": 1,
          "methods": [
            "prompt.submit",
            "session.interrupt",
            "approval.respond",
            "session.delete"
          ],
          "status_method": "mutation.status"
        },
        "interaction.lifecycle": {
          "version": 1,
          "kinds": ["approval"],
          "response_methods": ["approval.respond"]
        }
      },
      "authorization": {
        "subject": "<authenticated subject>",
        "provider": "<identity provider>",
        "audience": "hermes.mobile",
        "scopes": ["conversation.read"]
      }
    }
  }
}
```

The limits and scopes in this example illustrate their wire locations; use the values the server sends. `gateway.ready.authorization` is the effective server-enforced grant, even if it differs from what the client requested. Stop using the mobile contract if its audience is not `hermes.mobile`.

Gate each feature independently on the exact advertised capability version and required schema major. A contract major, Hermes release version, or observed behavior does not imply a capability. Treat an absent capability or an unknown major as unsupported; ignore unknown additive fields.

#### Mobile scope and parameter allowlist

Every mobile ticket requires `conversation.read`. The only supported mobile scopes are `conversation.read`, `conversation.write`, `conversation.control`, and `conversation.delete`. The server permits these exact method/parameter combinations:

| Method | Required scopes | Allowed parameters |
| --- | --- | --- |
| `session.list` | `conversation.read` | none |
| `session.active_list` | `conversation.read` | `current_session_id` |
| `session.activate` | `conversation.control` | `session_id` |
| `session.history` | `conversation.read` | `session_id` |
| `session.create` | `conversation.write`, `conversation.control` | `cols`, `title` |
| `session.resume` | `conversation.control` | `cols`, `cursor`, `session_id` |
| `prompt.submit` | `conversation.write` | `client_request_id`, `expected_stored_session_id`, `session_id`, `text` |
| `session.interrupt` | `conversation.control` | `client_request_id`, `expected_stored_session_id`, `session_id` |
| `approval.respond` | `conversation.control` | `approval_id`, `choice`, `client_request_id`, `expected_stored_session_id`, `reason`, `session_id` |
| `session.delete` | `conversation.delete` | `client_request_id`, `session_id` |
| `mutation.status` | `conversation.read` | `client_request_id` |

An unmapped method, unsupported parameter, or missing scope fails closed with JSON-RPC error `4030`. Its `error.data` reports `reason`, `method`, `required_scope`, `required_scopes`, `missing_scopes`, `granted_scopes`, and `grantable`; parameter denials also report `parameter`. Reasons are `missing_scope`, `method_not_available_to_mobile`, or `parameter_not_available_to_mobile`. `grantable: false` means that no additional defined mobile scope can authorize the operation.

#### Reconcile one authoritative conversation

`session.create` and `session.resume` return the same schema-major-1 `synchronization` object when `conversation.sync` version 1 is present:

```text
synchronization
├── schema_major
├── snapshot
│   ├── schema_major
│   ├── server_instance_id, stream_id, revision, watermark
│   ├── conversation_id, stored_session_id, live_session_id
│   ├── messages, inflight_turn, active_tools, pending_interactions
│   └── status
└── recovery
    ├── outcome, reason
    ├── cursor, events, snapshot_required
    └── available_after (gap only)
```

The three conversation identities are not interchangeable:

- `conversation_id` is the stable root of the conversation/compression lineage.
- `stored_session_id` is the current durable tip and can rotate after compression.
- `live_session_id` is the process-local route sent as `session_id` while addressing the attached live session.

The process-lifetime `server_instance_id` changes after restart. The `stream_id` changes whenever that live stream is rebuilt. A cursor is exactly `{server_instance_id, stream_id, sequence}`. Each synchronized event carries `schema_major`, `stream_id`, monotonic `sequence`, `type`, `session_id`, and an optional `payload`.

Snapshot `messages` are display records. User, system, and assistant records have `role` and `text`; assistant records can also carry additive reasoning fields. Tool records have `role: "tool"`, `name`, and `context`. `inflight_turn` is `null` or `{turn_id, user, assistant, streaming}`. `active_tools` and `pending_interactions` are server-owned descriptors. Snapshot `status` is `idle`, `working`, or `waiting`.

| Recovery outcome | Meaning | Client action |
| --- | --- | --- |
| `complete` | Every event after the supplied cursor through the returned watermark is in `events`; `snapshot_required` is `false`. | If local state exactly matches that cursor, replay events in sequence. Installing the returned snapshot instead is also safe. |
| `gap` | `reason` is `replay_evicted`; `events` is empty; `snapshot_required` is `true`; `available_after` marks the unavailable boundary. | Replace local state with the returned snapshot. |
| `reset` | `reason` is `cursor_missing`, `server_instance_changed`, `stream_changed`, or `cursor_invalid`; `events` is empty and `snapshot_required` is `true`. | Replace local state and discard state tied to the old server or stream. |

Buffer live events while create/resume is in flight. Choose complete replay or the returned snapshot, advance the local base to the snapshot watermark, discard duplicates at or below it, and then apply buffered events for the same server and stream in increasing order. Resume again if a later live gap appears.

`message.delta` additionally carries one `turn_id` and an absolute UTF-8-byte `offset`, as named by `conversation.sync.delta_offsets.unit`. Compare the already-applied UTF-8 prefix before appending an overlapping delta.

Replay retention is bounded by the advertised event and byte limits. It begins after a mobile-audience transport first attaches. A legacy transport can later attach without changing its legacy response or event shapes.

#### Retry only advertised durable mutations

Only methods in `mutation.idempotency.methods` have durable at-most-once receipts. Currently those are `prompt.submit`, `session.interrupt`, `approval.respond`, and `session.delete`. `session.create` is not covered, so never blindly retry a create whose result is uncertain.

Each covered call needs a `client_request_id` containing 1 to 256 characters after trimming surrounding whitespace. Receipts are scoped to the effective `provider` and `subject`. Reuse the exact identity and semantic fields after an uncertain disconnect. The same fingerprint replays its stored success or handler error with `mutation.deduplicated: true`; changed method, resource, or semantic fields produce `mutation_conflict`.

For prompt, interrupt, and approval response, retain one Hermes-issued member of the conversation lineage—normally `snapshot.conversation_id`—as `expected_stored_session_id`, and send that exact string for every retry. Hermes validates it against the current lineage before first execution. Approval choice is normalized by trimming and lowercasing; other semantic values should be resent exactly.

A mutation receipt is attached at `result.mutation` or `error.data.mutation`:

```json
{
  "client_request_id": "client-generated-id",
  "deduplicated": false,
  "state": "completed"
}
```

A prompt can initially report mutation state `in_progress` while Hermes proves the exact user turn reached durable history. `mutation.status` returns `{client_request_id, method, state, outcome}` for a receipt owned by the effective principal.

| Code | Wire condition | Meaning |
| --- | --- | --- |
| `-32602` | Validation failure. When present, `error.data.reason` is `client_request_id_required`, `invalid_client_request_id`, `durable_resource_id_required`, `durable_lineage_id_required`, or `approval_choice_required`. | Correct the request before retrying. |
| `4019` | No machine `reason`; the message reports a stored-session mismatch. | Reconcile and use a valid Hermes-issued lineage identity. |
| `4030` | `missing_scope`, `method_not_available_to_mobile`, or `parameter_not_available_to_mobile` | The effective grant cannot perform the call as sent. |
| `4040` | `mutation_not_found` | This principal has no matching receipt. |
| `4090` | `mutation_conflict` | The identity is already bound to different semantics. |
| `4091` | `mutation_outcome_unknown` | Execution may have begun and Hermes will not run it again automatically. Surface uncertainty. |
| `4092` | `mutation_in_progress` | Poll status or retry the same identity later. |
| `5037` | `mutation_store_unavailable`, `mutation_policy_unavailable`, `mutation_preflight_unavailable`, or `mutation_outcome_unknown` | Follow the returned reason; do not switch identities merely to force another execution. |

Completed receipts survive reconnect and restart. An unfinished receipt owned by a vanished process becomes `outcome_unknown`, not a new opportunity to execute.

#### Recover and resolve one exact approval

Approval recovery is available only when `interaction.lifecycle` version 1 names `approval`, its response methods include `approval.respond`, schema `approval.lifecycle` is major 1, and mutation idempotency also covers `approval.respond`.

`approval.request` carries Hermes-owned `approval_id`, `created_at`, `expires_at`, `state`, and `resolution`, plus opaque presentation fields that Hermes recursively force-redacts. The matching item in `snapshot.pending_interactions` adds `kind: "approval"`. Presentation fields never grant authority.

Terminal events are `approval.resolved`, `approval.expired`, and `approval.stale`. They retain the same `approval_id`; a terminal `resolution` contains `choice`, `resolved_at`, `reason`, and opaque `metadata`.

A mobile `approval.respond` must name the current `live_session_id`, retained lineage identity, Hermes-issued `approval_id`, choice, and a fresh durable client request identity. Valid choices are `once`, `session`, `always`, and `deny`; optional `reason` participates in durable semantics. The legacy ID-less FIFO and `all` response are unavailable to mobile grants. Exact response outcomes are `resolved`, `already_resolved`, `expired`, `stale`, `not_found`, and `invalid_choice`.

Pending approvals survive a transport disconnect only while the same server process and live stream retain them. Terminal tombstones are short-lived and process-local. On `reset`, discard old pending approval state and trust the new snapshot. A completed `approval.respond` receipt remains durable across restart.

#### Compact reconnect sequence

1. Authenticate, mint a minimum-scope mobile ticket, connect, and validate `gateway.ready` plus its effective grant.
2. Create a conversation and retain the snapshot identities and recovery cursor. Do not retry an uncertain create.
3. Submit a prompt with a fresh `client_request_id`, current `live_session_id`, and retained lineage identity.
4. If the socket drops before certainty, mint a fresh ticket, reconnect, resume with the durable session identity plus cursor, and apply complete replay or the gap/reset snapshot.
5. Retry the prompt with the identical durable identity and fields; accept the receipt and authoritative history as proof of one user turn.
6. Retain every `approval_id`. After another uncertain disconnect, reconnect with a fresh ticket and recover the same approval from replay or `pending_interactions`.
7. Resolve it once with a new durable identity. Reuse that identity for an uncertain retry and accept the stored outcome or matching terminal event.
8. After any gap/reset replacement, reconcile all live identities before enabling another consequential action.

### Legacy/full-authority method catalog (selected)

The catalog below describes the trusted generic gateway surface used by the TUI and dashboard. It is not the mobile allowlist above.

```
prompt.submit           prompt.background       session.steer
session.create          session.list            session.active_list
session.activate        session.close           session.interrupt
session.history         session.compress        session.branch
session.title           session.usage           session.status
clarify.respond         sudo.respond            secret.respond
approval.respond        config.set / config.get commands.catalog
command.resolve         command.dispatch        cli.exec
reload.mcp              reload.env              process.stop
delegation.status       subagent.interrupt      spawn_tree.save / list / load
terminal.resize         clipboard.paste         image.attach
```

`session.active_list`, `session.activate`, and `session.close` are the process-local live-session controls used by the TUI session switcher. Use `session.list` / `/resume` for saved transcript discovery; use the active-session methods only for sessions that are currently open in the TUI gateway process.

### Events streamed back

`message.delta`, `message.complete`, `tool.start`, `tool.progress`, `tool.complete`, `approval.request`, `approval.resolved`, `approval.expired`, `approval.stale`, `clarify.request`, `sudo.request`, `sudo.expire`, `secret.request`, `secret.expire`, `gateway.ready`, plus session lifecycle and error events. `sudo` and `secret` expiry events carry their original `request_id`; approval lifecycle events carry `approval_id`. External hosts should clear only the matching pending interaction.

### Pi-style RPC mapping

Every command in the Pi-mono RPC spec ([issue #360](https://github.com/NousResearch/hermes-agent/issues/360)) has a TUI-gateway equivalent:

This table describes the generic full-authority gateway. A mobile client may use only the equivalent that also appears in the mobile allowlist above.

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
| `ui_request` / `ui_response` | `clarify.respond` / `sudo.respond` / `secret.respond` / `approval.respond` |

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
POST /v1/runs/{id}/stop          Interrupt the run
GET  /v1/capabilities            Machine-readable feature flags
GET  /v1/models                  Lists hermes-agent
GET  /health, /health/detailed
```

Setup, headers (`X-Hermes-Session-Id`, `X-Hermes-Session-Key`), and frontend wiring: [API Server](../user-guide/features/api-server).

---

## Which one should I use?

- **You're writing an IDE plugin and the IDE already speaks ACP** → ACP. Zero protocol work on the IDE side.
- **You're writing a trusted custom desktop / web / TUI host and want the full gateway surface** (slash commands, approvals, clarify, multi-agent, session branching) → TUI gateway JSON-RPC with its legacy/full-authority grant.
- **You're writing a native/mobile client that reconnects over the network** → the capability-negotiated `/api/ws` Mobile Client Contract above.
- **You want any OpenAI-compatible frontend, a language-agnostic HTTP client, or curl-driven automation** → API server.
- **You want a Python in-process embed without a subprocess** → import `run_agent.AIAgent` directly. See [Agent Loop](./agent-loop).

---

## Model hot-swapping

Mid-session model switching is available on the full-authority surfaces below. The scoped mobile allowlist does not expose `command.dispatch`; do not infer model-control authority from this generic catalog.

- **CLI / TUI:** `/model claude-sonnet-4` or `/model openrouter:anthropic/claude-sonnet-4.6`
- **TUI gateway RPC:** `command.dispatch` with `{"command": "/model claude-sonnet-4"}`
- **ACP:** the IDE sends the slash command as a prompt; the agent dispatches it
- **API server:** include a `model` field in the request body or set `X-Hermes-Model`

Provider-aware resolution (the same model name picks the right format for whatever provider you're on) is built in. See `hermes_cli/model_switch.py`.

---

## A note on `--mode rpc`

Hermes does not have a `--mode rpc` flag. The three protocols above already cover the use cases — ACP for IDE-protocol clients, the TUI gateway for stdio JSON-RPC hosts, and the API server for HTTP. If you find a real gap that none of them fill, open an issue with the concrete consumer you're building.
