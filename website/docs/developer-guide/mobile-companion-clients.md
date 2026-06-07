---
sidebar_position: 26
title: "Mobile Companion Clients"
description: "Developer guide for building thin mobile and third-party companion clients against the Hermes dashboard WebSocket JSON-RPC surface"
---

# Mobile Companion Clients

Hermes runs on a desktop or server. Mobile and third-party clients are **thin companions**: they connect to an already-running Hermes dashboard, stream agent events, submit prompts, and handle interactive flows (approvals, clarify, sudo, secrets). They do not embed the full agent runtime.

## Scope

This guide covers companion clients that talk to Hermes over the dashboard WebSocket JSON-RPC surface exposed by `hermes dashboard` (or an equivalent `hermes_cli/web_server.py` deployment).

In scope:

- Native mobile apps (Android, iOS) and other third-party UIs that drive an existing Hermes session remotely.
- Wire protocol, authentication, minimum UX, security, and testing expectations for v1 companions.

Companion clients reuse the same RPC/event contract that Ink (`hermes --tui`) uses over stdio, routed through the dashboard WebSocket endpoint.

## Non-goals

- **Running Hermes on the phone.** The agent loop, tools, terminal backends, and model providers stay on the desktop/server host.
- **Reimplementing the dashboard embedded TUI.** `/api/pty` is a PTY-over-WebSocket bridge for the browser Chat tab's xterm.js terminal - not the recommended transport for native mobile clients.
- **Shipping a reference Android app in this tree.** The native Android reference app lives in a separate repository unless Hermes maintainers explicitly request otherwise.
- **Gateway messaging platforms.** Telegram, Discord, Slack, and other adapters are documented separately; this guide is for direct JSON-RPC companions only.

## Protocol

### Transport endpoint

`hermes_cli/web_server.py` exposes `/api/ws` and routes accepted upgrades to `tui_gateway.ws.handle_ws`:

```python
@app.websocket("/api/ws")
async def gateway_ws(ws: WebSocket) -> None:
    ...
    from tui_gateway.ws import handle_ws
    await handle_ws(ws)
```

### Wire format

`tui_gateway/ws.py` reuses the same JSON-RPC payload shape as Ink's stdio gateway. Over WebSocket, each text message carries one JSON-RPC object: the server reads with `receive_text()` and writes with `send_text()`. On connect, the server emits a `gateway.ready` event immediately, then processes inbound requests and streams responses/events.

Example connection flow:

1. Open `wss://<host>/api/ws?<auth-query-param>`.
2. Wait for `gateway.ready`.
3. Send one JSON-RPC request object per WebSocket text message (`method`, `params`, `id`).
4. Receive one JSON-RPC response or server-pushed event per WebSocket text message (`type` field on event frames).

### Reference client implementation

`apps/shared/src/json-rpc-gateway.ts` contains `JsonRpcGatewayClient` - a reusable client for WebSocket connect/disconnect, request/response correlation, event subscription, and `ConnectionState` tracking (`idle` -> `connecting` -> `open` -> `closed` / `error`). Desktop and web surfaces already use this module; mobile companions can port the same patterns.

### Useful v1 RPCs

Minimum companion surface for a usable chat experience:

| RPC | Purpose |
|-----|---------|
| `session.list` | Discover saved sessions |
| `session.resume` | Attach to a session |
| `prompt.submit` | Send a user message / start a turn |
| `session.interrupt` | Cancel the running agent |
| `approval.respond` | Answer tool-approval prompts |
| `clarify.respond` | Answer clarify prompts |
| `sudo.respond` | Answer elevated-permission prompts |
| `secret.respond` | Supply secrets requested mid-turn |
| `commands.catalog` | Slash-command discovery (empty query) |
| `complete.slash` | Slash-command completion while typing |

Listen for streaming events such as `message.delta`, `message.complete`, `tool.start`, `tool.progress`, `tool.complete`, `approval.request`, `clarify.request`, `sudo.request`, and `secret.request`. See [Programmatic Integration](./programmatic-integration) for the broader method catalog.

### What not to use

`/api/pty` is the dashboard embedded TUI/terminal transport (raw PTY bytes for xterm.js). Native mobile clients should use `/api/ws` JSON-RPC instead. `/api/pub` and `/api/events` exist for the dashboard's PTY sidecar event fan-out and are not required for standalone companions.

## Authentication modes

WebSocket auth is enforced in `hermes_cli/web_server.py` before `handle_ws` runs. Two modes matter for companions:

### Gated mode (OAuth / public bind)

When the dashboard auth gate is active (`app.state.auth_required`), companions authenticate with a **short-lived, single-use `?ticket=`** query parameter:

1. Establish an authenticated HTTP session (OAuth cookie flow via the dashboard login).
2. `POST /api/auth/ws-ticket` to mint a ticket (30s TTL, single use).
3. Connect immediately: `wss://<host>/api/ws?ticket=<ticket>`.

The legacy `?token=` path is **rejected** in gated mode. Server-spawned internal children use `?internal=` instead; third-party companions should use tickets.

### Non-gated loopback / `--insecure`

On loopback bind or with `--insecure`, companions may use the ephemeral **`?token=`** query parameter. The token is the process-lifetime `_SESSION_TOKEN` injected into the dashboard SPA (or set via `HERMES_DASHBOARD_SESSION_TOKEN`). Constant-time comparison is used server-side.

### Host, Origin, and peer checks

Host/Origin/peer checks also live in `hermes_cli/web_server.py`:

- `_ws_request_is_allowed` combines `_ws_host_origin_is_allowed` (Host/Origin must match the bound dashboard host - DNS-rebinding guard) and `_ws_client_is_allowed` (loopback peer restriction unless gated or non-loopback bind).
- Failed auth closes with code `4401`; failed boundary checks close with `4403`.

Packaged native apps connecting from non-browser origins should expect credential checks to be the primary auth boundary; see `_ws_host_origin_is_allowed` for how non-`http(s)` Origin values are handled.

## Minimum companion UX

A v1 companion should provide:

1. **Connect** - configurable base URL, auth flow appropriate to gated vs loopback mode, visible connection state.
2. **Session picker** - `session.list` + `session.resume`.
3. **Chat transcript** - render `message.delta` / `message.complete`; show tool activity from `tool.*` events.
4. **Composer** - `prompt.submit` with in-flight disable; `/stop` equivalent via `session.interrupt`.
5. **Interactive prompts** - modal or dedicated screens for `approval.request`, `clarify.request`, `sudo.request`, `secret.request`, wired to the corresponding `*.respond` RPCs.
6. **Slash commands (optional but recommended)** - `commands.catalog` + `complete.slash` for discoverability.

Companions should degrade gracefully on disconnect (surface `ConnectionState`, allow retry) and never cache tickets beyond their single-use TTL.

## Android reference app boundary

The Hermes **runtime stays on desktop/server**. An Android client is a remote control and notification surface - not a second agent host.

The reference native Android app is maintained **outside this repository** (companion repo) unless Hermes maintainers explicitly decide to merge it upstream. This tree documents the protocol and security contract; it does not ship the Android UI project.

When building the reference app:

- Target `/api/ws` + `JsonRpcGatewayClient` patterns, not `/api/pty`.
- Implement gated auth via the dashboard OAuth session + `POST /api/auth/ws-ticket`.
- Keep tool execution, file access, and model calls on the Hermes host.

## Security checklist

- [ ] Use `wss://` in production; never send tickets or tokens over plaintext unless on trusted loopback only.
- [ ] In gated mode, mint a fresh ticket per WebSocket connection; do not reuse or persist tickets.
- [ ] Do not embed the loopback `?token=` in release builds pointed at remote hosts.
- [ ] Store dashboard OAuth credentials in the platform secure store (Android Keystore, iOS Keychain).
- [ ] Validate server TLS certificates; pin or trust-on-first-use only with an explicit operator workflow.
- [ ] Treat `secret.respond` payloads as sensitive - mask input, avoid logging, clear buffers after send.
- [ ] Respect Host/Origin expectations when connecting through reverse proxies (honour `X-Forwarded-Prefix` if the dashboard is path-mounted).
- [ ] Fail closed on `4401` / `4403` WebSocket close codes; do not retry with stale credentials indefinitely.

## Testing checklist

- [ ] **Loopback dev** - connect with `?token=` against `hermes dashboard` on `127.0.0.1`, receive `gateway.ready`.
- [ ] **Gated auth** - complete OAuth login, mint ticket via `POST /api/auth/ws-ticket`, connect with `?ticket=`, confirm single-use enforcement (second connect with same ticket fails).
- [ ] **Session lifecycle** - `session.list`, `session.resume`, `prompt.submit`, verify `message.delta` / `message.complete`.
- [ ] **Interrupt** - start a long tool turn, call `session.interrupt`, confirm the turn stops.
- [ ] **Approval / clarify / sudo / secret** - trigger each prompt type on the host, verify the companion surfaces the request and `*.respond` unblocks the agent.
- [ ] **Slash helpers** - `commands.catalog` returns entries; `complete.slash` narrows as the user types.
- [ ] **Reconnect** - drop the socket mid-turn; companion recovers without duplicating in-flight prompts.
- [ ] **Negative paths** - wrong token, expired ticket, wrong Host/Origin behind a proxy; expect clean close and user-visible error.

For CI-parity server behavior, run dashboard tests via `scripts/run_tests.sh` when changing auth or WebSocket paths in `hermes_cli/web_server.py`.
