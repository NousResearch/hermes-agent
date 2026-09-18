# tui_gateway/ + ui-tui/ — the TUI and its JSON-RPC backend

Applies on top of the root `AGENTS.md`. The TUI fully replaces the classic prompt_toolkit CLI;
activate with `hermes --tui` or `HERMES_TUI=1`. `tui_gateway` is ALSO the backend the Desktop app
and the dashboard `/chat` talk to — changes here have three consumers.

## Process model

```
hermes --tui
  └─ Node (Ink)  ──stdio JSON-RPC──  Python (tui_gateway)
       │                                  └─ AIAgent + tools + sessions
       └─ renders transcript, composer, prompts, activity
```

TypeScript owns the screen. Python owns sessions, tools, model calls, and slash-command logic.
Never move agent behaviour into the renderer.

## Transport

Newline-delimited JSON-RPC over stdio, peer-to-peer: client→server method calls, server→client
**requests** (the agent asking the user something: `approval`, `clarify`, `sudo`, `secret`, `vault.*`,
`connection`, the desktop read/act bridges) and server→client `event` notifications. `tui_gateway/server.py`
is the facade with the method/event catalog; methods live in `methods_*.py` siblings (`methods_config`,
`methods_complete`, `methods_browser`, `methods_bot_relay`, ...), event publishing in
`event_publisher.py` / `event_replay.py`, server→client requests in `server_requests.py` (`send()` blocks
the agent thread until the response frame with the same `srq-<n>` id arrives; `cancel*` withdraws with a
`request.cancel` event; `open_requests(sid)` is what `session.resume` / `session.events.since` replay so a
reconnecting client re-renders the still-open questions). A client says once per connection that it
answers them (`client.capabilities {server_requests: true}`, sent by the shared channel on `gateway.ready`);
a WebSocket client that never did is an app build older than server→client requests, and `send()` fails
fast for it instead of stalling the agent for the deadline. Desktop reaches the same server over WebSocket
via `apps/shared` (`JsonRpcGatewayClient`, `onRequest`). New RPC = a new `methods_<topic>.py` or an entry
in an existing topical sibling, registered in the table — no `if method == ...` chain (root shape rules).

**The wire is declared in Python and generated for TypeScript** (`tui_gateway/contracts/`). Every method
has a `Params` + `Result` model, every server→client request a `Params` + `Result`, every event a
`Payload` — one Pydantic class each, `extra="forbid"` by default (`OpenModel` for producer-owned dicts).
`register_method` refuses an undeclared name at import; the dispatcher rejects unknown param keys
(`4000` + key path) and, under `HERMES_TEST_ISOLATION=1`, raises `ContractViolation` when a handler's
result or an emitted payload does not match its model (production only logs). `apps/shared/src/
gateway-contract.generated.ts` (`RpcMethods`, `ServerRequestMap`, `BackendGatewayEventMap` + every value
shape) and `gateway-contract.openrpc.json` are rendered by `scripts/gen_gateway_contracts.py`;
`tests/tui_gateway/contracts/test_generated.py` fails when they are stale, so the loop is: change the model →
regenerate → `tsc` shows every consumer the field moved. `apps/shared/src/gateway-events.ts` only adds
the client-local synthetic events and the `GatewayEvent` envelope on top.
New question for the user = `_ask("<method>", sid, params, timeout)` in the emitter, a handler in
`apps/desktop/.../gateway-event/server-requests.ts` and `ui-tui/src/app/createServerRequestHandler.ts`,
and a `server_request(...)` in `contracts/server_requests.py`.
New event = `event("<type>", Payload)` in `contracts/events.py`; the emitter is checked against it.

## Profile scope in RPC methods

One `serve` process may host sessions from several profile homes (Desktop pooled backends launch
under a profile; the dashboard serves several). The launch profile is a profile: "default" means
the launch home, never `~/.hermes`. The first non-launch home hosted flips
`launch_profile_policy.py` → `set_multiplex_active(True)`; without it every fail-closed guard is
silently off. Every method that reads or writes home-, config- or `.env`-derived state runs under
`server.py::@_profile_scoped` (resolved from the live session's `profile_home`, or the explicit
`profile` argument for sessionless calls) and, for tool/agent construction,
`methods_tools.py::_profile_scoped_rpc`; the tokens come from `model_switch.py::
_profile_runtime_scope_tokens(profile_home)` — home + secret scope + terminal scope together.
**A method that sets only `get_hermes_home_override()` is half-bound**: config paths resolve to the
right profile while credentials and `TERMINAL_*` policy still come from the launch profile.
Off-turn paths bind the same way: `session_lifecycle.py::_finalize_session` / `_teardown_session`
enter `_session_profile_runtime_scope(session)` around `on_session_end`, the memory commit and
`agent.close()` (their callers are unscoped reapers, Timers, atexit and pool threads); background
threads start via `agent.memory_provider.spawn_context_thread`, never bare `threading.Thread`;
children act for the served profile through `tools/environments/local.py::served_profile_child_env`
(`hermes -p X` workers, `key_cmd` helpers, browser drivers), never `dict(os.environ)`. Grep for
unscoped handlers before adding one: `rg -n "^(async )?def " tui_gateway/methods_*.py | rg -v
_profile_scoped`. Probe with two on-disk homes and a `.env` name present only in the secondary:
call the method for that session and assert the secondary's value resolves and the launch
profile's does not, and that `os.environ` is unchanged afterwards.

## Key surfaces

| Surface | Ink component | Gateway method / event |
|---|---|---|
| Chat streaming | `app.tsx` + `messageLine.tsx` | `prompt.submit` → `message.delta` / `message.complete` |
| Tool activity | `thinking.tsx` | `tool.start` / `tool.generating` / `tool.complete` |
| Approvals | `prompts.tsx` | server→client request `approval` → response `{choice}` |
| Clarify / sudo / secret | `prompts.tsx`, `maskedPrompt.tsx` | server→client requests `clarify` / `sudo` / `secret` (`server_requests.py`) |
| Session picker | `sessionPicker.tsx` | `session.list` / `session.resume` |
| Slash commands | local handler + fallthrough | `slash.exec` → `_SlashWorker`; `command.dispatch` |
| Completions | `useCompletion` hook | `complete.slash`, `complete.path` (under a non-local `terminal.backend`, `complete.path` lists the directory through the session's terminal backend — never the gateway host, whose same-named tree would look right and be wrong) |
| Theming | `theme.ts` + `branding.tsx` | `gateway.ready` carries skin data |
| Plugin compat notice | — | `plugins.compat_report` (see `plugins/AGENTS.md`) |
| Connection operations (desktop card) | desktop `store/connection-request.ts` | `connection.request` → `connection.update`* → `connection.respond {op_id}`; `connectors.operation.status`. The op lives in `tools/connectors/live.py`; the card never parks the tool thread (`methods_connectors.py`). |

## Shared subagent snapshots

`subagent.list({session_id})` returns `{subagents, delegations}` for the calling
transport's live session. Live child records are pinned to the exact session
record and transport. Child authority is resolved at RPC time against the owning session's
LIVE transport slot, so every authenticated reattach path (prompt.submit, queued drain,
resume, activate, viewer failover) carries it with no registry bookkeeping — never add a
per-record transport sync at an attach site; foreign or retired generations remain inaccessible. `last_tool` is the last started tool, not an in-flight
indicator. Async completion units are not agents and lack exact generation authority;
`delegations` remains an empty array for wire compatibility. No dispatch context,
results, callbacks, or routing keys are sent. Clients hydrate from this snapshot
on their existing poll and avoid updates when unchanged.

`subagent.tail({session_id, subagent_id})` returns
`{subagent_id, available, text, truncated}`: the last 16 KiB of the live child's
existing transcript. Poll only the selected detail. Missing/finished/foreign
children return an unavailable empty snapshot; no client-supplied path is opened.
This is live-only, not persisted completion history. Invalid session/transport
returns error 4001. `subagent.steer({session_id, subagent_id, text})` remains the
shared control: `status: queued` acknowledges acceptance, not delivery; final
boundary races are reported by the existing runtime as `missed_steer`.
`subagent.interrupt({session_id, subagent_id})` requires the same exact live
session/transport/generation ownership, including for subtree members. Missing
RPC session authority is rejected; direct in-process `interrupt_subagent(id)`
retains its legacy unscoped contract.

## Slash command flow

1. Built-in client commands (`/help`, `/quit`, `/clear`, `/resume`, `/copy`, `/paste`, ...) are
   handled locally in `app.tsx`.
2. Everything else → `slash.exec`, which runs in the persistent `_SlashWorker` subprocess →
   `command.dispatch` fallback, which the gateway resolves into a skill / alias / exec directive
   (a skill command resolves to `{type: "skill", message}` and is submitted as a normal prompt).

`commands.catalog` (empty-query list) and `complete.slash` (typed-query completions) already include
built-ins, user `quick_commands`, AND skill-derived commands (`scan_skill_commands()` /
`get_skill_commands()`) — clients do not need a new RPC to see skills. The command definitions
themselves come from `hermes_cli/commands.py` (`hermes_cli/AGENTS.md`).

## Dev commands

```bash
cd ui-tui
npm install       # first time
npm run dev       # watch mode (rebuilds hermes-ink + tsx --watch)
npm start         # production
npm run build     # full build (hermes-ink + tsc)
npm run typecheck # tsc --noEmit
npm run lint      # eslint
npm run fmt       # prettier
npm test          # vitest
```

Python tests: `tests/tui_gateway/` via `scripts/run_tests.sh`. TS tests: vitest in `ui-tui`. A
Python test that asserts about `package.json` / `.ts` sources will not run on a JS-only PR — keep
JS-side assertions in vitest (root testing rules). Root TypeScript style rules apply.

Related: `web/AGENTS.md` (dashboard embeds this TUI over a PTY), `apps/desktop/AGENTS.md` (own
renderer on the same backend).
