# TUI and JSON-RPC gateway

The root guidance applies. `tui_gateway` serves the Ink TUI, Desktop, and dashboard chat. Python owns sessions, tools, model calls, and slash behavior. Clients own presentation. Do not move agent behavior into a renderer.

## Transport contract

The TUI uses newline-delimited peer-to-peer JSON-RPC over stdio. Desktop uses the same server through `apps/shared` over WebSocket. Server-to-client questions are requests with matching response IDs, not events. Open questions remain replayable across reconnect through `open_requests(sid)`. A client declares `server_requests` capability once per connection. Unsupported clients fail fast instead of blocking the agent until timeout.

The wire is declared by Pydantic models under `tui_gateway/contracts/` and generated into `apps/shared/src/gateway-contract.generated.ts` and `gateway-contract.openrpc.json`.

- Every method has one Params and Result model. Every server request and event has one payload model. Models reject unknown fields unless the producer intentionally uses `OpenModel`.
- Registering an undeclared method fails. Test isolation raises on invalid handler results or emitted events. Production logs the violation.
- Change the Python model first, run `scripts/gen_gateway_contracts.py`, then fix every TypeScript consumer reported by typechecking.
- Add a method to a topical `methods_<topic>.py` table, not a dispatcher branch.
- Add a user question to the emitter, both client request handlers, and `contracts/server_requests.py`.
- Add an event to `contracts/events.py`. Emitters are checked against it.

## Profile scope

One `serve` process may host several profile homes. The launch profile means the launch home, not a hardcoded default path. Hosting the first non-launch home activates multi-profile fail-closed guards.

Every RPC that touches home-, config-, credential-, terminal-, tool-, or agent-owned state uses `server.py::@_profile_scoped` or the owning scoped helper. Bind home, secret, and terminal tokens together. A home-only override is incomplete.

Sessionless calls require an explicit profile. Off-turn session finalization, teardown, memory commit, and `agent.close()` re-enter the session owner's scope. Scoped threads and child processes use the repository helpers, never ambient environment copies.

Prove scope changes with two on-disk homes and a credential or setting present only in the secondary. Assert the secondary resolves it, the launch profile does not leak, and process environment remains unchanged.

## Completion and slash behavior

For non-local terminal backends, path completion queries the session's terminal backend rather than the gateway host.

Built-in client commands remain local. Other slash input goes to `slash.exec` in the persistent worker and then `command.dispatch`. Skill directives become normal prompts. `commands.catalog` and `complete.slash` already include built-ins, quick commands, and skills. Do not add a duplicate RPC or registry.

Connection operations are asynchronous. The card follows operation status and responds by operation ID. It does not park the tool thread.

## Subagent authority

`subagent.list({session_id})` is a read-only snapshot for the calling transport's live conversation. It may include children related through durable session lineage so reconnect and compression do not hide running work. It exposes no dispatch context, callback, result, path, or routing secret.

Control remains stricter than visibility. Tail, steer, and interrupt resolve exact live session, transport, and generation ownership at call time. Do not copy authority into per-record attach bookkeeping. Foreign, retired, missing, or finished children are unavailable or rejected. Tail reads only the existing transcript's bounded suffix and never opens a client-supplied path. A queued steer acknowledges acceptance, not guaranteed delivery.

Async completion units are not subagents and have no generation authority. Keep their wire-compatible placeholder separate.

## Tests

Python tests run from `tests/tui_gateway/` through `scripts/run_tests.sh`. TUI TypeScript tests and typechecks run through scripts declared in `ui-tui/package.json`. Generated-contract tests must pass. Keep JavaScript assertions in Vitest, not Python source-reading tests.

Related local guidance: `apps/desktop/AGENTS.md`, `apps/desktop/src/AGENTS.md`, and `web/AGENTS.md`.