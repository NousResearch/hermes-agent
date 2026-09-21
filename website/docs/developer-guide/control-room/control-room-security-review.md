# Control Room — Security Review (CR-605)

Reviewed: `control_room/actions.py`, `control_room/executors.py`,
`control_room/flows.py`, `tui_gateway/methods_control_room.py`, dashboard BFF
`backend/control_room.py`. Date: 2026-08-12.

## Verdict

No critical or high findings. All mutation paths route through existing
authoritative APIs; Control Room owns no second state machine and performs no
raw database writes. Remaining items are hardening notes, not blockers.

## Findings by control

| Control | Result | Evidence |
|---------|--------|----------|
| Authority checks | ✅ PASS | Router rejects `target.profile != scope_profile` unless `allow_cross_profile` explicitly set (`actions.py:81-86`); every registered executor is server-side, renderers only probe via `capability()`. |
| Confirmation bypass | ✅ PASS | `confirmation == "required"` returns `confirmation_required` with preview on the first dispatch; only `confirmed=True` executes (`actions.py:98-103`). Renderers style the preview but cannot skip it — the gate lives in the router, not the UI. |
| Cross-profile leakage | ✅ PASS | Snapshot rows carry `profile`; router scopes by profile; dashboard BFF exposes `profile=` param defaulted to `default` and never aggregates profiles into one ownership-less list (`control_room.py:snapshot`). |
| Malicious peer message content | ✅ PASS | `peer_send_executor` passes `message` verbatim to `hermes_peer.tools.peer_send_message` public API only; no shell interpolation anywhere in the module; text is treated as opaque data (`executors.py:225-250`). |
| Stale action replay | ✅ PASS | `expected_revision` re-fetches via verifier before mutation; mismatch returns `stale` (`actions.py:106-120`). Duplicate-submit guard rejects re-execution within TTL (`actions.py:124-132`). |
| Direct SQLite mutation | ✅ PASS | Zero raw SQL in `control_room/`. Kanban create/comment/move go through `hermes_cli.kanban_db.connect_closing()` locked write route with validated transitions only (`block`/`unblock`/`complete`) — no generic status setter (`executors.py:381-432`). Dashboard BFF is read-only. |
| Kill / interrupt scope | ✅ PASS | `process_kill_executor` targets the process registry by session id; unknown id returns `not_found` not a silent success (`executors.py:131-136`). Subagent interrupt/steer go through gateway RPC handlers. |
| Unavailable honesty | ✅ PASS | Executors degrade to typed `unavailable` when their backend import fails — no fake successes (`_unavailable`, `executors.py:passim`). `agent_run` requires a surface-injected runner (`context["agent_run_runner"]`), else unavailable. |
| Dashboard BFF | ✅ PASS | `/api/control-room` is GET-only (405 on POST/DELETE, tested); capabilities `approvals`/`peer_messages`/`delegation_control` are hardcoded `False` in v1 — no dead controls advertised. |

## Notes (non-blocking)

1. **TTL window is in-process**: `_executed` is bounded-memory per router
   instance. A router restart resets the duplicate guard. Acceptable for v1
   (the stale-revision check still protects cross-restart replays).
2. **No rate limiting on read routes**: `/api/control-room` and the gateway
   snapshot RPC are unauthenticated localhost. Same posture as the rest of the
   dashboard BFF; out of scope for this feature.
3. **Peer send target validation** happens inside the Peer plugin; Control Room
   trusts the plugin's target handling. Documented dependency.
