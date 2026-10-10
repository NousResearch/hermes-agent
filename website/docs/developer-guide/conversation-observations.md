---
title: Conversation observations
---

# Content-free conversation observations (v1)

`GET /api/session-observations?session_id=ID&session_id=ID&profile=NAME`
uses the dashboard's existing authentication and native profile resolver. It does
not create, resume, run, archive, repair or migrate a session. Valid responses
have `Cache-Control: no-store`. IDs are exact ASCII `[A-Za-z0-9_-]{1,100}`;
1–40 distinct IDs are required. Invalid IDs are rejected, not normalized or
prefix-resolved. The profile selector retains native profile validation.

The envelope is `{schema_version: 1, profile, observations: [...]}`. Each row has
`session_id` (the requested ID), `lineage_tip_id`, `profile`, `source`,
`observed_at` (actual UTC read time), `provenance` (`native`, `lease`,
`unavailable`), `revision`, `execution` (`running`, `idle`, `unknown`),
`turn_id`, `last_result` and `attention`. `last_result` is null or
`{turn_id, status: complete|interrupted|error, at}`. `attention` is
`{kind: none|question|approval|validation|unknown, request_id, turn_id, opened_at}`.
Nullable metadata stays null for an unknown identity. No title, message, prompt,
command, response, credential, private lease holder or request body is returned.

The additive optional `last_active` field is the same effective Unix-seconds
activity timestamp used by session lists: the freshest valid activity heartbeat
or message timestamp, falling back to creation time. It is not `observed_at`,
terminal sealing time or proof of execution/progress. Readers may use it for
recency filters without converting an older conversation into an idle one.

## Reader API

```python
from hermes_state_observations import read_session_observations_at_path
rows = read_session_observations_at_path(
    profile_home / "state.db", [exact_session_id], profile="default")
```

This exported adapter opens `SessionDB(..., read_only=True)` directly, never the
dashboard's schema-healing opener. A missing, legacy or unreadable store returns
explicit unknown rows. A caller already holding a DB can use
`SessionDB.read_session_observations(ids, profile=...)`. `validate_observation_ids`
and `unknown_session_observation` are also exported. Readers must bind the path
to the authorized profile themselves; a supplied profile label is **not** path
authorization. The dashboard does this through its existing profile resolver.
An external HTTP companion must apply its own identity middleware.

The batch reads identity, compression lineage, lease and observation in one
SQLite read transaction. It reads only session identity/routing metadata, native
leases, observations and the indexed maximum message timestamp, never message
content. Requested IDs remain exact.
A unique compression continuation is followed; explicit branch/reset/delegate
edges are not continuations. Cycles, multiple continuation siblings, missing
continuations, hidden/internal rows and profile mismatches are unknown, never
"newest sibling" or idle. Legacy `profile_name=NULL` is accepted only within the
caller's already authorized profile store. Internal listing sources plus
`subagent`, `cron`, `unknown`, absent sources and delegated rows are excluded.

- A live native lease proves admission, not progress. Without a covered record:
  `provenance=lease`, `execution=running`, attention unknown, no result.
- A generation covered by the current lease has native running proof.
- An uncovered running record is unknown, including a dead/expired/crashed owner.
- An explicitly sealed terminal without a newer admission is idle with its result.
- A newer admission invalidates the old terminal atomically even if an older
  producer never calls begin or dies before doing so. The old idle cannot reappear.
- Validation requests persist across turns/compression/restart. An explicit
  current-generation resolution can target an older request's original turn.
- Multiple pending requests are stored in opening order; the wire projects the
  oldest eligible request. Request bodies and answers are never part of that list.

## Producer API and wiring

`SessionDB.begin_session_observation(session_id, holder, attention_covered=True)`
mints an opaque UUID generation under the admitted native lease.
`finish_session_observation(session_id, holder, turn_id, status)` seals one
terminal (`complete`, `interrupted`, `error`). These are wired into
`AIAgent.run_conversation`'s admission/exit facade and `DurableTurnLease`.
Only the actual terminal result flags (`completed`, `interrupted`, `failed` /
`error`) or an exception branch prove a terminal. Prose alone does not. An
unclassified return remains unknown; observation persistence is fail-open for
execution and never invents a fallback error result.

Every producer write checks the lease digest, its original acquisition timestamp
and the exact generation in the same write transaction. Refreshes do not replace
the acquisition timestamp. An expired lease, a new holder or a stale turn cannot
close/rewrite a successor. Terminal results are sealed against duplicates.

`open_session_attention(session_id, turn_id, kind, request_id=None, holder=None)`
and `resolve_session_attention(session_id, turn_id, request_id,
request_turn_id=None, holder=None)` are the explicit content-free request APIs.
Native question/approval writes require the private lease holder; business
validation declarations can name the current generation while running or after
completion. Opening the same validation twice in a turn is idempotent. Resolution
requires the exact request ID and its turn; `turn_id` always fences the **current**
generation. Resolving a badge is **not approval to mutate any business data**.

`tui_gateway.server_requests` brackets real `send`/`send_async` requests, not
tool-name heuristics. The closed human-method list is `clarify`, `setup_choose`, `approval`,
`sudo`, `secret`, `vault.unlock_prompt`, `vault.save_login`, `vault.code`.
Publication precedes the request frame; full response, final clarify lock,
timeout, cancellation, undelivered refusal and frame-write failure settle it.
Partial clarify locks leave the request open. The owner is resolved through the
actual live session's agent and DB, not process-global profile environment.
Desktop `terminal.read`, `preview.read`, `window.read`, `preview.act` and `tour`
are technical RPCs and do not create attention.

**Coverage limit:** this patch instruments server→client requests in the shared
TUI/Desktop backend only. Classic CLI and messaging-specific question/approval
callbacks are not yet instrumented. Their running native observations retain
attention unknown unless an explicit validation is declared, never inferred none.
The facade marks request coverage for `desktop` / `tui`; direct producer callers
must pass `attention_covered=False` if they do not own native request settlement.
Compute-host agents whose requests are relayed outside the live local-agent slot
also need an explicit owner adapter before their waiting state can be proven.
No Voice HTTP route is installed here; the external companion consumes the
read-only adapter. No core model tool or system-prompt mutation is added.

## Explicit business validation via CLI

Given an exact native session and current `turn_id` from the reader:

```sh
hermes --profile default sessions attention open EXACT_SESSION_ID --turn-id EXACT_TURN_ID
hermes --profile default sessions attention resolve EXACT_SESSION_ID \
  --turn-id CURRENT_TURN_ID --request-id EXACT_REQUEST_ID \
  --request-turn-id ORIGINAL_REQUEST_TURN_ID
```

The response is JSON, exit 0 for an actual successful declaration/resolution,
exit 1 for an unknown identity, obsolete generation, wrong request or unavailable
store. The command does not create/resume a conversation. Use explicit IDs from
the requesting conversation, never titles or search-selected sessions. A skill
may use the existing terminal tool to call this CLI after intentionally asking
for business validation, and resolve on a specifically attached answer or an
explicit producer cancellation. This changes an **indicator**, not an action
authorization. No prose classification or automatic next-message resolution.

## Exact additive SQLite schema

Created by normal **writable** SessionDB initialization (`SCHEMA_SQL`); read-only
observation opens never create it. One current record per compression-root key.

```sql
CREATE TABLE IF NOT EXISTS session_observations (
    conversation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    lease_digest TEXT NOT NULL,
    lease_acquired_at REAL NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('running', 'terminal')),
    result_status TEXT CHECK (result_status IN ('complete', 'interrupted', 'error')),
    result_at REAL,
    attention_json TEXT NOT NULL DEFAULT '[]',
    attention_covered INTEGER NOT NULL DEFAULT 0 CHECK (attention_covered IN (0, 1)),
    updated_at REAL NOT NULL
);
```

`lease_digest` is SHA-256 of the native holder, not the holder itself; it and the
acquisition time are internal fence metadata, never wire fields. `attention_json`
is an ordered JSON array, initially `[]`. Every item has exactly:
`{"kind":"question|approval|validation","request_id":"opaque id",
"turn_id":"opaque generation","opened_at":"ISO UTC"}`. `result_at`,
`lease_acquired_at`, `updated_at` use Unix seconds. `session_id` records the
producer's physical segment, not a separately trusted lineage-tip identity.
`revision` increments on begin, request changes, terminal sealing and new-lease
invalidation; it is not a transcript count or permission token.

Three additive triggers (exact definitions in `hermes_state_common.py`):
`session_observations_delete` deletes records on owning root/segment deletion;
`session_observations_new_lease` invalidates result/state on a new lease INSERT;
`session_observations_reclaimed_lease` does the same on holder/acquisition change.
The latter two set state running, result fields NULL, increment revision, and
set updated_at to the new lease acquisition. Native validation requests remain
durable; an old native human question is never eligible under a different lease.
No archival event history, schema sidecar, heartbeat or progress timestamp is
introduced.

## Verification

Use `HERMES_PYTHON=<isolated-test-env>/bin/python scripts/run_tests.sh` for
`tests/hermes_state/test_conversation_observations.py`,
`test_observation_terminal.py`, `test_observation_identity.py`,
`tests/agent/test_conversation_observations.py`,
`tests/tui_gateway/test_conversation_observations.py`, `test_observation_native_requests.py` and
`tests/hermes_cli/test_session_observations.py`, plus native lease/protocol tests.
The coverage includes crash uncertainty, late generations, sealed terminals,
legacy admission invalidation, compression/reopen, profile A→B→A, authenticated
metadata-only reads, request settlement and technical-RPC exclusion.

For a real HTTP two-profile probe, isolate both `HOME` and `HERMES_HOME` and
assert `get_profile_dir('default') == isolated_home` before serving. Native
`get_default_hermes_root` treats homes **inside the native ~/.hermes** as belonging
to that native root, even if the scratch path is supplied as `HERMES_HOME`.
Changing only `HERMES_HOME` there is not an isolated native-profile test.
