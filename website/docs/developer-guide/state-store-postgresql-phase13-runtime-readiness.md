# Phase 13 — PostgreSQL runtime-activation readiness gate

## Scope

Phase 13 makes an explicit `state_store.backend: postgresql` selection measurable and fail-closed. It does **not** enable PostgreSQL as the normal Hermes runtime, alter the default SQLite configuration, create a SQLite fallback, change a service, or cut over data.

`state_store_runtime_readiness.py` owns three testable boundaries:

1. `static_raw_state_db_inventory()` parses the approved runtime modules and publishes each direct `SessionDB`, shared-registry `acquire`, `sqlite3.connect`, and shared `open_db` opener with its caller symbol and porting classification.
2. `trap_state_db_opens()` intercepts root/profile `state.db` access through both Python `open()` and `sqlite3.connect`, recording the exact caller path and symbol. It is test instrumentation only.
3. `SessionDB.__init__` calls `require_legacy_state_db_runtime()` **before** test isolation, directory creation, connection, pragma, or schema work. When PostgreSQL is selected, it refuses the legacy SessionDB path with a capability report instead of silently opening or creating `state.db`.
4. `react_to_message_tool()` calls the same guard before its shared-registry acquisition. A selected PostgreSQL profile returns a bounded structured tool error naming the missing capabilities; it does not open root/profile `state.db`, persist a reaction, or emit `message.reaction`. This is a fail-closed safety boundary, not PostgreSQL reaction support.

The static inventory is checked into `website/docs/developer-guide/state-store-postgresql-runtime-callsite-report.json`. Regenerate it from the feature worktree with:

```bash
python -c 'from pathlib import Path; from state_store_runtime_readiness import write_runtime_callsite_report; write_runtime_callsite_report(Path("website/docs/developer-guide/state-store-postgresql-runtime-callsite-report.json"))'
```

## Selected-backend capability report

A selected PostgreSQL profile reports its canonical profile home/name and derived tenant schema without retaining a DSN or opening a database. The supported slice is:

- narrow StateStore records; and
- profile-derived PostgreSQL tenant schema selection; and
- CLI fresh/resume acquisition through `cli_session_store.open_cli_session_store()`; and
- bounded `AIAgent` lazy recall acquisition and append-only persistence through that same facade; and
- the public inline `session_search` consumer when that facade is injected.

The inline consumer reuses the already selected tenant-bound facade; it does not
open `state.db`, reacquire an ambient DSN, or select a second tenant. Its discovery
response exposes PostgreSQL generated-search health, and its search/read/scroll/
browse shapes use the complete contextual contract. Explicit `profile=` searches
remain resolver-owned read-only acquisitions: root/default and named profiles use
their own canonical config, secret scope, and trusted tenant derivation. Repair
(`rebuild_search_index()`) is an explicit store-maintenance operation, not a tool
request shape.

The CLI facade persists a real session row, single-message and batch structured
messages (including identity, order, timestamps, and null content), system-prompt
snapshot, end/reopen transition, and bounded recent-session lookup. It rejects
unimplemented SessionDB methods with a capability error before any SQLite
fallback. The legacy SQLite CLI path continues to construct `SessionDB`.

The adapter-only `atomic-compression-rotation-v1` contract is separately proven
by the PG18 rotation acceptance harness. The real public `AIAgent` route is verified with deterministic offline providers for
selected-PG lazy acquisition → `run_conversation()` → `ContextCompressor` →
fenced parent/child publication, a post-commit lost-acknowledgement receipt
adoption, provider-cancellation reopen/retry with no receipt replay, and an
expired-owner server-clock successor rotation. Each public route runs under a
trapped `state.db` opener. This remains narrower than the adapter harness:
delivery retries and unrelated runtime-owner consumers retain their direct-adapter
acceptance coverage and are not claims that those runtime consumers are activated.

Activation remains blocked outside that bounded CLI/inline-search path by these capabilities:

- gateway session routing, peer identity, transcript persistence/deduplication,
  rewind, recovery, auto-archive/housekeeping, and shutdown routing;
- gateway delivery-ledger runtime/recovery routing (except the explicit,
  dependency-injected final-response consumer described below); and
- async-delegation ledger routing; and
- Bot Live Delivery owner lineage and durable mailbox admission/claim/completion.
  Every public mailbox operation and the direct lineage-match helper refuse with
  `PostgreSQLRuntimeActivationError` before `state.db`, receipt JSON, mailbox
  directories/locks, or a live-turn handoff. This preserves the SQLite mailbox's
  at-most-once protocol; it is not PostgreSQL mailbox support. The separately
  injected final-delivery ledger consumer remains the only supported delivery
  route described below.
- cron session/transcript lifecycle, including durable session creation, title,
  lineage, finalization, retry, and lease semantics.
- ACP session creation, load/resume, list, fork, and control operations. Their
  public `SessionManager` boundary rejects selected PostgreSQL before UUID/cwd
  registration, SessionDB, agent/provider/MCP discovery, protocol response, or
  transcript side effect. This is a typed `PostgreSQLRuntimeActivationError`
  naming `acp-session-transcript-lifecycle`; it is **not** PostgreSQL ACP
  support. Non-session handshake/status behavior remains outside this boundary
  only when it cannot create, load, control, or act on a session.
- TUI/API session list, resume, control, workdir ownership, cache, and
  agent-build runtime. These surfaces return a typed, actionable refusal before
  an agent, tool, transport, transcript, cache, or `state.db` fallback can run;
  this is a safety boundary, not PostgreSQL SessionDB support.
- hosted-room coordination, including rooms/events, peer grants and receipts,
  driver leases/tasks, replicas, policy projection, fences, and recovery. A
  selected PostgreSQL profile raises `HostedRoomCoordinationUnavailableError`
  before a hosted SQLite path, room worker, peer client, RPC/group operation,
  or transport is created. This is not a partial PostgreSQL room adapter.

The error names the missing capabilities and points here. This is deliberately
not a claim that gateway, cron, TUI/API, ACP, async-delegation, or hosted-room
runtime is PostgreSQL-ready.

## Gateway session-routing fail-closed boundary

`GatewayRunner` and `SessionStore` call the selected-backend activation gate
before transport startup, route generation, session-directory creation, or a
legacy SessionDB open. A selected PostgreSQL profile therefore raises the
public `PostgreSQLRuntimeActivationError` with the canonical active profile
scope and the missing `gateway-session-routing-transcript` contract. The
error contains capability names and this evidence path, never a DSN.

This is a safety boundary, **not** PostgreSQL gateway support. It intentionally
does not construct a partial SessionDB adapter. In particular, selected
PostgreSQL refuses every public and direct legacy mirror lookup/append,
shutdown flush/recovery/transcript-spool, and channel-directory build/read
boundary before SQLite acquisition, JSON/cache access, directory creation,
replay, adapter enumeration, or transport handoff. Those refusals propagate as
`PostgreSQLRuntimeActivationError`; they are never converted to an empty
routing directory, JSON fallback, in-memory durable success, or a best-effort
background action. The legacy SQLite mirror, recovery spool, and channel
directory remain available only when SQLite is selected. A root-created SQLite
`SessionStore` also rechecks the
active profile for every routing load/save boundary, so multiplexed selected
profiles cannot route into the root database or mirror.

## Cron fail-closed boundary

Cron is **not PostgreSQL supported**. `run_one_job()` and `run_job()` both
require the legacy transcript runtime before creating an execution row,
handing off a restart-safe worker, running either a pre-agent script or a
`no_agent` script, constructing an agent, saving output, delivering a result,
or finalizing a session. When PostgreSQL is selected, they return/log the
structured `CRON_TRANSCRIPT_UNAVAILABLE` refusal with the missing
`cron-session-transcript-lifecycle` capability. This intentionally creates no
`state.db`, cron output/cache, JSON transcript, SQLite fallback, delivery, or
finalization side effect.

The bounded SessionDB-init timeout remains a SQLite-only availability behavior:
SQLite jobs may continue without a transcript after that timeout, as before.
Do not treat that legacy availability fallback as a PostgreSQL lifecycle port.
A full port needs equivalent durable creation/title/lineage/finalization,
retry/resume, ownership/lease, output and delivery contracts before this guard
can be removed.

## Bounded injected final-delivery ledger consumer

`BasePlatformAdapter.send_final_ledgered()` is the sole gateway consumer that
may receive a PostgreSQL ledger, and only when a caller explicitly constructs
`open_configured_delivery_ledger()` and injects it as `adapter.delivery_ledger`.
The five-method port is restricted to obligation, fenced claim, delivered,
failed, and unsent-claim release transitions; no raw connection is available to
the consumer. PostgreSQL configuration alone does not inject it, start a
gateway, route sessions, or select a recovery sweep. SQLite remains the
default when no ledger is injected.

`tests/integration/test_postgresql_delivery_ledger.py` uses an owned PG18 UUID
schema and traps `sqlite3.connect`: final obligation/claim/ack completes with
no `state.db` open and a fake sender is never invoked by the ledger bracket.
This is consumer/ledger evidence only. Gateway startup recovery, session
persistence/routing, cron, TUI/API, ACP, hosted rooms, and async delegation remain
unported and fail closed under selected PostgreSQL. Contextual `session_search`
is limited to the verified CLI/inline consumer and explicit read-only profile
resolver; it does not activate those unrelated runtime surfaces.

## Sandbox fixture and proof

`tests/fixtures/postgresql-state-store-runtime-sandbox-config.yaml` is non-secret: it selects PostgreSQL and only names `HERMES_STATE_STORE_TEST_DSN`. The loopback PG18 fixture supplies the trust-only DSN in the test process; no DSN is committed to configuration.

`tests/test_state_store_runtime_readiness.py` proves:

- AST inventory retains raw unported runtime openers;
- a legacy SQLite open is trapped with exact caller information; SQLite test isolation may materialize the fixture before the trapped native connect;
- a selected named PostgreSQL sandbox profile derives its tenant schema, reports the supported contextual contract, opens no `state.db`, and has no fallback event;
- the public inline `session_search` path reuses the selected CLI facade and serializes CJK discovery/grammar rejection, anchored scroll, read, browse, health, and explicit rebuild evidence without SQLite;
- selected-PG `delegate_task(background=true)` reaches the async-dispatch boundary, rejects before its injected runner/external-child side effect, and opens no `state.db`;
- `tests/acp/test_selected_postgresql_session_boundary.py` proves selected-PG
  manager and available ACP server session entrypoints reject with that typed
  capability before agent factory, protocol update, `state.db`, or JSON output;
  it also proves a root SQLite manager cannot be reused under an active named
  selected-PG profile, while default SQLite ACP persistence/resume is unchanged.
- direct `SessionStore` and normal `GatewayRunner` construction reject a selected
  PostgreSQL root or named active profile before route generation, transport
  activity, `state.db`, `sessions.json`, or JSONL output; a root SQLite store
  is rechecked under an active selected named profile; and
- selected-PG `delegate_task(background=true)` reaches the async-dispatch boundary, rejects before its injected runner/external-child side effect, and opens no `state.db`;
- default SQLite construction still creates/opens its configured `state.db`; and
- the checked-in report contains no PostgreSQL DSN.

## Async-delegation boundary

The current durable `async_delegations` row is not a portable job/execution
protocol. It records a local daemon runner's routing metadata and completion
for replay into the originating process's in-memory completion queue. The
runner closure, future/executor state, interrupt callback, progress monitor,
process ownership, and queue admission/acknowledgement boundary are
process-local and cannot be reconstructed from the row. A PostgreSQL table
that accepts dispatches before it carries that complete protocol would claim
recovery it cannot provide and could either duplicate or lose an external
child execution.

Accordingly, `tools.async_delegation._persist_dispatch()` retains the legacy
SQLite adapter only, and its mandatory selected-backend gate raises
`PostgreSQLRuntimeActivationError` before directory creation, SQLite open,
executor submission, or runner invocation. This is direct-test evidence of a
safe failure, not PostgreSQL async-delegation routing. Porting may proceed only
with a backend-neutral ledger plus end-to-end fenced claim, cancellation,
completion, recovery, and completion-queue delivery contracts across every
caller; until then the selected PostgreSQL sandbox intentionally refuses the
operation.

`tests/integration/test_postgresql_cli_session_store.py` exercises a fresh
isolated `HERMES_HOME` against loopback PostgreSQL: session creation, ordered
single and batch message persistence, prompt snapshot, end, a new-store
resume/reopen, session lookup, unsupported-call refusal, and the file-open
trap. Its deterministic offline-agent harness constructs the real `AIAgent`
with local mocks only, proving lazy selected-PG acquisition, first-turn
persistence, public compression publication, lost-ack durable-receipt adoption,
provider-abort reopen/retry without replay, stale-owner successor rotation,
prompt/history restoration, end/reopen, and active lease-holder rejection
before a write. The v19 coordination slice additionally persists activity labels, cooldown snapshot/restore, anti-thrash counters, and fenced server-clock leases; migration v20 adds the adapter-owned atomic parent/child compression publication receipt. `tests/integration/test_postgresql_compression_rotation_acceptance.py` exercises its fenced/idempotent handoff without SQLite fallback. This does not make the gateway, cron, TUI, ACP, hosted, or async paths PostgreSQL-ready; it also proves the default SQLite factory path is unchanged.

Run the focused gate with:

```bash
scripts/run_tests.sh -o "addopts=" tests/integration/test_postgresql_cli_session_store.py tests/tools/test_react_to_message_tool.py; scripts/run_tests.sh tests/test_state_store_runtime_readiness.py tests/test_state_store_config.py tests/test_state_store_factory.py
```

## Remaining work

Do not route gateway, cron, TUI, ACP, async-delegation, or hosted-room startup
to PostgreSQL until every listed runtime consumer has an approved
backend-neutral contract and equivalent lifecycle, ownership, delivery,
contextual-search, and recovery behavior. SQLite remains the migration
source/reference only; it is not a selected-PG fallback.
