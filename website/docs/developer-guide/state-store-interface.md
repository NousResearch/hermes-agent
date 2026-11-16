---
title: "State Store Interface"
description: "The backend-neutral protocol plugins code against and its SQLite reference implementation"
---

# State Store Interface

`state_store_interface.py` defines `StateStoreInterface` — the formal,
backend-neutral contract for the session/state store surface that plugins may
rely on. Before it existed, plugins duck-poked the store: the session-titler
`getattr`-probed a dozen members with silent fallbacks, and hermes-evolve
shipped its own raw-SQLite system-prompt invalidation with a layout probe
copied out of core. A plugin written against raw SQLite internals breaks the
moment a second backend appears.

## Why a protocol: three compatibility states

A plugin that touches stored state can find itself in front of one of three
stores:

1. **Original raw-SQLite** (upstream without this layer): the plugin must
   probe defensively (`getattr`) and carry its own SQL for anything not
   exposed.
2. **Interface-layer SQLite** (this layer): the store satisfies
   `StateStoreInterface`, so the plugin can rely on the contracted face and
   call `clear_stored_system_prompts()` instead of hand-rolled SQL.
3. **Another backend**: the same face implemented over a different storage
   engine; a plugin written against the protocol works unchanged.

The protocol is what makes states 2 and 3 interchangeable from a plugin's
point of view.

## The protocol face

`StateStoreInterface` is a `typing.Protocol` decorated with
`@runtime_checkable` — **structural typing**. Implementations do *not*
inherit from it; they satisfy it by having the members. That is deliberate:
the existing SQLite `SessionDB` and the PostgreSQL facade both keep their own
class hierarchies, and neither needs to import or subclass anything to be
contract-conformant.

Members (extracted from what real plugins consume, not invented):

| Group | Members |
| --- | --- |
| Titles | `get_session_title`, `get_session_title_source`, `set_session_title`, `set_auto_title`, `sanitize_title`, `TITLE_SOURCE_LLM` |
| Reads | `get_session`, `get_messages_as_conversation`, `search_messages` |
| Availability probes | `is_search_available`, `fts_health` |
| Invalidation | `clear_stored_system_prompts` |
| Unit of work | `apply_atomic` |

How to use it as a compatibility check:

```python
from state_store_interface import StateStoreInterface

def attach(store) -> bool:
    # Dev-time / test-time validation: prove the store exposes the face.
    # isinstance on a @runtime_checkable protocol checks member *existence*;
    # signatures are checked by type checkers against the Protocol, not at
    # runtime.
    return isinstance(store, StateStoreInterface)
```

Because the protocol carries a data member (`TITLE_SOURCE_LLM`),
`issubclass` is not allowed by `runtime_checkable` semantics — use
`isinstance` on an instance.

### What is explicitly NOT in the contract

Private underscore members — `_execute_write`, `_read_one`,
`_is_compression_ancestor`, `_store_system_prompt`, ... — are implementation
details of the SQLite store. Some plugins reach for them defensively today;
that is tolerated, not contracted. A backend is free to change, rename, or
absent them. If a plugin needs a capability that only exists as a private
member, the right move is: propose it for the protocol first (see below), then
rely on it.

### The protocol is a check, not a gate

Nothing at runtime refuses a store for failing `isinstance`. The protocol
exists so plugin authors can *verify* compatibility (in tests and dev
validation) and so backend authors know exactly which surface must not drift.
Treat a failed `isinstance` as a bug report, not an error to raise.

### Error channels are part of the contract

Which failures raise and which are returned is contracted, not accidental:

- **Boolean writes** (`set_session_title`, `set_auto_title`) return `False`
  for every outcome a caller can sensibly retry or skip — missing row, a
  title of equal or higher authority already holding the row, a protected
  canonical Bot Chat — and raise `ValueError` only for outcomes the caller
  must handle before retrying: an invalid `source` (a programming error),
  a sanitized title over the store's length cap, or a title owned by
  another live session. The reference titler relies on exactly this split:
  it catches the collision `ValueError` to retry with a `#N` lineage name.
- **`sanitize_title`** returns `None` only for empty input; a normalized
  title over the cap raises by default. `sanitize_title(title,
  truncate=True)` is the guaranteed-value path for protocol-facing callers:
  the title is cut to the cap instead of raising, so `Optional[str]`
  means "empty or truncated-to-fit", never an exception.
- **Reads** never raise for missing data (`None` / `[]`) and raise only for
  genuine storage failures.

## `search_messages` availability: probe, don't infer

`search_messages` is fail-open by design: an index that is disabled,
detached after corruption, or latched stale does not error — the call
serves a canonical-row LIKE fallback (documented subset of the query
semantics, different ranking) or `[]`. That means `[]` alone cannot tell
"no match" from "no index". A caller that needs to know probes first:

```python
if store.is_search_available():
    rows = store.search_messages(q)   # indexed route
else:
    ...                                # degrade: narrower query, warn, skip
```

`is_search_available()` is advisory and point-in-time (another process can
latch fail-open between the check and the call); `fts_health()` returns the
diagnostic detail — `available`, `stale`, per-index `routes`, deferred
`rebuild` progress, and a machine-readable `reason` (`fts_unavailable`,
`fts_stale`, `probe_error`) — and never raises.

## `apply_atomic`: one transaction, expressed

Every protocol write is atomic per call; composing several is not — each
call is its own transaction, and a failure halfway leaves the earlier
writes committed. `apply_atomic(fn)` is the public unit-of-work:

```python
store.apply_atomic(lambda conn: (
    store._set_session_title_in_transaction(  # backend-specific; prefer the
        conn, sid, name, source="llm"),       # store's own public methods
    conn.execute("UPDATE sessions SET model = ? WHERE id = ?", (m, sid)),
))
```

Contract on backends: every write `fn` issues must land in one transaction —
all committed together or not at all, including mid-`fn` failure rollback.
Contract on callers: `fn` must be idempotent under replay (the backend may
re-run the whole callback after a lock-contention rollback; relative
mutations like `+=` are unsafe unless expressed as absolute writes), and
`fn` must not commit or roll back the connection itself — the backend owns
the transaction boundary. The handle passed to `fn` is backend-specific;
portable code calls only the store's own methods inside the unit and treats
the handle as an opaque last resort, because SQL written against it binds
the caller to one backend.

## `clear_stored_system_prompts`

The first capability born in the protocol rather than in a plugin's SQL. It
invalidates every stored system-prompt snapshot so each session rebuilds its
prompt from live configuration on the next run or resume — what hermes-evolve
needed after hot-reloading modules (a stale snapshot would pin the old prompt
forever).

```python
result = store.clear_stored_system_prompts()
# {"cleared": 3, "storage_mode": "out-of-line"}
```

Per-operation behavior of the SQLite reference (other backends must implement
an equivalent `clear_stored_system_prompts` operation, not this SQLite layout):

- **Never deletes sessions.** Session rows survive; only prompt snapshots are
  invalidated.
- **Both SQLite storage layouts.** `storage_mode` reports which was found:
  `"out-of-line"` (prompt text in a `system_prompts` table referenced by
  `sessions.system_prompt_hash` — references are NULLed before the
  unreferenced snapshot rows are deleted, so no foreign key ever dangles),
  `"inline"` (legacy text column on `sessions`, unset), or `"unknown"`
  (neither column exists; nothing to do).
- **The canonical unset is `None` on every layout.** The out-of-line branch
  NULLs both columns; a true legacy inline row is normalized to `NULL` too
  (the inline branch's UPDATE sets `system_prompt = NULL`, not `''`), so a
  cleared prompt reads back identically regardless of which layout a
  database happens to carry. An inline row that was already `''` before the
  clear is untouched history, not a cleared value.
- **Idempotent.** Nothing stored (or already cleared) → `cleared == 0`.
- **SQLite atomicity is scoped to this call.** SQLite's `_execute_write`
  wraps the reference implementation's UPDATE and snapshot sweep in one
  `BEGIN IMMEDIATE` transaction. Failure before commit rolls both back;
  after commit, both changes are durable. Composing this call with other
  store methods in one transaction is what `apply_atomic` is for — the
  per-method call remains its own unit. The private write helper may retry
  a callback after a lock/busy failure, so its callback must tolerate
  replay; an IOERR after the callback starts is not blindly replayed
  because commit status may be unknown.

For reads, the canonical unset is `None` on every layout and backend:
the reference out-of-line `get_session(...)["system_prompt"]` resolves to
`None`, and a true legacy inline row cleared by the inline branch is
normalized to `NULL` as well, so `is None` means unset regardless of the
storage layout. (An inline row that already held `''` before the clear is
untouched history — distinguish it from a cleared value by provenance, not
by sentinel.)

## Extension rules

1. **New capabilities enter the protocol first, then the implementations.**
   The protocol is the versioned contract; by landing it before a backend
   implements a capability, plugin code never has to probe for it.
2. **Private methods never enter the contract.** If a private member becomes
   load-bearing for plugins, promote a public, semantically-named member via
   rule 1 instead of contracting the implementation detail.
3. **The face stays narrow.** Every member should be traceable to a real
   consumer; speculative members freeze implementation details that no one
   needs yet.

## Relationship to other backends

Backend implementations are reviewed and shipped independently from this
interface layer. A backend may carry an equivalent protocol definition while
the changes are under separate review, then converge on this module after both
lines land. That temporary duplication does not make either pull request a
dependency of the other.

This layer contains only the protocol and the SQLite reference behavior: it
introduces no PostgreSQL dependency and changes no SQLite default behavior.
