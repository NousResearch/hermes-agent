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
| Invalidation | `clear_stored_system_prompts` |

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

Semantics, identical across backends:

- **Never deletes sessions.** Session rows survive; only prompt snapshots are
  invalidated.
- **Both storage layouts.** `storage_mode` reports which was found:
  `"out-of-line"` (prompt text in a `system_prompts` table referenced by
  `sessions.system_prompt_hash` — references are NULLed before the
  unreferenced snapshot rows are deleted, so no foreign key ever dangles),
  `"inline"` (legacy text column on `sessions`, blanked), or `"unknown"`
  (neither column exists; nothing to do).
- **Idempotent.** Nothing stored (or already cleared) → `cleared == 0`.
- **One transaction.** The whole invalidation is atomic; a crash mid-clear
  leaves the store fully intact.

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
