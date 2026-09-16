# Hermes Workstation Architecture

## Durable execution routing

Task Compiler (`task_compiler.py`) classifies structured work as interactive,
deterministic single/batch, browser transaction, prompt queue or reasoning exception.
The session-selected `desktop_ui` capability `work_execute` is outside the core
toolset. Quantified repetitive requests cannot execute isolated mutations: the
agent demands compilation and halts after a repeated uncompiled mutation attempt.
Read-only planning, schema discovery and clarification remain available.

The planner supplies one stable operation_key, structured items and verified steps.
An items_ref dataset can replace the entire inline record list.
Arguments bind through `$item.field`. DurableBatchRunner creates/reuses WorkPlan
and WorkItems in the canonical Kanban DB. Each step uses the existing sequential
agent dispatcher with scope, deferred-tool validation, middleware and approvals.
Private item messages do not become transcript turns; the outer assistant/tool
pair retains normal SessionDB persistence. A provider-free conversation test proves
100 executions with two provider boundaries, with zero model calls inside the runner.

Step intent precedes execution; committed verification refs precede advancement.
Completed items/steps are skipped after restart. Captured output resumes persistence
and validation without rerunning its worker. Uncertain interrupted mutations and
invalid results escalate; only read failures have automatic retries. Existing
failed, blocked, cancelled and suspect work is not silently replayed.

Reference-First Boundary: tool outputs go into ArtifactStore before model projection.
SUMMARY is the default; FULL explicitly returns item result records containing
source refs. The manifest contains all item result/evidence handles; model exceptions
are bounded. Verifiers consume content before legacy spillover. `reference_plane.py`
deduplicates outputs by hash and stores data-URI blobs as recoverable MIME/hash/refs.
Legacy multimodal messages remain compatible. No visual digest is fabricated;
automatic visual-digest generation/pixel reinjection is not implemented here.

Context State Ledger: `DurableTaskStore.operational_ledger()` derives objective ref,
phase, constraints, counts, artifact refs, blockers and next action from canonical
DB rows. It contains no historical reasoning and requires no transcript. Runtime
KanbanRun handles use the existing trusted operational-reference envelope through
compaction, reconnect and model changes.

Constraint Routing checks routes before dispatch/browser probes and persists the
constraints in plan metadata. Explicit JSON user constraints cannot be broadened
by compiled constraints. Unknown provider routes fail closed when exclusions cannot
be proved. The existing ModelRouter can prune provider candidates before scoring.

Read Cache performs authorized fresh durable reads, then returns hash/ref/cache_hit
for unchanged content. Same-size/restored-mtime edits invalidate it; sections are
retrievable. Existing file-state stamps remain intact. This avoids repeated context,
not all file I/O. Schema Cache extends Desktop deferred tool_describe with session,
profile and schema fingerprints; repeats return hash/ref/capabilities, full=true
reloads. Existing registry-generation caching and HW-011 remain authoritative.

Browser Transactions preserve the original BrowserTask ID and use internal browser
actions; controller loss never creates a fallback browser. Prompt Queue uses a
bounded read-only completion wait (deadline/polls/interval), captures artifacts and
advances only after verification. Browser/queue exceptions stop advancement. These
are generic step primitives, with no site-specific adapter or second browser store.

No-Progress Circuit Breaker ports restricted upstream period-2–4 cycles, identical
call halts, failure-tolerant shell paths, progress resets and unattended default hard
stops. Interactive defaults/poller exemptions remain. Verified runtime checkpoints
reset progress counters; arbitrary tool text cannot claim actual_delta authority.

Telemetry is local: dispatch/input/output bytes, latency, cache counts, verified
checkpoint transitions, replans, completed items and internal LLM interventions.
Executor usage is unknown unless the provider reported the compile-request usage;
token-per-transition then has explicit compile_request scope, excluding final response
usage, which remains in SessionDB. Unknown token counts are not invented. Existing provider
session accounting remains intact. Assistant flushes now preserve reported token_count
and canonical usage buckets in display metadata; absent usage stays null. No outbound
telemetry or wholesale compressor/SessionDB upstream refactor was introduced.

## Product boundary

### Final durable hardening: effects, shared graph and turn routing

`tools.effects.ToolEffect` is the canonical read/discovery/write/interactive/
cognitive taxonomy. Registry registrations accept `effect`, `idempotency_key`
and `routes`; plugins can also supply these in their schema metadata. Metadata
does not enter provider function schemas. MCP live/cache registration captures
readOnlyHint without bypassing the existing trust gates; missing/malformed effect
metadata is MUTATION. IDEMPOTENT_WRITE requires an explicit key contract and a
nonempty key argument, but still cannot bypass compilation or replay uncertainty.
Discovery and clarification can execute even in a response that also proposes
blocked individual mutations. Cognitive delegation must be resolved before work.

The backward-compatible `items + steps` format remains unchanged. Optional
`setup_steps` and `finalize_steps` create shared phase WorkItems in the same
WorkPlan, not new tables or a scheduler. `execution_graph.prepare_graph()` checks
unique IDs, missing/impossible references, implicit binding dependencies and
cycles before dispatch; ready nodes are sorted deterministically. Setup outputs
bind via `$setup.node.field`, previous phase-local outputs via `$steps.node.field`,
and finalization receives `$items_ref`, a manifest of item result refs. Finalize
waits for all setup/items to be completed. Shared mutations use the same
dispatch-intent/verified-result checkpoints and uncertainty protection as items.
The tool-owned DurableTaskStore closes its connection after every outer call.

The ledger derives setup/fan_out/finalize phase, step and item progress, pending,
failed, uncertain, review blockers and checkpoint artifact refs from persisted
rows and plan metadata. SUMMARY contains no raw shared outputs; FULL remains
explicit opt-in. It does not include chain-of-thought.

Initial batch detection uses directive grammar, counts, collection quantifiers
and homogeneous structured records. A per-turn detector also compares tool name
and argument shape, retaining action/operation/method discriminators. The third
distinct equivalent mutation is stopped before dispatch. Prior successful outputs
are held by artifact refs and adopted only after the compiled verifier matches;
the planner is told to compile the remaining work. This is per-turn detection,
not a global execution cache or a change to durable restart semantics.

TurnConstraintContext exists before planner calls, checks the selected provider,
filters configured fallbacks before credential/client resolution, and rechecks
after turn setup may restore the primary. Main streaming/non-streaming boundaries
and ambient auxiliary resolver/cache boundaries fail closed. Auxiliary automatic
or composite routes under constraints require an explicit permitted provider;
their internal destinations cannot be assumed safe. ModelRouter uses the same
provider-route normalization. Plan metadata/tool/browser constraints remain scoped.

The guardrail adaptation matches upstream failure-tolerant names and unattended
policy (`non_interactive_hard_stop_enabled`), preserving period-2–4 detection and
poller exemptions. Progress reset requires canonical landed file-write evidence,
a trusted caller verifier signal, or a durable verified checkpoint; an arbitrary
`actual_delta` field in tool text is insufficient.

Regression replay: `python -m workstation.benchmarks.trello_regression` uses the
real AIAgent dispatch with fake provider/remote adapter. Twelve creations measure
two provider calls, three shared setup calls, twelve verified mutations, zero
replays, 21 cache hits and zero compactions. Sampled inline bytes 2,687 versus
2,558,773 modeled raw-inline bytes; token_count is unknown. Physical work remains
O(N), setup O(1), planner boundaries O(1) absent exceptions. This is no claim of
live Trello or paid-token validation. Workstation CI installs the project with
`uv sync --locked --python 3.13 --extra dev` and runs contracts plus core seam tests.

```text
Hermes Desktop / Dashboard
        |
        v
Hermes Gateway + Sessions + Kanban + Memory + Skills
        |
        +---------------- Browser Router ----------------+
        |                    |                           |
        v                    v                           v
Internal Browser       extension/agent-browser     browser_exec
(Electron Chromium)    optional fallback lanes     adaptive/power
        |
        v
persistent Electron Session
        |
        v
BrowserTask + task binding + controller + journal + safety
```

## V3 runtime and control boundary

The long-lived runtime is split into operational projections around the same
Hermes owners:

```text
Independent RuntimeSupervisor
  -> starts, checks, restarts and rolls back the runtime process

RecoveryPlane / recovery_cli
  -> diagnoses and quarantines optional components when rich UI paths fail

EvidenceStateStore + RuntimeEventBus
  -> projects live evidence, bounded lifecycle events and recovery state

SessionLifecycleStore / WorkerRegistry / ProceduralMemory
  -> durable bindings, worker queues and typed temporal memory over canonical
     Hermes sessions, journal, Kanban and memory ownership

Typed resources + protocol adapters
  -> reconnectable client views; A2A/ACP/UHP remain adapters, never state owners
```

`RUNNING` is only valid while a task has live operational evidence. Expired
evidence reconciles to `STALLED`, and worker failure, missing approval or
external failure is never converted into successful completion. Event delivery
uses bounded subscriber queues so one slow client cannot block unrelated work.
Deadlines/cancellation are explicit at the operation boundary, and persistent
worker messages/results retain parent task and session lineage.

V3 durable artifacts are projections or recovery copies, not replacements for
SessionDB, Kanban, BrowserTask, the Hermes Memory owner or the agent core. The
Recovery Plane has a tiny CLI surface:

```text
python -m workstation.recovery_cli status
python -m workstation.recovery_cli quarantine <component> <reason>
python -m workstation.recovery_cli restore <component>
```

Native process-boundary evidence is versioned in
`context/engineering-journal/probes/v3-runtime-hardening-smoke.py`; Electron
browser evidence remains versioned separately in the H010 probe.

### Internal Browser

The primary Workstation browser is not a second application. Hermes Desktop is
Electron, therefore it already ships an open-source Chromium runtime. Workstation
creates a dedicated Electron `Session` and manages one or more
`WebContentsView`s in the Desktop main process.

This is deliberately different from the existing Preview `<webview>`:

- Preview is a chat-adjacent rendering surface.
- Workstation Browser is a long-lived runtime owned by main.
- route changes only detach the view; WebContents keep running.
- all tabs share one persistent browser profile.
- agent control uses the same WebContents through CDP via a loopback-only authenticated controller.
- the profile does not share Hermes Desktop cookies.

Default Windows profile:

`%LOCALAPPDATA%\HermesWorkstation\Browser\User Data`

Linux/macOS use platform-appropriate configuration roots.

### BrowserRuntime abstraction

The product-level browser contract is runtime-neutral:

- `electron-chromium` — V1 primary, visible + persistent + background
- `agent-browser` — deterministic disposable fallback
- `browser-exec` — adaptive/power mode via Hermes Browser Use path
- `lightpanda` — future ultra-light headless runtime

Routing is configurable. When routing is disabled, Workstation uses only the
internal browser.

### BrowserTask lifecycle

`BrowserTask` is the logical identity of one durable browser job. It is not a
second tab/page store. For the Electron runtime, the existing `taskTabs` map and
`BrowserEntry.ownerTaskId` remain authoritative for the process-local binding
between a task and its live `WebContentsView`.

The lifecycle contract is:

```text
create(taskId)
  -> one task-owned live page at most

show(taskId, host/bounds)
  -> expose that task's existing page
  -> park a previously visible BrowserTask when necessary

hide(taskId)
  -> remove the page from the visible host
  -> keep page/task alive

park(taskId)
  -> keep the page alive in the background parking strategy
  -> keep page/task state

destroy(taskId)
  -> explicit terminal operation
  -> close/remove the owned live page
  -> remove BrowserTask metadata
```

Within one Electron process, `hide -> show` and `park -> show` must retain the
same live page and current navigation state. Repeating creation for the same
`taskId` is idempotent and must not allocate a second independently navigated
page.

BrowserTask metadata is safe structural state, separate from the Chromium
profile. The promoted composite BrowserSessionState includes identifiers/host
linkage, lifecycle status, parked state, lease/recovery state and timestamps
alongside ordinary/task logical tabs, order and active selection. It is
versioned and written atomically under the Workstation runtime directory.

A `WebContentsView` is process-local. On Desktop restart, Workstation restores
BrowserTask metadata as parked before creating any task page. When that task is
used again, the runtime lazily creates/reconnects one page under the same logical
`taskId` and records recovery. This is not serialization of the previous
renderer, JavaScript heap, or `WebContents` object. Browser profile-managed state
(cookies/localStorage/IndexedDB and compatible login state) persists separately.

The V1 #1.5 Chat Browser View, Browser Hub, and Workstation-mode Preview
compatibility slices must consume this same BrowserTask/BrowserRuntime state. A second UI surface may
show metadata/thumbnail while another owns the native live page; it must not
create an independently navigated duplicate for the same BrowserTask.

### Work/Kanban task lifecycle

```text
chat request
  -> classify as multistep/asynchronous work
  -> create Hermes Kanban card
  -> bind session/run/card
  -> choose BrowserRuntime if browser capability is needed
  -> bind/create BrowserTask for durable browser work
  -> journal actions/evidence
  -> create follow-up cards when discovered
  -> execute child only when required for parent
  -> complete card with summary + structured metadata
  -> report back into the originating chat
```

A discovered task must retain:

`parent_task_id`, `discovered_by`, `reason`, `evidence`,
`origin_session_id`.

BrowserTask may reference Hermes session/run/card identities, but it does not own
a second SessionDB, Kanban database, or Memory system.

### Safety

A browser task already bound to the internal runtime never silently fails over.
If browser/controller connectivity is lost:

`pause -> health/reconnect -> rebind -> verify -> resume`

Sensitive side effects cross an approval gate. The default policy covers
payments, send/publish, delete, contracts/terms, credentials/permissions,
money movement, irreversible actions and secrets entering a new context.

### LAN/mobile

LAN reuses the official `hermes dashboard` backend. It must never create a
second Workstation server or a second state store. Non-loopback bind is
fail-closed unless an official Hermes dashboard auth provider is configured.

V1 target:

`Desktop toggle -> auth verification -> dashboard bind -> IP detection -> QR`

Tailscale is V1.1.

## Source reuse

We adapt concepts/code only where that is architecturally valuable:

- `browser-use/desktop` MIT: WebContentsView pool/parking/lifecycle patterns.
- `hermes-browser-extension` MIT: browser-controller protocol, leases, safety,
  approvals and reconnect patterns.
- `browser-memory` MIT: procedural memory lifecycle, interface only in V1.
- `lattice` MIT: compact perception contract, interface only in V1.
- BrowserTrace/Witness/Driftlock: journal/replay/drift design references.
- BrowserOS: UX reference only; AGPL code is not incorporated.

See `THIRD_PARTY_NOTICES.md`.

## V1 controller and routing order

```text
browser_* tool call
  -> Workstation router
     -> internal Electron Chromium when available
     -> if unavailable AND task is unbound AND routing is enabled:
          official Hermes extension router
          -> legacy local/cloud backend
     -> if task is already bound OR routing is disabled:
          fail closed and recover the internal runtime
```

The controller descriptor contains a random bearer token and loopback URL and is
written outside the repository with private-file permissions where supported.
It is never exposed through LAN mode. A successful internal browser action binds
the task/session to that runtime according to Workstation routing policy; a
bound BrowserTask must not silently fail over to a different browser lane with
different page/auth state.
