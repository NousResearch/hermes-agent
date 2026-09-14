# Hermes Workstation Architecture

## Product boundary

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
