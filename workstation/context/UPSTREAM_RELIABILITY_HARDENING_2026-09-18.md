# Upstream Reliability Hardening Intake — 2026-09-18

## Purpose

This document turns the 2026-09-18 upstream sweep into an implementation plan for
Hermes Work. It is deliberately **not** a cherry-pick list. The upstream corpus is
used as field evidence: invariants, failure cases, regression tests and focused
patches are transplanted only where they fit the current Hermes Work architecture.

The current Work architecture remains authoritative:

```text
intent
  -> canonical Task
  -> canonical TaskRun
  -> WorkPlan / WorkItems
  -> Browser / Worker / Host operations
  -> evidence
  -> acceptance / verification
  -> canonical commit
  -> journal / projections
  -> result
  -> learning / routine
```

This lane is reliability hardening, not feature expansion. It may proceed alongside
the active Adaptive Execution & Progressive Compilation correction as long as it
does not weaken TaskCompiler/canary/effect/uncertainty invariants or introduce a
second state owner.

## Executive disposition

### P0 — implement / verify immediately

| Upstream evidence | Current Hermes Work state | Disposition |
| --- | --- | --- |
| #114964 browser hide keepalive | BrowserTask already defines and unit-tests `hide/park != destroy`; real UI/WebContentsView composition can still regress independently | **Do not port production code. Add native integration regression immediately.** |
| #114897 bounded CDP reconnect | `tools/browser_supervisor.py` still reconnects indefinitely after a successful attach | **Directly adapt reconnect budget + registry eviction + tests.** |
| #111493 read-only resume on active writer | `active_sessions.py` limits total sessions but does not reject a second live writer for the same `session_id`; transfer can also move onto a foreign-owned session | **Port the invariant: one canonical writer, N observers.** |
| #115068 in-flight journal recovery | Current `mergeInFlightMessages` still uses the earlier `findIndex` live-projection selection and does not apply the upstream sealed-row dedupe at that point | **Direct focused port + upstream regression cases.** |
| #115085 optimistic messages survive resync | Current background sync graft does not call `preserveLocalPendingTurnMessages` | **Direct focused port + tests.** |
| #114785 Kanban session provenance | Current `kanban_create` still has provenance paths that can trust ambient/process state without proving the session exists in `state.db` | **Adapt to Work lineage; never let process-global env be authority.** |
| #114793 heartbeat fence truth | Current auto-heartbeat returns success for an attempted heartbeat even when one/both durable writes fail and lacks the delegated-child fence | **Directly harden return semantics and worker-scope fencing.** |
| #114904 dead-worker classification | Work already classifies reaped child exit codes, but the evidence is process-local; there is no durable worker exit trailer for a different dispatcher process to observe | **Port durable exit evidence; integrate with existing breaker rather than replacing it.** |

### P0-watch — audit now, code only if the Work path can reproduce it

| Upstream evidence | Current Hermes Work state | Disposition |
| --- | --- | --- |
| #114986 phantom Desktop turn lease | Current downstream `gateway.ts` does not contain the upstream `turnLeases/retainGatewayForSessionTurn` mechanism that the bug fixes | **No blind port. Record the invariant and re-audit when/if that routing lease mechanism lands.** |
| b9 Kanban provenance/link/heartbeat hardening | Work already has `current_run_id`, `expected_run_id`, CAS/fencing and a stronger canonical reliability gate | **Extract tests/invariants only after the direct P0 gaps above are closed.** |

### P1 — next reliability wave

| Upstream evidence | Work interpretation |
| --- | --- |
| #115009 / #115010 durable inbound delivery, receipts, dedup | Design one durable delivery rail for intervention/events that arrive while ownership is busy. Reuse existing canonical session/task identities; do not add a second notification state owner. |
| #114780 wake preservation across `/stop`, `/new`, `/reset` | Fold into the same durable-delivery design so lifecycle transitions cannot erase already-earned events. |
| `feat/session-owner-attach` | Mine the attach/fanout tests for observer semantics after the P0 one-writer invariant exists. |
| `feat/session-durable-admission` | Verify that accepted Work/TaskRun execution truly survives client disconnect/reconnect; reuse existing TaskRun/WorkPlan persistence rather than copying upstream admission state. |
| `feat/durable-delegation-completions` + #61332 | Audit the existing Agent Task -> Human Card/result projection for a “work completed but result delivery disappeared” gap. Extend the current rail; do not create a parallel completion queue. |
| #115056 one-call page snapshot | Native Work browser already performs the principal inventory in one `webContents.executeJavaScript` call. Import quality ideas/benchmarks only: hit-test/occlusion, geometry, mutation freshness and recall. |
| `feat/tool-search-tiered-disclosure` | Evaluate against the existing Schema Cache/deferred tools. Adopt only if it reduces provider-visible schema bytes without weakening capability discovery. |
| `opencode-port/tool-results-retention` | Evaluate as lifecycle/GC policy for the existing ArtifactStore/reference plane. |
| `bb/computer-use-provider-seam` / `bb/cua-desktop-bridge-provider` | Use as reference when native OS/computer control becomes an active milestone; keep separate from browser hardening. |

### P2 — deliberately deferred

- `feat/browser-vault-fill`: defer until a formal secret-boundary/threat model and
  credential approval UX exist.
- #108914 Bot Screen: preserve takeover/hand-back ideas; do not import Linux/Xfce
  streaming architecture into the Windows/Electron product.
- `cowork/workflow-record`: mine for UX ideas after RecipeStore/ProceduralMemory
  promotion is proven in dogfood.

## P0 implementation sequence

### P0.0 — Native Browser truth test before changing architecture

**Goal:** determine whether the current native-browser problem violates the
Workstation runtime itself, the Desktop composition surface, task ownership, or
the adaptive-execution gate.

Add a real Electron/WebContentsView regression that:

1. creates a BrowserTask and loads a fixture with:
   - a running timer;
   - an input with typed state;
   - scroll position;
   - a stable agent reference/automation target;
2. records `taskId`, tab id and WebContents identity;
3. executes `hide -> show`;
4. executes `park -> show`;
5. switches Desktop route/surface and returns;
6. proves:
   - no `destroy` occurred;
   - the same live page is reused inside the process;
   - timer advanced while hidden/parked;
   - input + scroll survive;
   - agent control still reaches the same task-owned page;
   - no second independently navigated page was allocated;
   - browser routing did not silently fall back to an external runtime.

If this passes while real dogfood still cannot use the browser, investigate the
Adaptive Execution/TaskCompiler admission path next; do **not** weaken BrowserTask.

### P0.1 — Session recovery and resync correctness

#### #115068 — in-flight journal live-tail selection

Current gap:
- `mergeInFlightMessages` selects the first live-looking assistant row after the
  matching user message.
- A sealed interim row may retain an `assistant-stream-*` id and therefore look
  live before the genuinely live tail.

Implementation:
- select the **last** live projection row;
- dedupe recovered sealed rows against base ids before insertion;
- preserve the existing Work journal/state ownership.

Acceptance:
- interim sealed row + later live row overlays the later live row;
- no duplicate message ids;
- stale journal does not re-render committed output;
- crash/resume preserves structured tool/reasoning parts.

#### #115085 — preserve local pending user intent

Current gap:
- background sync can replace the local optimistic `user-*` row before backend
  acknowledgement.

Implementation:
- compose `preserveLocalPendingTurnMessages(...)` inside both tile and active
  transcript reconciliation before assistant-error preservation.

Acceptance:
- background refresh during send does not make typed/sent intent disappear;
- once acknowledgement lands, the optimistic row converges to authoritative state
  without duplication.

### P0.2 — One canonical writer, N observers

Use #111493 as the reference invariant, but integrate at the Work ownership layer.

Implementation:
1. add a strict live-owner lookup for a `session_id`;
2. reject acquisition of a second writer lease for the same live session;
3. reject transfer of an existing writer lease onto a foreign-owned session;
4. when the Desktop/CLI opens an already-owned session, expose it as read-only
   observer instead of silently becoming another writer;
5. if liveness cannot be proven because the registry is unreadable/corrupt, fail
   closed rather than assuming “unowned”;
6. keep human Browser control lease separate from session writer ownership.

Acceptance:
- writer A + observer B is allowed;
- writer A + writer B is impossible;
- after A releases/dies and stale ownership is pruned, B may acquire;
- transfer cannot steal a live session;
- reconnect does not create phantom ownership.

### P0.3 — Kanban provenance and liveness truth

#### #114785 — persisted session provenance

Implementation:
- add a helper that validates session ids against the active profile's
  `state.db` using read-only SessionDB;
- prefer request-scoped ContextVar/session context over process-global environment;
- never use process-global `HERMES_SESSION_ID` as provenance authority;
- preserve canonical inherited task provenance only when it is internally
  consistent with Work Task/TaskRun lineage;
- surface invalid explicit provenance as bounded diagnostic evidence rather than
  writing a dangling session id.

Acceptance:
- stale process env X + request-scoped session Y => new card points to Y;
- nonexistent id is never persisted as valid provenance;
- child task inherits canonical parent origin without cross-session contamination.

#### #114793 — heartbeat means persisted heartbeat

Implementation:
- return success only when both claim-extension and worker-heartbeat writes succeed;
- do not let an in-process delegated child impersonate the dispatcher-owned worker;
- warn once when the board fence rejects inherited worker scope;
- keep the existing `expected_run_id` fencing.

Acceptance:
- first write succeeds + second fails => heartbeat result is false;
- delegated child activity cannot keep the parent worker artificially alive;
- reclaimed/superseded run cannot refresh a successor's liveness.

#### #114904 — durable worker exit evidence

Current gap:
- `_recent_worker_exits` is process-local. A separate dispatcher process can see
  the dead PID but not the exit code that another process reaped.

Implementation:
- have the worker write a bounded, machine-readable exit trailer to its own log;
- when process-local reap evidence is absent, read the final trailer and classify
  the same exit deterministically;
- strip the trailer from user-visible final worker output;
- preserve existing Work protocol-violation streak and systemic breaker semantics;
  do not replace them with upstream policy wholesale.

Acceptance:
- embedded dispatcher and one-shot dispatcher classify the same worker death alike;
- rate-limit sentinel remains neutral;
- clean exit while card remains running is a protocol violation;
- signal/OOM/no trailer remains a crash;
- machine trailer never leaks into the result presented to the user.

### P0.4 — Bounded browser recovery

#### #114897 — CDP supervisor reconnect cap

This applies directly to the legacy/fallback `tools/browser_supervisor.py`, which
still has an unbounded post-attach reconnect loop.

Implementation:
- count consecutive reconnect failures only after at least one successful attach;
- reset the counter after successful attach;
- stop after a small bounded budget (upstream uses 5);
- remove that dead supervisor from the registry;
- a later browser request may create a fresh supervisor;
- retain the redaction of CDP credentials in logs.

Acceptance:
- dead endpoint after successful attach produces at most the configured retry
  budget and one terminal warning;
- registry no longer retains the dead supervisor;
- later `get_or_start` creates a new supervisor;
- transient disconnect followed by successful reconnect resets the budget.

**Native Work browser note:** the internal Electron controller is not this WebSocket
supervisor. Do not invent a CDP reconnect loop for it. Its equivalent policy remains:

`pause -> health/reconnect -> rebind -> verify -> resume`

with bounded attempts and fail-closed routing.

## P1 integrated reliability design

After all P0 regressions are green, design one **Durable Delivery Rail** rather
than porting #115009, #115010, #114780 and delegation completion work separately.

Required properties:

- durable event id;
- origin session/task/run/card lineage;
- target owner/surface;
- pending -> claimed -> delivered / released state;
- lease with expiry;
- content-keyed or event-id dedup as appropriate;
- acknowledgement only after the destination state/transcript commit succeeds;
- lifecycle transitions (`/stop`, `/new`, `/reset`, reconnect) cannot erase
  an earned pending event;
- bounded payload, ArtifactStore reference for large results;
- no synthetic user-authored message;
- Agent Task result/evidence projection uses this rail or a proven existing
  equivalent, never a second completion mechanism.

## #115056 snapshot optimization plan

The native browser already performs its main inventory in one
`webContents.executeJavaScript(inventoryScript(...))` call, so the main upstream
optimization is already structurally present.

P1 benchmark only:

1. compare current Work inventory with the upstream one-call fixture set;
2. add center-point hit testing / occlusion signal if it improves action success;
3. add mutation sequence/freshness metadata so the agent can cheaply detect a
   stale snapshot;
4. measure:
   - round trips per observation;
   - p50/p95 snapshot latency;
   - interactive-control recall;
   - stale-ref rate after action;
   - bytes/tokens projected to the model;
5. keep iframe/shadow/canvas exceptions explicit.

Do not introduce a second `browser_exec_snapshot.py` authority for the native
Electron browser.

## Sequencing with the active Adaptive Execution gate

This hardening lane does not supersede
`context/ADAPTIVE_EXECUTION_COMPILATION.md`.

Order:

```text
P0.0 native browser truth test
  -> if BrowserTask/UI fails: repair browser composition/lifecycle
  -> if BrowserTask/UI passes but agent is blocked: continue Adaptive Execution fix

P0.1 recovery/resync
P0.2 one-writer ownership
P0.3 Kanban provenance/liveness/worker truth
P0.4 bounded fallback recovery

then

P1 Durable Delivery Rail
P1 snapshot quality benchmark
P1 owner-attach / durable-admission regression matrix
P1 delegation-result delivery audit

then

computer-use / secrets / workflow-learning feature expansion
```

## Definition of done for this intake

P0 closes only when:

- native browser hide/park/show is proven on a real WebContentsView path;
- no known unbounded CDP reconnect loop remains in the fallback supervisor;
- a session cannot have two canonical writers;
- in-flight journal recovery cannot bind onto the wrong live row;
- a local pending user message survives resync;
- Kanban never claims provenance from an unverified process-global session id;
- auto-heartbeat success means durable writes actually succeeded;
- worker death classification is stable across dispatcher process topology;
- focused upstream-derived regressions and the existing Workstation reliability
  suite are green.

Architecture docs should be updated from “planned” to “implemented” only after each
corresponding acceptance test passes.
