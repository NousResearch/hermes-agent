# Current State

Snapshot date: 2026-09-18.

The 2026-09-15 hardening snapshot below remains valid as implementation and test
history, but a 2026-09-17 forensic audit of current `main`, persisted Workstation
state and historical behavioral traces found a stronger cross-domain reliability
boundary that was not exercised by the existing green suites. The active next correction is therefore
[ADAPTIVE_EXECUTION_COMPILATION.md](ADAPTIVE_EXECUTION_COMPILATION.md),
which refines the already-implemented reliability gate before further feature expansion.

This file describes **observed implementation state**, not target architecture.
When it disagrees with code on current `main`, inspect the code and update this
file. “Implemented”, “contract layer validated” and “tests green” do not imply that
a product-level causal invariant has been proven across Task -> Run -> operation ->
evidence -> canonical commit -> projection.

## 2026-09-18 native-browser compiler obstruction — active correction

The Canonical Execution Reliability Gate remains implemented history, but real
native-browser dogfood exposed a new execution-policy regression: the durable
compiler can prevent safe stateful work from progressing when a repeatability
signal is treated as a broad mutation latch.

Current target architecture is defined in
[ADAPTIVE_EXECUTION_COMPILATION.md](ADAPTIVE_EXECUTION_COMPILATION.md).
The active work is to replace the binary/session-wide gate with operation-scoped
progressive compilation:

ADAPTIVE -> COMPILED SEGMENT -> COMPILED WORK -> PROMOTED ROUTINE, with compact
NEEDS_REASONING escalation on drift.

All prior safety invariants remain required. This is a correction to admission
and learning, not a rollback of durable execution.


## 2026-09-18 upstream reliability hardening — confirmed parallel P0 lane

A direct PR-to-current-code comparison on `main@03e06cfd8c94e5a7627c288c8eddfd5d4c5c8033`
confirmed a second, compatible reliability lane. It does not supersede the
Adaptive Execution correction and does not represent implemented behavior yet.

**Confirmed open downstream gaps:**

- #114897: `tools/browser_supervisor.py` has unbounded post-attach reconnect;
- #111493: active-session capacity exists, but same-`session_id` writer
  exclusivity and transfer fencing do not;
- #115068: in-flight recovery still selects the first live-looking projection
  row instead of the last true live tail;
- #115085: background resync does not preserve the local unacknowledged user row;
- #114785: Kanban provenance is not uniformly validated against persisted
  SessionDB/request-scoped ownership before persistence;
- #114793: automatic heartbeat reports attempted work rather than requiring both
  durable writes and lacks delegated-child liveness fencing;
- #114904: downstream worker failure policy is already richer, but exit-code
  evidence remains process-local when a different dispatcher observes death.

**Refined non-gap / watchlist evidence:**

- #114964 is not evidence that BrowserTask keepalive is missing. H004 already
  proved real Electron `BrowserWindow`/`WebContentsView` identity across
  hide/show and park/show; H013 covers the integrated Desktop/Browser path.
  P0.0 therefore reuses/extends those probes on current main with timer/input/
  scroll, controller action and explicit no-fallback assertions.
- #114986 is watchlist-only because the affected upstream `turnLeases`
  mechanism is absent from the downstream Desktop gateway.
- #115056 is P1 quality/benchmark work because native `snapshotForEntry()`
  already obtains its principal inventory in one `executeJavaScript` call.

Canonical plan:
[UPSTREAM_RELIABILITY_HARDENING_2026-09-18.md](UPSTREAM_RELIABILITY_HARDENING_2026-09-18.md).
Engineering evidence: H-065 in
[engineering-journal/CURRENT.md](engineering-journal/CURRENT.md).

## 2026-09-17 forensic reliability audit — current boundary

The audit changes the interpretation of the existing implementation without
invalidating the many mechanisms that are already useful.

### Resolved and Verified Invariants (Canonical Execution Reliability Gate)

- **Run Fencing Enforcement:** `workstation/kanban.py::complete_task_with_report()` now requires and enforces `expected_run_id` CAS checks against `tasks.current_run_id`. Stale runs are rejected immediately before acceptance evaluation or mutation, preventing concurrency violations and state corruption.
- **Canonical Commit Before Journal:** `TASK_COMPLETED` is only recorded in `ExecutionJournal` after the canonical completion commit succeeds in SQLite. On CAS failure, an `acceptance_commit_failed` event is recorded and `False` is returned.
- **Canonical Lineage:** `ExecutionEvent`, `BrowserTaskReport`, `TaskOutcome`, and `EvidenceRef` carry first-class `run_id` and `operation_id`.
- **Plan Lineage & Execution Key:** `WorkPlan` and `WorkItem` persist canonical `run_id`, separate `execution_key`, and `operation_id` via additive schema migrations.
- **Database Connection Portability:** `hermes_cli.kanban_db.connect(db_path=...)` normalizes `Path(db_path)` supporting both `str` and `Path`.
- **Terminal Tree Consistency:** `DurableTaskStore.update_plan_state` cascades terminal parent states (`interrupted`, `failed`, `cancelled`, `blocked`) to mark all live descendant items as `blocked`. `reconcile_terminal_plans()` provides startup and periodic reconciliation sweeps.
- **Human Takeover & Fencing:** `BrowserControlLeaseManager` generates monotonic lease generations and `fence_token`s. Takeover revokes agent mutation authority (`browser_click`, `browser_type`, `navigate`, `submit`, etc.), and rejects stale fence tokens from previous runs.
- **Constant-Time Journal Chaining:** `ExecutionJournal.append()` operates in $O(1)$ constant time via `_get_last_record()` without parsing entire files, verified by a 120-event streaming benchmark.
- **Cockpit Auditability:** `task_cockpit()` exposes full canonical lineage (`run_id`, `execution_key`, `operation_id`, `workplan_id`, `human_card_id`, `acceptance_status`, `acceptance_approved`).

### Existing mechanisms that must be reused, not rebuilt

- TaskCompiler canary-before-fan-out, mutation verifiers, idempotency contracts,
  dispatch checkpoints, recipe staleness/quarantine and uncertain-mutation
  escalation;
- canonical Kanban Task/TaskRun state and CAS transitions;
- BrowserTask and scoped human-control leases;
- EvidenceState, RuntimeEventBus, RuntimeSupervisor and RecoveryPlane;
- WorkerRegistry and host capability adapters;
- Execution Journal and artifact/reference plane;
- Hybrid Human Card -> Agent Task delegation with separate lifecycles;
- existing compaction, routine promotion, no-progress and evaluation machinery.

### Current architectural interpretation

`One Hermes State` means **one authority per domain + shared causal identity +
explicit lineage + reconcilable projections**, not one physical SQLite database.
The immediate work is to make TaskRun and operation identity flow through the
existing owners and to enforce canonical commit/acceptance ordering.

## Working now

On `main` plus the current Workstation V3 hardening working tree:

- Hermes Vault rejects cross-platform absolute/traversal paths before side
  effects, excludes symlink escapes, writes atomically, and shares one bounded
  process-wide manager/watcher between agent tools and Desktop plugin RPC.
- Workstation Chromium advertises the actual `process.versions.chrome` engine
  without Electron/Hermes product tokens.
- Desktop display/installer/shortcut naming is `Hermes Work`; update identity,
  `hermes://`, executable/artifact naming, AUMID and the historical `Hermes`
  userData directory remain stable for upgrades.

- Hermes Workstation is first-class in this downstream fork and Desktop exposes
  the built-in `/browser` route.
- The internal browser uses Electron Chromium through `WebContentsView` and a
  dedicated persistent Electron session/profile outside the repository.
- Navigation, ordinary tabs, task-owned tabs, background parking, pause/resume,
  focus and human/agent control primitives are implemented.
- The Workstation controller is loopback-bound and bearer-token authenticated.
- `browser_*` prefers the Workstation controller before allowed fallback and
  remains fail-closed after task binding.
- Desktop Browser schema capability is session-scoped and protected from
  process-global reachability/cache leakage.
- Contextual Chat Browser View (`WorkstationBrowserPane`), global Browser Hub (`BrowserView`),
  and single-host Viewport Transfer (`transferViewport`).
- Host-aware viewport geometry propagation rejects stale resize updates from a
  non-owner Chat/Hub pane while preserving Chromium zoom/clamp normalization.
- Live Task Rail (`TaskRail`) with task grouping (`active`, `waiting-for-human`, `background`, `recent`),
  individual task deletion, and clearing parked tasks.
- Responsive zoom DIP scaling ensuring Chromium viewport aligns flush with the UI window.
- Persistent Kanban (`kanbanCardId`) and run (`runId`) identity bindings with fail-closed enforcement in covered paths; the 2026-09-17 audit shows Run continuity is not yet universal across Workstation completion/contracts.
- Automatic multistep Kanban promotion, follow-up discovery with parent blocking, append-only
  Execution Journal (`ExecutionJournal`), and structured completion reports.
- Fail-closed LAN and Tailscale controller with auth preflight and network detection.
- V1.1 Multi-task scheduler (`MultiTaskScheduler`) enforcing the one-live-host invariant, with lease timeouts, heartbeats, and orphan task reaping.
- V2 Procedural Web Memory (`ProceduralMemory`) with dynamic intent discovery, reinforcement, and resilient multi-facet fallback anchoring (testid -> role -> text -> selector).
- V2 Compact Provenance-Aware Perception Engine (`PerceptionEngine`) with Lattice-inspired node summaries, hidden node filtering, and tiered smart budgeting that guarantees CTA preservation.
- V2 Drift Diagnosis & Governed Adaptation (`DriftGovernor`) with blocking cookie/modal overlay detection (`DISMISS_OVERLAY`) and strict financial/irreversible action boundaries.
- V2 Lightpanda headless stateless runtime adapter (`LightpandaAdapter`) with transparent gzip/deflate decompression and fail-closed auth redirect detection.
- Windows Filesystem atomic resilience in Electron session persistence (fallback on `EPERM`/`EBUSY` via `copyFileSync`).
- Desktop UIX additions: high-contrast attention styling for human intervention in `TaskRail` and interactive Downloads drawer in `BrowserView`.
- **V2.1 Chrome Web Store Extensions**:
  - `ChromeExtensionManager` in `workstation/extensions.py` with official Chromium update endpoint CRX downloader (`clients2.google.com/service/update2/crx`).
  - Native ZIP/CRX unpacking with `Cr24` header stripping into `~/.hermes/workstation/extensions/<id>/`.
  - Automatic loading in Electron Chromium via `session.loadExtension(..., { allowFileAccess: true })`.
- **V2.5 Agent Runtime & System Capability Control Plane**:
  - `WorkerRegistry` in `workstation/workers.py`: discovery, readiness, and bounded subtask delegation for Codex, Claude Code, Antigravity, OpenCode, and K-Tools-Neo while retaining canonical Hermes task/session/card lineage in `ExecutionJournal`.
  - System Capability Layer in `workstation/host.py`: `HostCapabilityProvider` contract with `WindowsHostCapabilityProvider`, `LinuxHostCapabilityProvider`, and `KToolsNeoCapabilityAdapter`.
  - System Event -> Hermes Task Pipeline in `workstation/events.py`: `SystemEventPipeline` converting system/host events into Kanban tasks or task enrichments.
  - Scoped Autonomy Policy Engine in `workstation/policy.py`: `ScopedPolicyEngine` evaluating actions across `ALLOW`, `SANDBOX`, `REQUIRE_APPROVAL`, `DENY` with auditable decision logs.
- **V3 Cross-Platform Agentic Workstation**:
  - Omarchy Linux reference host adapter in `workstation/omarchy.py` with launcher normalization and system skills.
  - `CrossPlatformHostManager` in `workstation/cross_platform.py` unifying capabilities across Windows and Linux.
  - `AgenticBenchmarkRegistry` tracking 8 major agentic environments (Omarchy, Hermes upstream, OpenHands, OpenCode, Claude Code, Codex, Antigravity, BrowserOS) against core architectural criteria.
- **V3.1–V3.4 runtime-hardening contract layer**:
  - `EvidenceStateStore`, bounded `RuntimeEventBus`, typed resources, deadlines,
    cancellation, human handoff and budget/model routing in `workstation/runtime.py`.
  - Independent child-process `RuntimeSupervisor`, checkpoint rollback,
    `RecoveryPlane` and dependency-light `recovery_cli.py`.
  - Versioned routine promotion/replay, persistent WorkerRegistry queue/control,
    temporal memory/snapshots, session ownership/migration/compaction,
    portable replay/fork, protocol adapters, isolation and evaluation gates.
  - Persistent worker results use a durable envelope/ACK handoff over the
    existing worker record: an executor result is persisted before it is
    journaled or published, unread results survive reconstruction, and claimed
    work remains explicit recovery work when persistence cannot be confirmed.
    Large spillover results are content-addressed within task scope and expose
    a reference-first metadata contract without becoming an argument-only
    execution cache.
  - Shared resource/event client boundary: Electron BrowserTask/page ownership
    publishes versioned `/resources` and bounded `/events` projections; Desktop
    IPC, Dashboard REST and TUI JSON-RPC consume the same read-only contracts,
    with controller loss represented as degraded state and control permissions
    failing closed.
- **Chat / Browser UX Hardening & WhatsApp Web Compatibility**:
  - Runtime-derived Chromium User-Agent (`getStandardChromeUserAgent(process.versions.chrome)`) in `apps/desktop/electron/workstation-browser-runtime.ts` across `browserSession.setUserAgent()`, `webRequest.onBeforeSendHeaders`, and `WebContentsView` instances; it preserves the platform token without exposing `Electron` or `Hermes` and cannot age behind a hardcoded Chrome version.
  - Session-scoped preview/browser pinning via `$sessionPreviewTabs` in `apps/desktop/src/store/preview.ts`: creating a new chat session presents a clean workspace with no lingering lateral panels from prior sessions, and switching back seamlessly restores that session's browser panels.
  - Browser Hub lateral rail suppression: `isBrowserHubRoute()` in `preview.ts`, layout effect in `apps/desktop/src/app/browser/index.tsx`, and event filtering in `use-preview-routing.ts` eliminate dual-rail collision when visiting `/browser`.
  - Friendly automatic task names in Browser Hub: `TaskRail` resolves chat conversation titles (`s.id === task.sessionHost || s.parent_session_id === task.sessionHost`) and page tab titles/domains, replacing raw task IDs with meaningful human context.
- **Hybrid Kanban Real-time Invalidation & Full Lifecycle (H-053)**:
  - Canonical domain functions `delete_card`, `delete_column`, `delete_board`, and `get_board_activity` in `hermes_cli/hybrid_kanban.py` with dense reindexing.
  - Realtime WebSocket invalidation over `/events`: emits `hybrid_events` and `hybrid_cursor` on mutation, prompting instant query invalidation in Desktop UI and eliminating the 8-second polling latency.
  - Desktop UI: horizontal column drag-and-drop reordering (`moveHybridColumn`), rich card activity drawer with human vs. agent provenance badges, and complete deletion lifecycle.
- **Human card → Agent Task delegation**: `delegate_card` creates a separate
  canonical Agentic task in the same Kanban database, persists the
  `human_card_id` ↔ `agent_task_id` link, projects compact status/result/evidence
  metadata, and keeps both lifecycles independent across restart and retry.
- **Hybrid Card checklists**: multiple checklists and ordered items live in
  additive `hybrid_*` tables in the canonical Kanban database. Completion,
  reorder and deletion enforce optimistic revisions, emit provenance through
  `hybrid_activity`, survive restart and are exposed in the existing Desktop
  card drawer.
- **Scoped human browser control**: `BrowserHumanControlLease` is owned by the
  bound `BrowserTask` and carries task/session/tab/page/profile scope, timestamps,
  renewal and expiry; stale leases are cleared during restore and never block an
  unrelated task.
- **Extension update safety**: `ChromeExtensionManager` stages updates behind a
  promotion journal and retains last-known-good content until Electron load and
  verification commit.
- **Operational compaction envelope and workload benchmark**: compaction retains
  bounded task/session/worker/browser/result/approval/evidence identifiers, and
  `workstation/benchmarks/workload_baseline.json` records deterministic structural
  counters without private databases or wall-clock thresholds.
- Automated contract coverage on the 2026-09-15 snapshot included **247/247 Workstation Pytests passing** and clean Desktop typecheck. These remain implementation evidence, not proof that the newly identified cross-domain invariants are closed.

### BrowserSessionState — promoted V1 #1

PR #11 accepted exact head `d5be442021ea0c744351622317eef5212219786d` and was merged as `e0a99ef3aba6e6d2b65c30cf3c908ee1d49c4d29`.
The exact-head native Windows/Electron probe emitted `H010_CLASSIFICATION=VALIDATED`.

## Partially implemented / reliability-hardening required

- Canonical TaskRun identity exists but is not yet propagated/fenced through every Workstation completion, event, evidence and effect boundary.
- Terminal WorkPlan/WorkItem tree consistency is not yet proven universally.
- External mutable-operation uncertainty/reconciliation is strong in TaskCompiler but not yet established as a universal Workstation effect invariant.
- Exact-once terminal projection from Agent Task into every journal/UI/Hybrid view still needs cross-restart regression evidence.
- Experimental non-Electron browser backends remain secondary fallbacks to the primary internal Chromium.
- External browser extensions operate strictly in unbound compatibility mode.

## Not implemented yet

- Autonomous multi-agent swarm arbitration across remote physical hosts without a local controller (long-horizon exploration beyond V3).

## Manual validation already observed

- H004 proved the promoted BrowserTask lifecycle with a real BrowserWindow,
  WebContentsView, renderer and two Windows/Electron processes.
- H010 on PR #11 exact head proved clean and abrupt two-process
  BrowserSessionState restart, profile separation, lazy exactly-one-page task
  recovery, failed-write convergence and explicit-destroy failure cleanup.
- A 2026-09-12 local rerun of the same versioned H010 probe passed on the
  current Windows/Electron toolchain and current working-tree product files;
  it remains explicitly classified as local dirty-tree evidence until the
  clean-checkout workflow gate runs.
- Native Electron dogfooding verified live Chromium rendering, synchronized chat right-rail,
  task deletion, and clear parked tasks in Browser Hub.
- The versioned V3 process-boundary smoke emitted
  `V3_RUNTIME_HARDENING_CLASSIFICATION=VALIDATED_CONTRACT_BOUNDARY`, proving
  independent child recovery, cross-process session ownership rejection and
  stale-evidence reconciliation on Windows 10.0.26200 / Python 3.13.
- The packaged Desktop headless boundary smoke covered the renderer, HUD,
  Desktop IPC and authenticated loopback controller for both resource and event
  projections; the exact packaged artifact passed **6/6** without revealing a
  native window.
- The hidden native H011 Browser runtime soak passed **39 process episodes / 936
  task cycles / 4 tasks** in 60 seconds, including real Chromium navigation,
  hub/chat host changes, hide/park transitions, resource lineage and composite
  restart persistence. It is explicitly bounded to the native runtime.
- The real headless H012 backend soak passed **12 cycles / 48 turns / 44
  reconnects / 2 backend restarts / 48 heartbeats**, with 192 streamed events,
  zero errors and four concurrent turns through the authenticated WebSocket
  gateway. It uses the deterministic synthetic-turn seam and does not claim
  real-provider quality or clean-machine release qualification.

## Known bugs / gaps

See `KNOWN_ISSUES.md` and the active reliability gate.

- The 2026-09-17 confirmed execution-identity/completion/journal contradictions are active P0 work even though the underlying subsystem tests are green.
- KI-003 is resolved by promoted BrowserSessionState.
- KI-002/KI-004 (Preview duplication and host overlap/ownership composition)
  are resolved by single-host viewport transfer and Workstation preview pane.
- KI-006 is resolved on the current working tree by the HW-018 Windows
  portability closure; the historical baseline comparison remains below as
  provenance, not as an open failure.
- KI-009 is resolved: the packaged Desktop GUI Playwright smoke passes after
  lazy Workstation Browser startup and per-sandbox state isolation.
- V3 product-level clean-machine release qualification, event/resource parity
  for future client surfaces and candidate-release confirmation of the
  full-duration/production-scale Desktop/Browser profile remain evidence
  gates; the Python contract layer does not silently promote those claims. The
  Chromium/Firefox/Edge Dashboard smoke is validated when the supported Edge
  browser is present. The read-only
  `python -m workstation.release_qualification` runner now requires a clean
  checkout, makes the local/native/migration gates reproducible and refuses to
  infer clean-machine evidence; the Windows workflow emits the candidate-
  matched external clean-install report with an isolated `RUNNER_TEMP`
  absolute `workstation_home` outside the checkout that install and doctor
  share.
- A local isolated candidate snapshot completed install, `npm ci`, Desktop
  production build, strict doctor, release qualification **6/6** and the
  hidden H013 profile (**2 passed in 3.2 minutes; 16 tasks, 173 rounds,
  120,262 ms, 8 chat turns**). This is stronger local evidence, but its
  synthetic candidate is not the official clean-machine CI gate.

## Latest automated validation state

The following results are retained as evidence for the implementation boundaries
they actually exercised; they do **not** close the 2026-09-17 reliability gate.

- **247/247 Pytest tests passed** across the earlier Workstation contract snapshot, LAN/Tailscale,
  Kanban/Journal, Procedural Memory, Perception Engine, Drift Governance,
  Lightpanda Runtime, Multi-Task Scheduler, Chrome Extensions, Worker Registry,
  Host Capabilities, System Events Pipeline, Scoped Policy, Cross-Platform/Omarchy
  and V3.1–V3.4 contracts.
- **Production dependency audit:** `npm audit --omit=dev --audit-level=moderate`
  passed with **0 vulnerabilities** after the lockfile-only refresh of the
  affected transitive packages.
- **Desktop typecheck passed with 0 errors** across app, Electron and E2E configs.
- **Desktop focused resource/runtime tests:** 20/20 passed; Dashboard typecheck
  and the Python resource-client tests also pass.
- **Dashboard cross-engine smoke:** Chromium, Firefox and the installed Edge
  browser all passed the provider-free `/workstation` route smoke (`3 passed
  in 9.2s`).
- **Canonical reconnect soak sample:** 3/3 fresh-process iterations passed with
  3 sessions, 0 failures, 36 journal events, 9 memory snapshots and 6 worker
  reconstructions, 6 model changes and 6 cold reloads; long-duration
  Desktop/Browser soak remains open.
- **Desktop UI suite:** 591 files / 5,669 tests passed with Vitest bounded to
  `--maxWorkers=4` on the current Windows audit host. The targeted messaging
  file also passed 8/8.
- **Desktop platform/Electron suite:** 126 files / 1,778 tests passed, 5 skipped.
- **KI-006 closure:** Windows path/permission/SSH/WSL/staging/locale contracts
  pass without disabling the broad suites.
- **Client parity boundary:** Dashboard REST and TUI JSON-RPC each read the
  same authenticated controller projection; the live HTTP adapter test passed
  with identical browser/task/journal identities, while H013 compares the full
  controller/Desktop IPC resource projection.
- **Event parity boundary:** Dashboard REST and TUI JSON-RPC each read the
  bounded event projection for the same task; the Electron runtime test covers
  canonical journal ordering and limits, while H013 compares the full bounded
  controller/Desktop IPC event payload.
- **Packaged projection/UI boundary:** the exact `win-unpacked` artifact passed
  the complete headless packaged GUI/HUD smoke (**6/6**) plus IPC/controller
  resource and event identity checks; visible desktop-session reveal remains a
  separate environment-specific gate.
- **Local isolated candidate qualification:** a clean temporary candidate with
  real install, `npm ci`, production build and external Workstation home passed
  the release runner **6/6**; its hidden H013 candidate profile also passed
  **2/2** and its report was accepted by `workstation.desktop_load_evidence`.
  Official candidate CI remains required for release promotion.
- **Windows workflow coverage:** the release workflow now installs the required
  Playwright browsers, runs the strict Workstation doctor, runs cross-engine
  Dashboard smoke, builds the unpacked artifact, runs the four-session
  180-second reconnect soak, runs the native H011 Browser runtime soak, and
  runs the H012 real headless backend reconnect soak, packaged GUI E2E and H013
  integrated Desktop/Browser load E2E, with explicit final outcome checks and uploaded
  evidence artifacts. H013 selects a bounded 16-task/120-second candidate
  profile with eight real backend chat turns, writes a bounded JSON evidence
  report validated by `workstation.desktop_load_evidence`, and is rejected
  when that report is missing or under-sized.
- **60-second local soak:** 77 fresh-process iterations and 0 failures across
  3 sessions; the duration budget ended normally with `timed_out=true`.
- **Latest 180-second local soak:** 128 fresh-process iterations and 0 failures
  across 4 sessions, with 2,560 actions, 2,048 journal events, 15,888
  cumulative memory records, 512 snapshots, 508 worker reconstructions, 508
  model changes and 508 cold reloads; live memory peaked at **32 records**
  (the explicit 8-record-per-session bound), and the duration budget ended with
  `timed_out=true`.
- **Native H011 Browser runtime soak:** 39 hidden Electron process episodes and
  936 task cycles across 4 tasks passed in 60 seconds, with composite state and
  resource lineage preserved across reconnects.
- **Headless H012 backend soak:** 12 real `hermes serve` cycles, 48 streamed
  turns, 44 session reconnects, 2 backend process restarts, 48 heartbeats and
  zero errors passed with four concurrent sessions.
- **H013 integrated Desktop/Browser E2E:** the hidden-window four-task
  controller/IPC scenario plus the default sustained eight-task/three-round
  backend load passed locally in **2 tests / 38.8s** on the latest rerun
  (36.6s on the prior run), including native Chromium navigation, host-aware
  viewport geometry/transfer, native maximize/restore reconciliation, three
  real chat turns, complete resource/event projection parity and
  hide/park/destroy cleanup. A
  local scaled 12-task/15-second profile also passed **2 tests / 55.4s**. The
  exact Windows profile also passed locally: **16 tasks / 170 rounds / 120,651
  ms / 8 chat turns / 2 tests in 3.0 minutes**. The Windows workflow
  additionally selects that bounded 16-task/120-second profile and still
  requires the headless gate and clean-machine evidence.
- The isolated clean candidate also passed H013 **2/2 in 3.2 minutes**:
  16 tasks, 173 completed rounds, 120,262 ms and 8 backend chat turns; the
  resulting report passed the shared evidence validator.

## Promotion status

- Implementation 4 BrowserTask: **PROMOTED / RESOLVED** through PR #9.
- V1 #1 BrowserSessionState: **PROMOTED / RESOLVED** through PR #11.
- V1 #1.5 sequencing + launcher: **PROMOTED** through PR #12 as historical implementation work.
- Pre-1.5 Mainline Consolidation Gate: **PASS**.
- V1 #1.5, V1.1, V2, V2.1, V2.5, and V3: substantial implementation exists and prior scoped verification remains valid for those boundaries.
- V3.1–V3.4: **CONTRACT LAYER IMPLEMENTED & VALIDATED** for their tested boundaries; this does not imply cross-domain causal completion correctness.
- **Canonical Execution Reliability Gate (2026-09-17): ACTIVE / BLOCKS FURTHER FEATURE EXPANSION.**

## Canonical Work Loop hardening (2026-09-17)

The working tree extends canonical acceptance/intent/liveness/journal and durable
execution owners. The full product program is still IN PROGRESS. See
[implementation evidence](CANONICAL_WORK_LOOP.md) for exact scope, additive
migrations, compatibility boundaries and remaining integrations.
