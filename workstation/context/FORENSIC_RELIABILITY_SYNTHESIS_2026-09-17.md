# 2026-09-17 Forensic Reliability Synthesis

Status: **ACTIVE INPUT TO THE CANONICAL EXECUTION RELIABILITY GATE**

This document integrates three independent 2026-09-17 investigations of Hermes Work and reconciles their conclusions against the current downstream `main` implementation. It is an evidence map and prioritization note, not a new runtime authority or a replacement for `CURRENT_STATE.md`, `DECISIONS.md`, or `CANONICAL_EXECUTION_RELIABILITY_GATE.md`.

## Sources integrated

The synthesis combines:

1. the Antigravity deep investigation of the current Hermes Work codebase and canonical-loop tests;
2. the first ChatGPT investigation/benchmarking and current-main gap analysis, including the product-level “killer loop” from human intent through verified outcome and reusable routine;
3. the second ChatGPT persisted-state/invariant investigation, including WorkPlan/WorkItem snapshots, cross-domain lineage, terminality, effect certainty and acceptance analysis.

The source reports are treated as hypotheses/evidence, not as architectural truth by themselves. When a report conflicts with current `main`, current code plus reproducible behavior wins. Historical findings that are already fixed become regression cases, not new implementation work.

## Converged thesis

All three investigations converge on the same next product problem:

> Hermes Work already has broad agentic capability. The immediate bottleneck is whether the state presented to the user is causally attributable to the execution that actually happened and is backed by evidence strong enough to justify the claimed outcome.

The next milestone is therefore not “Hermes can do more things.” It is:

> **Hermes can prove which work is active, which execution owns it, what external effects happened, whether those effects are certain, and why the reported result is accepted.**

The target operating loop is:

```text
human/system origin
  -> typed Intent Authority
  -> canonical Task / Human Card / Cron work identity
  -> canonical TaskRun
  -> WorkPlan / WorkItems when deterministic execution is appropriate
  -> Browser / Worker / Host operations
  -> operation identity + evidence + artifacts
  -> reconciliation when reality is ambiguous
  -> acceptance / verification
  -> canonical lifecycle commit
  -> journal / UI / Hybrid projections
  -> result or planned human handoff
  -> measured experience
  -> validated procedure / deterministic routine
```

## Four properties that outrank feature expansion

The investigations reduce the next phase to four cross-domain properties:

1. **Identity** — every externally mutating agent action is attributable to the authoritative TaskRun and operation identity, without reconstructing ownership from transcript prose.
2. **Terminality** — terminal parent state is a tree invariant, not a local flag; live descendants cannot silently survive a terminal parent.
3. **Effect certainty** — timeout/lost acknowledgement after dispatch is not proof that a side effect did not happen; uncertain effects reconcile before retry.
4. **Acceptance** — `execution ended`, `result exists`, `Task done`, and `verified outcome` are separate facts; canonical completion requires current-run authority plus the required acceptance evidence.

Everything else in the immediate gate — browser leases, recovery, failure cohorts, cron health, evidence provenance, Runtime State Resolver, Work 100 and Cockpit UX — exists to make these properties universal and observable.

## Current-main evidence classification

### Confirmed open on current `main`

- **`kanban_db.connect(db_path=...)` path-type crash.** `hermes_cli/kanban_db.py` currently assigns `path = db_path` and then uses `path.parent`; callers that provide a string can still raise `AttributeError`. The Antigravity regression `test_two_processes_migrate_same_trello_manifest_once` must be restored to green with a type-normalization fix rather than worked around.
- **Workstation completion is not fully run-fenced.** `WorkstationKanbanBridge.complete_task_with_report()` calls canonical completion without `expected_run_id` even though canonical Kanban already supports the fence.
- **Terminal journal can outrun canonical commit.** The same bridge records `TASK_COMPLETED` after the completion call regardless of whether the canonical transition returned `False`.
- **Run lineage is not first-class at every boundary.** `ExecutionEvent`, `BrowserTaskReport` and deterministic WorkPlan persistence still do not universally carry canonical `run_id`/operation lineage.

### Confirmed from persisted dogfood/state evidence; invariant still must be proven on current code

- Persisted WorkPlans were observed in terminal/interrupted state while descendants remained `running` or `pending`, including a 12-item plan with live descendants and a 100-item interrupted plan with 99 pending items. This makes terminal-tree reconciliation a P0 product invariant, even if individual local transition helpers already attempt cleanup.

### Existing mechanisms to reuse, not reimplement

- TaskCompiler canary-before-fan-out, verification requirements, dispatch checkpoints, idempotency contracts, recipe staleness/quarantine and uncertain-mutation escalation.
- canonical Kanban `tasks`, `task_runs`, `current_run_id` and run-aware CAS support.
- BrowserTask, scoped human-control leases and the dedicated Electron Chromium profile/runtime.
- semantic browser readiness primitives already present in `workstation/browser_readiness.py`; the open work is universal integration/coverage and safe drift/hydration/CAPTCHA behavior, not creating another readiness system.
- EvidenceState, RuntimeEventBus, RuntimeSupervisor, RecoveryPlane, WorkerRegistry, ExecutionJournal and the ArtifactStore/reference plane.
- Hybrid Human Card -> Agent Task delegation with deliberately separate lifecycles.
- procedural memory/routine-promotion machinery, no-progress guardrails and existing evaluation/soak infrastructure.

### Must be audited before changing; do not assume the source report is still current

- whether every Gateway/messaging/scheduler/recovery ingress preserves typed origin and Intent Authority;
- whether journal append/replay cost actually grows materially at long-run sizes and where the hot path is;
- whether active/compacted/FTS content still duplicates semantic history after current dedup changes;
- the exact authority split between cron definitions, scheduler projections, execution records and model/provider drift handling;
- whether Agent Task -> Human Card terminal result projection is exactly-once across restart/retry/re-delegation;
- whether every mutable Browser/Worker/Host path consumes TaskRun fencing and rejects stale commands.

## Immediate implementation order

This order supersedes feature/polish work until the Canonical Execution Reliability Gate closes.

### P0.0 — concrete blocker and regression restoration

1. Normalize `db_path` to `Path` in canonical Kanban connection setup and restore the reproduced multi-process migration regression.
2. Add the regression to the immediate gate so the fix cannot silently regress.
3. Re-run the focused canonical-loop/continuity suite before broader invariant work.

### P0.1 — canonical execution lineage

Make the chain explicit and queryable without transcript reconstruction:

```text
Task -> TaskRun -> WorkPlan/WorkItem -> operation -> evidence/artifact -> acceptance -> canonical outcome
```

Keep `work_<hash>` as `execution_key`/plan identity; never let it replace canonical Task or TaskRun identity.

### P0.2 — strict completion/commit ordering

- pass `expected_run_id` through Workstation completion;
- reject late completion from superseded runs;
- require acceptance evidence for verified completion;
- commit canonical Task state before `TASK_COMPLETED`, UI, Browser or Hybrid terminal projections;
- emit typed rejected/superseded completion events when the CAS fails.

### P0.3 — terminal-tree reconciliation

A terminal parent must reconcile descendants before the system exposes a clean terminal picture. While reconciliation is unresolved, expose `NEEDS_RECONCILIATION`/`UNCERTAIN` rather than contradictory state.

### P0.4 — operation identity and uncertainty

Every external mutation gets durable `operation_id`; retryable targets use idempotency identity when supported. Lost acknowledgement after dispatch enters `UNCERTAIN -> RECONCILING`, then resolves to applied/not-applied/human-needed before retry.

### P0.5 — Intent Authority everywhere

Origin such as HUMAN, SYSTEM, TOOL, SCHEDULER, WORKER, RECOVERY and INTERNAL is typed data. Content may describe intent; provenance grants authority. Non-human events may wake/enrich work or create explicitly policy-authorized system work, but must never silently masquerade as human `CREATE_WORK`.

### P1 — reliability, recovery and browser truth

- run/operation fencing for Browser, Worker and Host mutable leases;
- startup and periodic reconciliation for Task/Run/WorkPlan/worker/browser/evidence/journal/delegation/cron relations;
- exactly-once Agent Task -> Human Card result/evidence projection;
- failure fingerprints/cohorts so one structural failure does not become N identical effects;
- semantic Browser readiness integrated into mutation paths: re-acquire semantic state after navigation/reload/SPA hydration, classify CAPTCHA/auth walls as handoff/blockers, never reuse stale DOM refs as durable identity;
- audit cron health/migration semantics and preserve fail-closed model/provider drift handling rather than rebuilding cron.

### P2 — deterministic state and context efficiency

- evidence provenance tied to Task/Run/operation/verifier/criterion/environment;
- Runtime State Resolver for deterministic facts: current task/run, browser lease, last effect, artifacts, blockers, dependencies, approvals, verification;
- reference-first large outputs and artifact handles instead of repeated raw blobs;
- context/FTS canonicalization and measured `Context Reconstruction Overhead`;
- event-driven wait/wake where possible and bounded no-progress/polling;
- benchmark journal append/read/replay at >=5,000 events before choosing an incremental-hash optimization;
- expose typed capability/readiness/error contracts so normal agent execution does not need source-code archaeology to answer deterministic capability questions.

### P3 — evaluation, learning and trust UX

- grow Hermes Work 100 from corpus-derived failures, not generic synthetic quantity;
- measure AVCR together with False Completion, Zombie Running, Evidence Coverage, Human Rescue vs Planned Handoff, Recovery Success, Uncertain Mutation, Systemic Failure Amplification and cost/time/tokens/tool-calls per verified outcome;
- promotion lifecycle: `Experience -> Candidate -> Validate -> Promote -> Routine -> Drift`, with measured benefit and rollback/quarantine;
- Task Cockpit/Control Center remains a projection of canonical truth and only becomes the primary UX after P0/P1 invariants are proven.

## Provisional reliability guardrails

These are engineering targets for the gate, not marketing claims:

- False Completion Rate: **0** in the corpus-derived reliability suite;
- stale/superseded run terminal commits: **0**;
- terminal parent with live descendant after bounded reconciliation: **0**;
- blind retry while a mutable operation is `UNCERTAIN`: **0**;
- structural failure amplification after detection/canary: **<= 1 additional exposed equivalent item**;
- journal/projection claiming completion before canonical commit: **0**;
- non-human event silently becoming human work intent: **0**.

Zombie-running and efficiency targets should be set from a measured baseline rather than copied blindly from a report.

## Hermes Work 100 seed priorities

The initial suite should include at least:

- terminal parent + live child;
- stale run late completion;
- completion CAS rejected while journal attempts terminal event;
- mutable timeout after dispatch;
- effect applied + ACK lost;
- restart between dispatch and checkpoint;
- system event cannot become human intent;
- 12x and 100x shared-failure amplification;
- worker dies while task says running;
- provider 429/backoff;
- Browser restart + stale DOM ref;
- two sessions competing for one mutating page lease;
- human takeover fencing;
- SPA hydration / DOM drift / CAPTCHA/auth wall;
- stale procedure/Skill before fan-out;
- cron provider/model drift;
- missing/corrupt artifact reference;
- clean E2E test-state isolation;
- Hybrid delegation result projection after restart;
- clean Windows/Electron restart/recovery with a real BrowserTask.

Each case records expected behavior, forbidden behavior, final canonical state, events, evidence coverage, tool calls, tokens/cost when available, elapsed time and human interventions.

## Explicit defer rule

Until the active gate closes, all historical V1/V1.1/V2/V2.1/V2.5/V3/V3.1-V3.5/V4 feature expansion, polish and research items remain preserved but **deferred**. Existing implementation labels remain historical facts; they do not override the immediate reliability sequencing.

Do not add more parallelism merely to improve throughput before N=1 execution is causally trustworthy. Do not create another Kanban, SessionDB, Run store, browser abstraction, memory framework, scheduler or “super database” to solve cross-domain consistency. Strengthen lineage and reconciliable invariants across the owners that already exist.
