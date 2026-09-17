# Canonical Execution Reliability Gate

Date established: 2026-09-17

Status: **ACTIVE — immediate next implementation work**

This gate supersedes every feature-expansion milestone in `../ROADMAP.md` as the
next engineering priority. Existing V1/V1.1/V2/V3/V3.5/V4 work is preserved, but
new feature work resumes only after this gate's exit criteria are proven.

The gate exists because the 2026-09-17 forensic audit found that Hermes Work
already has most of the required capabilities, but cross-domain causal identity is
not yet strong enough to guarantee that the state shown to the user corresponds to
the execution that actually produced the external result.

The objective is not a rewrite and not another state store. It is to make the
existing owners compose into one trustworthy operational loop:

```text
intent
  -> canonical Task
  -> TaskRun
  -> WorkPlan / WorkItems
  -> Browser / Worker / Host operations
  -> evidence
  -> acceptance / verification
  -> canonical commit
  -> journal / projections
  -> result
  -> learning / routine
```

## Evidence boundary

The gate integrates four evidence classes:

1. current `main` code and tests;
2. persisted production/dogfood state from Kanban, WorkPlan/WorkItem, sessions,
   verification, browser, cron, artifacts and journals;
3. historical conversation traces used as a pre-hardening behavioral baseline;
4. the 2026-09-17 independent analyses, rechecked against current code rather than
   accepted as design truth.

Important current-code facts:

- `hermes_cli.kanban_db.complete_task()` already supports `expected_run_id`, so
  run fencing has an existing canonical owner and must be reused.
- `workstation/kanban.py::complete_task_with_report()` currently does **not** pass
  that run identity and writes `TASK_COMPLETED` to the Execution Journal even
  when the canonical completion call returns `False`.
- `ExecutionEvent` and `BrowserTaskReport` currently carry task/session identity
  but no first-class `run_id`.
- `WorkPlan`/`WorkItem` persist `task_id` but no explicit canonical `run_id` or
  separate `execution_key` field.
- TaskCompiler already implements important protections — canary-before-fan-out,
  explicit mutation verifiers, idempotency requirements for declared idempotent
  writes, persisted dispatch checkpoints, recipe staleness/quarantine and
  uncertain-mutation escalation. These mechanisms must be reused and hardened,
  not rebuilt.
- BrowserTask, human-control leases, WorkerRegistry, Recovery Plane, EvidenceState,
  Execution Journal, Hybrid Kanban delegation and canonical Kanban TaskRun state
  already exist. The gate connects these authorities instead of creating peers.

## Canonical interpretation of One Hermes State

`One Hermes State` does **not** mean one SQLite database.

It means:

> one canonical authority per domain + shared causal identity + explicit lineage +
> reconcilable projections.

Examples:

- Kanban Task is authority for durable agent work responsibility;
- Kanban TaskRun/current_run_id is authority for an execution attempt;
- WorkPlan/WorkItem own deterministic compiled-plan progress;
- BrowserTask owns durable browser-work identity and one-live-page semantics;
- browser/worker/process handles are live leases, not durable proof;
- Execution Journal is an auditable record, not a competing lifecycle owner;
- Hybrid Card remains authority for the human workflow while delegated Agent Task
  remains authority for the agent workflow;
- Evidence/projection stores may describe state but do not become authoritative
  merely because a canonical owner is unavailable.

## P0 — causal execution invariants

P0 is dependency ordered. Do not start feature-expansion work between these items.

### P0.1 — Canonical Task -> Run -> Execution identity

**Problem:** Task, Run, WorkPlan and BrowserTask identities are present, but the
compiled execution boundary does not persist enough first-class run lineage.

**Change:**

- keep `tasks.id` as the canonical Agent Task identity;
- keep existing `task_runs.id` / `tasks.current_run_id` as the canonical attempt;
- make every new WorkPlan persist both canonical `task_id` and `run_id`;
- treat the deterministic `work_<hash>` identity as an `execution_key` / plan
  identity, never as a replacement Task identity;
- preserve `session_id`, BrowserTask id and delegation/card links as lineage;
- use additive migration/backfill only. Legacy rows that cannot be safely bound
  become explicitly `legacy_unbound` / reconciliation-required rather than guessed.

**Do not build:** another Run table, another Task store, another SessionDB.

**Definition of Done:** every new WorkPlan can answer, without transcript
reconstruction: `which Task? which TaskRun? which session? which BrowserTask/worker
if any? which operation?`.

### P0.2 — Run-aware Workstation contracts

Propagate `run_id` through the existing contracts that can claim or mutate execution
state:

- `ExecutionEvent` / Execution Journal;
- verification/evidence records;
- typed resources and runtime events;
- BrowserTask report/completion metadata;
- Worker execution/result envelopes;
- host/process/browser mutation context;
- Recovery Plane reconciliation events;
- Human Card delegation result projection.

Task-scoped browser persistence remains Task-scoped. Run identity is fencing and
provenance for **effects**, not a reason to create a second BrowserTask for each
attempt.

**Definition of Done:** every externally visible mutation/evidence item can be
attributed to exactly one canonical TaskRun or is explicitly classified as a human
or system action outside agent-run authority.

### P0.3 — Strict completion and acceptance gate

`execution ended`, `result exists`, `Task done`, and `Verified Completion` are
separate facts.

**Change:**

- Workstation agent completion must call canonical Kanban completion with
  `expected_run_id`;
- stale/superseded runs cannot complete a Task;
- required acceptance criteria and evidence must be satisfied before
  `verified_completed`/equivalent projection;
- human/manual completion remains possible only as an explicit separately audited
  authority path;
- failure to commit canonical completion must remain failure/superseded state, not
  a successful narrative result.

**Definition of Done:** a run A that loses ownership to run B cannot mark the Task
complete after B claims it, even if A produces a late result.

### P0.4 — Canonical commit before terminal journal/projection

The journal records facts that committed; it must not race ahead of authority.

**Change:**

- emit `TASK_COMPLETED` only after the canonical Task transition succeeds;
- unsuccessful CAS/expected-run completion emits a typed superseded/rejected event,
  never a completion event;
- terminal projections to Browser/Hybrid Card/UI follow canonical commit;
- use stable event/idempotency identity so recovery may safely replay projection;
- if cross-store atomicity later requires it, introduce a transactional outbox in
  the **existing canonical owner**, not another lifecycle store.

**Definition of Done:** no test can produce `journal=completed` while canonical Task
remains non-terminal because the completion transition returned `False`.

### P0.5 — Terminal-tree reconciliation barrier

Persisted audit data contained terminal WorkPlans with live descendants. Existing
recovery/reconciliation machinery must enforce a transversal invariant:

```text
parent.status in TERMINAL
  => no descendant is externally exposed as LIVE after reconciliation barrier
```

`LIVE` includes pending/ready/claimed/running/retrying equivalents.

**Change:**

- terminal parent transition computes its reconciliation set;
- cancellation/interruption intent propagates to descendants using existing
  cancellation/recovery owners;
- late child completion is reconciled against parent/run authority instead of
  silently resurrecting the tree;
- while the barrier is unresolved, expose `NEEDS_RECONCILIATION` / `UNCERTAIN`
  rather than a false terminal picture.

**Definition of Done:** corpus-derived interrupted-plan regressions produce zero
live descendants after bounded reconciliation.

### P0.6 — Mutable operation identity and uncertainty protocol

Timeout is an observation by the caller, not proof that an external mutation did
not happen.

Every mutation must have a durable `operation_id`; effects that can be retried must
also have an idempotency identity where the target supports one.

Minimum effect lifecycle:

```text
PREPARED -> DISPATCHED -> ACKNOWLEDGED -> VERIFIED
                    \-> UNCERTAIN -> RECONCILING
                                  -> VERIFIED_APPLIED
                                  -> VERIFIED_NOT_APPLIED
                                  -> NEEDS_HUMAN
```

Reuse TaskCompiler's existing dispatch checkpoints and
`uncertain_mutation_requires_review`; generalize the invariant to other mutable
Workstation surfaces rather than replacing that implementation.

**Definition of Done:** timeout/network loss/runtime death after dispatch can never
cause a blind automatic duplicate mutation.

### P0.7 — Intent Authority coverage

Typed origin/authority must be enforced at every work-creation boundary, not only
inside one path.

Audit Gateway/message adapters, system events, scheduler/cron, worker/internal and
recovery inputs so they preserve origin semantics such as HUMAN, SYSTEM, TOOL,
SCHEDULER, WORKER, RECOVERY and INTERNAL.

A system notification or tool observation may enrich/wake existing work; it must not
become new user intent unless an explicit policy authorizes that transition.

**Definition of Done:** a corpus-derived system-notification regression cannot
create an Agent Task as if it were a human request.

## P1 — reliability, fencing and recovery

### P1.1 — Run fencing for Browser, Worker and Host mutations

- one live mutable resource lease has at most one mutating TaskRun owner;
- commands carry task/run/operation + fencing epoch/token;
- human takeover explicitly suspends/revokes agent mutation authority;
- stale commands from a pre-takeover or superseded run are rejected;
- read-only observation may be shareable when policy allows.

### P1.2 — Startup and periodic reconciliation

Extend existing RuntimeSupervisor/RecoveryPlane/EvidenceState paths so startup can
reconcile:

- Task <-> TaskRun;
- WorkPlan <-> WorkItem;
- TaskRun <-> worker/process;
- TaskRun <-> BrowserTask/page lease;
- operation <-> evidence;
- canonical Task <-> journal terminal projection;
- Agent Task delegation <-> Human Card projection;
- cron definition <-> scheduler/execution health.

Recovery must classify ambiguous state; it must not silently guess a successful
outcome.

### P1.3 — Hybrid delegation exactly-once projection

Preserve the existing two-lifecycle design:

`Human Card -> Delegation -> Agent Task`.

Add/reinforce an idempotent reconciler that projects terminal Agent Task
result/evidence to the original Human Card exactly once across restart, retry and
re-delegation. Human Card status/columns remain independent unless an explicit user
policy says otherwise.

### P1.4 — Systemic failure cohort / circuit-breaker coverage

TaskCompiler already implements canary admission for mutable batches. Treat this as
an existing mechanism and extend regression coverage to fan-out surfaces that can
still amplify one shared failure.

Requirements:

- shared failure fingerprint/cohort;
- canary/small cohort before large equivalent fan-out where appropriate;
- repeated auth wall/provider failure/stale skill/schema drift halts the cohort;
- one structural cause must not become N identical external effects/failures.

### P1.5 — Cron autonomy health and migration semantics

Audit current cron definition/execution/scheduler authorities before changing them.
Provider/model/config drift must become an actionable state such as
`NEEDS_MIGRATION`, `BLOCKED_CONFIG_DRIFT` or equivalent, not an infinite failed
loop. User-visible automation health must distinguish disabled, paused, blocked,
failed, degraded and healthy work.

## P2 — efficiency and auditability

### P2.1 — Evidence provenance contract

Evidence must answer:

- what claim does it support?
- which Task/Run/operation produced it?
- which verifier/acceptance criterion evaluated it?
- when and in which environment/profile/provider/tool version?
- is it live evidence, durable proof or a projection?

Do not infer Verified Completion from the mere existence of an evidence-state row.

### P2.2 — Runtime State Resolver

Move objective bookkeeping out of the model. Provide bounded structured resolution
for current Task, Run, plan/item, browser/resource lease, last operation, blockers,
artifacts, dependencies, approvals and verification state.

The model should reason about ambiguity and exceptions, not repeatedly rediscover
IDs or ask deterministic questions such as "did this process finish?" or "which tab
was bound?".

### P2.3 — Reference-first tool/artifact boundary

Reuse the existing ArtifactStore/reference plane. Large tool outputs and repeated
structured observations should be stored once and passed by trusted references with
bounded summaries. No second artifact store.

Instrument bytes/tokens avoided so the optimization is measured, not assumed.

### P2.4 — Context/FTS canonicalization

Audit active/inactive/compacted messages, snapshots, FTS and spillover for duplicate
historical content. Prefer references or canonical compacted representations over
indexing the same semantic text repeatedly.

Measure `Context Reconstruction Overhead`: tokens/tool calls/time spent recovering
state that was already available structurally.

### P2.5 — No-progress and bounded polling

Reuse current no-progress/circuit-breaker logic and event bus. Prefer event-driven
wakeups. Fingerprint repeated effective tool + args + observed state + failure and
escalate after bounded no-progress cycles.

### P2.6 — Journal scalability and integrity benchmark

Measure append/read/replay behavior at long-run sizes (>=5,000 events). Preserve
integrity/auditability while ensuring append does not become progressively expensive.
Do not optimize based only on one external analysis; profile current implementation
first and add a regression budget.

## P3 — evaluation, learning and UX

### P3.1 — Hermes Work 100 seed suite

Do not invent 100 generic cases immediately. Start with corpus-derived failures and
promote them into durable regressions. Initial seed must include at least:

- terminal parent + live child;
- stale run late completion;
- canonical completion rejected but journal tries to say completed;
- timeout after mutable dispatch;
- applied mutation + lost ACK;
- restart between dispatch and checkpoint;
- system event cannot become human intent;
- 12x/100x shared failure amplification;
- worker dies while task says running;
- provider 429/backoff;
- Browser restart and stale DOM ref;
- two sessions / one page mutation race;
- human takeover fencing;
- SPA hydration / DOM drift / CAPTCHA;
- stale procedure/Skill before fan-out;
- cron model/provider drift;
- artifact reference missing/corrupt;
- test-state isolation;
- Hybrid delegation projection after restart;
- clean Windows restart/recovery with real BrowserTask.

Each case records situation, evidence origin, expected behavior, forbidden behavior,
metrics and pass/fail.

### P3.2 — North-star and guardrail metrics

Use **AVCR — Autonomous Verified Completion Rate** only as a north-star candidate,
not as the sole score. Its denominator must be eligible delegated work, and a
completion counts only after acceptance/evidence verification.

Instrument together:

- Verified Completion Rate;
- False Completion Rate;
- Zombie Running Rate;
- Human Rescue Rate vs Planned Handoff Rate;
- Recovery Success Rate;
- Uncertain Mutation Rate;
- Systemic Failure Amplification;
- Evidence Coverage;
- Tool Calls / Tokens / Cost / Time per Verified Outcome;
- Context Reconstruction Overhead;
- Routine Reuse and Routine Savings;
- regression/rollback rate after procedure promotion.

### P3.3 — Experience -> Candidate -> Validate -> Promote -> Routine -> Drift

Skills and procedures should improve Hermes only when their benefit is measured.
For each version record usage, verified success/failure, tokens, tool calls, wall
time, exception/fallback count and drift incidents.

Promote a procedure to deterministic routine only after replay/validation proves
benefit. Drift invalidates/quarantines the affected version and returns control to
Hermes; it must not silently rewrite historical routine behavior.

### P3.4 — Task Cockpit / Control Center projection

Only after P0/P1 state truth is proven, make the trust model obvious in UX. The
cockpit should project, not own:

`INTENT | TASK | RUN | NOW | EVIDENCE | BLOCKER | HUMAN ACTION | OUTCOME`.

Expose concise progress, active resources, cost, latest verified transition,
artifacts and handoff controls. Deep technical logs remain inspectable.

## Immediate regression/implementation targets

Expected primary touch points include, but are not limited to:

- `hermes_cli/kanban_db.py` and canonical task/run transitions;
- `workstation/kanban.py`;
- `workstation/contracts.py`;
- `workstation/durable_tasks.py`;
- `workstation/task_compiler.py`;
- `workstation/runtime.py` / evidence/resources/event reconciliation;
- `workstation/journal.py`;
- BrowserTask/runtime/controller mutation paths;
- `workstation/workers.py` and host capability adapters;
- Hybrid Kanban delegation/reconciliation;
- cron scheduler/jobs health paths;
- context/session compaction + FTS paths;
- Workstation contract tests, canonical Kanban regressions and real Windows/Electron
  E2E probes.

## Gate exit criteria

Feature roadmap work resumes only when all of the following are true:

1. **TaskRun continuity:** all new mutable Workstation execution has canonical
   `task_id + run_id + operation_id` lineage.
2. **Stale-run safety:** a superseded run cannot mutate canonical task completion or
   reuse a revoked mutable resource fence.
3. **Commit truth:** no terminal journal/UI/Hybrid projection can precede or
   contradict canonical commit.
4. **Tree consistency:** terminal parents reconcile all live descendants or expose
   an explicit reconciliation state.
5. **Mutation safety:** uncertain external mutations never blind-retry.
6. **Recovery:** restart can classify/reconcile running state without reconstructing
   truth from transcript prose.
7. **Intent Authority:** non-human inputs cannot silently become human work intent.
8. **Regression evidence:** the initial Hermes Work 100 seed passes, including
   corpus-derived stale-run, timeout, restart, browser and delegation cases.
9. **Real environment evidence:** relevant clean/isolated Windows + Electron paths
   pass; unit tests alone do not close the gate.
10. **Metrics integrity:** Verified Completion and false-completion/zombie/uncertain
    metrics are derived from canonical events and acceptance evidence, not narrative
    responses.

## Explicit non-goals during this gate

Do not build:

- a second Kanban, SessionDB, TaskRun store, BrowserTask store or Memory system;
- a "super database" that merges unrelated authorities;
- a new LLM orchestration framework around existing runtime state;
- a second browser abstraction;
- a new generic logging platform;
- unrestricted self-improvement;
- large new Skill catalogs;
- additional parallelism before N=1 execution is trustworthy;
- feature-polish dashboards whose state semantics are not yet proven.

The smallest architectural change that removes the largest failure class remains
the governing principle: strengthen shared invariants and causal identity before
patching isolated symptoms.