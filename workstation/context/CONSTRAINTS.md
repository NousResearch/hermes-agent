# Workstation Constraints

These constraints apply to Workstation changes in addition to the repository-wide rules in `AGENTS.md`.

## State and ownership

- Do not create a second SessionDB, Kanban, Memory system, approval system, or browser-routing control plane when Hermes already owns that concern.
- Do not duplicate Browser state to keep two independent pages “in sync”. BrowserTask is the semantic owner; UI surfaces are views/hosts.
- A live `WebContentsView` belongs to at most one active host at a time.
- Distinguish process-scoped, profile-scoped, Hermes-session-scoped, BrowserTask-scoped, TaskRun-scoped, operation-scoped, and renderer/view-scoped state explicitly.
- Closing/hiding a view must not imply destroying the BrowserTask unless the user or lifecycle explicitly requests destruction.
- `One Hermes State` means one canonical authority per domain + shared causal identity + explicit lineage + reconcilable projections. It does **not** mean one physical SQLite database.

## Canonical execution reliability

The 2026-09-17 Canonical Execution Reliability Gate is the active sequencing authority. Until it closes:

- every agent-owned mutating effect must be attributable to the authoritative canonical `TaskRun` plus a durable operation identity, or be explicitly classified as a human/system action outside agent-run authority;
- deterministic WorkPlan identity such as `work_<hash>` is an `execution_key`/plan identity, never a replacement Task or TaskRun identity;
- stale/superseded runs must not be able to commit task completion or reuse a revoked mutable-resource fence;
- canonical lifecycle commit must succeed before `TASK_COMPLETED`, terminal UI/resource state, or Hybrid Human Card result projection is emitted;
- a terminal parent must not expose live descendants after the reconciliation barrier. Ambiguous trees remain explicit `UNCERTAIN`/`NEEDS_RECONCILIATION` work rather than a false clean terminal state;
- timeout or lost acknowledgement after a mutable dispatch is not proof that the side effect did not happen. Such an effect enters uncertainty/reconciliation and must not blind-retry;
- origin/authority is typed data. SYSTEM/TOOL/SCHEDULER/WORKER/RECOVERY/INTERNAL events must never silently masquerade as HUMAN `CREATE_WORK`;
- N=1 causal correctness outranks more parallelism, feature expansion and UX polish.

Reuse the current canonical Kanban TaskRun/CAS support, TaskCompiler canary and dispatch checkpoints, BrowserTask, ExecutionJournal, EvidenceState, RecoveryPlane, WorkerRegistry, Hybrid delegation and ArtifactStore/reference plane. Do not solve reliability by introducing a parallel orchestration framework or state owner.

## Progressive-compilation correction (2026-09-18)

- Safety/authority/evidence gates may block work; **lack of an already-compiled
  procedure alone may not**.
- A repeatability hint is an optimization signal, not session-wide mutation
  authority. Compilation requirements are scoped to an operation fingerprint /
  target family.
- Novel/drifted state may use bounded adaptive execution. Existing approvals,
  TaskRun fencing, BrowserTask leases, no-progress limits and uncertain-effect
  rules remain authoritative.
- True homogeneous mutable fan-out still requires TaskCompiler/canary admission.
- Structural shape alone does **not** prove homogeneity. Mandatory compilation
  requires positive semantic family evidence: stable operation + canonical
  route/provider + owner-declared or safely derived target-family/contract
  identity. Shape-only repetition may suggest learning/compilation but may not
  independently block bounded adaptive work.
- Native browser mutations without such family evidence must not become
  `REQUIRE_COMPILE` merely because the same tool name/schema appears three
  times against different semantic UI targets.
- Deterministic work must be able to return a compact NEEDS_REASONING handoff and
  resume from confirmed checkpoints rather than looping on refusal.
- Native-browser tool constraints normalize to native_browser; do not compare
  route names and tool names as if they were one namespace.
- tools.effects is the single effect taxonomy. Do not maintain conflicting
  browser idempotent/read/write lists in guardrail code.
- Arbitrary browser_console remains potentially mutating. Use structured browser
  reads for observation.
- Reuse RecipeStore, ProceduralMemory, RoutinePromotionService, ArtifactStore and
  ExecutionJournal for learning; do not add a parallel procedure/task/evidence
  store.
- The target lifecycle is adaptive -> compiled segment -> compiled work ->
  promoted routine -> drift -> adaptive exception.

See [Adaptive Execution & Progressive Compilation](ADAPTIVE_EXECUTION_COMPILATION.md).

## Reasoning vs deterministic execution boundary

- The LLM owns semantic ambiguity, planning, interpretation, exception handling and replanning.
- Deterministic runtime code owns IDs, leases, deadlines, retries, checkpoints, progress accounting, artifact lookup, dependency resolution, idempotency, reconciliation and other objective bookkeeping.
- Quantified/repetitive mutable work should use the existing TaskCompiler/DurableBatchRunner path rather than placing the LLM between every equivalent item.
- If the same objective state question must repeatedly be rediscovered from transcript prose, prefer a structured runtime resolver/reference rather than adding more prompt instructions.

## Browser profile and secrets

- Never reuse a user's personal Chrome/Edge profile for Workstation automation.
- Persistent Chromium profile data stays outside Git/source control and is managed by the dedicated Electron session/partition.
- BrowserSessionState may persist safe structural metadata only. Do not persist passwords, typed secrets, page-extracted access tokens, sensitive form values, or screenshots in that state file.
- Local controller secrets/tokens must not be logged or committed.

## Network and control

- Local Workstation controllers bind to loopback by default. Expanding exposure beyond localhost requires an explicit security design and is out of scope for the browser-foundation phase.
- A BrowserTask already bound to the Workstation controller is fail-closed if that controller disappears. Do not silently fall back to another browser/runtime.
- `routing.enabled: false` remains internal-only behavior; it must not unexpectedly route to external automation.

## Agent/tool contract

- GUI/browser surface availability is resolved from the Hermes session/platform, not `HERMES_DESKTOP` or another process environment proxy.
- Process-wide `check_fn` caching must not encode per-session surface identity.
- Keep prompt/tool schemas stable during a conversation in accordance with Hermes prompt-caching rules.
- `web_search` remains distinct from visible/authenticated browser interaction; do not eliminate it to force browser use.
- Existing semantic browser-readiness primitives must be reused and hardened. Navigation/reload/SPA hydration invalidates stale element/ref assumptions; mutation resumes only after semantic readiness/reacquisition, and CAPTCHA/auth walls become explicit blocker/handoff states rather than blind retries.

## Human control and approvals

- Human `Take Control` and agent control operate on the same BrowserTask/page.
- Existing Hermes approval/security gates remain authoritative for sensitive actions. Workstation must not bypass approvals merely because the page is locally visible.
- Pause/resume/stop/focus/control ownership semantics must remain recoverable and explicit.
- Human takeover suspends/revokes the active agent mutation fence for that resource until authority is deliberately returned.

## V3 operational hardening

- `RUNNING` requires live, verifiable operational evidence; stale evidence must
  degrade to `STALLED` rather than leave a zombie operation.
- Event subscribers are bounded and isolated: a slow client must not block
  unrelated task/worker events, and structural failures must be observable.
- Runtime calls, worker waits, MCP pagination/execution and persistence use
  explicit deadlines/cancellation or bounded failure behavior.
- Persistent worker messages/results retain sender, parent-task, session,
  sequence and semantic status; worker stop/failure is never normal completion.
- Session ownership is cross-process exclusive; migration is backup → validate →
  promote with rollback, and hot/warm/cold plus compaction markers remain
  reconstructable from durable state.
- Recovery and release gates are evidence-driven. Unit tests, typechecks and
  native browser smoke are separate claims and must not be conflated.

## Upstream maintenance

- Prefer extending existing code over adding parallel managers or frameworks.
- Every edit to an upstream-owned integration point must have a concrete consumer, a behavior test, and an entry in `workstation/UPSTREAM_DELTA.md` when it changes the maintained downstream delta.
- Normal install/CI validates committed source; it must not auto-heal missing downstream edits before tests run.
- Keep dependencies pinned and license policy intact.

## Active milestone scope boundary

V1 #1 BrowserSessionState and the pre-1.5 Mainline Consolidation Gate remain promoted history. The **Canonical Execution Reliability Gate** is now the active handoff and supersedes the historical V1 #1.5/later feature sequence until its executable exit criteria pass.

During this gate:

- feature-expansion work in V1 #1.5, V1.1, V2, V2.1, V2.5, V3, V3.1–V3.5 and V4 is preserved but deferred unless a narrowly scoped change is required to satisfy a reliability invariant;
- historical “Completed”/“Implemented contract layer” labels remain implementation history, not proof that Task → Run → operation → evidence → canonical commit is causally trustworthy end to end;
- existing mechanisms are hardened and connected rather than rebuilt;
- no additional parallelism is promoted solely for throughput before single-run execution is trustworthy;
- the Task Cockpit/Control Center may be improved only as a projection of canonical state; UX must not become a competing lifecycle authority.

After the reliability gate closes, the deferred roadmap resumes from the sequencing explicitly recorded in `ROADMAP.md`.

## Mainline handoff

- New milestone work branches from current `main`, never from a retained
  diagnostic/validation ref.
- Do not merge a formatter/bot branch wholesale; reproduce a required scoped
  formatting change on the active branch.
- Close or formally classify superseded PRs before the next milestone.
- Run the Mainline Consolidation Review after every major milestone and keep the
  full pre-1.5 ledger in `context/MAINLINE_CONSOLIDATION.md`.

## Work Loop safety boundaries

Do not relax a task acceptance policy at completion. Uncertain dispatched effects
require reconciliation, never blind retry. Stored running without a current live
handle is recovery-required. Journal corruption fails explicitly before accepted
completion; legacy unverified evidence must remain distinguishable. Workstation
tests isolate homes before collection and for each test; production AVCR excludes
test/benchmark/e2e/replay and unknown provenance.
