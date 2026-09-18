# Architectural Decisions

These are settled decisions for the Hermes Workstation downstream architecture. They narrow implementation choices; they are not a substitute for the detailed design in [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## D-001 — Workstation is first-class in this downstream fork

Hermes Workstation is part of this fork's product architecture, not a temporary ZIP overlay. The committed `main` tree is the canonical integrated source for this downstream. Upstream synchronization remains deliberate and tracked.

## D-002 — Preserve Hermes upstream and extend existing ownership

Use Hermes Sessions, Gateway, tool registry/toolsets, approvals, memory, Kanban, profile handling, and browser routing instead of introducing parallel stores or control planes. Downstream edits to upstream-owned files must stay small, explicit, tested, and recorded in `UPSTREAM_DELTA.md`.

## D-003 — The internal browser is Electron Chromium

The primary Workstation Browser uses Electron's Chromium runtime through `WebContentsView` and a dedicated persistent Electron session/partition. Do not reuse the user's personal Chrome/Edge profile.

## D-004 — Browser profile state and BrowserSessionState are different

The Chromium profile owns browser-managed state such as cookies, localStorage, IndexedDB, cache, and compatible authentication state. Workstation BrowserSessionState owns only safe structural metadata such as logical tabs, active tab, BrowserTask linkage, URL/title/order/status, and related identifiers. Never conflate the two stores.

## D-005 — One BrowserTask owns one live page

A BrowserTask represents the durable semantic ownership of an automated web task. It may be hidden, parked, focused, or moved between hosts without navigating again. The same task must not be represented by two independently navigated live pages.

`taskTabs`/`ownerTaskId` remain the authoritative live-page binding inside the Electron process. BrowserTask lifecycle metadata wraps those existing primitives; it does not introduce a second map that owns `WebContentsView` instances.

`hide` and `park` never mean destroy. While the Electron process remains alive, a later `show` must re-expose the same live page and preserve its current URL/page state. `destroy` is an explicit lifecycle operation.

A `WebContentsView` object is process-local and is not serializable across a Desktop restart. Restart recovery therefore preserves the BrowserTask identity and safe metadata, normalizes the task to `parked`, and lazily recreates/reconnects one page under the same `taskId` when the task is used again. Do not describe that as preserving JavaScript heap or `WebContents` object identity across process restart.

## D-006 — Chat Browser View and Browser Hub are views of the same runtime

The contextual Chat Browser View and the global Browser Hub expose the same BrowserRuntime/BrowserTask state. A live `WebContentsView` has one active host at a time; the other surface represents the task with state/card/thumbnail rather than duplicating the page.

## D-007 — Do not create second SessionDB, Kanban, or Memory systems

BrowserTask may reference Hermes session/run/agent identifiers, but it does not own a replacement session database. Workstation planning/execution features must reuse the existing Kanban and Memory abstractions when those features are later introduced.

## D-008 — Bound Browser tasks fail closed

Once a task is bound to the Workstation Browser/controller, loss of that controller must not silently move the task to a different browser/runtime with different authentication or page state. Recovery is explicit. Unbound requests may use configured fallback according to routing policy.

## D-009 — Surface capability is session-scoped

Whether a Desktop/GUI session should know about a GUI/browser surface is a property of the session/platform contract, not of process environment variables or a process-wide cached reachability probe. Reachability may gate execution/recovery, but must not silently erase a valid session surface from the model schema.

## D-010 — BrowserRuntime remains an abstraction boundary

Do not make Workstation business logic depend irreversibly on one concrete browser implementation. Electron Chromium is the current primary runtime, while the BrowserRuntime/controller boundary should remain explicit enough for future specialist runtimes without duplicating state or changing BrowserTask semantics.

## D-011 — Main is tested as committed

Installation and CI must validate the source committed in this downstream `main`. Migration/rebase helpers may exist, but normal install/test paths must not silently rewrite tracked source before validation; otherwise a missing integration can be hidden by the test harness itself.

## D-012 — Main is the only milestone handoff line

Accepted product code, required tests/probes/workflows, current state, and the
decision record must be reachable from `main` before the next milestone starts.
Temporary validation branches and diagnostic PRs may preserve evidence, but they
are never implicit dependencies or alternate product lines.

The extraordinary pre-V1 #1.5 Mainline Consolidation Gate inventories and
classifies the repository's accumulated history. After each later major
milestone, a smaller Mainline Consolidation Review verifies promotion,
classifies new lateral work, reconciles canonical documents and confirms that
the next milestone can branch exclusively from `main`.

Known, causally classified debt may remain open when its scope and evidence are
explicit. An unclassified material delta, stale active predecessor, or required
artifact available only on another branch blocks the handoff.

## D-013 — V3 operational state is a projection, not a new canonical owner

EvidenceState, typed resources, event delivery, session lifecycle metadata,
worker persistence, Recovery Plane records and evaluation traces are
operational projections over Hermes' canonical SessionDB, Kanban, Memory,
BrowserTask and ExecutionJournal owners. They may persist identities, leases,
evidence and recovery copies, but they must not become a second task/session/
memory/browser database or agent core.

The independent supervisor is allowed to own runtime process liveness and
restart/rollback metadata. The Recovery Plane is allowed to quarantine optional
components. Neither is allowed to perform ordinary task work or weaken Policy
Engine/approval boundaries. External A2A/ACP/UHP protocols are adapters into
the canonical event/resource contracts, never alternative state owners.

## D-014 — Canonical TaskRun is the execution-attempt authority

A canonical Agent Task, its execution attempt, a compiled WorkPlan and a
BrowserTask are related identities with different lifecycles. They must not be
collapsed into a mega-entity and must not substitute for one another.

- `tasks.id` remains the canonical Agent Task responsibility identity.
- Existing Kanban `task_runs.id` / `tasks.current_run_id` remains the canonical
  execution-attempt authority. Do not add another Run store.
- WorkPlan/WorkItem own deterministic compiled-plan progress and must persist
  their canonical TaskRun lineage.
- A deterministic `work_<hash>` or equivalent durable-plan identity is an
  `execution_key`/plan identity, never a replacement canonical Task id.
- BrowserTask remains the durable semantic identity for browser work; its live
  page is a process-local lease. BrowserTask persistence may remain Task-scoped,
  while mutations are fenced by TaskRun/operation identity.
- Worker/process/browser handles are live operational evidence, not durable
  proof of successful outcome.

Every mutating Workstation effect must be attributable to one canonical TaskRun
or explicitly classified as a human/system action outside agent-run authority.
Legacy records that cannot be bound safely must be classified for reconciliation;
the runtime must not invent lineage from convenience or transcript prose.

## D-015 — Canonical commit precedes terminal journal and projections

Execution, acceptance, verification, canonical lifecycle commit and UI/journal
projection are separate stages.

For agent-owned completion:

1. the active TaskRun must still own the Task;
2. required acceptance/evidence must be satisfied;
3. the canonical Kanban transition must commit using the existing run-fencing
   mechanism (`expected_run_id` or its canonical successor);
4. only after that commit may `TASK_COMPLETED`, Human Card writeback or terminal
   UI/resource projections be emitted.

A failed/stale CAS is a superseded/rejected completion attempt, not a completed
Task. Human/manual override remains possible only as a separate explicit and
audited authority path.

For external mutable effects, timeout or loss of acknowledgement after dispatch
must enter an explicit uncertainty/reconciliation state. It must never be treated
as proof that the effect did not happen and must never authorize a blind retry.

These rules are the execution-level meaning of `One Hermes State`: one authority
per domain, shared causal lineage and reconcilable projections — **not** one
physical database.

## D-016 — Reliability gate precedes further feature expansion

The 2026-09-17 `CANONICAL_EXECUTION_RELIABILITY_GATE.md` is the immediate roadmap
handoff. Historical V1 #1.5 and all later feature/polish work remain preserved but
do not outrank the gate.

The gate is closed only by invariant/regression evidence, including real
restart/Windows/Electron paths where relevant. A green unit suite or an
"Implemented contract" label is not sufficient to promote product-level causal
reliability.

## D-017 — Adaptive execution precedes progressive compilation

Durable compilation is an optimization and reliability mechanism for understood
repetitive work; it is not a universal permission gate for every mutation inside
a request that happens to look repetitive.

Novel or drifted work may execute through a bounded ADAPTIVE mode under the
existing TaskRun, BrowserTask/worker/host lease, approval, policy and uncertainty
contracts. As stable operation segments are observed, the runtime progressively
moves them through compiled execution and the existing
Experience -> Candidate -> Validate -> Promote -> Routine lifecycle.

A compilation requirement is scoped to a concrete operation fingerprint/target
family, not to the whole session or turn. The current _work_batch_candidate-style
global latch is therefore not an architectural invariant and must be replaced by
operation-scoped policy.

AEPC-E002 clarifies what “operation-scoped” means: **structural call shape alone
does not establish one repeatable operation family**. `REQUIRE_COMPILE` needs
positive semantic homogeneity evidence across the operation and canonical
route/provider plus an owner-declared or safely derived target-family/contract
identity. When that evidence is absent, repeated shape may suggest compilation
or learning but may not, by itself, block bounded adaptive execution. This is
especially important for native `browser_type`, `browser_click` and
`browser_press` sequences whose semantic targets can change as page state
changes. Real same-family fan-out remains compiler/canary gated.

The deterministic runner must return a compact NEEDS_REASONING/drift handoff when
reality no longer matches its assumptions. It must not trap the agent in a
durable_compile_required <-> PREFLIGHT_REQUIRED refusal loop.

For stateful browser work, transient UI interaction and external commit are
distinct. Evidence strength is chosen at the effect boundary; independent
persisted readback is required where the risk/effect contract demands it, not for
every focus/type/open intermediate UI step.

tools.effects is the canonical effect taxonomy. Route policy compares canonical
route identities (native_browser for the internal Workstation Browser) after
tool-to-route normalization. Arbitrary browser_console remains potentially
mutating.

Detailed contract:
[ADAPTIVE_EXECUTION_COMPILATION.md](ADAPTIVE_EXECUTION_COMPILATION.md).

## D-018 — Capability is the reusable deterministic unit; work_execute is runtime infrastructure

D-017 remains valid: adaptive execution precedes progressive compilation. D-018
defines the reusable unit and execution layering that follows from that rule.

A **Capability** is the canonical abstraction between primitive Tool execution and
larger Recipe/Routine workflows. It represents versioned deterministic
operational knowledge with typed inputs, effects, scope, preconditions,
postconditions/verifiers, dependencies, provenance and validation lifecycle.

A Capability is not a Skill. Skills are LLM-facing knowledge/strategy and may
call capabilities. Capabilities may call other capabilities and trusted runtime
primitives. A promoted deterministic capability must not silently call a Skill or
LLM on its success path; drift returns compact NEEDS_REASONING to the adaptive
layer.

`work_execute` is the deterministic execution entry point/infrastructure, not a
global permission gate and not primarily an instruction forcing the model to
manually rewrite repeated work. Exact compatible promoted Capability/Recipe/
Routine reuse should be selected by the harness whenever possible.

The low-level execution substrate is a shared **Operational Kernel**. Browser,
filesystem, process/shell, HTTP/API and future desktop control are backends under
the same effect/evidence/policy/capability contracts. This architecture is not
browser-only.

Native Browser semantic identity must be recoverable and stable enough for
compilation. Transient element refs, tab ids, WebContents ids and arbitrary page
prose are not durable operation identity. Structural call similarity can support
discovery but cannot independently establish semantic homogeneity or mandatory
compilation.

Capability persistence/learning must extend existing RecipeStore,
ProceduralMemory, ExecutionJournal and ArtifactStore ownership where practical.
Do not introduce another canonical SessionDB, Kanban, TaskRun, BrowserTask or
Memory system merely to host capabilities.

Detailed target and acceptance criteria:
[PROGRESSIVE_OPERATIONAL_COMPILATION.md](PROGRESSIVE_OPERATIONAL_COMPILATION.md).

## Changing a decision

A replacement decision must state which decision it supersedes, why the old invariant no longer holds, how migration/backward compatibility is handled, and which tests prove the new contract. Do not silently drift architecture through implementation-only changes.

## Canonical Work Loop contracts (2026-09-17)

Intention authority comes from a trusted MessageEnvelope, never request prose.
Workstation/Hybrid done requires a verified outcome accepted against the policy
captured at creation. Live handles require bounded liveness; durable proof cannot
prove running. Canonical execution lineage and Task Cockpit are projections over
Kanban, WorkPlans, BrowserTask refs and journal, with no new task store. Routine
execution stop remains distinct from accepted task completion. Implementation
boundaries are documented in [Canonical Work Loop](CANONICAL_WORK_LOOP.md).
