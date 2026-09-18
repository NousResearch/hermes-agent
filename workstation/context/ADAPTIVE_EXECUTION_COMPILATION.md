# Adaptive Execution & Progressive Compilation

Date established: 2026-09-18

Status: **IMPLEMENTED BASELINE / SEMANTIC-HOMOGENEITY HARDENING OPEN — native product validation pending**

This document refines the durable-execution boundary after real native-browser
dogfood showed that the baseline compiler guard could prevent legitimate work.
It does **not** revoke the Canonical Execution Reliability Gate, canary admission,
uncertain-mutation handling, TaskRun fencing, acceptance, circuit breaking or
canonical evidence requirements.

## Architectural extension — Capability runtime

AEPC defines **when adaptive execution may continue and when compilation is
admissible/required**. The follow-up
[PROGRESSIVE_OPERATIONAL_COMPILATION.md](PROGRESSIVE_OPERATIONAL_COMPILATION.md)
defines **what should be compiled and reused**.

The new target introduces Capability as a reusable deterministic unit between
primitive Tool calls and larger Recipe/Routine workflows. It also repositions
`work_execute` as deterministic runtime infrastructure selected by the harness,
rather than primarily as a refusal/replan instruction.

This extension does not weaken AEPC-E002. Semantic homogeneity remains required
before mandatory compilation. Structural signatures remain discovery hints. The
Capability layer must reuse the existing evidence, recipe, procedural-memory,
WorkPlan/WorkItem, uncertainty and promotion owners rather than creating parallel
authority.



## Implemented boundary — 2026-09-18

Baseline/main audited: `c04906aacee568bb6480287717c76afd230cf4a7`.
Remote implementation: `9e7292ab7825e5ce1ea294490eec57ba1f286069`.
Remote documentation/evidence: `5e1b22527fd40d732ee4fa7a1035e6366953f6b7`.
Local pre-publish implementation SHA: `365794e29d66cd63a6134c5c67ecc1ef603d70a6`.
Follow-up audit head: `c4234200145162eefb60f6070c9f170b4bf79321`.
The original reproduction below remains historical evidence.

`execution_policy.py` projects CompilationCandidates from the existing mutation
ledger. ExecutionMode names ADAPTIVE, COMPILED, ROUTINE and HUMAN;
CompilationDecision distinguishes ALLOW_ADAPTIVE, SUGGEST_COMPILE,
REQUIRE_COMPILE and REQUIRE_HUMAN. Repeatability prose is only
`_work_repeatability_hint`. The first implementation permits bounded adaptive execution for the first two
structurally equivalent mutations, suggests compilation on the second and can
require compilation on the third. Signatures incorporate owner metadata when it
exists. Follow-up audit AEPC-E002 found that **structural equivalence is not
sufficient semantic proof of one repeatable operation family**: built-in native
browser mutations currently lack a concrete target-family contract, so distinct
stateful UI actions may still collide at the shape level. Mandatory compilation
must therefore be conditioned on proven semantic homogeneity, not merely a
threshold over shape-equivalent calls. Exact duplicates remain under ordinary
guardrails; uncertainty overrides adaptive permission.

The central dispatcher rechecks final middleware arguments and records mutable
dispatch in ArtifactStore before I/O, inside the existing ordered start section.
Restart can recover uncertainty without a new database. Canonical run checks,
BrowserTask lease generations, approvals and session tool scope remain authority.
Guardrail effects come from tools.effects, including registry declarations;
unknown effects and browser_console remain MUTATION. Legacy guardrail effect-list
configuration slots are retained but no longer override the canonical taxonomy.

Known native tool constraints normalize to native_browser. This is an explicit
tool/runtime mapping, not a browser-prefix admission exemption. Recognized
browser_transaction phases require tool-owner registry metadata for an exact
operation, semantic anchors, bounded read-only preflight and effect-boundary
verifiers. A planner cannot grant itself INTERACT/COMMIT authority. No blanket
transaction contract was granted to native actions or arbitrary JavaScript.

EvidenceStrength distinguishes E0 ACK, E1 same-session semantic observation,
E2 persisted semantic readback and E3 independent persisted readback. Native
snapshot is E1 and is never recorded as persisted proof. Recognized external
COMMIT requires E2 or stronger; explicit external work cannot substitute a native
snapshot for persisted readback. HTTP status or action ACK alone is insufficient.

Adaptive dispatch captures bounded sanitized traces and raw-result references
through ArtifactStore/ExecutionJournal during execution. Only semantic,
reacquirable actions enter a candidate after canonical acceptance. Transient refs,
tab/WebContents IDs and credentials do not enter durable procedure arguments;
typed text is parameterized. Trace truncation prevents learning an incomplete
procedure. ProceduralMemory owns versioned candidates; existing promotion still
requires validation. Changed or already-promoted procedures create revisions.

TaskCompiler captures verified graphs in RecipeStore without requiring recipe_key
and selects exact compatible fingerprint/scope/target/preflight matches.
Promoted native routines with structured semantic pre/postconditions can be
selected by operation_fingerprint + recipe_scope, then lowered into existing
WorkPlan/WorkItem checkpoints. Click/type anchors are reacquired from the live
native snapshot inventory; untrusted page text is excluded. Unsupported routine
formats need adaptive reasoning; similarity never grants replay authority.

Compiler drift returns NEEDS_REASONING with completed_until, expected, observed,
state_ref and safe_to_resume. Expected/observed details stay in artifacts. Resume
checks the owning session, retains confirmed steps and resets only a diagnosed
unexpected-state item. Outstanding mutable dispatch, stale authority or systemic
failure is not safe to resume. Routine drift exposes the same compact handoff;
its procedure/version/content fingerprint must remain compatible after diagnosis.

Validation on this implementation: final Workstation + adjacent core gate
**570 passed, 2 skipped in 346.25s**; Work100 **30 PASS / 0 FAIL / 0 gaps**;
Desktop owner contracts **36 passed**. Provider-free learning demonstrates
simulated calls **3 -> 0** on promoted replay, and only the unresolved segment
after drift. Efficiency calculations retain null for unknown observations and
denominators, including Guardrail Obstruction Rate. These are contract/simulated
results, not paid-provider savings or native Electron product proof. Electron
executable and built main bundle were absent; packaged/native authenticated smoke
remains open. Full commands, intermediate failures and limits are in
[the journal](engineering-journal/CURRENT.md) and [TESTING.md](TESTING.md).

## AEPC-E002 — semantic homogeneity hardening

Follow-up audit of remote `main@c4234200145162eefb60f6070c9f170b4bf79321`
confirmed that the global latch is gone and the major reliability contracts are
sound, but exposed one residual false-positive class.

Current `structural_signature()` intentionally abstracts scalar values. That is
useful for batch discovery, but a shape such as
`browser_type(ref=str, text=str)` does not prove that two calls target the same
semantic operation. The built-in `browser_type`, `browser_click` and
`browser_press` registrations do not currently provide a concrete
`mutation_target` / `target_family_fields` contract. Therefore a long adaptive
workflow can perform three semantically different UI actions with the same tool
shape and accidentally satisfy the current threshold for `REQUIRE_COMPILE`.

New invariant:

> **same call shape != same repeatable operation**

`REQUIRE_COMPILE` is permitted only when the runtime has positive semantic
evidence that the calls belong to one compilable family. At minimum, the family
must be stable across the canonical operation, route/provider and an
owner-declared or safely derived target-family/contract identity. If that evidence
is absent, structural repetition may support `SUGGEST_COMPILE`/learning, but it
must not by itself block bounded adaptive work.

Native browser behavior is especially important:

- three different textbox edits are not homogeneous merely because all use
  `browser_type`;
- three different semantic clicks are not homogeneous merely because all use
  `browser_click`;
- state changes and newly reacquired anchors may represent different operations
  even when argument schemas match;
- real repeated fan-out with the same semantic family must still transition to
  TaskCompiler/canary and may not be exempted by tool name.

Required paired regression:

~~~text
long stateful browser:
navigate -> snapshot
type target A -> snapshot
click target B -> snapshot
type target C -> snapshot
click target D -> snapshot
type target E
=> stays bounded ADAPTIVE when semantic families differ

true homogeneous fan-out:
update same operation/family target 1
update same operation/family target 2
update same operation/family target 3
=> third distinct mutation REQUIRE_COMPILE
~~~

Do not fix this by raising the threshold, resetting counters around navigation,
hard-coding `if browser: allow`, or disabling compiler/canary. Fix the meaning
of the candidate itself.

`CompilationCandidate.successful_occurrences` must also stop counting
`executed_unverified` as semantic success. Rename that metric to
`executed_occurrences`, or increment `successful_occurrences` only after the
existing verification/acceptance path proves success. The metric may never grant
mutation authority by itself.

Closure requires the paired regressions above, the existing uncertainty/fencing/
route/evidence/replay suites, Work100, and the separate packaged/native browser
gate when the environment supports it.

## Executive principle

The optimization goal is:

> use the least reasoning necessary to complete work correctly, but never make
> the absence of an already-deterministic path, by itself, a reason that safe,
> authorized work cannot proceed.

Correctness and completion come first, constrained by safety, authority and
evidence. Within those constraints, minimize LLM calls, tokens, redundant tool
calls, latency and context reconstruction.

The invariant is:

> **never pay twice for the same operational reasoning.**

A novel task may be expensive once. Repeated compatible tasks should become
progressively cheaper and more deterministic.

## Why this correction exists

A 2026-09-18 native-browser reproduction established a clean failure mode:

- the internal Electron Chromium route opened an authenticated page successfully;
- browser_snapshot and browser_extract_items continued to read the same page;
- mutating interactions such as browser_click / arbitrary console execution were
  intercepted by durable_compile_required;
- the compiler then required mutation authority plus durable readback/verifier
  semantics better suited to homogeneous external fan-out than to one stateful UI
  interaction;
- one attempt to express the restriction through tool names conflicted with the
  compiler route name native_browser;
- the session-level _work_batch_candidate boolean allowed a repeatability signal
  for the larger request to contaminate unrelated later mutations;
- an attempted authority-recording path could itself be intercepted, demonstrating
  a logical bootstrap/deadlock risk.

The browser controller, authenticated profile and read path were healthy. The
failure was the execution-policy boundary.

## Architectural correction: execution is a ladder, not a binary switch

The runtime must distinguish four progressively cheaper execution levels:

~~~text
L0  ADAPTIVE / DISCOVERY
    Hermes reasons and uses tools to understand a novel or drifted situation.

        ↓ stable segment discovered

L1  COMPILED SEGMENT
    A bounded known fragment executes without another LLM call.

        ↓ repeated verified success

L2  COMPILED WORK
    work_execute / DurableBatchRunner executes the known workflow with
    checkpoints, canary/fan-out rules and compact exception returns.

        ↓ validated reuse + measured benefit

L3  PROMOTED ROUTINE
    DeterministicRoutineRunner replays a versioned promoted procedure with
    zero repeated planning.

        ↓ drift / unexpected state

ESCALATION
    Return only the smallest unresolved state to Hermes, adapt, then resume.
~~~

The first execution may use more LLM reasoning. The second should normally use
less. Stable repeated work should approach zero planning calls.

## Compilation policy

Replace the effective binary policy ("batch candidate => any later mutation must
compile") with an explicit decision:

~~~text
ALLOW_ADAPTIVE
SUGGEST_COMPILE
REQUIRE_COMPILE
REQUIRE_HUMAN
~~~

### ALLOW_ADAPTIVE

Use when work is novel, drifted, stateful, exploratory or not yet representable by
a validated deterministic contract. Examples include navigating a new authenticated
UI, locating the correct control, testing a bounded interaction and observing the
semantic result.

Adaptive does **not** mean unbounded. Existing budgets, approvals, route policy,
leases, effect uncertainty, no-progress guards and security policy still apply.

### SUGGEST_COMPILE

Use when a stable operation/segment has repeated successfully and the runtime can
capture a deterministic candidate. This should preferably be handled by the
harness rather than by asking the model to remember to optimize itself.

### REQUIRE_COMPILE

Reserve this for true homogeneous fan-out / repetitive mutations where continuing
with one LLM-mediated mutation per equivalent item would amplify cost or failure.
A repeatability hint alone is insufficient. The requirement should be scoped to a
specific operation fingerprint / target family and should normally have:

- multiple distinct equivalent items;
- a stable structural operation signature;
- a known target family / route;
- known or discoverable verifier semantics;
- non-trivial fan-out/blast radius;
- no unresolved semantic ambiguity that requires agent reasoning.

### REQUIRE_HUMAN

Use existing approval / handoff policy for sensitive, ambiguous or policy-gated
actions. This is orthogonal to compilation.

## Scope compilation to an operation, never to the whole session

_work_batch_candidate must not behave as a session/turn-wide mutation latch.

Introduce an operation-scoped candidate concept, reusing existing structural
signatures and canonical stores rather than adding a new task database:

~~~text
CompilationCandidate
  pattern_id
  operation_fingerprint
  route
  target_family
  occurrences
  successful_occurrences
  verifier_capability
  confidence
  blast_radius
  compile_status
  metrics
~~~

A candidate belongs to the repeated operation/segment. A different browser action,
a human handoff, a novel exception or unrelated mutation must not inherit the
candidate merely because the parent request looked repetitive.

## Stateful native-browser work

A browser workflow over one BrowserTask is not automatically a durable batch.

The adaptive path must support the canonical stateful loop:

~~~text
observe -> reason -> act -> observe -> adapt -> continue
~~~

while retaining the same BrowserTask/authenticated profile, TaskRun/operation
lineage, one mutating lease owner, approvals/policy and fail-closed behavior if a
bound Workstation Browser disappears.

### Browser transaction semantics

Known/replayable UI workflows should be modeled as a transaction with semantic
boundaries:

~~~text
PREPARE
  reacquire page, readiness and semantic anchors

INTERACT
  focus, type, select, scroll, open intermediate UI

COMMIT
  perform the action expected to create the user-visible/external effect

VERIFY
  observe the semantic postcondition at the evidence strength required by policy
~~~

Do not require independent persistent readback for every transient UI interaction.
Require the appropriate evidence at the effect/commit boundary.

This does not make arbitrary browser_console read-only. Arbitrary JS remains
potentially mutating. Prefer browser_snapshot, browser_extract_items and other
structured reads for observation.

## Evidence strength

Verification policy should distinguish evidence strength:

~~~text
E3  INDEPENDENT_PERSISTED_READBACK
    API/DB/provider-independent read confirms persisted state.

E2  SEMANTIC_PERSISTED_READBACK
    reload/navigation/reacquisition confirms semantic state.

E1  SAME_SESSION_SEMANTIC_OBSERVATION
    current DOM/application state demonstrates expected postcondition.

E0  TOOL_ACK_ONLY
    transport/tool reports success; insufficient for durable external acceptance.
~~~

Policy chooses the minimum admissible level by effect/risk. Tool acknowledgement
alone must never be promoted to durable external proof.

## Deterministic runner must have an official escape

work_execute and promoted routines must be allowed to stop with a compact reasoning
handoff instead of entering a refusal loop.

Canonical result shape:

~~~json
{
  "status": "NEEDS_REASONING",
  "completed_until": "step_7",
  "expected": "...",
  "observed": "...",
  "state_ref": "artifact://...",
  "safe_to_resume": true
}
~~~

The LLM receives only unresolved state/evidence needed for diagnosis. After
adaptation, execution may resume from the checkpoint. Uncertain external mutations
remain subject to reconciliation and may set safe_to_resume=false.

## Route and effect authority

There must be one canonical effect taxonomy: tools.effects.

- remove/avoid parallel lists that disagree with tool_effect(...);
- unknown actions remain conservatively mutating;
- browser_console remains potentially mutating;
- browser tool names normalize to the canonical route native_browser before route
  constraints are compared;
- tool-name restrictions and route restrictions remain distinct concepts.

Reuse the existing BrowserControlLease/TaskRun fencing and policy plane. Do not
create a second browser-authority store.

## Learning while executing

Learning must start during successful adaptive execution, not only after a final
chat response.

Reuse ExecutionJournal, ArtifactStore, RecipeStore, ProceduralMemory and
RoutinePromotionService. Add a bounded trace/candidate layer that can capture:

- action/tool and canonical route;
- semantic target/anchor when available;
- parametrizable arguments;
- before/after state references when available;
- evidence/result references;
- duration and outcome;
- LLM calls/tokens used to reach the decision.

Do not promote ephemeral DOM refs, tab IDs or renderer node IDs as durable
knowledge. Learned browser procedures reacquire semantic anchors after
navigation/restart/drift.

Lifecycle:

~~~text
Experience
  -> Candidate Procedure
  -> compatible replay / validation
  -> Promote
  -> versioned deterministic Routine
  -> Drift
  -> Hermes re-explores only the failed segment
  -> new validated version
~~~

## Automatic reuse

A verified recipe/routine should be selected by the runtime when scope,
preconditions, fingerprint and policy match. The model should not need to remember
a recipe key simply to obtain savings already proven by the system.

Fail closed on scope mismatch or drift, then escalate to adaptive reasoning.
Do not silently mutate a historical routine in place.

## Guardrails that remain mandatory

This correction must preserve:

- canary-before-fan-out;
- no blind retry after uncertain mutable dispatch;
- durable operation identity and TaskRun lineage;
- systemic-failure circuit breaking;
- canonical completion/acceptance ordering;
- terminal-tree reconciliation;
- one mutating browser lease owner and human takeover fencing;
- route/security/approval policy;
- evidence-backed completion;
- reference-first large outputs and bounded context.

The change is **where the compiler gate applies**, not removal of protections.

## Efficiency metrics

Track efficiency per verified outcome:

- LLM calls / verified outcome;
- tokens / verified outcome;
- tool calls / verified outcome;
- context reconstruction overhead;
- deterministic replay rate;
- routine reuse rate;
- drift rate;
- human rescue rate;
- discovery cost vs replay cost;
- estimated savings after promotion.

Add Guardrail Obstruction Rate:

~~~text
eligible tasks with a safe authorized path blocked by harness policy
/
eligible tasks
~~~

Target: approach zero without weakening safety invariants.

## Implementation map

Primary code owners expected to change:

- run_agent.py
- workstation/work_intent.py
- workstation/batch_detection.py
- workstation/task_compiler.py
- workstation/routing.py
- tools/effects.py
- agent/tool_guardrails.py
- tools/workstation_work.py
- tools/browser_workstation.py / browser registry metadata as required
- workstation/recipes.py
- workstation/memory.py
- workstation/routines.py
- ExecutionJournal / ArtifactStore integration points only as needed

Prefer a small workstation/execution_policy.py (or equivalently narrow owner) for
ExecutionMode, CompilationDecision and operation-scoped candidate policy rather
than scattering another set of ad-hoc booleans.

No second SessionDB, Kanban, task scheduler, BrowserTask store, memory authority,
recipe store, evidence DB or approval plane.

## Required regressions / acceptance

At minimum prove:

1. Native browser adaptive interaction: authenticated Workstation Browser can
   navigate -> snapshot -> type/click/press -> snapshot under one BrowserTask even
   when the larger request contains repeatability signals.
2. Homogeneous fan-out still compiles: three-or-more distinct equivalent mutations
   are stopped before uncontrolled fan-out and require durable compiler/canary.
3. Candidate scope isolation: one repeated operation does not force an unrelated
   later mutation into compilation.
4. Route normalization: allowed native-browser use cannot produce
   "Route forbidden by task constraints: native_browser" due to namespace mismatch.
5. Effect taxonomy consistency: guardrails and compiler agree; browser_console is
   not mislabeled read-only.
6. Reasoning escape: deterministic drift returns NEEDS_REASONING with compact state
   reference and resumes without replaying confirmed work.
7. Uncertain mutation safety: lost acknowledgement after commit still enters
   UNCERTAIN/RECONCILING and never blind-retries.
8. Learning curve: provider-free scenario demonstrates adaptive first execution,
   validated deterministic reuse with fewer simulated provider calls,
   promotion/replay, and drift returning only the failed segment to reasoning.
9. No authority-store duplication.
10. Existing TaskCompiler, read-only preflight, durable-hardening, routine,
    BrowserTask, canonical-work-loop and Work100 regressions remain green.

## Definition of Done

A safe novel workflow can always make bounded progress, equivalent repeated work
is progressively moved out of the LLM loop, known workflows replay
deterministically, drift returns only the unresolved segment to reasoning, and all
existing reliability/safety invariants remain demonstrably green.

Product rule:

> **explore safely when necessary; capture what was learned; compile what becomes
> stable; promote what proves reusable; return to reasoning only on novelty or
> drift.**
