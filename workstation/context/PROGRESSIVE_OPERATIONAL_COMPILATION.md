# Progressive Operational Compilation — Capability Runtime

Date established: 2026-09-18

Status: **TARGET ARCHITECTURE / IMPLEMENTATION REQUIRED**

This document extends
[ADAPTIVE_EXECUTION_COMPILATION.md](ADAPTIVE_EXECUTION_COMPILATION.md).
AEPC established that deterministic compilation must not block novel or stateful
adaptive work. This document defines the next abstraction boundary: **what Hermes
should learn, compile, persist, compose and replay so operational reasoning is paid
for once and reused across tasks, sessions and workflows.**

It does not replace the Canonical Execution Reliability Gate. TaskRun lineage,
effect authority, BrowserTask ownership, human takeover fencing, uncertain
mutation reconciliation, canary-before-fan-out, acceptance, canonical commit,
evidence strength and circuit breaking remain mandatory.

## Executive invariant

> **Do not force determinism before Hermes knows the procedure. Once a procedure
> is known and validated, do not pay an LLM to rediscover it.**

The optimization target is not merely fewer model calls inside one batch. It is
**amortized operational reasoning**:

~~~text
novel work
  -> reason / observe / act
  -> capture useful experience
  -> compile reusable capability
  -> validate
  -> promote
  -> compose
  -> replay deterministically
  -> invoke reasoning only on novelty, ambiguity or drift
~~~

The original force-work_execute patch optimized repetition inside a task. The
target architecture must also optimize repetition across tasks, sessions and
larger workflows.

## Why the current abstraction is insufficient

The current progressive-compilation baseline still projects candidates primarily
from tool-call structure. In current `execution_policy.py`, repeated writes are
grouped through a structural signature and the third distinct equivalent write
may reach `REQUIRE_COMPILE`. AEPC-E002 already establishes that:

> **same call shape != same repeatable operation**

The deeper issue is that **a batch is not the right reusable unit**.

A tool call such as:

~~~text
browser_type(ref=@e12, text=...)
~~~

is too low-level and too transient to be durable operational knowledge, while a
whole user request may be too large and heterogeneous to compile safely as one
unit.

Hermes needs a reusable layer between primitive tools and full routines.

## Canonical new abstraction: Capability

A **Capability** is a versioned deterministic operational unit that can execute
without LLM reasoning when its contract matches current state.

Examples:

~~~text
browser.find_semantic_target
browser.fill_field
browser.wait_for_semantic_state

github.open_pull_request
github.merge_pull_request_ui
github.verify_pull_request_merged

filesystem.find_latest_file
filesystem.rename_using_pattern
filesystem.move_to_project_folder

git.sync_local_main
git.verify_clean_state
~~~

A Capability is not merely a renamed Tool and is not a Skill.

### Capability contract

A capability must be able to declare, at minimum:

- stable `capability_id` and version;
- input schema and parameterization rules;
- effect taxonomy using the canonical `tools.effects` model;
- runtime route/backend;
- scope and compatibility constraints;
- preconditions;
- deterministic steps or implementation reference;
- postconditions;
- evidence/verifier contract;
- idempotency semantics when applicable;
- dependencies on primitives or other capabilities;
- policy/approval requirements;
- failure/drift classification;
- provenance to the experience/recipe/procedure that created it;
- validation evidence and lifecycle state;
- success/failure/reuse metrics;
- estimated reasoning/tool-call savings.

Transient browser refs, WebContents ids, tab ids, secrets and raw credentials are
not valid durable capability identity.

## Capability lifecycle

Reuse the existing canonical owners rather than create a second memory/task
system.

~~~text
Experience
  -> Capability Candidate
  -> compatible deterministic replay
  -> validation
  -> PROMOTED Capability
  -> composition into larger Capability / Routine
  -> drift
  -> NEEDS_REASONING for the smallest unresolved segment
  -> revised candidate/version
~~~

Existing `ExecutionJournal`, `ArtifactStore`, `RecipeStore`,
`ProceduralMemory`, `RoutinePromotionService` and WorkPlan/WorkItem machinery
must be extended/reused. Do not introduce a parallel SessionDB, task store,
BrowserTask store or memory database.

## Operational Kernel

Generated capabilities should normally be composed from a small trusted set of
deterministic primitives instead of arbitrary generated Python/JavaScript.

The **Operational Kernel** is the stable low-level execution substrate.

### Browser primitives

Target primitives should include equivalents of:

~~~text
navigate
snapshot
find_semantic
click_semantic
fill_semantic
press
scroll
wait_semantic
extract_structured
verify_semantic
~~~

The implementation may reuse existing `browser_*` tools. The requirement is a
stable machine contract, not a duplicate browser.

### Filesystem primitives

Target primitives should include:

~~~text
stat
list
find
read
write
copy
move
rename
mkdir
hash
compare
~~~

### Process primitives

Target primitives should include:

~~~text
spawn
wait
stdout
stderr
exit_code
terminate
~~~

### Additional backends

HTTP/API, downloads/clipboard and later native desktop/UI control may implement
the same capability contract. The architecture is therefore **not browser-only**.

A composite capability may cross backends, for example:

~~~text
filesystem.find
  -> spreadsheet.parse
  -> browser.navigate
  -> browser.upload
  -> browser.verify
  -> filesystem.move
~~~

## Native Browser Semantic Operation Contract

The internal Browser runtime already owns the correct live page through
BrowserTask and supports structured snapshot/action execution. The missing
compiler boundary is a first-class semantic operation contract.

Actions such as `browser_click`, `browser_type` and `browser_press` should
make enough structured information available to identify the operation
semantically without relying on transient `@eN` refs.

A normalized action observation should be able to carry concepts equivalent to:

~~~json
{
  "operation": "fill",
  "target": {
    "role": "textbox",
    "name": "Search",
    "testid": "search-input",
    "page_family": "github.repository"
  },
  "semantic_family": "github.repository.search.query",
  "target_fingerprint": "...",
  "before_state_ref": "artifact://...",
  "after_state_ref": "artifact://...",
  "semantic_effect": "field_value_changed"
}
~~~

Exact field names may differ after code review. The invariants are:

1. transient element refs are observations, not durable target identity;
2. semantic anchors are reacquirable after navigation/restart/drift;
3. a target family must distinguish unrelated stateful actions that happen to
   use the same tool schema;
4. structural similarity may seed learning but cannot independently grant
   `REQUIRE_COMPILE`;
5. semantic family identity must be owner-declared or safely derived from trusted
   runtime metadata, not arbitrary page prose;
6. current BrowserTask/lease/controller authority remains unchanged.

Existing `ProcedureStep.resolve_anchor()` and promoted-routine semantic anchors
are foundations to generalize, not parallel mechanisms to replace.

## work_execute becomes runtime infrastructure

`work_execute` must stop being primarily a guardrail instruction that tells the
model to manually rewrite repeated work as a durable batch.

Its target role is the deterministic execution entry point for already-resolved
operational work.

It should be able to execute, directly or through a resolver:

- a promoted `capability_id` + typed inputs;
- an exact compatible Recipe;
- a promoted Routine;
- an explicit validated graph/WorkPlan;
- bounded inline deterministic work when no promoted object exists.

Target resolution flow:

~~~text
intent / proposed operation
  -> Capability Resolver

compatible promoted Capability exists?
  -> yes: execute through work_execute without another planning call

else compatible Recipe/Routine exists?
  -> yes: execute through work_execute

else
  -> bounded ADAPTIVE execution
  -> trace capture
  -> candidate compilation
  -> continue work
~~~

The model should not need to remember a recipe key or know that `work_execute`
was selected when the harness can resolve an exact compatible deterministic path.

## Compilation policy after this change

The execution policy should distinguish at least:

~~~text
ALLOW_ADAPTIVE
SUGGEST_COMPILE
USE_COMPILED
COMPILE_OR_USE
REQUIRE_COMPILE
REQUIRE_HUMAN
~~~

Exact enum names may be simplified if existing compatibility makes that safer.

Semantics:

- **ALLOW_ADAPTIVE**: novel, drifted, stateful or semantically unresolved work;
- **SUGGEST_COMPILE**: useful repeated pattern exists but no validated reusable
  deterministic unit yet;
- **USE_COMPILED**: an exact compatible promoted capability/recipe/routine exists;
- **COMPILE_OR_USE**: semantic homogeneity and deterministic representation are
  established and compilation/reuse is economically useful;
- **REQUIRE_COMPILE**: reserve for true homogeneous mutable fan-out where
  continuing item-by-item LLM mediation would amplify cost/risk and all current
  safety admission requirements are satisfied;
- **REQUIRE_HUMAN**: existing sensitive/uncertain authority boundary.

**Absence of a compiled capability must never by itself block safe adaptive
discovery.**

## Skills versus Capabilities

They are separate layers.

**Skill**
- knowledge/instructions for the LLM;
- helps choose strategy;
- handles novelty and exceptions;
- may create or call capabilities;
- may invoke `work_execute`.

**Capability**
- deterministic executable operational contract;
- should not require an LLM during the successful replay path;
- may call primitives;
- may call other capabilities;
- may be composed into larger deterministic flows.

Allowed direction:

~~~text
Skill -> Capability
Skill -> work_execute
Capability -> Capability
Capability -> Operational Kernel
~~~

Avoid a nominally deterministic replay path that silently becomes:

~~~text
Capability -> Skill -> LLM -> Capability
~~~

When deterministic execution cannot continue, return a compact
`NEEDS_REASONING` handoff to the adaptive layer, then compile only the newly
learned segment/revision after verification.

## Capability storage and registry

Do not use one monolithic mutable knowledge file.

A profile-scoped registry may physically use JSON/SQLite/files, but the logical
model should support independent versioning and atomic updates per capability.

A human-readable projection may resemble:

~~~text
~/.hermes/workstation/capabilities/
  browser/
  filesystem/
  process/
  github/
  composed/
~~~

This path is illustrative, not yet an implementation mandate. Reuse existing
stores when they already satisfy ownership/persistence requirements.

The Capability Registry must support:

- exact lookup by semantic operation fingerprint;
- dependency resolution;
- lifecycle: discovered/validated/promoted/retired;
- immutable historical versions;
- current promoted version;
- scope/runtime compatibility checks;
- provenance/evidence links;
- invalidation/quarantine on contract drift;
- usage, failure, drift and savings metrics.

## Semantic operation fingerprint

`structural_signature()` remains useful as a discovery hint.

It must not remain the authority for mandatory compilation.

A semantic operation fingerprint should be based on trusted stable dimensions
such as:

~~~text
canonical operation
+ canonical route/runtime/provider
+ target/contract family
+ effect contract
+ relevant scope
+ capability/precondition contract version
~~~

Parameter values that identify individual fan-out items should normally be
separated from the operation family. Secrets and ephemeral ids must never enter
the durable fingerprint.

## Automatic reuse and composition

The resolver should prefer the **smallest exact validated deterministic unit**
that avoids repeated reasoning.

Atomic capabilities are expected to produce the highest reuse because they can be
shared across many larger workflows.

Composite capabilities should call capabilities rather than duplicate their
implementation. A mature flow can therefore evolve approximately as:

~~~text
primitive
  -> atomic capability
  -> composite capability
  -> full routine
~~~

Promotion of a larger workflow must not make its atomic dependencies inaccessible
for reuse elsewhere.

## Token-efficiency objective

Measure savings at the verified-outcome level, not from raw call count alone.

Track:

- LLM calls / verified outcome;
- input/output tokens / verified outcome;
- deterministic capability reuse rate;
- atomic capability reuse count;
- composite reuse rate;
- discovery cost vs replay cost;
- context reconstruction overhead;
- tool calls / verified outcome;
- drift/escalation rate;
- failed replay rate;
- human rescue rate;
- Guardrail Obstruction Rate.

The desired curve is qualitative until measured with real provider accounting:

~~~text
first novel execution: expensive adaptive reasoning
later compatible execution: less reasoning
promoted stable replay: zero operational planning calls
drift: reasoning only for the unresolved segment
~~~

Do not claim paid-token savings from simulated provider tests.

## Required implementation sequence

### P0 — fix the current wrong abstraction without weakening safety

1. close AEPC-E002:
   - structural shape is discovery-only;
   - semantic homogeneity is required for `REQUIRE_COMPILE`;
   - `executed_unverified` is not counted as verified success;
2. add/extend owner metadata needed to identify semantic target/operation family;
3. preserve paired regression:
   - long heterogeneous stateful browser flow remains adaptive;
   - true homogeneous fan-out still compiles/canaries.

### P1 — introduce Capability as a first-class reusable contract

4. define Capability model/schema/lifecycle;
5. implement Capability Registry by extending existing canonical persistence
   owners where possible;
6. implement exact Capability Resolver;
7. allow `work_execute` to execute a resolved `capability_id` + inputs;
8. preserve explicit graph/Recipe/Routine backward compatibility.

### P2 — operational kernel and browser semantic contract

9. normalize trusted primitive contracts for browser/filesystem/process;
10. add native Browser semantic target/action observations and reacquirable anchors;
11. generate semantic operation fingerprints from owner/runtime metadata;
12. parameterize learned traces and reject ephemeral refs/secrets from candidates.

### P3 — continuous learning and promotion

13. convert accepted adaptive trace segments into Capability Candidates;
14. validate compatible replay before promotion;
15. auto-promote only through existing evidence/acceptance policy;
16. version on drift; never mutate historical promoted behavior in place.

### P4 — composition and automatic reuse

17. allow Capability -> Capability composition;
18. auto-select the smallest exact promoted capability before model planning;
19. allow larger composed capabilities/routines to reference atomic dependencies;
20. surface only compact unresolved state to the LLM on drift.

### P5 — expand beyond browser

21. exercise filesystem and process capabilities end to end;
22. add cross-backend composite flow tests;
23. integrate HTTP/API and later native desktop control through the same contract.

## Acceptance criteria

The milestone is not complete until all of the following are proven:

1. three structurally identical but semantically different browser mutations do
   not trigger mandatory compilation;
2. true same-family fan-out still reaches compiler/canary admission;
3. a learned atomic capability can be replayed with **zero new LLM planning
   calls** under exact compatible preconditions;
4. the same atomic capability can be reused by at least two different composite
   flows without duplicating its implementation;
5. a composite capability can call another capability deterministically;
6. drift returns compact `NEEDS_REASONING` and resumes without replaying
   confirmed mutable effects;
7. browser anchors are reacquired semantically rather than persisting `@eN`;
8. filesystem/process capabilities use the same lifecycle/resolver model;
9. no new parallel SessionDB/Kanban/BrowserTask/TaskRun/Memory authority exists;
10. all uncertainty, effect, approval, lease, canary, acceptance and canonical
    completion invariants remain green;
11. Work100 and focused adaptive/compiled/routine/capability suites remain green;
12. native packaged/authenticated Browser qualification is run when Browser
    product code is changed and the environment supports it;
13. efficiency telemetry can distinguish discovery cost from replay cost without
    inventing token savings when provider usage is unavailable.

## Non-goals / prohibited shortcuts

Do not:

- fix AEPC-E002 by increasing the threshold;
- exempt all browser mutations from compilation;
- reset counters after navigation/snapshot merely to avoid the obstruction;
- make page prose authoritative for mutation identity;
- persist transient browser refs as reusable targets;
- create a second browser automation runtime;
- create a second memory/task/run database;
- allow arbitrary generated code to bypass effect/policy/approval contracts;
- silently call an LLM from a promoted deterministic capability;
- promote on one successful unverified dispatch;
- treat tool ACK as durable external effect proof;
- require users or the LLM to know internal recipe/capability keys when exact
  compatible reuse can be selected by the harness.

## Relationship to existing components

The intended mapping is:

~~~text
Tool Registry / browser_* / filesystem / process
  = primitive executable substrate

Operational Kernel
  = normalized trusted primitive contracts

Capability
  = smallest reusable deterministic operational knowledge

Recipe
  = validated deterministic graph representation

ProceduralMemory / WebProcedure
  = learned procedural history and promoted procedure ownership

Routine
  = higher-level promoted deterministic workflow

TaskCompiler / DurableBatchRunner / WorkPlan / WorkItem
  = durable execution/checkpoint/canary machinery

work_execute
  = public/runtime entry point for deterministic execution

Skill
  = LLM-facing knowledge and strategy, above capabilities

ExecutionJournal / ArtifactStore
  = evidence/provenance/trace plane
~~~

Implementation should consolidate these roles where safe rather than duplicate
them. The new Capability layer is an abstraction and contract boundary first; its
physical persistence may be implemented by extending existing Recipe/Procedural
stores.

## Final architectural statement

Hermes Work should behave less like an agent that repeatedly decides every
operational click and more like a system that **writes, validates and reuses its
own operational programs**.

The LLM remains responsible for novelty, interpretation and exception handling.
The runtime becomes responsible for repeatable mechanics.

The long-term optimization loop is:

~~~text
experience
  -> learn capability
  -> validate
  -> promote
  -> compose
  -> reuse automatically
  -> work_execute runtime
  -> reason only on novelty/drift
~~~
