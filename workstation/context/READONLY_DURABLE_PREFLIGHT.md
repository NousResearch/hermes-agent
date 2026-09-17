# Read-only discovery before durable compilation

Discovery may prepare a mutation plan, but discovery may never become an
untracked mutation channel.

Successful read capability does not establish write capability.

Fan-out is authorized by verified persistence of a representative canary,
not by successful execution of a preparation step.

## Caller protocol

`DISCOVERING -> COMPILED -> CANARY -> VERIFIED -> FAN_OUT -> CHECKPOINT -> RESUME`
extends TaskCompiler, WorkPlan/WorkItem, registry ToolEffect and ArtifactStore;
there is no second execution state owner.

After `durable_compile_required`, preserve confirmed semantic identities and
inspect uncertain effects. Tool completion counts do not prove external
persistence. Local scratch writes do not prove provider resources were updated.
Use `read_file`, `search_files`, `browser_snapshot`, `browser_extract_items`,
tool descriptions and `work_execute action=status` where available. These
PURE_READ/DISCOVERY calls neither count toward mutation fan-out nor require
compilation. Repeated refused compilation does not itself close discovery;
existing no-progress and tool-budget guards remain active.

Use `work_execute action=discover` with up to eight `preflight` probes when
the graph is not yet known. Results are owner-scoped artifact references;
`PREFLIGHT_COMPLETE` means the requested probes succeeded, not that write
authority or a recipe has been verified. No frozen plan is created.
`PREFLIGHT_REQUIRED` and the guard response expose requirements and permitted
read tools. Arbitrary shell/JS cannot gain read authority from caller-supplied
`effect` or `read_only` labels; `GUARD_BOOTSTRAP_BLOCKED` identifies that boundary.

Workstation DOM inspection reuses `browser_extract_items` with `mode=inspect`,
`selector`, optional `attributes` (at most 16), and `limit` (at most 100).
It returns presence/count, node text, attributes and URL/title. Fixed code runs
in an isolated Electron world; selectors/attribute names are serialized data.
There is no click, submit, keyboard, arbitrary evaluation or network method.
Unsupported runtimes fail closed and suggest `browser_snapshot`. Default card
extraction remains available. Generic authenticated GET is not added: use an
existing trusted read provider when available. `browser_console`, `browser_exec`
and arbitrary `terminal` remain write-capable; use the structured alternatives.

Compile pending items with the actual mutation and an independent read step
whose `verifies` references the mutation and whose `expect` compares persisted
resource fields to `$item` values. A mutable batch without a read verifier is
rejected before any dispatch. Transport-only `ok/success/status_code/exit_code`
checks do not admit a verifier. Preparation alone cannot admit fan-out.
Declare the intended `mutation_target` (`scope`, `provider`, `kind`, `field`) for
external transactions. The compiler rejects preparation-only graphs and local
file verifiers for those targets. When the provider serializes the resource field
under another name, use verifier `readback={field, path}` and compare that exact
path in `expect`. Recipes pin this intended target in their body/fingerprint;
changing it requires a corrected graph and fresh canary.
The existing canary gate still blocks all remaining items on failed/uncertain
readback. Each dispatch intent/result/verifier is checkpointed; confirmed steps
are skipped on resume and uncertain writes require review, never blind retry.

## Independent channels and operation evidence

`constraints.mutation_allowed_routes` and `mutation_forbidden_routes` restrict
mutation tools and their declared provider routes only. For example forbid
`fake_cards.api` for mutations and use UI mutation plus API GET verification.
Existing global `allowed_routes/forbidden_routes` still restrict all dispatch,
including discovery; their semantics are unchanged.

Owner-supplied registry schema metadata can declare `capability` with `provider`,
`channel`, and `method` (or `method_field`). Evidence keys include all three.
Only structured HTTP status results establish VERIFIED/REJECTED evidence;
arbitrary JS text is not parsed into authority. GET 200 leaves PUT UNKNOWN.
PUT 403 rejects PUT while GET remains VERIFIED; a later GET cannot restore PUT.
Rejected capability prevents automatic mutable dispatch on that route.
Unknown capabilities still require the representative canary; evidence is not
a replacement for approvals or route policy.

Owner-supplied `mutation_target` describes `scope`, `provider`, `kind`,
`identifier_field`, `operation` and optional `identity_fields` (resource plus desired
value/version, excluding incidental selectors). Unknown tools stay unknown rather than
being attributed to Trello from a session count. The existing mutation result
references retain replay identity; effect records add target, lineage when
known, evidence, timestamp, verifier and certainty. Durable records live in
WorkItem checkpoints; external persistence becomes true only after readback.
Summaries contain bounded identities, effect totals and truncation indication.
Historical incident effects are fake regression data only.
Full durable effect evidence is available through `mutation_ledger_ref`; bounded
inline samples never imply that every item has the same certainty. Arbitrary
tools without owner metadata remain unknown. Provider adapters must supply honest
method/effect and readback semantics; DOM text alone can be optimistic UI state
and must not be treated as independent persistence without a fresh trusted source.

## Configuration and validation

No config.yaml key, enable switch or threshold changes are required. The
structural effect exclusion already exists; this change supplies a safe browser
inspection path and keeps discovery available while the compiler is repaired.
The guardrail stays enabled with its existing safe mutation threshold.

Regression: `workstation/tests/test_readonly_preflight.py` exercises twelve fake
cards, unknown selector discovery, UI writes/API readback, canary admission,
checkpoint/reconstruction, no confirmed replay, uncertain-write refusal,
method-specific capabilities and local/external effect summaries.
Electron's runtime-task tests execute the fixed inspector against a fake DOM,
including injection-shaped selector data and mutation sentinels.
