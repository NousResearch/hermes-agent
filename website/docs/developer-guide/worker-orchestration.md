---
title: Worker Orchestration Architecture
sidebar_label: Worker orchestration
---

# Worker orchestration

The parent-facing delegation tool and public subagent lifecycle API share worker
resolution, construction, lifecycle policy, and durable state. Worker profiles are
configuration in the active Hermes profile, not a process-global team definition.
The existing provider resolver remains responsible for authentication and runtime
transport selection.

Start with the [illustrated tour](../user-guide/features/worker-orchestration-tour.md)
for the user workflow, before/after comparison and visual explanations.

## Identity and authority

A worker ID identifies a retained conversation; a run ID identifies one assignment.
Parent session ownership is checked independently of knowing either identifier.
The plugin API's opaque handle remains a capability, rather than making arbitrary
IDs sufficient authority. Parent lineage, nested-depth policy, and effective tool
permissions are resolved before execution. Configured instructions do not grant
permissions, and narrowing a schema alone is not an execution security boundary.

The effective execution catalog is distinct from model-visible schemas. Tool
Search may replace permitted tools with bridge schemas; that presentation change
must neither revoke those tools nor grant hidden ones. Compile the executable
identities from the current user, ancestor, request, profile, and backend limits,
then apply the same ceiling at native, deferred MCP, and `execute_code` dispatch.
Explicit denial also applies to `delegate_task`; default worker controls are not
an exception to an explicit user restriction.

User-enabled model routes govern dynamic routing. A profile may narrow that menu,
but cannot expand global policy. The resolver validates the complete launch batch
before it creates children. Per-task settings take precedence only within allowed
overrides; limits are applied after resolution.

## Cache and conversation boundaries

### Model-facing interfaces

`agent/worker_interfaces.py` selects and binds the model-facing vocabulary.
The precedence is an explicit `orchestration.interface` setting, a qualified
automatic provider/model match, then canonical Hermes. The production registry
starts empty. Entries require recorded live qualification; transport fixtures do
not establish model usability.

Selection and collision-aware aliases freeze when the agent builds its tool
catalog. Existing native and MCP names retain their original meaning; colliding
worker names receive a `hermes_worker_` alias. Refresh, executable-tool admission,
native dispatch and execute-code/MCP dispatch use that same binding. An alias
cannot grant an operation denied through the canonical `delegate_task` capability.

`AIAgent._dispatch_worker_interface` translates the advertised request and calls
`_dispatch_delegate_task`. The adapter owns no worker state or message queue.
Its receipt records interface version, selection and qualification source,
advertised name and canonical operation, separately from execution route evidence.
The legacy dispatcher and plugin lifecycle API remain available.

The shared service distinguishes guidance from `start_turn`. Guidance queues a
message without waking an idle worker. `start_turn` creates an ordered linked
run, including when another assignment is active; it does not silently become
guidance. Exact-run interruption and descendant-tree cancellation are separate.
Completion acknowledgment and uncertainty reconciliation remain reachable through
each style. Unsupported transcript forking and graceful process shutdown are
explicit errors, not inferred vendor behavior.

See [orchestration interfaces](../user-guide/features/orchestration-interfaces.md)
for configuration and plain-English examples. The adapters provide familiar
vocabulary over Hermes; they are separate from optional vendor execution backends.

### Discovery and retained conversation

Discovery is an action of the existing delegation tool. The tool schema does not
grow or change when profile definitions or provider catalogs refresh. The catalog
reports capability provenance and unknown availability without authenticating or
probing providers just to list options.

Worker system prompts remain fixed for a conversation. Messages arrive at supported
tool boundaries or as subsequent conversation turns, not by editing the cached
system prefix or inserting synthetic user messages into the middle of a tool round.
Resume rechecks current authority; it must not silently continue with revoked tools
or rebuild the old conversation under a different permission contract.

## Database and recovery protocol

`agent/worker_store.py` uses the existing `SessionDB` transaction and read-context
primitives. It does not open a separate database or resolve credentials. Schema
creation is additive and idempotent. The tables hold workers, ordered runs, and
ordered internal messages. The service explicitly initializes them before use.
Per-run tool-effect records bind admission and settlement to the exact tool-call
identity, so a blocked call cannot settle a different concurrent action.

Run admission and lease acquisition occur inside `BEGIN IMMEDIATE`. A partial
unique index allows only one running assignment per worker; admission also counts
the owner's running assignments to enforce concurrency across nested calls. A
waiting orchestrator must not strand descendants behind its own capacity slot.

Each explicit root-worker assignment has a durable budget identity. Its nested
workers inherit the same aggregate iteration and tool-call allocation and absolute
deadline, in addition to their own profile ceilings. Reserve spending transactionally
before the corresponding execution boundary. Restart does not erase spending or
extend the deadline. A subsequent explicit root assignment receives a new budget;
descendants and nested follow-ups from an older assignment retain the older budget.
Owner-wide active-run concurrency applies across these assignment trees.

Every launch, including a queued follow-up with an existing process record, passes
through current-authority admission immediately before execution. Public handles
and parent-tool controls use the same FIFO recovery path. The triggering actor is
authorized before any queue mutation; it cannot schedule or fail unrelated owner
subtrees. Revalidation uses the retained worker's correct parent authority rather
than whichever actor happened to ask for status.

Each executor receives a unique lease token. Heartbeats, conversation checkpoints,
message acknowledgments, and completion require a live matching lease. An expired
executor cannot overwrite state after recovery or a replacement run. Leases fence
database writes; execution must check the lease at tool boundaries too.

Before a tool action, persist its in-flight state. After a confirmed result, persist
the conversation checkpoint and clear that state. A crash between those operations
creates an uncertain side-effect outcome. Recovery marks the run interrupted and
blocks further work until reconciliation; it does not infer that an external action
failed merely because Hermes did not record the response.

Run request IDs deduplicate enqueue retries. Reusing an ID with different content
is an error. Message delivery is acknowledged atomically with the conversation
checkpoint that includes it. Completions remain available until acknowledged.
Reconciliation clears the worker's resume barrier while retaining the interrupted
run's historical uncertainty record.

Messages from a top-level worker to the main parent use the same profile-scoped
store. Their states distinguish `QUEUED`, `PUBLISHED`, and `ACKNOWLEDGED`.
Publishing a message makes it durably available to the parent; worker completion
alone is not proof that the parent consumed it. Unacknowledged messages remain
available after restart. Nested child-to-parent and policy-enabled sibling
messages retain their lineage and ownership checks.

The reconciliation annotation records an explicit disposition, nonempty decision
note, affected tool-call IDs, prior statuses, and time. The top-level uncertainty
flag indicates a currently unresolved barrier; historical uncertainty remains in
the annotation. Reconciliation itself performs no provider or tool dispatch.

No recovery protocol here promises exactly-once behavior from external tools or
message transports. The supported guarantee is one leased executor plus explicit
handling of uncertain outcomes and deduplicated internal delivery.

## Route receipts

Preserve the distinction between requested, resolved, transmitted, and
provider-reported settings. A model's self-description is not provider evidence.
A configured model ID is not proof of which model an upstream router executed.
Unknown actual model, thinking metadata, and cost remain unknown. Fallback and
normalization policy must remain visible in the receipt.

Only allowlisted nonsecret policy and receipt fields enter worker metadata.
Credentials stay in the existing provider subsystem. Conversation checkpoints
have the same local-state privacy boundary as Hermes session history; they do not
belong in a public test report or PR artifact.

## Acceptance boundaries

Storage tests exercise real SQLite transactions, ownership, racing claims,
checkpoint/message atomicity, database reopening, lease fencing, and uncertain
recovery. Routing and lifecycle tests must also exercise the actual delegation
dispatch and public plugin APIs using a temporary Hermes home.

A two-provider live test is separate evidence from fixtures. Neither establishes
customer runtime readiness or complete parity with every Codex feature.

| Capability | Codex comparison reference | Hermes acceptance requirement |
| --- | --- | --- |
| Discovery | Agent descriptions and model/effort metadata exposed by the harness | Compact discovery plus on-demand profile details preserve unknown metadata |
| Custom profiles | Custom agents and descriptions | User-defined profiles appear in discovery and affect execution |
| Model routing | Per-agent model selection | Different provider requests match selected routes |
| Per-worker effort | Supported reasoning-effort overrides | Requested, resolved, and transmitted effort agree or show an explicit configured transformation |
| Permissions | Agent configuration and runtime permission controls | Native/MCP enforcement cannot exceed parent/user/profile authority |
| Messaging | Parent/child messaging and waiting | Durable message IDs, ownership, and supported delivery boundaries are exercised |
| Follow-up | Subsequent assignments to retained agents | A new run retains the worker conversation without rebuilding its system prefix |
| Nested orchestration | Harness limits and role configuration | Tree-wide limits hold without nested-wait deadlock |
| Cancellation | Cooperative interrupt controls | Cancellation reaches owned active descendants and does not imply external effects stopped |
| Restart recovery | Deployment-specific; no universal comparison claimed | Conversation, queue, lease and uncertain-action tests pass |
| Execution evidence | Harness/model metadata | Requested/resolved/transmitted/reported values remain distinct |

Codex reference: [official subagent documentation](https://developers.openai.com/codex/subagents/).
The matrix defines what to test, not a blanket superiority claim.
