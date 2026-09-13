---
title: "Worker Orchestration: Illustrated Tour"
sidebar_label: Orchestration illustrated
---

# Worker orchestration: illustrated tour

An **orchestrator** is the main agent that divides a task, assigns the pieces, checks progress, and combines the answers. A **worker** is another agent doing one of those pieces. Each worker has its own conversation and can use the tools it is allowed to use.

For example, ask Hermes to compare three proposals. It can give one worker the job of extracting the facts and another the job of checking contradictions. While they work, the parent can add guidance, wait for their answers, and ask the same worker a follow-up. You choose the worker definitions and which models they use.

This page explains the changes in [PR #106268](https://github.com/NousResearch/hermes-agent/pull/106268). It is a feature guide, not a claim that every provider or deployment has passed the same tests. For copyable configuration, start with [Worker profiles](worker-profiles.md). For Codex, Claude Code, OpenClaw, Bot Mode and Kanban coverage, see the [capability comparison](orchestration-capability-comparison.md).

## The whole system at a glance

<img src="/img/worker-orchestration/overview.svg" width="1100" alt="A user defines profiles and limits. A parent selects researcher, checker and optional coordinator workers. All use one service with retained state and execution receipts." />

Read the diagram from top to bottom:

1. **You define the available team.** A profile combines a purpose, instructions, a provider/model, an optional thinking level, tool restrictions, context choices and limits. “Researcher” and “checker” are examples; there are no compulsory roles or model families.
2. **The parent makes the assignments.** It discovers the available profiles, selects suitable workers, and supplies each with a clear task and the context needed to complete it.
3. **Workers execute independently.** Different workers may use different providers. An optional coordinator worker can delegate further when nesting is enabled.
4. **Hermes manages the shared rules and records.** The parent tool and plugin API use the same worker service. The service tracks ownership, run order, budgets, messages and recovery state.
5. **The parent receives evidence as well as answers.** A receipt records what was requested, what was sent, and what the provider reported. These are separate facts.

A powerful model alone does not provide this machinery. Selecting an OpenAI Codex model in Hermes's default runtime still uses the Hermes harness—the software that supplies tools, permissions and the execution loop. Hermes also has an [optional Codex app-server runtime](codex-app-server-runtime.md), with different supported callbacks. This PR does not add or expand that backend integration.

## Codex, existing Hermes, and this change

<img src="/img/worker-orchestration/comparison.svg" width="1100" alt="Three columns compare the Codex reference, existing Hermes delegation, and the retained worker capabilities added by this PR." />

**Comparison scope:** “Hermes before” means the inspected upstream baseline `c076d653a216939b97a93ab0c10ce709bacde667`, not a claim about every subsequent release. The Codex column summarizes [official subagent documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents), checked on September 9, 2026, and the agent tool contracts exposed during this work. Client, account, model and policy differences still apply. “Not established” means this comparison has no supporting evidence; it does not mean the other product cannot do it.

| Capability | Codex reference | Hermes before | Hermes with this PR |
| --- | --- | --- | --- |
| Divide work in parallel | Parallel subagents | Already supported batches and summaries | Preserved; tasks can select different profiles |
| Reusable worker definitions | Custom agents | Delegation defaults and leaf/orchestrator roles | Named user profiles with descriptions, instructions and restrictions |
| Discover the available team | Agent metadata | No unified worker-profile catalog | Compact `discover`, with details fetched when needed |
| Select model and thinking level | Per-agent settings | Provider/model overrides and request settings | Profile selection or policy-gated per-task provider/model/effort routing |
| Control routing freedom | Configuration-dependent | No profile-only/dynamic worker policy | Profile-only by default; dynamic choices limited to a user-enabled menu |
| Add guidance during work | Steering | Already supported `steer` | Durable message IDs and delivery state alongside legacy steering |
| Ask the same worker another question | Follow-up controls | Fresh conversation for ordinary delegation; plugin records retained in process | Ordered new runs with a retained worker conversation |
| Wait, inspect and stop | Thread controls | List/stop, background delivery and plugin lifecycle controls | Shared status, wait, inspect, cancel and completion acknowledgment |
| Restrict tools | Permission inheritance | Inherited toolsets and role blocks | Explicit ceilings across native tools, MCP and calls through code |
| Delegate another level | Harness/policy-dependent | Optional nested orchestrators already existed | Profiles narrow depth; descendants share durable execution budgets |
| Limit concurrent work | Configurable concurrency | Existing concurrency/depth controls | Owned active-run limits plus durable tree iteration/tool/time budgets |
| Cancel descendants | Target interruption observed; subtree behavior not established here | Cancellation followed ownership | Shared service propagates cancellation to owned descendants |
| Separate editing workspaces | Environment-dependent | Optional Git worktrees already existed | Preserved; context settings do not promise filesystem containment |
| Recover after a process restart | Exact recovery contract not established here | Some completion delivery was durable; plugin worker registry was in memory | Retained worker/run state, checkpoints, queues, leases and explicit uncertain outcomes |
| Prove the execution route | Configured metadata; independent identity not established | Configured model and basic result metadata | Requested/resolved/transmitted/provider-reported fields kept distinct |
| Configure without a prescribed model family | Custom configuration | Existing provider support | Provider-independent worker profiles through existing Hermes resolution |

The main gap was a **unified, retained worker workflow**. Hermes could already delegate useful work. This change connects user-defined routing, ongoing communication, shared capability enforcement and restart-safe records into one service.

## What the parent can now do

The parent uses actions on the existing `delegate_task` tool. These are instructions the agent can issue; a person can ask for the same workflow in ordinary language.

| Parent action | Plain-English meaning | Example use | Important boundary |
| --- | --- | --- | --- |
| `discover` | Show the available worker profiles and routes | “Who can inspect these files?” | Listing does not authenticate or prove a provider is available |
| `spawn` | Start workers for task items | “Have my researcher and checker work in parallel” | Invalid selections reject the batch before launch |
| `status` | Read compact progress and result summaries | “Which assignment is still running?” | Use returned worker/run IDs, not a profile name as identity |
| `message` | Add guidance to an existing worker | “Focus on the dates, not the formatting” | Delivery happens at supported boundaries, not instantly inside a request |
| `wait` | Wait for a selected worker or run | “Wait for the checker before deciding” | Waiting is bounded; it need not copy the transcript |
| `resume` | Give the same worker a new assignment | “Recheck your answer against this new fact” | It creates a new run, with current permissions rechecked |
| `inspect` | Request more detail and visible conversation | “Show how the checker reached that result” | Hidden reasoning and system prompts are excluded; ownership and size limits apply |
| `cancel` | Request that a run and its owned descendants stop | “We no longer need that branch of work” | Cooperative cancellation cannot undo an external action |
| `completions` | Find terminal results awaiting acknowledgment | “Collect any answers I have not consumed” | A stored result is not the same as delivered external communication |
| `ack` | Confirm receipt of a completed run | “I have received and processed that result” | Internal acknowledgment does not change an external transport's guarantees |
| `reconcile` | Record a decision about an uncertain tool outcome | “The operation completed; do not repeat it” | Reconciliation does not itself execute or replay the tool |

Legacy `list`, `steer` and `stop` remain available. Plugins use the [shared lifecycle API](../../developer-guide/subagent-lifecycle-api.md), so the parent and plugin paths do not keep competing worker state.

## How model routing and permissions fit together

<img src="/img/worker-orchestration/routing.svg" width="1100" alt="Profile-only and dynamic routing converge on batch validation, the narrowest effective permissions, and execution receipts." />

A **provider** is the service handling the model request. A **model** is the selected model identifier. **Thinking level**, also called reasoning effort, is a model-supported request setting. A higher level can change latency and usage; it is not a guarantee of a better answer.

In **profile-only mode**, the parent chooses a profile and uses its route. In **dynamic mode**, you also allow the parent to choose among specific enabled provider/model combinations. A profile can narrow that menu further. It cannot add routes outside the user's menu.

Settings resolve from permitted task overrides, the selected profile, delegation defaults, then parent defaults. User ceilings still apply after that resolution. Explicit unsupported effort combinations fail visibly; the parent is not silently moved to another model or a lower thinking level. Unknown metadata remains unknown. Any permitted fallback or normalization must follow configured policy and appear in the receipt.

The service then intersects the allowed capabilities. Think of this as several lists of permitted tools: a worker only receives a tool if all applicable restrictions allow it. A model switch cannot supply new credentials, undo a parent's denial, or expand the tool list.

**MCP** is a way of connecting external tools. **Tools called through code** are tool requests made inside `execute_code` rather than directly by the model. Both routes receive the same worker restrictions. Hiding a tool from the visible menu alone would not be sufficient enforcement.

Instructions such as “never edit files” describe intended behavior. A tool policy removes access to editing tools. A filesystem sandbox is a separate enforcing mechanism; suppressing context files or choosing a working directory does not create one. Requested guarantees are rejected when no enforcing backend is available.

## Messages, follow-ups and nested work

<img src="/img/worker-orchestration/lifecycle.svg" width="1100" alt="A first run receives a message, then a second run starts with retained context. Optional descendants share limits, and the parent collects compact results." />

A **worker ID** identifies the retained conversation. A **run ID** identifies one assignment within it. Two workers using the same profile still have different identities. Only one run executes on a worker at a time; follow-ups queue in order.

A message can be accepted into the queue before the worker has consumed it. Consumption is recorded with a saved conversation checkpoint. This distinction matters if the process stops between receiving guidance and acting on it.

A worker can also message its parent. Top-level worker-to-parent messages retain `QUEUED`, `PUBLISHED` and `ACKNOWLEDGED` states. The parent can inspect queued messages; publication includes them with the result; acknowledgment marks published messages as received. Optional sibling messaging lets workers communicate within policy. It does not allow one worker to inspect, resume or cancel an unrelated worker.

When nesting is enabled, a coordinator worker may start descendants. They share the root assignment's iteration/tool allowance and deadline; a child's own profile can narrow them further. Restart does not replenish the allowance. A new explicit assignment to a top-level worker receives a new budget, while descendants from the old assignment keep the old budget. The owner's active-run concurrency limit spans those trees.

Context is also deliberate. The parent should pass the information needed for the assignment. Startup memory/context-file inclusion follows the profile's context policy. Later follow-ups retain the worker's own conversation. Profile edits do not rewrite that conversation's original instructions; current permissions and required capabilities are rechecked before another run.

## What happens if Hermes stops halfway through?

<img src="/img/worker-orchestration/recovery.svg" width="1100" alt="After a restart, Hermes restores records and reclaims expired leases. Known checkpoints can resume after validation; uncertain tool outcomes pause for reconciliation." />

The durable source is the active Hermes profile's database. **Durable** means recorded so it can survive process exit. It does not mean the old Python thread or an in-flight provider request survives.

A **lease** is a time-limited ownership claim on a run. It prevents two executors from owning the same run at once and prevents an expired executor from overwriting recovered state. A **checkpoint** is the last committed conversation state from which supported continuation can start.

For a known checkpoint, Hermes restores its own conversation, rechecks credentials, route and tool availability, and starts a new linked run when continuation is safe. This does not require a provider-side resumable session. Queued messages and unacknowledged results remain recorded.

The difficult case is an external effect: a tool might create a file or submit an operation, then Hermes might stop before saving the result. Automatically repeating the call could perform the action twice. The worker therefore becomes interrupted/uncertain until the parent records an explicit decision:

- `confirmed_applied`: external evidence shows the operation happened.
- `confirmed_not_applied`: external evidence shows it did not happen.
- `accepted_unknown_no_replay`: the outcome remains unknown, and the decision is to avoid replay.

The record keeps the decision, note and affected tool-call identities. A separate `resume` starts the next assignment. This is careful handling of uncertainty, not a promise of exactly-once execution across arbitrary external services.

## Reading an execution receipt

| Receipt field group | The question it answers |
| --- | --- |
| Requested profile/provider/model/effort | What did the parent ask for? |
| Resolved settings | What did the allowed configuration select? |
| Transmitted settings | What did Hermes send at the observed provider dispatch boundary? |
| Provider-reported model | What model identity did the provider report, if any? |
| Fallback, normalization and metadata provenance | Was the selection transformed, and where did capability information come from? |
| Tools, lineage and run identity | What could the worker access, and which assignment/tree did it belong to? |
| Status, termination and usage | How did the run finish and what usage was recorded? |
| Known or unknown cost | Is there supported cost information, or is it unavailable? |

For example, a requested profile can resolve to model A, send model A, and receive model B in provider metadata. Those values must stay separate. A configured name or a model's own self-description cannot independently verify the service's internal execution. Credential values and copied live configuration do not belong in receipts.

## What has been exercised, and what remains outside the claim?

Implementation source `cce21f3ac8ba844544f18c4e82d7b85a08a22dce` passed its controlled two-provider live workflow. [Canonical CI](https://github.com/NousResearch/hermes-agent/actions/runs/34315825965) passed on merge checkout `d4f51e901297da5920487762a760b6c8617d58a7`: **46,385 passed, zero failed, 447 skipped**. Documentation-only additions after that source do not imply a new runtime test.

| Evidence | What it exercises |
| --- | --- |
| `test_delegation_model_routing.py`, `test_worker_profile_dispatch.py` | Profile and dynamic routing, unsupported selections, batch preflight and transport receipts |
| `test_worker_tool_enforcement.py`, `test_worker_dispatch_budget.py` | Native/MCP/code tool ceilings and request budget charging |
| `test_worker_admission_v2.py`, `test_worker_tree_budget_v2.py`, `test_worker_execution_limits.py` | Ownership, current-authority admission, limits, nested cancellation and parent messages |
| `test_worker_store.py`, `test_worker_process_acceptance.py` | Real database state, leases, ordered queues, cold restart, uncertain effects and completion acknowledgment |
| `test_subagent_lifecycle.py`, `test_delegate.py`, CLI `test_workers.py` | Plugin/parent compatibility, retained follow-ups, configuration and CLI behavior |
| Controlled OpenAI-Codex + ZAI workflow | Parent discovery and selection, distinct supported thinking levels, running-worker message, retained follow-up and three completion acknowledgments; request/receipt agreement |

The live workflow used controlled scheduling to ensure the message reached a running worker. The crash tests used synthetic provider transports and fresh processes. Neither proves every provider, filesystem backend, customer deployment or workload. The bounded review history has corrected findings but no final independent PASS or merge-readiness claim.

This PR does not add a Codex app-server backend, a universal sandbox, a new credential store, automatic cost optimization, a graphical worker editor, or a customer rollout. It does not establish superiority over Codex. It provides a tested set of Hermes orchestration capabilities with user-controlled profiles and explicit boundaries.

For exact fields and examples, continue to [configuration](worker-profiles.md). For implementation details, read the [architecture](../../developer-guide/worker-orchestration.md).
