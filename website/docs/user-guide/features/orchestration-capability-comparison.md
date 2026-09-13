---
title: Orchestration capability comparison
sidebar_label: Orchestration comparison
---

# Orchestration capability comparison

An agent **harness** supplies the model's tools, permissions, conversations and execution loop. An **orchestrator** uses those facilities to divide work, select participants, coordinate them and check the result. Selecting a powerful model does not automatically supply its vendor's harness.

Hermes already has four useful building blocks: delegated workers, full-profile Bots, Bot rooms and Kanban tasks. The worker changes in [PR #106268](https://github.com/NousResearch/hermes-agent/pull/106268) add retained worker conversations, configurable routes and durable receipts. The follow-up work described below connects these capabilities without replacing their separate ownership and recovery rules.

**Evidence date: September 9, 2026.** The comparison uses official vendor documentation and inspected Hermes source. It is not a benchmark or a claim that a draft PR is installed. A checkmark means the named capability exists within its stated limits; it does not mean identical behavior across products.

## The big picture

<img src="/img/worker-orchestration/service-map.svg" width="1100" alt="The planned model-facing interface connects to retained workers, Bots, rooms and Kanban. Each existing service keeps authority over its own state. Dashed connections are planned." />

Think of these as different objects:

| Object | What it represents | Who keeps the authoritative record? |
| --- | --- | --- |
| Worker | A retained conversation for delegated assignments | Shared worker service in the parent's profile-scoped session database |
| Run | One assignment executed by a worker | Worker service, including ownership lease and execution receipt |
| Bot | A complete Hermes profile with its own configuration, tools, skills and conversation | Bot/profile services and the canonical Bot Chat |
| Room | A shared discussion involving multiple Bots | Room service and, for hosted rooms, its fenced coordinator |
| Task | Work that can have dependencies, assignees and a review lifecycle | Kanban board, including claims and acceptance state |
| Workflow — planned | A saved, bounded arrangement of tasks, branches, joins and review gates | A workflow record referencing the existing task/run owners |

A worker finishing is not necessarily a task being accepted. A message being queued is not necessarily the recipient consuming it. Keeping these facts separate makes progress and recovery understandable.

## Feature chart

**✓ supported · ◐ conditional or fragmented · ✗ absent in the named mode · ? not established · Planned: not an implemented claim**

The Codex reference is its documented custom-agent controls plus the exposed harness controls inspected during this work. Claude's column distinguishes ordinary subagents from teams when behavior differs. OpenClaw's native workers, Swarm and ACP are different paths. Hermes main is the inspected baseline `bf53ff00a7360826ec2c9e2949533160068a8fc8`; the worker column describes PR #106268 at `ca48a5678f976f4313bcd826b3493315b6648da0`.

### Selecting and controlling workers

| Capability | Codex | Claude Code | OpenClaw | Hermes main | Worker PR | Planned follow-up |
| --- | --- | --- | --- | --- | --- | --- |
| User-defined worker profiles | ✓ | ✓ | ✓ | ◐ Bots and delegation defaults | ✓ | Preserve user choice |
| Per-worker model | ✓ | ✓ | ✓ | ◐ Execution-path dependent | ✓ | Preserve route across interfaces |
| Per-worker thinking level | ✓ | ✓ Subagents; ◐ teams inherit | ✓ | ◐ Execution-path dependent | ✓ Where supported | Qualify model/interface combinations |
| Parallel delegation | ✓ | ✓ | ✓ | ✓ | ✓ | Expose through selected interface |
| Running-worker guidance | ✓ | ✓ | ✓ Parent controls | ◐ Steer/Bot paths | ✓ | Explicit delivery and wake semantics |
| Follow-up with retained context | ✓ | ✓ | ✓ Retained sessions | ◐ Bots/Kanban | ✓ Workers | Same controls across eligible objects |
| Nested delegation | ✓ Policy-dependent | ✓ Subagents; ✗ nested teams | ✓ | ◐ Path-dependent | ✓ Tree limits | Preserve limits in every adapter |
| Tool and permission restrictions | ✓ | ✓ | ◐ Native/ACP differ | ✓ Path-dependent | ✓ Native/MCP/code ceilings | Recheck at the owning executor |

Sources: [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents), [Claude subagents](https://code.claude.com/docs/en/sub-agents), [Claude teams](https://code.claude.com/docs/en/agent-teams), [OpenClaw subagents](https://docs.openclaw.ai/tools/subagents), [OpenClaw ACP](https://docs.openclaw.ai/tools/acp-agents), [worker configuration](worker-profiles.md).

### Collaboration and repeatable work

| Capability | Codex | Claude Code | OpenClaw | Hermes main | Worker PR | Planned follow-up |
| --- | --- | --- | --- | --- | --- | --- |
| Peer messages | ✓ Inspected harness | ✓ Named subagents/teams | ◐ Native children restricted | ✓ Authorized Bots | ✓ Policy-controlled siblings | Typed targets and per-recipient outcomes |
| Shared task dependencies and claims | ? Dedicated team protocol | ✓ Teams | ? Shared claim protocol | ✓ Kanban | ✓ Existing Kanban | Connect tasks to workers and rooms |
| Grouped collaboration | ◐ Parent coordinates | ✓ Experimental teams | ✓ Experimental Swarm | ✓ Rooms/Kanban | ✓ Existing systems plus workers | One coherent parent workflow |
| Repeatable scripted orchestration | ? Dedicated runtime | ✓ Dynamic workflows | ✓ Swarm/TaskFlow | ◐ Task automation | ◐ Unified interface missing | Saved bounded workflow definitions |
| Independent-session communication | ✓ App controls inspected; deployment-dependent | ✓ Local and remote sessions | ✓ Authorized session routes | ✓ Eligible Bot/peer routes | Preserved | Discover only eligible targets |
| One discovery surface for workers, Bots, rooms and tasks | Not applicable | ◐ Multiple modes | ◐ Multiple runtimes | ✗ | ✗ | Typed, permission-filtered references |
| Model-appropriate interfaces over one Hermes core | Not applicable | Not applicable | Not assessed | ✗ | ✗ | Canonical, Codex-style and Claude-style adapters |

Sources: [Claude teams](https://code.claude.com/docs/en/agent-teams), [Claude cross-session messaging](https://code.claude.com/docs/en/cross-session-messaging), [Claude dynamic workflows](https://code.claude.com/docs/en/workflows), [OpenClaw Swarm](https://docs.openclaw.ai/tools/swarm), [OpenClaw session tools](https://docs.openclaw.ai/concepts/session-tool), [OpenClaw TaskFlow](https://docs.openclaw.ai/automation/taskflow), [Kanban](kanban.md).

### Persistence and evidence

| Capability | Codex | Claude Code | OpenClaw | Hermes main | Worker PR | Planned follow-up |
| --- | --- | --- | --- | --- | --- | --- |
| Restart-safe worker continuation | ? Full contract | ◐ Mode-dependent | ◐ Mode-dependent | ◐ Bots/Kanban contracts | ✓ Tested checkpoints; uncertainty pauses | Preserve each owner's recovery rules |
| Durable communication evidence | ? Full contract | ◐ Mailbox/transcript contracts | ◐ Path-specific receipts | ✓ Bot ingress, with limits | ✓ Worker queues and ACKs | Cross-link distinct receipts |
| Requested vs transmitted route evidence | ◐ Configured metadata | ◐ Configured metadata | ◐ Audit/runtime dependent | ◐ Path-dependent | ✓ Separate fields | Include selected interface/version |
| Cross-host shared task board | Not assessed | Not established by team docs | Not established by reviewed docs | ✗ Single-host Kanban | ✗ | Outside this follow-up |

Do not infer an exactly-once guarantee from any checkmark. Hermes worker leases, Bot claims and hosted-room fencing are different contracts. A Bot ingress consumer's crashed claim must not be recycled by applying the worker lease-expiry rule. Hosted-room authority must not be transferred merely because a host is temporarily unreachable. See the [worker recovery explanation](worker-orchestration-tour.md#what-happens-if-hermes-stops-halfway-through), [Kanban](kanban.md) and [Claude session storage](https://code.claude.com/docs/en/agent-sdk/session-storage).

## What each platform contributes to the design

**Codex:** reusable profiles, explicit per-agent model/effort, parent controls and configurable context inheritance are useful patterns. The inspected controls distinguish messaging an agent from assigning a new turn. Hermes adapters must preserve that difference.

**Claude Code:** ordinary subagents, teams, independent-session messaging and dynamic workflows solve different problems. Teams add shared tasks and peer coordination, but remain experimental; their in-process teammates are not restored by session resume. The Agent SDK is not a direct team API. The current team documentation also retires `TeamCreate`/`TeamDelete`: copying old tool names would not reproduce current behavior.

**OpenClaw:** native delegation, Swarm and ACP demonstrate several execution and coordination modes. The shared lesson is to identify the active path and its limits. Native completion delivery and an ACP session are not interchangeable evidence of sandbox enforcement or replay safety.

**Hermes:** reuse Bot profiles and communication, rooms, Kanban claims/review, provider resolution and the retained worker service. The missing link is a consistent way for the parent to discover and invoke these capabilities. It does not require a second Bot database, task board or credential store.

## A team workflow in ordinary language

Imagine asking: “Compare these proposals, have a second participant check the result, then revise it if needed.” In the planned integrated workflow:

1. The parent discovers the user-created profiles and their permitted actions.
2. It creates a comparison task and a dependent review task in Kanban.
3. It assigns the comparison to an eligible worker or Bot. The participant's own route and tool restrictions still apply.
4. While the participant works, the parent sends guidance. The result identifies whether that guidance is queued or consumed.
5. The participant supplies a result. Kanban makes review available; it does not equate that result with final acceptance.
6. The reviewer requests a correction. The parent follows up with the original worker, retaining its context.
7. Once the review passes, the task is accepted and the parent presents the answer and supporting receipts.

<img src="/img/worker-orchestration/team-sequence.svg" width="1100" alt="Planned team sequence: parent creates dependent tasks, a worker supplies a result, a reviewer requests changes, the worker follows up with retained context, and Kanban records acceptance." />

If Hermes restarts, each service restores its own records. A known completed step is collected, not executed again. If a tool may already have performed an external action, the workflow pauses for reconciliation. A saved workflow adds repeatability to these steps, not permission to bypass them.

## Model-facing interface is not execution backend

| Choice | What changes | What does not follow from it |
| --- | --- | --- |
| Provider/model selection | Which configured model receives requests | It does not install the vendor's harness |
| Planned Codex-style/Claude-style interface | The tool vocabulary and documented request semantics presented to the model | It does not grant extra tools, change the model, or imply native runtime parity |
| Hermes native worker | Hermes owns the loop, tool enforcement, conversation and receipts | A provider-side session is not required for the Hermes-owned checkpoint |
| Existing optional Codex app-server runtime | Codex owns its execution loop and sandbox; Hermes projects events and offers supported callbacks | Stateless callbacks currently do not expose `delegate_task`; this worker PR does not remove that boundary |

The [Codex app-server runtime guide](codex-app-server-runtime.md) documents the existing backend and its Kanban callback path. Extending that integration is separate from this interface work.

The proposed automatic selection policy is explicit user setting, then a qualified model/profile match, then the canonical Hermes interface. Unknown models retain their chosen provider/model. A qualified entry needs live evidence; a matching model-family name is not enough. Interface schemas stay fixed within a conversation to preserve prompt caching.

## Implementation and proof boundaries

| Stage | Deliverable | Required observable proof |
| --- | --- | --- |
| Worker PR #106268 | Profiles, retained workers, messages, limits and receipts | Existing two-provider workflow and recorded recovery/CI tests; see illustrated tour |
| Follow-up 1 | Model-appropriate interfaces | Equivalent authorized worker events and route receipts across interfaces |
| Follow-up 2 | Shared discovery and references | Read-only discovery; eligible actions succeed and unauthorized controls fail |
| Follow-up 3 | Integrated teams | Dependency/review/correction workflow with guidance, retained context and restart |
| Follow-up 4 | Saved bounded workflows | Repeat with new input; pause/restart/collect/cancel without duplicate owned execution |

Follow-up rows remain planned until linked implementation and acceptance evidence establish otherwise. Compare each model against itself when qualifying interfaces. Measure task completion, invalid tool calls, corrections, tokens and latency; do not claim that familiar naming necessarily improves performance.

The worker PR's [recorded CI run](https://github.com/NousResearch/hermes-agent/actions/runs/34315825965) and controlled OpenAI-Codex/ZAI smoke cover the recorded candidate, not these planned additions or every provider. Source/CI evidence does not establish installation, release, customer readiness or market superiority. The useful target is a broad, testable set of workflows that users can configure themselves.
