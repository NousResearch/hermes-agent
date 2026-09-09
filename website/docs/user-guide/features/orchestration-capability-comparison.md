---
title: Orchestration capability comparison
sidebar_label: Orchestration comparison
---

# Orchestration capability comparison

An agent **harness** supplies the model's tools, permissions, conversations and execution loop. An **orchestrator** uses those facilities to divide work, select participants, coordinate them and check the result. Selecting a powerful model does not automatically supply its vendor's harness.

Hermes already has four useful building blocks: delegated workers, full-profile Bots, Bot rooms and Kanban tasks. The worker changes in [PR #106268](https://github.com/NousResearch/hermes-agent/pull/106268) add retained worker conversations, configurable routes and durable receipts. The follow-up work described below connects these capabilities without replacing their separate ownership and recovery rules.

**Competitor evidence date: September 9, 2026. Hermes delivery status refreshed September 10.** The comparison uses official vendor documentation and inspected Hermes source. It is not a benchmark or a claim that a draft PR is installed. A checkmark means the named capability exists within its stated limits; it does not mean identical behavior across products.

## The big picture

<img src="/img/worker-orchestration/service-map.svg" width="1100" alt="The draft model-facing interface connects to retained workers, Bots, rooms and Kanban. Each existing service keeps authority over its own state. Dashed connections are draft integrations." />

Think of these as different objects:

| Object | What it represents | Who keeps the authoritative record? |
| --- | --- | --- |
| Worker | A retained conversation for delegated assignments | Shared worker service in the parent's profile-scoped session database |
| Run | One assignment executed by a worker | Worker service, including ownership lease and execution receipt |
| Bot | A complete Hermes profile with its own configuration, tools, skills and conversation | Bot/profile services and the canonical Bot Chat |
| Room | A shared discussion involving multiple Bots | Room service and, for hosted rooms, its fenced coordinator |
| Task | Work that can have dependencies, assignees and a review lifecycle | Kanban board, including claims and acceptance state |
| Workflow — draft | A saved, bounded arrangement of tasks, branches, joins and review gates | Versioned templates and invocation controls in the Kanban board; references to existing task/run owners |

A worker finishing is not necessarily a task being accepted. A message being queued is not necessarily the recipient consuming it. Keeping these facts separate makes progress and recovery understandable.

## Feature chart

**✓ supported · ◐ conditional or fragmented · ✗ absent in the named mode · ? not established · Planned: not an implemented claim**

The Codex reference is its documented custom-agent controls plus the exposed harness controls inspected during this work. Claude's column distinguishes ordinary subagents from teams when behavior differs. OpenClaw's native workers, Swarm and ACP are different paths. Hermes main is the inspected baseline `bf53ff00a7360826ec2c9e2949533160068a8fc8`; the worker column describes PR #106268 at `ca48a5678f976f4313bcd826b3493315b6648da0`.

### Selecting and controlling workers

| Capability | Codex | Claude Code | OpenClaw | Hermes main | Worker PR | Implemented follow-ups (draft) | Remaining / limits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| User-defined worker profiles | ✓ | ✓ | ✓ | ◐ Bots and delegation defaults | ✓ | ✓ Preserved | User-selected profiles remain the default |
| Per-worker model | ✓ | ✓ | ✓ | ◐ Execution-path dependent | ✓ | ✓ Route preserved by adapters | No compulsory model family |
| Per-worker thinking level | ✓ | ✓ Subagents; ◐ teams inherit | ✓ | ◐ Execution-path dependent | ✓ Where supported | ✓ Supported profile effort preserved | Automatic interface qualification disabled |
| Parallel delegation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ Shared worker service | Parallel branches and joins pass focused fixtures |
| Running-worker guidance | ✓ | ✓ | ✓ Parent controls | ◐ Steer/Bot paths | ✓ | ✓ Worker controls and team guidance | Live Bot/room delivery not established |
| Follow-up with retained context | ✓ | ✓ | ✓ Retained sessions | ◐ Bots/Kanban | ✓ Workers | ✓ Retained team corrections exercised | Saved workflow process fixture passes; final cumulative gate pending |
| Nested delegation | ✓ Policy-dependent | ✓ Subagents; ✗ nested teams | ✓ | ◐ Path-dependent | ✓ Tree limits | ✓ Existing tree limits preserved | No broader native-harness parity claim |
| Tool and permission restrictions | ✓ | ✓ | ◐ Native/ACP differ | ✓ Path-dependent | ✓ Native/MCP/code ceilings | ✓ Adapter and service authorization tests | Recheck final cumulative candidate |

Sources: [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents), [Claude subagents](https://code.claude.com/docs/en/sub-agents), [Claude teams](https://code.claude.com/docs/en/agent-teams), [OpenClaw subagents](https://docs.openclaw.ai/tools/subagents), [OpenClaw ACP](https://docs.openclaw.ai/tools/acp-agents), [worker configuration](worker-profiles.md).

### Collaboration and repeatable work

| Capability | Codex | Claude Code | OpenClaw | Hermes main | Worker PR | Implemented follow-ups (draft) | Remaining / limits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Peer messages | ✓ Inspected harness | ✓ Named subagents/teams | ◐ Native children restricted | ✓ Authorized Bots | ✓ Policy-controlled siblings | ✓ Typed targets; individual team guidance outcomes | Live Bot/room recipients not tested |
| Shared task dependencies and claims | ? Dedicated team protocol | ✓ Teams | ? Shared claim protocol | ✓ Kanban | ✓ Existing Kanban | ◐ Team connection implemented and tested | Two actual-process team fixtures pass |
| Grouped collaboration | ◐ Parent coordinates | ✓ Experimental teams | ✓ Experimental Swarm | ✓ Rooms/Kanban | ✓ Existing systems plus workers | ◐ Team review/correction live pilot passed | Final team delivery gates pending |
| Repeatable scripted orchestration | ? Dedicated runtime | ✓ Dynamic workflows | ✓ Swarm/TaskFlow | ◐ Task automation | ◐ Unified interface missing | ◐ Saved workflow runtime implemented | Versioned bounded definitions, branches and joins |
| Independent-session communication | ✓ App controls inspected; deployment-dependent | ✓ Local and remote sessions | ✓ Authorized session routes | ✓ Eligible Bot/peer routes | Preserved | ✓ Eligible targets discoverable | Existing recipient authority remains required |
| One discovery surface for workers, Bots, rooms and tasks | Not applicable | ◐ Multiple modes | ◐ Multiple runtimes | ✗ | ✗ | ✓ #106647: read-only typed references | No discovery-triggered recovery |
| Model-appropriate interfaces over one Hermes core | Not applicable | Not applicable | Not assessed | ✗ | ✗ | ✓ #106463: three interface styles | Automatic registry empty |

Sources: [Claude teams](https://code.claude.com/docs/en/agent-teams), [Claude cross-session messaging](https://code.claude.com/docs/en/cross-session-messaging), [Claude dynamic workflows](https://code.claude.com/docs/en/workflows), [OpenClaw Swarm](https://docs.openclaw.ai/tools/swarm), [OpenClaw session tools](https://docs.openclaw.ai/concepts/session-tool), [OpenClaw TaskFlow](https://docs.openclaw.ai/automation/taskflow), [Kanban](kanban.md).

### Persistence and evidence

| Capability | Codex | Claude Code | OpenClaw | Hermes main | Worker PR | Implemented follow-ups (draft) | Remaining / limits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Restart-safe worker continuation | ? Full contract | ◐ Mode-dependent | ◐ Mode-dependent | ◐ Bots/Kanban contracts | ✓ Tested checkpoints; uncertainty pauses | ✓ Worker and integrated-team process fixtures | Team and saved-workflow process fixtures pass; cumulative gate pending |
| Durable communication evidence | ? Full contract | ◐ Mailbox/transcript contracts | ◐ Path-specific receipts | ✓ Bot ingress, with limits | ✓ Worker queues and ACKs | ✓ Distinct task/run/message references | No external exactly-once guarantee |
| Requested vs transmitted route evidence | ◐ Configured metadata | ◐ Configured metadata | ◐ Audit/runtime dependent | ◐ Path-dependent | ✓ Separate fields | ✓ Interface/version plus live team wire receipts | Actual provider identity and cost may remain unknown |
| Cross-host shared task board | Not assessed | Not established by team docs | Not established by reviewed docs | ✗ Single-host Kanban | ✗ | ✗ | Outside this plan |

Do not infer an exactly-once guarantee from any checkmark. Hermes worker leases, Bot claims and hosted-room fencing are different contracts. A Bot ingress consumer's crashed claim must not be recycled by applying the worker lease-expiry rule. Hosted-room authority must not be transferred merely because a host is temporarily unreachable. See the [worker recovery explanation](worker-orchestration-tour.md#what-happens-if-hermes-stops-halfway-through), [Kanban](kanban.md) and [Claude session storage](https://code.claude.com/docs/en/agent-sdk/session-storage).

## What each platform contributes to the design

**Codex:** reusable profiles, explicit per-agent model/effort, parent controls and configurable context inheritance are useful patterns. The inspected controls distinguish messaging an agent from assigning a new turn. Hermes adapters must preserve that difference.

**Claude Code:** ordinary subagents, teams, independent-session messaging and dynamic workflows solve different problems. Teams add shared tasks and peer coordination, but remain experimental; their in-process teammates are not restored by session resume. The Agent SDK is not a direct team API. The current team documentation also retires `TeamCreate`/`TeamDelete`: copying old tool names would not reproduce current behavior.

**OpenClaw:** native delegation, Swarm and ACP demonstrate several execution and coordination modes. The shared lesson is to identify the active path and its limits. Native completion delivery and an ACP session are not interchangeable evidence of sandbox enforcement or replay safety.

**Hermes:** reuse Bot profiles and communication, rooms, Kanban claims/review, provider resolution and the retained worker service. The missing link is a consistent way for the parent to discover and invoke these capabilities. It does not require a second Bot database, task board or credential store.

## A team workflow in ordinary language

Imagine asking: “Compare these proposals, have a second participant check the result, then revise it if needed.” In the draft integrated team workflow:

1. The parent discovers the user-created profiles and their permitted actions.
2. It creates the comparison task and any dependent tasks in Kanban. Review is a separate phase of a task's lifecycle.
3. It assigns the comparison to an eligible worker or Bot. The participant's own route and tool restrictions still apply.
4. While the participant works, the parent sends guidance. The result identifies whether that guidance is queued or consumed.
5. The participant supplies a result. Kanban makes review available; it does not equate that result with final acceptance.
6. The reviewer requests a correction. The parent follows up with the original worker, retaining its context.
7. Once the review passes, the task is accepted and the parent presents the answer and supporting receipts.

<img src="/img/worker-orchestration/team-sequence.svg" width="1100" alt="Draft team sequence: parent creates dependent tasks, a worker supplies a result, a reviewer requests changes, the worker follows up with retained context, and Kanban records acceptance." />

If Hermes restarts, each service restores its own records. A known completed step is collected, not executed again. If a tool may already have performed an external action, the workflow pauses for reconciliation. A saved workflow adds repeatability to these steps, not permission to bypass them.

The implemented draft task-to-worker connection records the assignment before scheduling
execution. That ordering matters: if Hermes stops between creating a worker and
remembering which task it belongs to, a retry must find that worker rather than
launch a second copy. The task claim, worker assignment and execution reference
keep their own identities and permission checks.

<img src="/img/worker-orchestration/task-run-link.svg" width="1100" alt="Draft restart-safe assignment: claim task, prepare pending worker, attach its reference, schedule that recorded run, and separately review the result. Each restart point reuses records or pauses for reconciliation." />

## What a saved workflow adds

A saved workflow is a recipe for work you want to repeat: compare two sets of information in parallel, review their results, then combine the accepted answers. Users choose the participant profiles and set a maximum number of corrections. The recipe is versioned so a later edit does not silently change work already in progress.

<img src="/img/worker-orchestration/saved-workflow.svg" width="1100" alt="Draft saved workflow: independent tasks run in parallel, each result is reviewed, accepted prerequisites release the combined task, and a new invocation repeats the recipe. Separate controls explain pause, restart, uncertain effects and cancellation." />

A **pause** prevents new task claims; work already claimed may finish. A **restart** reloads the recorded task and execution identities instead of starting the recipe from the beginning. If the outcome of an external action is unknown, continuation waits for reconciliation. A **cancellation** keeps already accepted results but blocks unfinished tasks, so cancelling a prerequisite cannot accidentally release its dependants.

These are the implemented draft increment-four contracts; the [workflow guide](orchestration-workflows.md) gives executable examples. They add control records to the existing board and reuse the team execution path. They do not add a second task engine or restore an arbitrary script's interpreter stack.

A [sanitized acceptance receipt](/evidence/orchestration-followup-acceptance.json) keeps exact source identities, failed samples and remaining gates together.

## Capability-to-test map

This map names the executable contracts behind the draft capabilities. Passing
fixtures establish their named behavior, not every provider or external delivery path.
The final cumulative candidate must rerun these contracts together.

| What a person can ask the parent to do | Contract fixture | Evidence boundary |
| --- | --- | --- |
| Choose a user profile and supported thinking level | `test_delegation_model_routing.py`, `test_worker_profile_dispatch.py`, `test_worker_receipts.py` | Invalid selections fail; requested, resolved, transmitted and reported model fields remain separate |
| Use familiar agent controls without changing permissions | `test_worker_interfaces.py`, `test_worker_admission_v2.py` | Canonical and styled controls share authorization; automatic qualification remains disabled |
| Find eligible workers, Bots, rooms and tasks | `test_shared_discovery.py` | Read-only temporary stores and trusted-session fixtures; live Bot/room delivery untested |
| Guide work, request review and retain the original worker for corrections | `test_team_orchestration.py` and the [controlled pilot](orchestration-team-pilot.md) | Nine focused cases and revised live sample 8/8; earlier failed sample preserved |
| Recover a held assignment or interrupted review after Hermes exits | `test_team_process_acceptance.py` | Two actual-process tests passed on the recorded team candidate; uncertain tool effect pauses without replay |
| Start one saved recipe despite simultaneous retries | `test_atomic_identical_admission_conflict_and_partial_graph_rollback` | One graph, stable references, changed-content conflict and rollback of partial admission |
| Run parallel branches, review them and combine accepted results | `test_parallel_review_correction_join_and_restart_safe_resume` | Real team/store services with controlled execution; its initial restart is service reconstruction |
| Pause new work while an already-claimed run settles | `test_pause_and_claim_share_one_board_serialization_point` | Board transaction orders pause and claim; later claims are denied |
| Cancel a recipe without releasing dependent work | `test_exact_cancel_is_sticky_and_does_not_replay_uncertain_interrupt` and native-block/stale-recovery regression | Exact-run interruption, terminal evidence, retained accepted steps and blocked external dependent; task status alone cannot prove a worker stopped |
| Keep another parent or stale board from controlling the recipe | `test_readonly_and_styled_paths_preserve_owner_board_and_capability` | Owner, board and dispatcher exclusions; inspection cannot create a missing board |

## Model-facing interface is not execution backend

| Choice | What changes | What does not follow from it |
| --- | --- | --- |
| Provider/model selection | Which configured model receives requests | It does not install the vendor's harness |
| Draft Codex-style/Claude-style interface | The tool vocabulary and documented request semantics presented to the model | It does not grant extra tools, change the model, or imply native runtime parity |
| Hermes native worker | Hermes owns the loop, tool enforcement, conversation and receipts | A provider-side session is not required for the Hermes-owned checkpoint |
| Existing optional Codex app-server runtime | Codex owns its execution loop and sandbox; Hermes projects events and offers supported callbacks | Stateless callbacks currently do not expose `delegate_task`; this worker PR does not remove that boundary |

The [Codex app-server runtime guide](codex-app-server-runtime.md) documents the existing backend and its Kanban callback path. Extending that integration is separate from this interface work.

The implemented automatic selection policy is explicit user setting, then a qualified model/profile match, then the canonical Hermes interface. Unknown models retain their chosen provider/model. A qualified entry needs live evidence; a matching model-family name is not enough. Interface schemas stay fixed within a conversation to preserve prompt caching.

## Implementation and proof boundaries

The saved-workflow runtime and its correction deltas received independent
review PASS at `7068e5814e8bc44ea1465e122882877c836490c3` after one initial
and two targeted passes. Nine focused workflow tests pass locally. The corrected
process fixture and final cumulative hosted validation retain separate receipts;
see the draft PR validation section for their final status.


The corrected team runtime is `c7fb32110494054fe0ed06bb8267b7794dce8c34`.
Its revised two-provider pilot passed 8/8 checks; the earlier 5/8 sample is retained.
The dependent worker succeeded, but its task was not separately accepted. The
pilot used temporary profiles, synthetic assignments and no worker tools or live
Bot/room recipients. It does not enable automatic interface qualification.

[Exact-target hosted validation](https://github.com/100yenadmin/hermes-agent-for-upstream-PR-only/actions/runs/34378493480)
passed every mapped orchestration test file, Windows and Nix, but the complete
Python job failed two quickstart assertions (46,503 passed; 449 skipped).
[Baseline comparison](https://github.com/100yenadmin/hermes-agent-for-upstream-PR-only/actions/runs/34385960828)
reproduced the same failures on upstream main's recorded baseline and the team
candidate. A test-only fixture correction then [passed all six quickstart tests](https://github.com/100yenadmin/hermes-agent-for-upstream-PR-only/actions/runs/34386428723). The team PR now advertises `1dc2c96218e6554617146fddb3f298a815289a6f`, which adds that fixture correction and the two process fixtures to the reviewed runtime. Historical fork checks retain their exact target identities.

Two [integrated-team process tests](https://github.com/100yenadmin/hermes-agent-for-upstream-PR-only/actions/runs/34387885271) now kill and restart actual Python processes using temporary Hermes stores. They exercise held attachment, retained review correction and a single uncertain tool effect that is not replayed. Saved-workflow focused tests pass atomic admission, immutable version and new-input reuse, all three styled action paths, shared native-transition limits, cancellation, control and task integration; their initial recovery case reconstructs services within one process. The subsequent [workflow process scenario](https://github.com/100yenadmin/hermes-agent-for-upstream-PR-only/actions/runs/34389291000) also passed: pause, initial branch attachment, reviewer attachment and retained correction survive process termination before the join is accepted. Final cumulative CI remains a separate gate.


The interface follow-up is now published as [draft PR #106463](https://github.com/NousResearch/hermes-agent/pull/106463).
Its [controlled live pilot](orchestration-interface-pilot.md) includes passing and
failing samples at a recorded source identity. The chart above retains its named
main/worker baselines; a later draft does not retroactively change those columns.
Shared discovery is published as [draft PR #106647](https://github.com/NousResearch/hermes-agent/pull/106647).
At `a90eb6202b6a0090aaa68ad981e196e9370c3fb3`, its six focused acceptance
cases passed, independent review passed, and [hosted CI](https://github.com/NousResearch/hermes-agent/actions/runs/34366948971),
[Docker](https://github.com/NousResearch/hermes-agent/actions/runs/34366947969)
and [Nix](https://github.com/NousResearch/hermes-agent/actions/runs/34366947927)
passed. These tests exercise real temporary profile databases and trusted-session
fixtures; they do not establish delivery to a live Bot or room.

For a quick view of the follow-up work, read this table alongside the fixed
main/worker baselines above:

| Capability added by follow-up | Draft implementation | Evidence and remaining limit |
| --- | --- | --- |
| One selected worker interface per conversation | ✓ #106463 | Deterministic conformance and independent review passed; live pilot has both passes and failures; automatic qualification registry remains empty |
| Discover permitted workers, runs, Bots, rooms and tasks | ✓ #106647 | Read-only discovery, typed references and action-time checks exercised; room grants permit inspection only |
| Coordinate dependency tasks, review and retained corrections | ◐ Draft #106696 | Corrected runtime c7fb3211 passed 9 focused tests, targeted independent review and an 8/8 two-provider pilot; two actual-process fixtures and the CI fixture correction pass; PR head reconciled; cumulative CI pending |
| Repeat a saved bounded workflow after restart | ◐ Implemented draft | Nine focused cases pass; initial service reconstruction and later actual-process workflow fixture pass; cumulative CI remains pending |

Discovery answers “what can this parent see and use?” It does not create a room,
assign a task, wake a Bot, or recover a worker. A discovered run points back to
its owning worker. Every later control request still has to pass the owning
service's permission checks. Integrated teams have their own focused and live evidence; saved workflows have a committed implementation with nine focused cases passing. Their remaining process-recovery and delivery gates are separate.

| Stage | Deliverable | Required observable proof |
| --- | --- | --- |
| Worker PR #106268 | Profiles, retained workers, messages, limits and receipts | Existing two-provider workflow and recorded recovery/CI tests; see illustrated tour |
| [Follow-up 1: #106463](https://github.com/NousResearch/hermes-agent/pull/106463) | Model-appropriate interfaces | Deterministic authorized event/receipt conformance passed; live qualification remains limited as recorded in the pilot |
| [Follow-up 2: #106647](https://github.com/NousResearch/hermes-agent/pull/106647) | Shared discovery and references | Read-only discovery, authorization, owner relationships and no-write identity handling passed at the head recorded above |
| [Follow-up 3: #106696](https://github.com/NousResearch/hermes-agent/pull/106696) | Integrated teams | Dependency/review/correction workflow with guidance, retained context and restart; acceptance remains pending |
| Follow-up 4 | Saved bounded workflows | Repeat with new input; pause/restart/collect/cancel without duplicate owned execution |

Unproven behavior remains marked pending or unknown. A draft implementation and its tests do not change what is available on main. Compare each model against itself when qualifying interfaces. Measure task completion, invalid tool calls, corrections, tokens and latency; do not claim that familiar naming necessarily improves performance.

The worker PR's [recorded CI run](https://github.com/NousResearch/hermes-agent/actions/runs/34315825965) and controlled OpenAI-Codex/ZAI smoke cover the recorded candidate, not these planned additions or every provider. Source/CI evidence does not establish installation, release, customer readiness or market superiority. The useful target is a broad, testable set of workflows that users can configure themselves.
