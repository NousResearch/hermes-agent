---
title: Orchestration Interfaces
sidebar_label: Orchestration interfaces
---

# Orchestration interfaces

An orchestration interface is the set of tool names and instructions a parent
model uses to manage workers. You can choose the standard Hermes interface or a
style with familiar Codex or Claude terminology. All styles use the same Hermes
worker service, permissions, conversations and execution receipts.

For example, a parent can ask a researcher to find evidence, send a correction
while the researcher works, and later ask the same researcher a follow-up. The
interface changes how the parent requests those actions. Your worker profiles
still determine which models run and what they may do.

These interfaces run **Hermes workers**. Selecting a style does not start Codex
CLI, Claude Code, an agent team in another application, or a vendor backend.
See the [capability comparison](orchestration-capability-comparison.md) for those
separate execution paths.

## Choose an interface

Add this setting to the active Hermes profile's `config.yaml`:

```yaml
orchestration:
  interface: auto
```

| Setting | Behavior |
| --- | --- |
| `auto` | Use an exact qualified provider/model match when available; otherwise use Hermes. |
| `hermes` | Use the canonical `delegate_task` interface. |
| `codex` | Explicitly select the Codex-style worker tools. |
| `claude` | Explicitly select the Claude-style worker tools. |

An explicit setting takes precedence over automatic selection. An unknown model
keeps its selected provider and model and receives the canonical Hermes interface.
Selection never changes the route to obtain a familiar tool style.

Styled interfaces are experimental unless their exact model/profile and interface
version have passing live qualification evidence. An empty qualification registry
means `auto` selects Hermes for every model. A configured model name, a fixture
test or an assumption about model training is insufficient to enable a registry
entry.

The interface and its advertised tool names are fixed for the session. Start a
new session after changing the setting. This keeps the conversation's tool
definitions consistent and avoids rewriting its cached prompt prefix.

## The actions behind the names

The available schemas describe the exact arguments. The following operations
retain the same meaning across styles:

| Operation | What the parent can accomplish | Important limit |
| --- | --- | --- |
| Discover | Read worker profiles, their uses and supported routing settings. | Discovery grants no additional access. Unknown availability stays unknown. |
| Spawn | Start a worker from a permitted profile, with an assignment and explicit context. | The complete parent transcript is not automatically forked. |
| Message | Send guidance to an existing worker. | Guidance alone does not wake an idle worker. |
| Follow up | Give the same worker another assignment with retained conversation context. | Only one run executes at a time; a queued assignment remains distinct from guidance. |
| Inspect or list | Read authorized status, results and execution evidence. | Knowing a worker ID does not grant access to another parent's worker. |
| Wait | Wait for a selected worker run with a bounded timeout. | A timeout is not worker failure or cancellation. |
| Interrupt | Ask one current run to stop at a supported execution boundary. | This does not cancel the worker's entire descendant tree. |
| Cancel owned work | Cancel a worker and its owned descendants. | Cancellation is cooperative and cannot undo an external side effect. |
| Acknowledge | Record that the parent consumed a terminal result. | Worker completion and result delivery remain separate events. |
| Reconcile | Record a decision about an interrupted action with an uncertain outcome. | Reconciliation does not automatically replay that action. |

Requests for unsupported behavior fail visibly. In particular, familiar naming
does not add transcript forking or graceful process shutdown. An interruption,
tree cancellation and graceful shutdown request are different operations.

### Messages and new assignments

Suppose a worker has completed its research. Sending “Use the updated exchange
rate” adds guidance to that conversation; it does not spend another model call
merely because the worker is idle. A separate follow-up such as “Now revise the
comparison using that correction” starts the next authorized run.

A new assignment sent while work is active must retain its identity as queued
work. It must not silently become a steering message for the current assignment.
The worker service owns ordering, admission, retained context and execution limits
for both operations.

### Existing tool names remain available

Some tools already use names that a styled interface would otherwise advertise.
For example, an installed messaging tool may be named `send_message`. Hermes
preserves that tool and gives the worker operation a collision-free alias, such
as `hermes_worker_send_message`.

The parent should follow the tool definitions actually advertised in its session.
The frozen alias map applies to dispatch as well as presentation. It cannot
redirect an unrelated native or MCP tool into worker management or bypass the
permission checks on `delegate_task`.

## Routing and permissions remain yours

Interface style is independent of [worker profiles](worker-profiles.md). A parent
using one style can select workers from different providers and choose supported
thinking levels within your configured routing policy.

In profile-only mode, the parent chooses your named profiles. Dynamic routing
additionally permits the configured provider/model overrides. Neither style can
expand tools, MCP access, credentials, filesystem permissions or delegation depth.
An unsupported explicit model or effort fails instead of being silently replaced.

Context is also independent of style. Supply task-specific context deliberately;
worker context-file and memory settings still follow the selected profile. A
follow-up retains the worker's own conversation and rechecks its current execution
contract before launching.

## Read the receipts

Worker results carry an `orchestration_interface` receipt identifying the selected
style, interface version, selection source, qualification status, advertised tool
and canonical operation. This answers “Which interface did the parent use?”

The worker's route receipt separately records requested, resolved and transmitted
provider/model/effort settings, plus provider-reported information when available.
This answers “What did Hermes ask the provider to execute?” A configured model
identifier is not independent verification of the provider's actual model.

Use these receipts when comparing interfaces. Successful tool translation proves
the tested semantics; a controlled live qualification run establishes usability
only for its tested profiles, versions and scenarios. Neither result establishes
universal superiority over another harness.

## Compatibility and recovery

The existing delegation dispatcher and plugin lifecycle API remain supported.
The parent sees one consistent orchestration interface, while legacy callers
continue through the same underlying service.

Worker identity, run checkpoints, ordered messages and completion acknowledgments
remain owned by that service. An interface adapter does not keep a second worker
registry or recovery queue. Restart recovery preserves the existing uncertainty
rule: a possibly completed side effect pauses for reconciliation before another
run can continue.
