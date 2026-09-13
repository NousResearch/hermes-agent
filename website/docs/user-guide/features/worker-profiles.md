---
title: Worker Profiles and Orchestration
sidebar_label: Worker profiles
---

# Worker profiles and orchestration

Worker profiles describe the roles you want Hermes to delegate to:
their purpose, instructions, model, thinking level, tools, and execution limits.
They are independent of provider. You can use one model everywhere or combine
models from different providers. Hermes does not install a prescribed team.

A **Hermes profile** selects an independent configuration and state directory.
A **worker profile** is a delegation definition inside that configuration.
Worker definitions do not inherit live settings from other Hermes profiles.

For a plain-English overview, feature comparison and diagrams, see the
[illustrated orchestration tour](worker-orchestration-tour.md).

## Configure a worker

Use your existing configured provider and its model identifier in a YAML file:

```yaml
description: Inspect supplied files and report concrete inconsistencies.
instructions: Cite the evidence for each finding. Keep the result concise.
provider: openai
model: your-enabled-model
reasoning_effort: high
tool_policy:
  allowed_toolsets: [file]
  blocked_tools: [write_file, patch]
execution_limits:
  max_iterations: 30
workspace_context:
  mode: inherit
  include_context_files: false
  include_memory: false
```

Choose a thinking level supported by your model; omit it to inherit the applicable
default. Tool names must match your installed tool catalog. Instructions such as
“read only” are not permission controls: narrow actual tools and use an enforcing
backend when filesystem isolation is required.

```bash
hermes workers set analyst --file analyst.yaml
hermes workers validate
hermes workers list --json
hermes workers inspect analyst --json
hermes workers default analyst
```

`set` creates or replaces the named definition. `default -` clears the default
worker selection and restores legacy delegation defaults. These commands operate
on the active Hermes profile; the normal `hermes -p NAME` selector still applies.
Listing and validation do not contact providers or prove account availability.
Credentials remain in Hermes's existing provider authentication system; do not
put keys, tokens, or endpoints containing credentials in worker definitions.

### Tools, context, and limits

`allowed_toolsets` selects named tool groups. `allowed_tools` narrows exact tool
names, and `allowed_mcp_tools` adds an exact-name restriction for MCP tools.
`blocked_tools` removes named tools. Use the names reported by your installed
tool catalog. Omitted restrictions inherit the applicable authority; an explicit
empty allowlist grants no tools in its scope. A descendant cannot regain a tool
denied by its parent. The same restrictions apply when a tool is discovered later
or called through `execute_code`.

Worker instructions describe behavior; tool policies govern access. Likewise,
`workspace_context.mode: none` suppresses startup context files and memory, but
does not create a filesystem sandbox. Supported context modes are `inherit` and
`none`. A requested filesystem guarantee needs a backend that can enforce it;
unsupported guarantees are rejected.

Profiles support these execution ceilings:

| Field | Meaning |
| --- | --- |
| `max_iterations` | Positive iteration limit |
| `timeout_seconds` | Positive execution time limit in seconds |
| `max_followups` | Nonnegative number of subsequent assignments; `0` disables follow-ups |
| `max_tool_calls` | Nonnegative tool-call limit; `0` disables tool execution |
| `max_spawn_depth` | Nonnegative descendant-depth limit; `0` disables spawning descendants |
| `max_concurrent_children` | Nonnegative active-child limit; `0` disables child launches |

Omit a limit to inherit the applicable default. Limits narrow the user's and
ancestors' ceilings; they do not authorize additional tools or deeper delegation.
Nesting is optional. Enable it through the existing
`delegation.orchestrator_enabled` and `delegation.max_spawn_depth` settings, then
narrow individual worker profiles as needed.

An assignment to a top-level worker starts a shared execution budget for its tree.
Nested workers share the iteration and tool-call budget and inherit its deadline;
their own profiles may narrow these limits further. Restarting a process does not
replenish that budget. A new explicit assignment to the top-level worker starts a
new budget, while descendants still finishing an older assignment keep the old
one. The owner's active-run concurrency ceiling also applies across these trees.

## Choose the parent's routing freedom

The default `delegation.routing_mode: profile_only` lets the parent select named
profiles. To also permit per-task model selection, enable dynamic routing and
list permitted provider/model combinations:

```yaml
delegation:
  routing_mode: dynamic
  enabled_models:
    - provider: openai
      model: your-enabled-openai-model
    - provider: anthropic
      model: your-enabled-anthropic-model
  profiles:
    analyst:
      description: Review evidence and explain discrepancies.
      provider: openai
      model: your-enabled-openai-model
```

These identifiers are placeholders, not model recommendations. Configure both
providers through the existing Hermes setup/authentication flow first.

Routing chooses within your policy. It cannot add tools, change credentials,
or grant filesystem access. Resolution uses allowed task overrides, the selected
profile, delegation defaults, then parent defaults; user limits apply last.
An invalid profile or route rejects the batch before workers launch.

For example, after configuring the menu above, the parent can request:

```json
{
  "action": "spawn",
  "tasks": [{
    "profile": "analyst",
    "provider": "anthropic",
    "model": "your-enabled-anthropic-model",
    "reasoning_effort": "high",
    "goal": "Check the supplied evidence and explain any inconsistencies."
  }]
}
```

Use an effort that the selected model supports. A profile's optional
`enabled_routes` list can further narrow its choices, including a specific
`reasoning_effort` on a route. It cannot add a provider/model outside the global
enabled menu. An explicit unsupported selection fails with an explanation;
Hermes does not silently choose a cheaper model or lower thinking level.

The parent discovers profiles and model metadata through `delegate_task` rather
than having a large model catalog inserted into every prompt. Missing capability
or availability metadata remains unknown. Requested effort, resolved effort,
transmitted effort, and provider-reported model are separate evidence fields.
Explicit substitutions must follow configured policy and remain visible.

## Work with retained workers

A worker has a durable identity. Every assignment or follow-up creates a separate
run. The parent can inspect status and results, send a message, queue a follow-up,
wait, cancel, and resume a worker with retained conversation context. Steering
is delivered at supported execution boundaries; it does not interrupt an HTTP
request or terminal command instantly.

Only one run executes per worker at a time. Follow-ups queue in order. Worker
permissions and tree limits still apply to nested delegation. Cancellation is
cooperative and reaches owned descendants; a cancellation request does not prove
that an external side effect stopped.

Existing conversations retain their original instructions. Changes to profile
permissions, available tools, credentials, and models are rechecked on resume.
Status and completion results are summaries; inspect the conversation explicitly
when you need detail instead of copying every worker transcript into the parent.
Use `delegate_task(action="inspect", worker_id="...")` for the retained visible
conversation and receipt metadata. Conversation entries include user, assistant,
and tool text, capped at 32,000 characters per entry. System prompts, hidden
reasoning, and provider session objects are excluded. Ownership checks apply to
this inspection just as they do to worker controls.

The parent uses these actions on `delegate_task`:

| Action | Purpose |
| --- | --- |
| `discover` | List worker profiles and routing metadata |
| `spawn` | Start the selected profiles through task items |
| `status` | Read worker/run summaries, optionally selecting `worker_id` and `run_id` |
| `inspect` | Explicitly inspect retained worker details |
| `completions` | List terminal results awaiting acknowledgment |
| `message` | Send `message` to a selected `worker_id` |
| `wait` | Wait for a selected worker/run, bounded by `timeout_seconds` |
| `resume` | Submit `message` as a new assignment with retained conversation |
| `cancel` | Request cancellation of a selected worker/run |
| `reconcile` | Record an explicit decision about an uncertain tool outcome |
| `ack` | Acknowledge receipt of a terminal completion |

Legacy `list`, `steer`, and `stop` actions remain available. Retain the returned
worker and run IDs to target subsequent controls; do not infer identity from a
profile name when several workers use the same profile.

Message acceptance and message consumption are different events. Inspect delivery
state to see whether the message has reached a committed conversation checkpoint.
Sibling messaging is optional (`delegation.allow_sibling_messaging`); it does not
grant permission to resume, cancel, or inspect an unrelated worker.

A worker can send a message back to its parent through the same `message` action.
For a top-level worker, the parent can read `messages_to_parent` through `inspect`,
including while the message is `QUEUED`. Terminal publication changes it to
`PUBLISHED` and includes it with the result. The parent's `ack` of that terminal
run changes published messages to `ACKNOWLEDGED`. These states survive restart;
publication alone does not mean the parent has consumed the message.

## After a restart

Worker conversations, messages, and result delivery state live in the active
Hermes profile's database. Python threads and active requests do not survive a
process exit. Recovery restores durable state and reconciles interrupted runs.

If Hermes cannot tell whether an external tool action completed, the run is
marked interrupted/uncertain. The parent must reconcile that outcome before
continuing; Hermes does not blindly repeat it. Resume creates a new run linked to
the interrupted run. Hermes-owned conversation checkpoints work even when a
provider offers no resumable server-side session.

After checking the external action, record the decision against the latest
uncertain run:

```json
{
  "action": "reconcile",
  "worker_id": "worker-id-from-status",
  "run_id": "interrupted-run-id",
  "reconciliation_disposition": "confirmed_applied",
  "message": "Checked the operation by its external reference; it completed. Do not repeat it."
}
```

The other dispositions are `confirmed_not_applied` and
`accepted_unknown_no_replay`. The latter records that the outcome remains unknown;
it is not verification. Supply a nonempty note describing the evidence or decision.
Reconciliation records the original tool-call identities and prior statuses without
executing the tool. Then use a separate `resume` action with the next assignment.

Queued assignments are revalidated immediately before launch, including those
queued before a restart or a configuration change. A saved provider session handle
does not replace Hermes-owned conversation state or current authorization.

Delivery acknowledgments prevent duplicate internal consumption. An external
messaging transport may still provide at-least-once delivery; this feature does
not manufacture exactly-once guarantees for remote services.

## Compatibility

With no worker profiles configured, existing delegation defaults remain valid.
The provider/model you select does not replace the Hermes harness. Choosing an
OpenAI Codex model still uses Hermes orchestration; running the actual Codex
app-server is a separate runtime integration.
