---
sidebar_position: 16
title: Kanban recipes
---

# Kanban recipes

A recipe is a reusable JSON definition that creates a fresh graph of native
Kanban tasks. Each invocation stores its definition, inputs, resolved plan and
task mapping atomically. The existing dispatcher claims the tasks.

## Start a graph

Save this as `brief.json`:

```json
{
  "schema_version": 1,
  "recipe_id": "research-brief",
  "inputs": {"topic": {"type": "string", "required": true}},
  "roles": ["researcher", "writer"],
  "nodes": [
    {"key": "research", "role": "researcher", "title": "Research {{input.topic}}", "body": "Write cited findings."},
    {"key": "brief", "role": "writer", "title": "Prepare brief", "body": "Read the parent handoff and summarize supported claims.", "needs": ["research"]}
  ]
}
```

Save `inputs.json` as `{"topic":"SQLite transactions"}` and `bindings.json` as
`{"roles":{"researcher":"default","writer":"default"}}`. Roles may bind to
different installed profiles. `hermes profile list` lists available profiles.

```bash
hermes kanban --board default recipe validate brief.json --inputs inputs.json --bindings bindings.json --json
hermes kanban --board default recipe run brief.json --inputs inputs.json --bindings bindings.json --key brief-request-001 --json
```

Validation does not initialize or migrate a board, create workspaces, or call a
model. Run creates the entire graph in one transaction. Roots become ready
immediately, so validate is the preview command, not a paused run.

The result contains `instance_id`, `replayed`, `definition_digest`,
`request_digest`, and `tasks`, a node-key-to-task-ID map. Save the result.

```bash
hermes kanban --board default recipe show <instance-id> --json
hermes kanban --board default recipe export <instance-id> --output portable.json
```

Export writes only the portable definition. It excludes inputs, bindings, task
IDs, paths, claims and execution evidence. An existing output file requires
`--overwrite`. The same commands work through `/kanban`.

## Definition rules

- **Identity:** recipe, input, role and node identifiers match
  `[a-z][a-z0-9_-]{0,63}` exactly. Names are not trimmed or case-normalized.
- **Graph:** each node requires `key`, `role` and `title`. Optional `body`,
  `needs` and `task` default to empty values. Dependencies name other nodes.
  Duplicate keys/edges, unknown references and cycles are errors.
- **Inputs:** types are `string`, `integer`, `boolean`, `object` and `array`.
  Declarations may set `required` or `default`, not both required=true and a
  default. Unknown inputs fail. Missing optional inputs stay absent unless a
  default exists. Null is only allowed inside objects/arrays.
- **Interpolation:** only title/body accept `{{input.NAME}}`. Only strings,
  integers and booleans can substitute. `{{{{` escapes literal `{{`. There is no
  expression language or second evaluation of input text. Objects and arrays
  appear in a separate untrusted-data section in worker context.
- **JSON:** UTF-8 only, with duplicate object keys, invalid Unicode, non-finite
  numbers, unknown fields and trailing content rejected. Integers must be
  integer tokens in the safe JSON range, not booleans, strings, fractions or
  exponent notation.

The loader executes no code or shell interpolation. Task bodies still instruct
agents after dispatch. Inspect untrusted recipes before running them, and keep
credentials out of inputs because receipts retain invocation data.

## Native task controls

| Control | Recipe policy |
|---|---|
| workspace_kind | scratch or worktree |
| priority | Integer -2147483648 through 2147483647, default 0 |
| max_runtime_seconds | Integer 1 through 604800, omitted means native unset |
| max_retries | Integer 1 through 100, omitted means native failure policy |
| goal_mode | Boolean, default false |
| goal_max_turns | Integer 1 through 1000, only with explicit goal_mode=true |
| model, provider | Native task overrides, provider requires a nonempty model |
| reasoning_effort, skills, completion_contract | Existing native validators |

With goal mode enabled and turns omitted, creation freezes the native goal
default. Omitted runtime/retry controls retain native unset semantics, not an
immutable snapshot of future dispatcher policy. Explicit null controls fail.
`max_retries=1` is the native first-failure threshold, not one additional retry.

Bindings contain required `roles` and optional `project` and `tenant`. A project
ID/slug resolves through the native project registry, or the board's project
default. Project-linked tasks get independent native worktrees and branches.
A worktree without a resolvable project fails. Shared `workspace_path`, `dir`
workspaces and recipe-embedded project paths are unsupported in version 1.

Profiles must exist before creation. If a bound profile disappears before
dispatch, recipe tasks block with a diagnostic instead of silently rebinding.
Profile credentials/configuration are not copied. Explicit task overrides remain
pinned, otherwise ordinary profile configuration applies at dispatch.

## Retry and recovery

A run key is mandatory, contains 1-128 printable non-whitespace ASCII characters,
and is unique across the board. Different keys always create fresh identities.
Identical retries return the stored mapping, even after archive or changes to
ambient defaults. Changed explicit inputs, bindings or definition under the same
key return `IDEMPOTENCY_CONFLICT`.

A process lost before commit leaves no graph. A process lost after commit can be
recovered with the same key. On storage/uncertain-outcome errors, inspect or replay
that key before attempting new work. Missing membership is `INSTANCE_DAMAGED`,
not permission to reconstruct part of an invocation.

| Exit | Meaning |
|---|---|
| 0 | Successful validation, run, show or export |
| 2 | Invalid definition, binding or output request |
| 3 | Identity conflict, missing/damaged instance or imported-key refusal |
| 4 | Storage failure or uncertain outcome, inspect/replay before retry |
| 5 | Authority or board-context denial |

With `--json`, failures contain `error` with `code`, `path`, `message` and
`retryable`. Explicit board, context and worker DB selections must agree.
Delegated child contexts cannot instantiate recipes.

## Limits and compatibility

Definitions are limited to 1 MiB, 256 nodes, 2048 edges, 64 roles and 64 inputs.
JSON nesting is limited to 16 containers. Canonical invocation inputs are at
most 64 KiB, rendered titles 1024 UTF-8 bytes, bodies 64 KiB, and the effective
plan 2 MiB.

Task statuses, all-parent dependencies, same-card review, goal behavior and PR
completion contracts remain native. Existing readiness accepts archived parents
as well as done parents. Archive is not proof that a producer passed acceptance.
Version 1 rejects custom statuses, schedules, gates, protected reviewer policy,
outputs, joins and cycles rather than storing declarations it cannot enforce.

Board export/import still transfers execution history. Imported recipe receipts
retain origin digests and plans, while active member tasks are parked in triage
before destination dispatch. Historical local paths may remain in the immutable
plan but are not reused as runtime authorization. An imported key cannot replay.
Export its definition and run with a new key and local bindings. Version 1 has
no resume/rebind surface for imported instances.
