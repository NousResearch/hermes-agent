---
sidebar_position: 11
title: "Workflow Graph Engine"
description: "Durable declarative graphs for multi-step orchestration."
---

# Workflow Graph Engine

Workflows are durable, declarative graphs for multi-step orchestration. A workflow definition is YAML, deployed into `~/.hermes/workflows.db`, then advanced by the workflow dispatcher. Nodes can transform data locally, branch, wait, fan out, and create durable Kanban agent tasks.

Workflows are not arbitrary scripts. Conditions and templates use a small data-path DSL; they do not run Python, shell, JavaScript, or model-generated code.

:::note Current CLI name
The command is singular today: `hermes workflow ...`.
:::

## Workflows vs Kanban vs Cron vs `delegate_task`

| Primitive | Best for | Durable? | Runs agents? | Main interface |
|---|---|---:|---:|---|
| `delegate_task` | Short fork/join subagent work inside the current turn | No | Yes, in-process child agents | `delegate_task` tool |
| Kanban | Long-lived task queue across profiles, humans, retries, and workspaces | Yes | Yes, as full worker processes | `hermes kanban`, `/kanban`, Kanban tools |
| Cron | Time-based prompts or scripts | Yes | Yes for prompt jobs; no for `--no-agent` scripts | `cronjob`, `hermes cron`, `/cron` |
| Workflows | A declarative graph with branching, waits, fan-out, and Kanban agent steps | Yes | Only through `agent_task` nodes | `hermes workflow`, Workflows dashboard |

Use a workflow when the shape matters: "do A, run B and C, wait for both, route on reviewer output, then either finish or create a revision task." Use Kanban directly when you just need a queue of work cards. Use Cron when the main question is "when should this run?" Use `delegate_task` when the parent agent needs a quick answer before continuing.

Workflows can include schedule triggers, but scheduled workflows still run through the workflow dispatcher and workflow state database. Cron jobs remain the general-purpose scheduled prompt/script system.

## Quick start

The examples live under `examples/workflows/`; this quick start uses `examples/workflows/code-change-review.yaml`.

That example leaves `agent_task.workspace_kind` unset, so Kanban uses scratch workspaces; each prompt passes `repo` and tells the worker to use that path. Set `agent_task.workspace_kind`/`workspace_path` on a workflow node when that node should run in a specific worktree or directory. To pin the default dispatcher working directory for a board, set board metadata with `hermes kanban boards create <slug> --default-workdir <path>` or `hermes kanban boards set-default-workdir <slug> <path>`.

Validate and deploy it:

```bash
hermes workflow validate examples/workflows/code-change-review.yaml
hermes workflow deploy examples/workflows/code-change-review.yaml
hermes workflow list
hermes workflow show code-change-review --json
```

Start a manual execution. `--input` must point to a JSON file containing an object:

```bash
cat > /tmp/workflow-input.json <<'JSON'
{
  "repo": "/home/me/project",
  "branch": "feature/workflow-demo",
  "change_request": "Add input validation and tests for the signup endpoint."
}
JSON

hermes workflow run code-change-review --input /tmp/workflow-input.json --json
```

The CLI run command creates a queued execution. Advance queued local nodes manually with:

```bash
hermes workflow tick --limit 10 --json
```

For unattended dispatch, run the gateway dispatcher instead:

```bash
hermes config set workflow.dispatch_in_gateway true
hermes config set workflow.tick_interval_seconds 30
hermes config set workflow.max_executions_per_tick 50
hermes gateway restart
```

`workflow.dispatch_in_gateway` defaults to `false`, so a gateway will not tick workflows until you opt in.

## Build workflows with Hermes

You do not have to write a workflow definition by hand. Ask Hermes to use the
`hermes-workflow-builder` skill:

```text
Use hermes-workflow-builder to create a workflow that researches a topic,
reviews source quality, summarizes findings, and routes high-value results to
an analyst profile.
```

The skill decomposes the goal into workflow cells, writes text-first cell
prompts, creates YAML under `.hermes/workflows/`, validates it with
`hermes workflow validate`, and can deploy it when you approve.

## CLI reference

```bash
hermes workflow init
hermes workflow validate <file.yaml>
hermes workflow deploy <file.yaml> [--json]
hermes workflow list [--json]
hermes workflow show <workflow_id> [--json]
hermes workflow run <workflow_id> [--input <input.json>] [--json]
hermes workflow executions list [--workflow <workflow_id>] [--json]
hermes workflow executions show <execution_id> [--json]
hermes workflow executions cancel <execution_id>
hermes workflow tick [--limit N] [--json]
```

There is no standalone `hermes workflow events` CLI command yet. Use `hermes workflow executions show ... --json` for execution state, or inspect the dashboard timeline for recorded events.

## Dashboard workflow screen

Open the dashboard and select the **Workflows** tab:

```bash
hermes dashboard
```

The bundled `workflows` dashboard plugin mounts at `/workflows`, after Kanban in the sidebar. It provides:

- workflow definition list
- validate/deploy editor for YAML, plus import/export/copy
- manual run form with JSON input
- execution list
- execution detail timeline from `workflow_events`
- React Flow visual graph editor for nodes and edges, with an HTML fallback when React Flow is unavailable
- node inspector that can edit JSON drafts; YAML drafts stay untouched until you convert to JSON

The dashboard run action starts an execution and nudges one dispatcher tick. Long-running `agent_task` nodes still wait for their Kanban task to complete, just like CLI-started executions.

## Workflow definition YAML reference

A workflow file is a YAML object with these supported top-level fields:

| Field | Required | Meaning |
|---|---:|---|
| `id` | Yes | Stable id, lowercase letters/digits/underscore/hyphen, starts with a letter, max 64 chars. |
| `name` | Yes | Human-readable name. |
| `version` | Yes | Integer version, minimum `1`. Deployed definitions are keyed by `id` + `version`. |
| `enabled` | No | Defaults to `true`. Disabled definitions deploy but do not create schedule rows. |
| `max_node_runs` | No | Loop guard, defaults to `500`. |
| `triggers` | No | List of trigger specs. |
| `nodes` | Yes | Mapping of node id to node spec. Node ids use lowercase letters/digits/underscore/hyphen, start with a letter, max 64 chars. |
| `edges` | No | List of `{from, to}` edges. |

Unknown keys may validate because the internal models allow forward-compatible extras, but only the fields documented here have runtime behavior. Do not rely on unknown keys in production workflows.

### Triggers

| Type | Fields | Runtime behavior |
|---|---|---|
| `manual` | `id`, `input` | Run with `hermes workflow run` or the dashboard run form. The CLI/dashboard run input is the execution input; trigger `input` is for external launchers and is not merged by the CLI. |
| `schedule` | `id`, `cron` or `schedule`, `input` | Deploy creates a row in `workflow_schedules`; dispatcher starts executions when due. Uses cron syntax accepted by `croniter`. |
| `webhook` | `id`, `input` | Schema-accepted for future/external launchers; no built-in webhook launcher in this CLI surface yet. |
| `kanban_event` | `id`, `input` | Schema-accepted for future/external launchers; not wired to Kanban events by default. |

Schedule trigger example:

```yaml
triggers:
  - type: schedule
    id: weekday_research
    cron: "0 9 * * 1-5"
    input:
      topic: "AI agent research"
      min_score: 0.75
```

### Edges and ports

Plain edges run after a node succeeds:

```yaml
edges:
  - from: start
    to: review
```

`switch` and `parallel` use dotted source ports:

```yaml
edges:
  - from: route.approved
    to: success
  - from: route.default
    to: revise
  - from: fork.research
    to: gather
```

A dotted edge source is valid only from a `switch` or `parallel` node. A `parallel` node must use dotted branch edges; `from: fork` is rejected.

Root nodes are the nodes with no incoming edge, no `default` target, and no `catch` target. Multiple roots are allowed.

## Node type reference

Common fields on node specs:

| Field | Meaning |
|---|---|
| `type` | One of `pass`, `switch`, `agent_task`, `wait`, `parallel`, `join`, `send_message`, `fail`, `subworkflow`. |
| `catch` | Node id to run after supported node execution failures once retries are exhausted. Some validation, render, or setup errors can fail the execution directly. |
| `retry` | `{max_attempts, delay_seconds, backoff_seconds, multiplier}` for retrying supported node execution failures. Some validation, render, or setup errors are not retryable node attempts. |

The schema also accepts `workspace: {cwd, env}` for forward compatibility, but the bundled dispatcher does not use it today. For Kanban workers, use `agent_task.workspace_kind` and `agent_task.workspace_path`.

### `pass`

Computes a local output with safe templates and immediately continues.

```yaml
start:
  type: pass
  output:
    topic: "${ input.topic }"
    workflow_id: "${ workflow.id }"
```

The output is stored at `$.node.<node_id>.output`.

### `switch`

Evaluates cases in order. The first true case selects a port with the case `name`. If no case matches, the selected port is `default`; you can route that with an edge from `route.default` or set `default: <node_id>`.

```yaml
route:
  type: switch
  cases:
    - name: approved
      when:
        op: eq
        left: {path: "$.node.review.output.verdict"}
        right: approved
edges:
  - from: route.approved
    to: success
  - from: route.default
    to: revise
```

### `agent_task`

Creates a Kanban task and waits for it. Required fields are `profile` and `prompt`.

```yaml
review:
  type: agent_task
  profile: reviewer
  title: "Review implementation"
  skills: [github-code-review]
  workspace_kind: worktree
  max_retries: 2
  goal_mode: true
  goal_max_turns: 10
  prompt:
    task: "Review this implementation. Complete with JSON."
    implementation: "${ node.implement.output }"
```

Supported `agent_task` fields:

| Field | Meaning |
|---|---|
| `profile` | Kanban assignee profile. Required. |
| `prompt` | String/list/object rendered with safe templates, then used as the Kanban task body. Required. |
| `title` | Kanban task title. Defaults to `<workflow name>: <node id>`. |
| `workspace_kind` | Passed to Kanban, e.g. `scratch` or `worktree`. Defaults to `scratch` when omitted. |
| `workspace_path` | Optional Kanban workspace path. This field is not templated. |
| `skills` | Skills to load for the worker. |
| `model_override` | Optional model override stored on the Kanban task. |
| `max_retries` | Kanban task retry limit. |
| `goal_mode` | Run the Kanban worker in goal mode. |
| `goal_max_turns` | Goal-mode turn budget. |

When the Kanban task reaches `done`, the workflow resumes. If the task result or latest summary is valid JSON, that object becomes the node output. Plain text becomes `{"result": "..."}`. If the Kanban task is blocked, the workflow execution becomes `blocked` with the block reason.

### `wait`

Pauses the workflow until a future dispatcher tick.

```yaml
cooldown:
  type: wait
  seconds: 300
```

After the wait, the node output is `{"waited": true}`.

### `parallel` and `join`

`parallel` fans out along dotted branch edges. `join` waits until all reachable branch work has completed, then stores branch outputs under `$.node.<join_id>.output.branches`.

```yaml
fork:
  type: parallel
gather:
  type: agent_task
  profile: researcher
  prompt: "Gather sources."
scan:
  type: agent_task
  profile: researcher
  prompt: "Scan for risks."
merge:
  type: join
edges:
  - from: fork.gather
    to: gather
  - from: fork.scan
    to: scan
  - from: gather
    to: merge
  - from: scan
    to: merge
```

### `fail`

Emits a node execution failure; if retries are exhausted, a configured `catch` may run.

```yaml
reject:
  type: fail
  output:
    reason: "${ node.review.output.reason }"
```

The error payload includes the node id, type `fail`, and rendered output.

### `send_message` and `subworkflow`

These node types are schema-accepted waiting nodes. The bundled dispatcher does not yet include a built-in message sender or subworkflow runner to complete them, so they are extension points. For normal user workflows, prefer `agent_task`, `pass`, `wait`, `parallel`, `join`, `switch`, and `fail`.

## Condition DSL reference

Conditions are YAML objects. There is no Python `eval`; unsupported operators raise validation errors during execution.

### Paths

Paths start with `$.` and read from workflow context:

```text
$.input.topic
$.workflow.id
$.node.review.output.verdict
$.node.gather.output.items[0].url
$.branches.fork.research
```

Only mapping keys, dots, and numeric list indexes are supported. Missing paths make comparison/string operators return `false`; `missing` can test for them explicitly.

### Values

`left`, `right`, and `arg` values may be literals or a path object:

```yaml
left: {path: "$.node.review.output.score"}
right: {path: "$.input.min_score"}
```

A mapping is treated as a path only when it is exactly `{path: "$.some.path"}`.

### Operators

| Operator | Shape | Meaning |
|---|---|---|
| `and` | `args: [cond, ...]` | All child conditions true. All children are evaluated. |
| `or` | `args: [cond, ...]` | Any child condition true. All children are evaluated. |
| `not` | `arg: cond` or one-item `args` | Negates a child condition. |
| `exists` | `path`, `arg`, or `left` | True when the value resolves. `null` counts as existing. |
| `missing` | `path`, `arg`, or `left` | True when the value does not resolve. |
| `eq`, `ne` | `left`, `right` | Equality / inequality. |
| `gt`, `gte`, `lt`, `lte` | `left`, `right` | Numeric or naturally comparable values. Type errors return false. |
| `contains` | `left`, `right` | Python-style membership, e.g. substring or list membership. Type errors return false. |
| `starts_with`, `ends_with` | `left`, `right` | String prefix/suffix checks. |
| `regex` | `left`, `right` | Regex search on strings; invalid regex raises an error. |

Example:

```yaml
when:
  op: and
  args:
    - op: eq
      left: {path: "$.node.review.output.verdict"}
      right: approved
    - op: gte
      left: {path: "$.node.review.output.confidence"}
      right: 0.8
```

## Template reference

Templates are also path-only. A string is templated only when the whole string is a `${ ... }` expression:

```yaml
output:
  ok: "${ node.review.output.verdict }"       # replaced
  text: "Verdict: ${ node.review.output.verdict }"  # literal string, not interpolated
```

Inside `${ ... }`, the leading `$.` is optional. Lists and objects are rendered recursively. A missing path raises an error; depending where rendering happens, it may fail the execution directly rather than route through `catch`.

This is intentional safety: templates can only copy values out of the workflow context. They cannot call functions, read files, run shell commands, or evaluate code.

## Safety model

- Definitions are data stored in SQLite, not executable scripts.
- Conditions use an explicit operator allowlist.
- Templates only resolve dotted data paths.
- Agent work happens by creating Kanban tasks; the workflow engine itself does not run terminal commands.
- Unknown YAML keys may be accepted for forward compatibility but are not behavior.
- The dispatcher uses claim locks and idempotent Kanban task creation so repeated ticks do not duplicate `agent_task` cards.

Normal Hermes tool safety still applies inside the Kanban worker that picks up an `agent_task` card: profile config, toolsets, command approvals, workspace selection, and model behavior all come from Kanban/agent runtime.

## How `agent_task` maps to Kanban

When a workflow reaches an `agent_task` node, the dispatcher creates or reuses a Kanban task with:

| Kanban field | Source |
|---|---|
| `title` | node `title`, or `<workflow name>: <node id>` |
| `body` | rendered node `prompt` |
| `assignee` | node `profile` |
| `workspace_kind`, `workspace_path` | node fields |
| `skills`, `max_retries`, `goal_mode`, `goal_max_turns`, `model_override` | node fields |
| `created_by` | `workflow:<execution_id>` |
| `workflow_template_id` | workflow `id` |
| `current_step_key` | node id |
| `idempotency_key` | `workflow:<execution_id>:<node_id>` |

The workflow execution stays `waiting` while the Kanban task is open. On later ticks:

- `done` task → node succeeds and its parsed result becomes `$.node.<node_id>.output`.
- `blocked` task → workflow execution becomes `blocked` and records the reason.
- missing/still-running task → execution remains waiting.

You can inspect workflow-created Kanban cards directly:

```bash
hermes kanban list --workflow-template-id code-change-review
hermes kanban list --workflow-template-id code-change-review --step-key review
```

## Examples

Two copyable examples ship in the repo:

```bash
hermes workflow validate examples/workflows/code-change-review.yaml
hermes workflow validate examples/workflows/research-triage.yaml
```

Before deploying them, edit the `profile` names (`implementer`, `reviewer`, `researcher`, `analyst`) to match profiles installed on your machine.
