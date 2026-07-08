---
sidebar_position: 11
title: "Workflows"
description: "Prompt-first durable automations for multi-step orchestration."
---

# Workflows

Workflows are named, versioned automations that Hermes can draft from a plain-language goal, validate, deploy, and run durably. You normally start by describing what you want to automate; YAML/JSON remains available as an advanced import/export/debug format.

Workflows are not arbitrary scripts. Conditions and templates use a small data-path DSL; they do not run Python, shell, JavaScript, or model-generated code.

:::note Current CLI name
The command is singular today: `hermes workflow ...`.
:::

## Build from plain language

Start with the outcome, not a YAML file. In the dashboard or through workflow draft tooling, describe:

- the goal you want automated
- inputs a run should ask for
- profiles or workers that should handle each part
- review/routing rules
- expected final output

Example prompt:

```text
Draft a workflow that researches a topic, scores source quality, sends high-value findings to an analyst profile, and produces a concise final report.
```

Hermes drafts a workflow spec from that goal. Review the draft, ask for corrections with workflow refine tooling, validate it, then deploy. If the assistant provider fails or returns an invalid draft, dashboard/API responses return typed, privacy-safe remediation hints so you can retry, adjust the request, or switch to Advanced YAML without exposing provider secrets. Use YAML/JSON only when you are importing, exporting, debugging, or intentionally authoring the definition by hand.

## When to use Workflows vs Kanban

| Primitive | Best for | Durable? | Runs agents? | Main interface |
|---|---|---:|---:|---|
| `delegate_task` | Short fork/join subagent work inside the current turn | No | Yes, in-process child agents | `delegate_task` tool |
| Kanban | Long-lived task queue across profiles, humans, retries, and workspaces | Yes | Yes, as full worker processes | `hermes kanban`, `/kanban`, Kanban tools |
| Cron | Time-based prompts or scripts | Yes | Yes for prompt jobs; no for `--no-agent` scripts | `cronjob`, `hermes cron`, `/cron` |
| Workflows | A named graph with branching, waits, fan-out, and Kanban agent steps | Yes | Only through `agent_task` nodes | Dashboard Workflows tab, `hermes workflow` |

Use a workflow when the shape matters: "do A, run B and C, wait for both, route on reviewer output, then either finish or create a revision task." Use Kanban directly when you just need a queue of work cards. Use Cron when the main question is "when should this run?" Use `delegate_task` when the parent agent needs a quick answer before continuing.

Workflows can include schedule triggers, but scheduled workflows still run through the workflow dispatcher and workflow state database. Cron jobs remain the general-purpose scheduled prompt/script system.

## Create a workflow in the dashboard

Open the dashboard and select the **Workflows** tab:

```bash
hermes dashboard
```

The bundled `workflows` dashboard plugin mounts at `/workflows`, after Kanban in the sidebar. The dashboard flow is prompt-first:

1. Use **Describe workflow**.
2. Describe the automation in plain language.
3. Add known inputs, profiles, constraints, and expected outputs.
4. Let Hermes create the draft spec.
5. Review the generated graph before validating or deploying.

The screen also provides definition list, import/export/copy controls, a visual graph view with an HTML fallback, and Advanced JSON/YAML panels for debugging or manual edits.

## Review and refine the draft

Treat the generated spec as a draft. Review:

- workflow id, name, version, and enabled state
- manual or scheduled trigger inputs
- each cell objective and node type
- profile assignment for each `agent_task`
- text-first prompts with `${ input.foo }` and `${ node.cell.output.field }` placeholders
- explicit `result_contract` mappings for downstream `agent_task` output keys
- switch cases, parallel branches, joins, waits, and terminal states
- workspace settings when a worker must use a specific worktree or directory

Use `workflow_refine` from the dashboard/API when the draft is close but needs corrections, such as "make the reviewer run before deployment" or "add a blocked path when confidence is below 0.8". The useful cell-table view is a review aid, not a requirement to hand-write YAML first:

| Cell id | Type | Profile | Objective | Output contract | Next |
| --- | --- | --- | --- | --- | --- |
| research | `agent_task` | researcher | Gather sources | `{ "sources": [...] }` | score |
| score | `agent_task` | reviewer | Score quality | `{ "approved": true/false }` | route |

For a single agent cell, the Prompt assistant can draft or improve the text prompt from the workflow goal, cell objective, available context placeholders, output contract, and constraints. Review the prompt before applying it.

## Validate and deploy

Always validate before deploy. The validation surface checks deployable workflow shape, including:

- required top-level fields (`id`, `name`, `version`, `nodes`)
- node ids and workflow id are stable and valid
- every referenced edge target exists
- switch and parallel edges use dotted ports where required
- the graph is acyclic
- each `agent_task` has `profile` and `prompt`
- only currently implemented primitives are used (`manual`/`schedule` triggers and `pass`, `switch`, `agent_task`, `wait`, `parallel`, `join`, `fail` nodes)
- optional `result_contract` entries use enforced flat types (`string`, `number`, `boolean`, `array`, `object`) or enum strings such as `approved|rejected`

Also review the draft manually before deploy for semantic issues that graph validation cannot fully prove yet, such as whether prompt placeholders refer to the intended input/upstream node output and whether scheduled trigger strings express the schedule you meant.

The dashboard shows validation results before deployment. It also shows a dispatcher readiness banner so you can see whether gateway dispatch is enabled; deployed workflows will not advance automatically until the dispatcher is ready.

Deploy only after validation is green. After deployment, verify the deployed definition is visible by reopening it in the dashboard or running:

```bash
hermes workflow show <workflow_id> --json
```

## Run a test execution

For a manual trigger, the dashboard generates a run form from the workflow's expected input shape. Fill the form, start the test run, and watch the execution enter the queue. Workflow inputs and outputs are stored locally in workflow state and Kanban history. Do not paste secrets; common secret-looking keys are redacted in dashboard display responses, but raw execution data remains in local storage for dispatcher correctness.

The equivalent CLI path uses a JSON object as input:

```bash
hermes workflow run <workflow_id> --input /path/to/input.json --json
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

## Monitor execution and linked worker tasks

Use the dashboard execution list to find a run, then open execution detail. The detail view shows the event timeline, node-runs, node outputs, waits, failures, and current status.

When a workflow reaches an `agent_task` node, Hermes creates or reuses a linked Kanban worker task. The execution detail node-runs drill-down shows `Linked worker task: <id>` and worker status when a run is waiting on an agent. Use that task id in Kanban views or CLI commands to inspect the worker card, logs, result, blocked reason, workspace, and assignee. Long-running `agent_task` nodes wait for their Kanban task to complete before the workflow resumes.

CLI inspection commands:

```bash
hermes workflow executions list --workflow <workflow_id> --json
hermes workflow executions show <execution_id> --json
hermes kanban list --workflow-template-id <workflow_id>
hermes kanban list --workflow-template-id <workflow_id> --step-key <node_id>
```

There is no standalone `hermes workflow events` CLI command yet. Use `hermes workflow executions show ... --json` for execution state, or inspect the dashboard timeline for recorded events.

## Advanced: YAML definitions

YAML/JSON definitions remain the portable representation for import, export, debugging, code review, and manual advanced authoring. Most users should draft and refine in the dashboard first.

The examples live under `examples/workflows/`:

```bash
hermes workflow validate examples/workflows/code-change-review.yaml
hermes workflow validate examples/workflows/research-triage.yaml
```

Before deploying examples, edit the `profile` names (`implementer`, `reviewer`, `researcher`, `analyst`) to match profiles installed on your machine.

A minimal manual definition looks like this:

```yaml
id: research-triage
name: Research triage
version: 1
triggers:
  - type: manual
    id: manual
nodes:
  start:
    type: pass
    output:
      topic: "${ input.topic }"
  research:
    type: agent_task
    profile: researcher
    title: Research topic
    result_contract:
      summary: string
      sources: array
    prompt: |
      You are the researcher profile executing workflow cell research.
      Research: ${ node.start.output.topic }
      Return JSON only: {"summary": "string", "sources": ["url"]}
edges:
  - from: start
    to: research
```

Definitions are data stored in SQLite after deploy, not executable scripts. Unknown YAML keys may validate because the internal models allow forward-compatible extras, but only the fields documented here have runtime behavior. Do not rely on unknown keys in production workflows.

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

## Schema reference

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

### Triggers

| Type | Fields | Runtime behavior |
|---|---|---|
| `manual` | `id`, `input` | Run with `hermes workflow run` or the dashboard run form. The CLI/dashboard run input is the execution input; trigger `input` is for external launchers and is not merged by the CLI. |
| `schedule` | `id`, `cron` or `schedule`, `input` | Deploy creates a row in `workflow_schedules`; dispatcher starts executions when due. Uses cron syntax accepted by `croniter`. |
| `webhook` | `id`, `input` | Declared for future/external launchers; built-in validate/deploy reject it until a launcher is available. |
| `kanban_event` | `id`, `input` | Declared for future/external launchers; built-in validate/deploy reject it until Kanban event launching is available. |

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

### Node type reference

Common fields on node specs:

| Field | Meaning |
|---|---|
| `type` | Declared node type. Built-in validate/deploy currently accept `pass`, `switch`, `agent_task`, `wait`, `parallel`, `join`, and `fail`; declared future nodes such as `send_message` and `subworkflow` are rejected until implemented. |
| `catch` | Node id to run after supported node execution failures once retries are exhausted. Some validation, render, or setup errors can fail the execution directly. |
| `retry` | `{max_attempts, delay_seconds, backoff_seconds, multiplier}` for retrying supported node execution failures. Some validation, render, or setup errors are not retryable node attempts. |

The schema also accepts `workspace: {cwd, env}` for forward compatibility, but the bundled dispatcher does not use it today. For Kanban workers, use `agent_task.workspace_kind` and `agent_task.workspace_path`.

#### `pass`

Computes a local output with safe templates and immediately continues.

```yaml
start:
  type: pass
  output:
    topic: "${ input.topic }"
    workflow_id: "${ workflow.id }"
```

The output is stored at `$.node.<node_id>.output`.

#### `switch`

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

#### `agent_task`

Creates a Kanban task and waits for it. Required fields are `profile` and `prompt`.

```yaml
review:
  type: agent_task
  profile: reviewer
  provider: openai-codex       # optional; omit to use profile default
  model: gpt-5.5               # optional; omit to use profile default
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
| `provider` | Optional provider override for this cell. Passed to the Kanban worker as `--provider`; omit to use the selected profile's default provider. Legacy `provider_override` is still accepted on input. |
| `model` | Optional model override for this cell. Passed to the Kanban worker as `-m`; omit to use the selected profile's default model. Legacy `model_override` is still accepted on input. |
| `prompt` | String/list/object rendered with safe templates, then used as the Kanban task body. Required. |
| `result_contract` | Optional mapping of required output keys to flat types (`string`, `number`, `boolean`, `array`, `object`) or enum strings like `approved|rejected`. Dispatcher blocks the cell if the completed Kanban result does not match. |
| `title` | Kanban task title. String titles support `${ ... }` template placeholders like prompts. Defaults to `<workflow name>: <node id>`. |
| `workspace_kind` | Passed to Kanban, e.g. `scratch` or `worktree`. Defaults to `scratch` when omitted. |
| `workspace_path` | Optional Kanban workspace path. This field is not templated. |
| `skills` | Skills to load for the worker. |
| `model_override` | Legacy alias for `model`; prefer `model` in new workflow specs. |
| `max_retries` | Kanban task retry limit. |
| `goal_mode` | Run the Kanban worker in goal mode. |
| `goal_max_turns` | Goal-mode turn budget. |

A workflow may mix providers/models across cells. `profile` still selects the worker identity and profile-scoped config; `provider` and `model` only override inference routing for that one worker process. Deploy validation does not check credentials because credentials are profile-local and resolved when the worker starts.

When the Kanban task reaches `done`, the workflow resumes. If the task result or latest summary is valid JSON, that object becomes the node output. Plain text becomes `{"result": "..."}`. If `result_contract` is present, the dispatcher requires every listed key to be present and to match its declared type or enum before the cell succeeds. Contract failures block the workflow instead of silently routing with malformed data. If the Kanban task is blocked, the workflow execution becomes `blocked` with the block reason.

#### `wait`

Pauses the workflow until a future dispatcher tick.

```yaml
cooldown:
  type: wait
  seconds: 300
```

After the wait, the node output is `{"waited": true}`.

#### `parallel` and `join`

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

#### `fail`

Emits a node execution failure; if retries are exhausted, a configured `catch` may run.

```yaml
reject:
  type: fail
  output:
    reason: "${ node.review.output.reason }"
```

The error payload includes the node id, type `fail`, and rendered output.

### Condition DSL reference

Conditions are YAML objects. There is no Python `eval`; unsupported operators raise validation errors during execution.

#### Paths

Paths start with `$.` and read from workflow context:

```text
$.input.topic
$.workflow.id
$.node.review.output.verdict
$.node.gather.output.items[0].url
$.branches.fork.research
```

Only mapping keys, dots, and numeric list indexes are supported. Missing paths make comparison/string operators return `false`; `missing` can test for them explicitly.

#### Values

`left`, `right`, and `arg` values may be literals or a path object:

```yaml
left: {path: "$.node.review.output.score"}
right: {path: "$.input.min_score"}
```

A mapping is treated as a path only when it is exactly `{path: "$.some.path"}`.

#### Operators

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

### Template reference

General workflow templates are path-only. Outside `agent_task` prompt text, a string is templated only when the whole string is a `${ ... }` expression:

```yaml
output:
  ok: "${ node.review.output.verdict }"       # replaced
  text: "Verdict: ${ node.review.output.verdict }"  # literal in general templates
```

Inside `${ ... }`, the leading `$.` is optional. Lists and objects are rendered recursively. A missing path raises an error; depending where rendering happens, it may fail the execution directly rather than route through `catch`.

`agent_task` prompts use `workflows_prompts.py` instead: text prompts render `${ ... }` placeholders inline, and list/object prompts keep backward-compatible recursive rendering before they become the Kanban task body.

This is intentional safety: templates can only copy values out of the workflow context. They cannot call functions, read files, run shell commands, or evaluate code.

### Safety model

- Definitions are data stored in SQLite, not executable scripts.
- Conditions use an explicit operator allowlist.
- Templates only resolve dotted data paths.
- Agent work happens by creating Kanban tasks; the workflow engine itself does not run terminal commands.
- Unknown YAML keys may be accepted for forward compatibility but are not behavior.
- The dispatcher uses claim locks and idempotent Kanban task creation so repeated ticks do not duplicate `agent_task` cards.

Normal Hermes tool safety still applies inside the Kanban worker that picks up an `agent_task` card: profile config, toolsets, command approvals, workspace selection, and model behavior all come from Kanban/agent runtime.

### How `agent_task` maps to Kanban

When a workflow reaches an `agent_task` node, the dispatcher creates or reuses a Kanban task with:

| Kanban field | Source |
|---|---|
| `title` | rendered node `title` (templates supported), or `<workflow name>: <node id>` |
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

## Limitations / unsupported primitives

- Validation and deploy reject primitives that are declared in the schema but not implemented in the bundled runtime. Today, `manual` and `schedule` triggers are implemented; `webhook` and `kanban_event` triggers are rejected by built-in validation until a launcher is available.
- `send_message` and `subworkflow` nodes are declared for forward compatibility but are rejected by built-in validation/deploy until the dispatcher has a built-in message sender and subworkflow runner.
- There is no standalone `hermes workflow events` CLI command yet.
- Workflows do not run arbitrary Python, shell, JavaScript, or model-generated code.
- `workflow.dispatch_in_gateway` defaults to `false`; a deployed workflow will not advance unattended until dispatch is enabled.
- Unknown YAML keys may validate for forward compatibility but do not imply runtime behavior.
