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

### Dashboard modes

The Workflows tab has three modes:

- **Build** — draft, refine, validate, and deploy workflow definitions. Use the structured cell editor for common fields (profile, prompt, output contract) or switch to Advanced YAML for full control. The structured editor covers `pass`, `switch`, `agent_task`, `wait`, `parallel`, `join`, and `fail` nodes; `send_message` and `subworkflow` are not available until their runtimes ship.
- **Run** — start a workflow execution with structured input. The run form is generated from the workflow's `input_schema`. Fill required and optional fields, then start the run. The execution enters the queue and, if the dispatcher is ticking, advances automatically.
- **History** — browse, filter, and drill into past executions. Filter by workflow id, status, and version. Open any execution to see the event timeline, per-node runs, linked Kanban tasks, inputs, outputs, and errors. Cancel non-terminal executions or rerun a workflow with the same or different input.

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
- every `agent_task` profile is available on this machine (not a typo or uninstall)
- only currently implemented primitives are used (`manual`/`schedule` triggers and `pass`, `switch`, `agent_task`, `wait`, `parallel`, `join`, `fail` nodes)
- optional `result_contract` entries use enforced flat types (`string`, `number`, `boolean`, `array`, `object`) or enum strings such as `approved|rejected`
- switch-case `when` and trigger `intake.ready_when` condition trees use the supported condition DSL shape at deploy time

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

`run` (CLI, dashboard, and the `workflow_run` tool alike) advances the execution one dispatcher tick immediately, so cheap all-local graphs finish inline and `agent_task` nodes create their Kanban cards right away. Anything still `queued`/`waiting` after that is advanced by the gateway dispatcher, or manually:

```bash
hermes workflow tick --limit 10 --json
```

`workflow.dispatch_in_gateway` defaults to `true` (matching `kanban.dispatch_in_gateway`), so a running gateway advances schedules, waits, retries, and completed agent tasks unattended. Tune or opt out with:

```bash
hermes config set workflow.dispatch_in_gateway false   # opt out; tick manually
hermes config set workflow.tick_interval_seconds 30
hermes config set workflow.max_executions_per_tick 50
hermes gateway restart
```

If dispatch is off (or no gateway is running), the CLI prints a warning after `run`, the dashboard shows a stall banner on queued/waiting executions, and `hermes workflow status` reports what would advance them.

## Monitor execution and linked worker tasks

Use the dashboard execution list to find a run, then open execution detail. The detail view shows the event timeline, node-runs, node outputs, waits, failures, and current status.

When a workflow reaches an `agent_task` node, Hermes creates or reuses a linked Kanban worker task. The execution detail node-runs drill-down shows `Linked worker task: <id>` and worker status when a run is waiting on an agent. Use that task id in Kanban views or CLI commands to inspect the worker card, logs, result, blocked reason, workspace, and assignee. Long-running `agent_task` nodes wait for their Kanban task to complete before the workflow resumes.

CLI inspection commands:

```bash
hermes workflow status
hermes workflow executions list --workflow <workflow_id> [--limit N] --json
hermes workflow executions show <execution_id> --json
hermes workflow executions node-runs <execution_id> [--json]
hermes workflow executions events <execution_id> [--json]
hermes kanban list --workflow-template-id <workflow_id>
hermes kanban list --workflow-template-id <workflow_id> --step-key <node_id>
```

Executions list newest-first. The same drill-downs are available in chat via the `/workflow` slash command (`/workflow status`, `/workflow executions show <id>`, ...) and to agents through the `workflow_execution_show` tool's `include_node_runs` / `include_events` flags.

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

Definitions are data stored in SQLite after deploy, not executable scripts. Validation is strict about field names: unknown or typo'd keys (e.g. `result_contarct`) are rejected at validate/deploy/draft time with a did-you-mean hint, so a misspelled field can never silently no-op. A `description` field is accepted at the workflow, trigger, and node level for human notes.

## CLI reference

```bash
hermes workflow init
hermes workflow validate <file.yaml>
hermes workflow deploy <file.yaml> [--bump] [--json]
hermes workflow list [--json]
hermes workflow show <workflow_id> [--version N] [--json]
hermes workflow enable <workflow_id> [--version N]
hermes workflow disable <workflow_id> [--version N]
hermes workflow run <workflow_id> [--input <input.json>] [--json]
hermes workflow executions list [--workflow <workflow_id>] [--limit N] [--json]
hermes workflow executions show <execution_id> [--json]
hermes workflow executions node-runs <execution_id> [--json]
hermes workflow executions events <execution_id> [--json]
hermes workflow executions cancel <execution_id>
hermes workflow tick [--limit N] [--json]
hermes workflow status [--json]
```

Deploying the same `id` + `version` with identical content is an idempotent no-op. Deploying changed content at an existing version errors by default; pass `--bump` to redeploy as the next version instead (the dashboard's Deploy button always auto-bumps; the `workflow_deploy` tool takes `auto_bump`).

The `/workflow` slash command exposes the run/inspect verbs (`status`, `list`, `show`, `enable`, `disable`, `run`, `executions`, `tick`) in the interactive CLI and messaging platforms.

## Schema reference

A workflow file is a YAML object with these supported top-level fields:

| Field | Required | Meaning |
|---|---:|---|
| `id` | Yes | Stable id, lowercase letters/digits/underscore/hyphen, starts with a letter, max 64 chars. |
| `name` | Yes | Human-readable name. |
| `version` | Yes | Integer version, minimum `1`. Deployed definitions are keyed by `id` + `version`. |
| `enabled` | No | Defaults to `true`. Disabled definitions deploy but block new runs (manual and scheduled) and create no schedule rows. Toggle later with `hermes workflow enable|disable`. |
| `description` | No | Free-form human note; no runtime behavior. Also accepted on triggers and nodes. |
| `max_node_runs` | No | Loop guard, defaults to `500`. |
| `triggers` | No | List of trigger specs. |
| `nodes` | Yes | Mapping of node id to node spec. Node ids use lowercase letters/digits/underscore/hyphen, start with a letter, max 64 chars. |
| `edges` | No | List of `{from, to}` edges. |

### Triggers

| Type | Fields | Runtime behavior |
|---|---|---|
| `manual` | `id`, `input` | Run with `hermes workflow run` or the dashboard run form. The CLI/dashboard run input is the execution input; trigger `input` is for external launchers and is not merged by the CLI. |
| `schedule` | `id`, `cron` or `schedule`, `input` | Deploy creates a row in `workflow_schedules`; dispatcher starts executions when due. Uses cron syntax accepted by `croniter`; expressions are validated at `workflow validate` time. Schedules evaluate in the **server's local timezone** — `0 9 * * *` means 9am where the dispatcher runs. |
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

### Manual and continuous input

Manual runs and continuous input feeds use the same input-schema and readiness rules.

```yaml
triggers:
  - type: manual
    id: intake
    input_schema:
      repo_path:
        kind: repo_path
        required: true
      criteria:
        kind: criteria
        default: "focus on release blockers"
    intake:
      mode: continuous
      dedupe_key: "$.input.repo_path"
      ready_when:
        op: and
        args:
          - op: exists
            path: "$.input.repo_path"
          - op: exists
            path: "$.input.criteria"
```

The dashboard trigger inspector exposes scalar input-schema fields, `intake.mode`, `dedupe_key`, and `ready_when` without requiring Advanced YAML. Public run surfaces (CLI, dashboard, and tools) validate `input_schema` and `ready_when` before starting a manual run or admitting a feed item. The raw dedupe source value is hashed before it is stored in `workflows.db`; duplicate detection still works without persisting the source string in `workflow_input_items.dedupe_value`.

Continuous feed lifecycle:

- `open` feeds accept new items.
- `paused` feeds do not accept new items; resume returns them to `open`.
- `closed` feeds are **terminal** — no further transitions or writes are accepted. To continue receiving items, open a new feed for the same workflow and trigger.
- Items start as `queued` or `needs_input`; the dispatcher claims ready queued items fairly after already queued executions.
- Linked items become terminal when their execution reaches `succeeded`, `failed`, `cancelled`, or `blocked`.

Phase 1 supports scalar manual and continuous input items. Batch splitting, document upload/splitting, `intake.item_source`, and non-`none` `split_strategy` are not supported in this release; validate/deploy rejects those runtime semantics instead of silently ignoring them.

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

The schema declares `workspace: {cwd, env}` for forward compatibility, but no runtime consumes it yet, so validate/deploy **reject** nodes that set it (the same declared-but-unimplemented treatment as `webhook` triggers) — user intent must never silently no-op. For Kanban workers, use `agent_task.workspace_kind` and `agent_task.workspace_path`.

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

Conditions are YAML objects. There is no Python `eval`; unsupported operators and malformed condition shapes are rejected during workflow validate/deploy for switch cases and `intake.ready_when`, before a run starts.

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
- Unknown YAML keys are rejected at ingestion (validate/deploy/draft) so typos fail loudly; previously stored definitions always load.
- The dispatcher uses claim locks and idempotent Kanban task creation so repeated ticks do not duplicate `agent_task` cards.
- Dashboard **display** responses redact secret-looking keys in execution inputs/contexts, node runs, and events. Definition specs are returned as authored (unredacted) because the builder round-trips them for editing — don't paste secrets into workflow specs; use profile-local credentials instead.

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
| `created_by` | `workflow:<execution_id>:version:<version>:node:<node_id>` (older cards may still show `workflow:<execution_id>`) |
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
- The `workspace: {cwd, env}` node field is declared but unimplemented and is rejected at validate/deploy.
- Workflows do not run arbitrary Python, shell, JavaScript, or model-generated code.
- Schedule triggers evaluate in the server's local timezone.
- Unattended advancement requires a running gateway (dispatch is on by default) or an external `hermes workflow tick` loop; the workflow dispatcher is not a standalone daemon.
