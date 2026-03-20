---
sidebar_position: 6
title: "AutoResearch"
description: "Install the AutoResearch optional skill to run bounded, manifest-driven research cycles and generate reports only for interesting runs"
---

# AutoResearch

AutoResearch is an official optional Hermes skill for running structured research loops against a workspace-defined project.

It is designed for cases where you want Hermes to:

- evaluate anchor strategies or baseline candidates
- generate bounded variants
- review them against explicit edit rules
- score them with your own evaluator command
- select a champion using holdout-aware rules
- write a Markdown report only when the run is interesting
- optionally prepare or send a short summary for Telegram or other Hermes messaging targets

AutoResearch is no longer a native toolset. Users opt into it by installing the skill:

```bash
hermes skills install official/research/autoresearch
```

Once installed, Hermes can load the `autoresearch` skill when relevant, and you can also run its bundled helper script directly.

## What AutoResearch does

The bounded research loop is:

`anchor -> generate -> review -> evaluate -> select -> report -> publish summary`

In practice:

1. Hermes loads your AutoResearch project and family manifests.
2. It evaluates the declared anchor candidates first.
3. It generates new candidates with either:
   - `param_mutation`
   - `agent_patch`
4. It reviews generated candidates against your editable-file rules.
5. It runs your evaluator command, which must write structured JSON.
6. It ranks and filters candidates with your selector rules.
7. It writes one report for the run if the final result is interesting.
8. It can prepare or send a short publishable summary.

## What to expect

AutoResearch is not a free-form autonomous coding mode. It is intentionally bounded.

What it expects from you:

- a workspace with `.hermes/autoresearch/project.yaml`
- one or more family manifests under `.hermes/autoresearch/families/`
- evaluator commands that Hermes can run from the workspace
- evaluator output written to a declared JSON file
- explicit selection and interestingness rules

What it gives you back:

- isolated candidate workspaces
- run metadata under `.hermes/autoresearch/runs/<run_id>/`
- candidate artifacts under the run folder
- optional Markdown reports under `research/YYYY-MM-DD/`
- a short summary string suitable for `send_message`

## Install and Entry Point

Install the skill from the official optional catalog:

```bash
hermes skills install official/research/autoresearch
```

The skill installs a helper script at:

```text
~/.hermes/skills/research/autoresearch/scripts/autoresearch.py
```

That script prints JSON and exposes the same bounded workflow actions the old native tool used.

## Recommended workflow

Always inspect the manifests before running a research cycle. The validation and evaluator commands come from the workspace and will be executed directly.

A reliable sequence is:

1. read `.hermes/autoresearch/project.yaml` and the target family manifest
2. run `inspect-project`
3. run `validate-project`
4. run `research-cycle`
5. run `inspect-run`
6. run `publish-summary` if you want a digest or delivery target

## Workspace contract

AutoResearch looks for these paths inside your project:

```text
.hermes/autoresearch/project.yaml
.hermes/autoresearch/families/*.yaml
.hermes/autoresearch/runs/<run_id>/
.hermes/autoresearch/workspaces/<run_id>/<candidate_id>/
research/YYYY-MM-DD/<project-id>--<run-id>.md
```

## Project manifest

`project.yaml` defines the shared project-level contract:

- `project_id`
- `description`
- `default_cwd`
- `datasets`
- `benchmarks`
- `evaluator`
- `report_output_dir`
- `publish_target`

Example:

```yaml
project_id: mean-reversion-demo
description: Research bounded variants of a mean-reversion strategy
default_cwd: .
datasets:
  - data/btc_1h.csv
benchmarks:
  - baseline
report_output_dir: research
publish_target: telegram
evaluator:
  evaluation: "python evaluate.py --candidate {candidate_json} --out {result_json}"
  result_json: "{result_json}"
```

## Family manifest

Each family defines one bounded search space.

Important fields:

- `family_id`
- `thesis`
- `commands.validation`
- `commands.evaluation`
- `commands.result_json`
- `mutation`
- `selection`
- `interesting_if`
- `anchors`
- `editable_files`
- `editable_markers`

Example:

```yaml
family_id: threshold-sweep
thesis: Search threshold combinations without changing the whole strategy
commands:
  validation: "python validate.py --candidate {candidate_json}"
  evaluation: "python evaluate.py --candidate {candidate_json} --out {result_json}"
  result_json: "{result_json}"
mutation:
  mode: param_mutation
  population: 8
  survivors: 3
  parameter_space:
    entry_threshold: [0.5, 0.75, 1.0]
    exit_threshold: [0.1, 0.2, 0.3]
selection:
  primary_metric: metrics.holdout.sharpe
  goal: maximize
  secondary_metrics:
    - metric: metrics.validation.sharpe
      min_delta: 0.0
interesting_if:
  mode: all
  rules:
    - metric: champion.primary_delta
      op: ">"
      value: 0.1
anchors:
  - candidate_id: baseline
    label: Baseline
    parameters:
      entry_threshold: 0.5
      exit_threshold: 0.1
```

## Mutation modes

### `param_mutation`

Use this when your research space is a bounded parameter search.

Hermes mutates only the declared parameter space and evaluates each candidate with your command contract.

Use this when:

- your strategies are mostly configuration-driven
- you want deterministic and easy-to-audit variation
- you do not want the model editing source files

### `agent_patch`

Use this when Hermes should propose file edits, but only within a declared mutable surface.

For `agent_patch`, define:

- `editable_files`
- optional `editable_markers`

Hermes will reject candidates that:

- touch forbidden files
- exceed editable marker bounds
- fail validation
- fail evaluation

This is the safer path for code mutation because AutoResearch evaluates candidates in isolated workspaces and never mutates the live workspace directly during the research run.

## Evaluator contract

Your evaluator command is the source of truth.

It must:

- accept the placeholders you define in the manifest
- run from the workspace
- write a JSON object to the declared `result_json` path

AutoResearch expects metrics to be addressable by dotted paths such as:

```text
metrics.holdout.score
metrics.validation.sharpe
metrics.stress.max_drawdown
```

Those paths are used by:

- `selection.primary_metric`
- `selection.secondary_metrics`
- `interesting_if.rules`

## Helper script actions

Use the helper script from the project root unless you pass `--project-root`:

```bash
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py list-projects
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py inspect-project
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py validate-project
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py research-cycle --family-id threshold-sweep
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py status --run-id ar-...
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py list-runs
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py inspect-run --run-id ar-...
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py publish-summary --run-id ar-...
```

Useful flags:

- `--project-root /path/to/project`
- `--population N`
- `--survivors N`
- `--seed N`
- `--model provider/model` for `agent_patch`
- `--target telegram` or another Hermes messaging target for `publish-summary`
- `--send` to send the prepared summary immediately

### `list-projects`

Finds discoverable AutoResearch projects and their family IDs.

Use this first when you are not sure which workspace Hermes should target.

### `inspect-project`

Returns the parsed project manifest and family definitions.

Use this to confirm:

- the project root
- available families
- mutation mode
- metrics used for selection

### `validate-project`

Validates the manifests before you run anything.

Use this before every first run in a new workspace or after editing manifests.

### `research-cycle`

Runs the full bounded loop for one family.

Useful parameters:

- `--family-id`
- `--population`
- `--survivors`
- `--seed`
- `--model` for `agent_patch`

### `status`

Returns a compact run status:

- `status`
- `phase`
- `error`
- `report_path`
- `summary`

### `list-runs`

Shows recent runs for the current AutoResearch workspace.

### `inspect-run`

Returns the full run record and a preview of the generated report if one exists.

### `publish-summary`

Builds a short messaging-ready summary for a completed run.

It can also send the summary if you provide `--send` and a valid messaging target is configured.

## Reports

AutoResearch writes one report per interesting run, not one report per candidate.

The report includes:

- project and family information
- anchor summary
- shortlisted candidates
- selected champion
- key metric deltas
- artifact paths
- interestingness verdict and reason

If a run is not interesting, AutoResearch still stores the run metadata but does not write a report file.

## Candidate isolation

Each candidate is evaluated in its own isolated workspace under:

```text
.hermes/autoresearch/workspaces/<run_id>/<candidate_id>/
```

Behavior:

- uses `git worktree` when the project is in a git repo and worktree creation succeeds
- falls back to a filesystem copy when worktrees are unavailable
- ignores `.hermes` during copy fallback to avoid recursive workspace nesting

This keeps candidate evaluation away from the live working tree.

## Selection and interestingness

Selection is not just “best score wins”.

By default, AutoResearch requires:

- a candidate to beat its parent on the primary metric
- secondary metrics to stay within configured tolerances

Interestingness is a separate gate. Even a valid champion does not produce a report unless the run passes `interesting_if`.

This separation is useful when you want:

- quiet storage of mediocre runs
- reports only for genuinely notable improvements
- publishable summaries only for high-signal results

## Messaging and scheduling

AutoResearch does not implement its own scheduler.

Use existing Hermes features for that:

- `publish-summary` for one-off summaries
- `send_message` for delivery
- `cronjob` for recurring research runs or recurring digests

Recommended pattern:

1. run `research-cycle`
2. run `publish-summary`
3. optionally deliver through Hermes messaging
4. if you want automation, schedule that workflow with `cronjob`

## Limitations in v1

AutoResearch is intentionally conservative.

Current limitations:

- projects integrate through command contracts, not Python plugin hooks
- `param_mutation` is bounded by explicit parameter spaces
- `agent_patch` is bounded by explicit editable files and markers
- no free-form whole-repo mutation
- no built-in scheduling layer
- no RL training loop inside AutoResearch itself

These constraints are deliberate. They make the feature safer to run, easier to understand, and easier to upstream.

## When AutoResearch is a good fit

AutoResearch works best when:

- you already have an evaluator script or benchmark command
- you can express success with structured metrics
- you want many bounded experiments, not one giant autonomous rewrite
- you want Markdown reports for high-signal outcomes
- you want Hermes to help orchestrate research, not replace your evaluator

## When it is not a good fit

It is not the best tool when:

- your task has no evaluator command
- success is entirely subjective
- the model needs unrestricted code mutation
- the search space is undefined or open-ended

In those cases, regular Hermes coding workflows or a custom skill may be a better fit.
