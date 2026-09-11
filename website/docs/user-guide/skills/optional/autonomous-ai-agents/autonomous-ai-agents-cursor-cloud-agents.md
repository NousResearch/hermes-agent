---
title: "Cursor Cloud Agents — Launch and track Cursor Cloud Agents for repository work"
sidebar_label: "Cursor Cloud Agents"
description: "Launch and track Cursor Cloud Agents for repository work"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cursor Cloud Agents

Launch and track Cursor Cloud Agents for repository work.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/cursor-cloud-agents` |
| Path | `optional-skills/autonomous-ai-agents/cursor-cloud-agents` |
| Version | `0.1.0` |
| Author | Iven Simon (ivenms), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Coding-Agent`, `Cursor`, `Cloud-Agents`, `GitHub`, `Pull-Requests`, `Automation` |
| Related skills | [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Cursor Cloud Agents Skill

Delegate repository coding work to Cursor Cloud Agents through Cursor's public-beta v1 API. This optional integration uses a dependency-free Python helper and returns durable agent/run identifiers, branches, and pull-request URLs. It does not replace local tests, code review, merge approval, or deployment controls.

## When to Use

- The user explicitly asks Cursor Cloud Agent to implement, fix, refactor, test, or document code.
- A long-running coding task should run remotely against an accessible GitHub repository.
- A bot needs to continue an existing Cursor agent conversation with a follow-up prompt.
- A second technical bot needs to inspect or continue work from an existing `agent-id` and `run-id`.

**Do not use for:** local-only edits, a review that does not require Cursor coding, production deployment, or a task without a repository and acceptance criteria.

## Prerequisites

- A Cursor API key with Cloud Agents access, stored only as `CURSOR_API_KEY` in the active Hermes profile's `.env` or process environment.
- Configure the persistent key through Hermes; do not hand-edit YAML:

```bash
hermes config set CURSOR_API_KEY 'YOUR_CURSOR_API_KEY'
```

This writes the secret to the active profile's `.env`, not `config.yaml`. Replace the placeholder locally; never place the real key in chat, prompts, repository files, URLs, or logs. Start a new profile session after changing the key so Hermes loads it.
- The repository must be accessible through Cursor's GitHub App installation.
- The repository URL and starting branch/ref must be known.
- For an explicit model, query `models` first and use an ID returned by Cursor. Otherwise omit `--model` and use Cursor's configured default.

The Cloud Agents API v1 is public beta and may change. Reference: &lt;https://cursor.com/docs/cloud-agent/api/endpoints>.

## How to Run

Run the bundled helper through the Hermes `terminal` tool. The helper is relative to the active profile's `$HERMES_HOME`:

```bash
AGENT="$HERMES_HOME/skills/autonomous-ai-agents/cursor-cloud-agents/scripts/cursor_cloud_agent.py"
```

Launch work on a new Cursor branch and request a pull request:

```bash
python "$AGENT" launch \
  --repo https://github.com/ORG/REPO \
  --ref main \
  --prompt "Implement the approved task. Add or update tests and report verification commands." \
  --auto-create-pr \
  --wait
```

For a prompt in a file, use `--prompt-file PATH`. The helper prints JSON for non-streaming commands; preserve `agent.id`, `run.id`, final status, pushed branch, and PR URL in the handoff.

## Quick Reference

```text
launch       Create an agent and enqueue its first run.
follow-up    Send a prompt to an existing agent.
status       Read one run's current or terminal state.
stream       Print SSE status, assistant, and tool events.
cancel       Cancel an active run; cancellation is terminal.
models       List model IDs and supported parameters.
```

Examples:

```bash
python "$AGENT" follow-up --agent-id AGENT_ID --prompt "Address the failing test and rerun the focused suite." --wait
python "$AGENT" status --agent-id AGENT_ID --run-id RUN_ID
python "$AGENT" stream --agent-id AGENT_ID --run-id RUN_ID
python "$AGENT" cancel --agent-id AGENT_ID --run-id RUN_ID
python "$AGENT" models
```

## Procedure

### 1. Define the task

State the repository URL, starting ref, desired change, constraints, acceptance criteria, and required tests. Completion criterion: the prompt is specific enough for the remote agent to work without an interactive clarification loop.

### 2. Check credentials

Confirm `CURSOR_API_KEY` is present without printing its value. Completion criterion: the helper passes credential preflight and no secret appears in the prompt or output.

### 3. Choose the model

Run `models` when explicit model selection matters. Completion criterion: every selected model ID and parameter is present in Cursor's response.

### 4. Launch safely

Use `launch` with `--auto-create-pr` unless a branch-only result was requested. Keep `--work-on-current-branch` disabled by default. Completion criterion: the API returns both an agent ID and a run ID.

### 5. Track execution

Use `--wait` for bounded tasks or `status`/`stream` for separately monitored work. Completion criterion: the run reaches `FINISHED`, `ERROR`, `CANCELLED`, or `EXPIRED`; never infer completion from an HTTP 2xx response alone.

### 6. Inspect the result

Read the terminal result and `git.branches[]`. Completion criterion: the final response, pushed branch, and PR URL (if created) are recorded, or the API error is reported verbatim without credentials.

### 7. Verify before merge

Inspect the branch or PR and run the repository's actual tests locally or in CI. Load `requesting-code-review` before committing or merging. Completion criterion: tests and review state have real evidence; the agent's prose alone is not proof.

### 8. Continue or stop

Send a follow-up only after the current run is terminal; Cursor rejects concurrent runs with `agent_busy`. Completion criterion: each follow-up returns a new run ID and is recorded separately.

## Pitfalls

- The v1 API separates a durable agent from per-prompt runs; never substitute one identifier for the other.
- `agent_busy` means another run is active. Wait or cancel it instead of retrying blindly.
- `--work-on-current-branch` can push directly to the supplied ref. Use it only with explicit approval.
- `--auto-create-pr` opens a PR but does not mean the code is reviewed, tested, or safe to merge.
- Run-level `git` is an agent-scoped snapshot. Use the agent's `latestRunId` or stream when attributing work across multiple runs.
- The API key is a secret. Do not pass it through prompts, `envVars`, repository files, URLs, or reports unless the user explicitly approves the scope and lifetime.
- If an SSE stream expires or disconnects, use `status` to read terminal state rather than retrying the expired stream indefinitely.
- Profiles isolate credentials and skills. A bot that should invoke this skill needs both the skill and `CURSOR_API_KEY` in its own active profile.

## Verification

Run these through the Hermes `terminal` tool for a setup-only check:

```bash
python "$AGENT" --help
python "$AGENT" launch --help
python "$AGENT" status --help
```

For a live integration check, run `models` and verify that the response is valid JSON with an `items` array. Do not launch a coding agent merely to test credentials unless the user approves the API usage.

**Sources:**

- &lt;https://cursor.com/docs/cloud-agent/api/endpoints>
- &lt;https://cursor.com/docs/api>
