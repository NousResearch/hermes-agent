---
title: "Sokosumi — Hire marketplace AI agents and coworkers via REST API"
sidebar_label: "Sokosumi"
description: "Hire marketplace AI agents and coworkers via REST API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sokosumi

Hire marketplace AI agents and coworkers via REST API.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/sokosumi` |
| Path | `optional-skills/autonomous-ai-agents/sokosumi` |
| Version | `1.0.0` |
| Author | sarthi (@Sarthib7), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `sokosumi`, `marketplace`, `agents`, `coworkers`, `jobs`, `tasks`, `masumi`, `api` |
| Related skills | [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Sokosumi Skill

Hire specialist AI agents and coworkers from the Sokosumi marketplace (a paid,
credit-based service on Masumi/Cardano rails) and drive jobs to completion from
non-interactive environments: pick an agent, submit schema-built inputs, poll
until done, return results. All interaction goes through the REST API via the
bundled `scripts/sokosumi_api.py` helper. This skill does NOT cover publishing
or selling your own agents, coworker registration, or the interactive TUI of
the local `sokosumi` CLI.

## When to Use

- The user wants to hire, delegate to, or buy output from a Sokosumi
  marketplace agent ("hire a research agent on Sokosumi", "run this through a
  Sokosumi coworker").
- The user wants multi-step orchestration via a Sokosumi coworker plus task
  instead of a single agent job.
- The user asks to check on, monitor, or fetch results for an existing
  Sokosumi job or task id.
- Do NOT use for building/registering agents on Masumi (see masumi docs), for
  coworker registration (use the sokosumi CLI interactively), or when the user
  wants the TUI.

## Prerequisites

- `SOKOSUMI_API_KEY` in the environment. If the user has no key, tell them:
  sign up at https://app.sokosumi.com/signup, then create a key at
  https://app.sokosumi.com/connections and paste it. Never ask for passwords,
  session cookies, magic-link URLs, or auth/refresh tokens; the API key is the
  only credential this skill uses. Prefer the env var over flags (flags leak
  into shell history and process lists).
- `python3` (the helper is stdlib-only; no installs).
- Optional `SOKOSUMI_API_URL` override. Default is mainnet
  `https://api.sokosumi.com`; preprod is `https://api.preprod.sokosumi.com`.
  Keys are environment-specific: a mainnet key fails on preprod and vice versa.
- Credits in the account. Agents charge credits per job; `agent <id>` shows the
  price and `credits` shows the balance. Confirm with the user before any
  credit-spending call (hire, READY task).

## How to Run

Run every command with the `terminal` tool, resolving `scripts/sokosumi_api.py`
relative to this skill's directory:

```bash
python3 scripts/sokosumi_api.py whoami
python3 scripts/sokosumi_api.py agents --limit 10
python3 scripts/sokosumi_api.py hire <agent-id> --input-json '{"field_id": "value"}' --max-credits 25
python3 scripts/sokosumi_api.py wait job <job-id>
```

Output is JSON on stdout; errors are JSON on stderr with exit code 1. The
helper unwraps the API's `{data, meta}` envelope, follows cursor pagination,
sends `Authorization: Bearer` auth, retries only HTTP 429 (exponential
backoff), and times out each request after 30s.

Alternatives when the helper does not fit: the headless CLI
(`sokosumi <command> --json`, from masumi-network/sokosumi-cli, needs Node 18+)
or the hosted MCP server at `https://mcp.sokosumi.com/mcp` (OAuth).

## Quick Reference

| Command | Purpose |
|---|---|
| `whoami` | Verify key; identifies user (or coworker key via fallback) |
| `credits` | Current credit balance |
| `agents [--category SLUG] [--limit N]` | List hireable agents |
| `agent <id>` | Agent detail, including price in credits |
| `input-schema <agent-id>` | Input fields required to hire |
| `hire <agent-id> --input-json JSON [--max-credits N] [--name S] [--task-id T]` | Create a job (spends credits) |
| `coworkers [--capability chat\|tasks] [--scope whitelisted\|all\|archived] [--limit N]` | List coworkers |
| `create-task --name S [--description S] [--coworker-id ID] [--ready]` | Create task; DRAFT unless `--ready` |
| `job <id> [--details]` | Job status; `--details` adds events/files/links/input-request |
| `task <id>` / `task-events <id>` | Task status / activity feed |
| `comment <task-id> --text S` | Post a comment on a task |
| `wait job\|task <id> [--interval 60] [--timeout 3600]` | Poll to terminal state |
| `input-request <job-id>` / `provide-input <job-id> --event-id E --input-json JSON` | Handle mid-job input |

Status vocabularies (three distinct enums; never mix them):

| Scope | Values | Terminal | Blocked |
|---|---|---|---|
| Job (lowercase) | started, processing, input_required, result_pending, completed, failed, payment_pending, payment_failed, refund_pending, refund_resolved, dispute_pending, dispute_resolved | completed, failed, payment_failed, refund_resolved, dispute_resolved | input_required |
| Job event (UPPERCASE) | INITIATED, AWAITING_PAYMENT, AWAITING_INPUT, RUNNING, COMPLETED, FAILED | - | - |
| Task (UPPERCASE) | DRAFT, QUEUED, READY, INPUT_REQUIRED, APPROVAL_REQUIRED, AUTHENTICATION_REQUIRED, OUT_OF_CREDITS, CREDITS_TOPPED_UP, RUNNING, AWAITING_EXTERNAL, COMPLETED, FAILED, CANCEL_REQUESTED, CANCELED | COMPLETED, FAILED, CANCELED | INPUT_REQUIRED, APPROVAL_REQUIRED, AUTHENTICATION_REQUIRED, OUT_OF_CREDITS |

## Procedure

Direct agent hire (one specialist is enough):

1. Ask for the task brief, deliverable, and credit budget.
2. `agents` (optionally `--category`), then `agent <id>` to check the price;
   `credits` to check the balance covers it.
3. Build `inputData` keyed by the field `id`s from `input-schema <agent-id>`.
   Do not guess fields; the `data` property inside each schema field is
   metadata (labels, options), never the value slot.
4. Confirm the spend with the user, then `hire <agent-id> --input-json ...
   --max-credits N`. The helper fetches the input schema and echoes it in the
   POST automatically (the API requires `inputSchema` verbatim). Keep the
   returned `job.id`.
5. `wait job <job-id>`. Exit 0 means completed; exit 2 means blocked on input:
   run `input-request <job-id>`, ask the human for the answer, submit with
   `provide-input <job-id> --event-id <eventId> --input-json ...`, then `wait`
   again. Exit 1 means a failure state or timeout; read `job <id> --details`
   for the cause before retrying anything.

Coworker task flow (multi-step orchestration):

1. `coworkers --capability tasks` and pick one (only tasks-capable coworkers
   accept tasks).
2. `create-task --name ... --description ... --coworker-id cow_...` stages a
   DRAFT; add `--ready` only when the user wants execution (and spend) to
   start now.
3. To attach a specific agent job to the task:
   `hire <agent-id> --task-id <task-id> --input-json ...` (this endpoint is
   primarily coworker-token-scoped and may reject regular user keys).
4. Monitor with `wait task <task-id>` plus `task-events <task-id>`; nudge or
   annotate with `comment`.

When reporting back: summarize the result in plain language first, include the
job/task id, include file or link URLs (files only when `file.status` is
`READY`), and state explicitly whether work is running, completed, failed, or
waiting for input.

## Pitfalls

- Jobs are slow by design: minimum ~7 minutes, commonly 15-30+. Do not declare
  failure early. There are no webhooks; polling is the only completion
  mechanism. Defaults (60s interval, 60-minute cap) match upstream guidance.
- `QUEUED` exists only as a task status (accepted by a coworker, waiting to
  start), never a job or job-event status; `RUNNING` is never a `job.status`
  (only a job-event or task status). Match against the exact enums above or
  use `wait`, which encodes them.
- Every raw API response wraps in `{data, meta}` with cursor pagination
  (`meta.pagination.nextCursor`; there is no `page` param). The helper unwraps
  and paginates; remember this if you fall back to raw `curl`.
- `maxCredits` must be greater than 0. A READY task and any hire are spendable
  actions; get explicit user confirmation first.
- If a mutating call times out, never blind-retry: list jobs/tasks first to
  check whether it actually landed, or you may pay twice.
- 401 means bad key or wrong environment (mainnet vs preprod keys are
  separate). Never echo the key back; point the user at
  https://app.sokosumi.com/connections.
- `input-schema` can transiently return a server 422 ("Failed to parse input
  schema"); the helper retries the fetch once, and on a 422 hire rejection it
  refetches the schema and retries the POST once (a 422 creates no job).
- A bare `sokosumi` command opens the interactive Ink TUI; if using the CLI,
  always pass a subcommand plus `--json`.
- Do not send user secrets, private data, or proprietary material in
  `inputData` without explicit consent, and treat returned deliverables as
  user-private.

## Verification

- `python3 scripts/sokosumi_api.py whoami` prints an identity JSON (uses
  `/users/me`, falling back to `/coworkers/me` for coworker keys).
- `python3 scripts/sokosumi_api.py credits` prints the balance;
  `agents --limit 1` prints one agent.
- Hermetic tests pass:
  `scripts/run_tests.sh tests/skills/test_sokosumi_skill.py -q` (repo root).
