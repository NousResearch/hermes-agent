---
sidebar_position: 13
title: "Kanban REST API"
description: "Safe external task/workflow API over the existing Hermes Kanban store"
---

# Kanban REST API

Hermes exposes a small, authenticated REST adapter for external control planes
at `/api/plugins/kanban/v1`. It uses the same `hermes_cli.kanban_db` functions and
database as the CLI, dashboard, tools, gateway dispatcher, and workers. It does
not create a second queue or schema.

The API is deliberately workflow-oriented and safe by default. Task bodies,
results, comment text, workspace paths, claim locks, worker PIDs, session IDs,
raw event payloads, run metadata/errors/summaries, and unredacted logs are not
returned. The log endpoint returns only a bounded excerpt after Hermes secret
redaction and absolute-path removal. The one exception is the
[run transcript](#run-transcripts-opt-in), which is off unless the operator
enables it.

## Authentication

External controllers authenticate with a dedicated service credential.
Generate the shared secret with a cryptographically secure generator —
`python -c "import secrets; print(secrets.token_urlsafe(32))"` (256 bits, 43
characters) — and
export it as `HERMES_KANBAN_API_SECRET` in the Hermes deployment's
environment. The bundled `kanban_api` dashboard-auth plugin then accepts
`Authorization: Bearer <secret>` on every endpoint documented here — on any
bind, including gated (non-loopback) deployments where the rest of the
dashboard requires a cookie session. A short or obviously structured secret
(fewer than 16 distinct characters, or one block repeated) is rejected at
startup (fail-closed) and the credential stays disabled. This check only
catches degenerate values — it cannot tell how a secret was generated, so a
hand-typed or derived value that passes it is still not a strong secret.

The credential is scoped to this API only: it cannot drive other service
surfaces (for example gateway drain control), and other service credentials
cannot drive this one. It also cannot open the interactive dashboard's own
routes next to it under `/api/plugins/kanban`.

Note that once the secret is set, this external surface accepts *only* the
service credential — the dashboard session token no longer authenticates
here, even on a loopback bind. Without `HERMES_KANBAN_API_SECRET` the
surface falls back to dashboard-session auth (loopback: the SPA session
token; gated: a cookie session), which no headless external controller can
use — so set the secret for any real integration.

```bash
export HERMES_URL="http://127.0.0.1:9119"
export AUTH="Authorization: Bearer $HERMES_KANBAN_API_SECRET"
```

## Endpoints

Paths below are relative to `/api/plugins/kanban/v1`.

| Area | Endpoints |
|---|---|
| Status | `GET /health`, `GET /capabilities` |
| Boards | `GET /boards`, `GET /boards/{id-or-name}` |
| Profiles | `GET /profiles` |
| Tasks | `GET /tasks`, `POST /tasks`, `GET /tasks/{id}`, `PATCH /tasks/{id}` |
| Actions | `POST /tasks/{id}/comment`, `/complete`, `/block`, `/unblock`, `/archive` |
| Dependencies | `POST /tasks/{parent}/links/{child}`, `DELETE /tasks/{parent}/links/{child}` |
| Observation | `GET /tasks/{id}/events`, `/runs`, `/log`, `/transcript` (opt-in) |

`GET /profiles` returns the sanitized assignee roster: each entry carries only
`name`, `description` (the operator-facing text from `profile.yaml`, the same
signal the built-in decomposer routes on), and `has_description`. Models,
providers, filesystem paths, env/config state, and skill inventories are not
exposed. Use it to populate an assignee picker; there is still no endpoint
that manages or executes a profile.

For attribution, task payloads include `created_by` — the profile (or surface,
e.g. `api:kanban-api` / `dashboard`) that created the card — alongside
`assignee`, so an external control plane can visualise which orchestrator
created which work, not just who executes it. Each entry in
`GET /tasks/{id}/runs` likewise names the `profile` that executed that
attempt (relevant when a task was reassigned between retries).

Cards created and comments posted through this API are attributed to the
authenticated principal with an `api:` prefix — `api:kanban-api` for the service
credential, or `api:external` when the request carried no service token. The prefix keeps
the identity from ever matching a profile name. The caller cannot
choose this identity: `POST /tasks/{id}/comment` accepts only `body` and
rejects an `author` field with 422. A running worker skips comments authored
under its own profile name, so a caller-chosen author could silence a real
operator note.

All task endpoints accept `?board=<board-id>`. Omit it to use the current board.
`GET /tasks` also supports `status`, `assignee`, `tenant`, `include_archived`,
and `limit` filters.

`POST /tasks` accepts an `Idempotency-Key` header or an `idempotency_key` JSON
field. Repeating a request with the same key returns the existing non-archived
task with HTTP 200 and `created: false`; a new task returns HTTP 201 and
`created: true`. The guarantee is enforced by a partial `UNIQUE` index on the
board's store (scoped to live, non-archived tasks), so even two concurrent
`POST`s with the same key resolve to a single task — the loser of the race is
answered with the existing task (HTTP 200, `created: false`) rather than
creating a duplicate. Archiving a task frees its key for reuse.

`PATCH /tasks/{id}` rejects edits to a task's `title` or `body` once the task
is completed (`done`) or `archived` with **HTTP 409** — the finished card text
is a historical record. `priority` and `assignee` may still be adjusted (the
latter subject to its own "not while running" rule).

`POST /tasks/{id}/complete` needs completion evidence: send a non-empty
`summary` (unless the task already carries a stored result), otherwise the
request is refused with **HTTP 400** and the task stays open.

Linking references two existing tasks: `POST /tasks/{parent}/links/{child}`
returns **HTTP 404** when either the parent or the child does not exist
(consistent with unlink), and HTTP 400 only for a genuinely invalid link
(self-dependency or a cycle).

There is intentionally no endpoint that directly starts a Hermes profile. An
assigned task becomes eligible through normal Kanban state/dependency rules,
and the gateway's existing Kanban dispatcher claims and launches it.

## Example: parent operation and dependent child work

A dependency link means **the parent is a prerequisite for the child**. The
example creates a short planning/approval parent, creates two children, links
the parent to each child, and completes the parent so the dispatcher may pick
up the children.

```bash
# 1. Create the parent operation.
PARENT=$(
  curl -fsS -X POST "$HERMES_URL/api/plugins/kanban/v1/tasks?board=default" \
    -H "$AUTH" -H 'Content-Type: application/json' \
    -H 'Idempotency-Key: ops-2026-07-10-parent' \
    -d '{
      "title": "Approve and fan out catalog maintenance",
      "body": "Validate scope, then release the dependent tasks.",
      "tenant": "catalog-maintenance",
      "priority": 20
    }' | jq -r '.task.id'
)

# 2. Create child tasks. Assignees are ordinary Hermes profile names; the API
# does not execute them directly.
CHILD_A=$(
  curl -fsS -X POST "$HERMES_URL/api/plugins/kanban/v1/tasks?board=default" \
    -H "$AUTH" -H 'Content-Type: application/json' \
    -H 'Idempotency-Key: ops-2026-07-10-child-a' \
    -d '{
      "title": "Process workstream A",
      "body": "Execute the first approved work package.",
      "assignee": "worker-a",
      "tenant": "catalog-maintenance"
    }' | jq -r '.task.id'
)

CHILD_B=$(
  curl -fsS -X POST "$HERMES_URL/api/plugins/kanban/v1/tasks?board=default" \
    -H "$AUTH" -H 'Content-Type: application/json' \
    -H 'Idempotency-Key: ops-2026-07-10-child-b' \
    -d '{
      "title": "Process workstream B",
      "body": "Execute the second approved work package.",
      "assignee": "worker-b",
      "tenant": "catalog-maintenance"
    }' | jq -r '.task.id'
)

# 3. Add prerequisite links. Each child moves to todo while PARENT is open.
curl -fsS -X POST \
  "$HERMES_URL/api/plugins/kanban/v1/tasks/$PARENT/links/$CHILD_A?board=default" \
  -H "$AUTH"
curl -fsS -X POST \
  "$HERMES_URL/api/plugins/kanban/v1/tasks/$PARENT/links/$CHILD_B?board=default" \
  -H "$AUTH"

# Release the children after the parent approval/planning work is complete.
curl -fsS -X POST \
  "$HERMES_URL/api/plugins/kanban/v1/tasks/$PARENT/complete?board=default" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"summary":"Scope approved and child work released."}'
```

## Poll task state and events

```bash
# Poll the workstream without receiving private task bodies or worker output.
curl -fsS \
  "$HERMES_URL/api/plugins/kanban/v1/tasks?board=default&tenant=catalog-maintenance" \
  -H "$AUTH" | jq '.tasks[] | {id, title, status, assignee, created_by, links}'

# Read the sanitized append-only event timeline and run state.
curl -fsS \
  "$HERMES_URL/api/plugins/kanban/v1/tasks/$CHILD_A/events?board=default" \
  -H "$AUTH" | jq
curl -fsS \
  "$HERMES_URL/api/plugins/kanban/v1/tasks/$CHILD_A/runs?board=default" \
  -H "$AUTH" | jq

# A bounded, redacted diagnostic excerpt. No filesystem path is returned.
curl -fsS \
  "$HERMES_URL/api/plugins/kanban/v1/tasks/$CHILD_A/log?board=default&tail_bytes=8192" \
  -H "$AUTH" | jq
```

The interactive Kanban dashboard keeps its richer internal routes directly under
`/api/plugins/kanban` (outside `/v1`). That surface is for the first-party
operator UI and may change without notice; external integrations should use only
the sanitized `/v1` endpoints documented here.

## Run transcripts (opt-in)

`GET /tasks/{id}/transcript` returns a worker's step-by-step session for one
run: reasoning, tool calls and results, and replies. That includes the task
body (the worker's first prompt) and its output, so it is **disabled by
default** — the endpoint returns 404 and `GET /capabilities` omits
`transcript` from `observability`. Enable it in the Hermes deployment's
`config.yaml`:

```yaml
kanban:
  api_expose_transcripts: true
```
The setting is per profile, and a transcript is served only when the profile
that **ran the worker** has it on — a `?profile=` on the request does not
change whose opt-in counts. With workers in named profiles, set it in each of
their `config.yaml` files as well as the one serving the API.


Every text field goes through the same secret redaction and absolute-path
removal as the log excerpt. Tool results are capped at 4,000 characters and
other fields at 20,000 (`truncated: true` marks a cut). System messages are
never returned, and neither is the worker's session ID.

| Query | Default | Meaning |
|---|---|---|
| `run_id` | latest run | A run of this task (`GET /tasks/{id}/runs`); a run of another task is 404. |
| `after_id` | `0` | Return steps after this cursor. Pass the previous `next_after_id` to poll. |
| `limit` | `200` | Page size, 1–500. |
| `latest` | `false` | Return the newest `limit` steps instead (`after_id` is ignored; `has_more` then means older steps exist). |

The transcript is available while the run is still going: the worker links
its session to the run when it starts, and continuations after context
compression are followed. A run started before this link existed, or one
whose worker has not started yet, returns an empty `messages` list.

```bash
# Poll a running task: repeat with after_id = the previous next_after_id.
curl -fsS \
  "$HERMES_URL/api/plugins/kanban/v1/tasks/$CHILD_A/transcript?board=default&after_id=0" \
  -H "$AUTH" | jq '{run_status, next_after_id, has_more, steps: [.messages[] | {id, role, tool_calls: [.tool_calls[].name], truncated}]}'
```

Each message carries `id`, `uid`, `role` (`user`, `assistant`, or `tool`),
`content`, `reasoning`, `tool_calls` (`id`, `name`, `arguments`),
`tool_name`, `tool_call_id`, `timestamp`, and `truncated`. The response
wraps them with `task_id`, `run_id`, `run_status`, `next_after_id`, and

`id` is only the paging cursor. When the worker compacts its context mid-run,
the steps it carries forward are stored again under new ids, so a poller can
receive a step it already has; `uid` stays the same, so deduplicate on it.
`has_more`.
