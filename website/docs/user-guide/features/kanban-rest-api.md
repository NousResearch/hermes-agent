---
sidebar_position: 13
title: "Kanban REST API"
description: "Safe external task/workflow API over the existing Hermes Kanban store"
---

# Kanban REST API

Hermes exposes a small, authenticated REST adapter for external control planes
at `/api/plugins/kanban`. It uses the same `hermes_cli.kanban_db` functions and
database as the CLI, dashboard, tools, gateway dispatcher, and workers. It does
not create a second queue or schema.

The API is deliberately workflow-oriented and safe by default. Task bodies,
results, comment text, workspace paths, claim locks, worker PIDs, session IDs,
raw event payloads, run metadata/errors/summaries, and unredacted logs are not
returned. The log endpoint returns only a bounded excerpt after Hermes secret
redaction and absolute-path removal.

Use the same authentication required by the Hermes dashboard deployment. In
the examples below, `HERMES_DASHBOARD_TOKEN` represents the bearer credential
issued for that deployment:

```bash
export HERMES_URL="http://127.0.0.1:9119"
export AUTH="Authorization: Bearer $HERMES_DASHBOARD_TOKEN"
```

## Endpoints

| Area | Endpoints |
|---|---|
| Status | `GET /health`, `GET /capabilities` |
| Boards | `GET /boards`, `GET /boards/{id-or-name}` |
| Tasks | `GET /tasks`, `POST /tasks`, `GET /tasks/{id}`, `PATCH /tasks/{id}` |
| Actions | `POST /tasks/{id}/comment`, `/complete`, `/block`, `/unblock`, `/archive` |
| Dependencies | `POST /tasks/{parent}/links/{child}`, `DELETE /tasks/{parent}/links/{child}` |
| Observation | `GET /tasks/{id}/events`, `/runs`, `/log` |

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
  curl -fsS -X POST "$HERMES_URL/api/plugins/kanban/tasks?board=default" \
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
  curl -fsS -X POST "$HERMES_URL/api/plugins/kanban/tasks?board=default" \
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
  curl -fsS -X POST "$HERMES_URL/api/plugins/kanban/tasks?board=default" \
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
  "$HERMES_URL/api/plugins/kanban/tasks/$PARENT/links/$CHILD_A?board=default" \
  -H "$AUTH"
curl -fsS -X POST \
  "$HERMES_URL/api/plugins/kanban/tasks/$PARENT/links/$CHILD_B?board=default" \
  -H "$AUTH"

# Release the children after the parent approval/planning work is complete.
curl -fsS -X POST \
  "$HERMES_URL/api/plugins/kanban/tasks/$PARENT/complete?board=default" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"summary":"Scope approved and child work released."}'
```

## Poll task state and events

```bash
# Poll the workstream without receiving private task bodies or worker output.
curl -fsS \
  "$HERMES_URL/api/plugins/kanban/tasks?board=default&tenant=catalog-maintenance" \
  -H "$AUTH" | jq '.tasks[] | {id, title, status, assignee, links}'

# Read the sanitized append-only event timeline and run state.
curl -fsS \
  "$HERMES_URL/api/plugins/kanban/tasks/$CHILD_A/events?board=default" \
  -H "$AUTH" | jq
curl -fsS \
  "$HERMES_URL/api/plugins/kanban/tasks/$CHILD_A/runs?board=default" \
  -H "$AUTH" | jq

# A bounded, redacted diagnostic excerpt. No filesystem path is returned.
curl -fsS \
  "$HERMES_URL/api/plugins/kanban/tasks/$CHILD_A/log?board=default&tail_bytes=8192" \
  -H "$AUTH" | jq
```

The interactive Kanban dashboard continues to use its richer internal API under
`/api/plugins/kanban/dashboard`. That surface is for the first-party operator UI;
external integrations should use only the sanitized endpoints documented here.
