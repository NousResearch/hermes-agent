---
title: Todoist read-only integration
sidebar_position: 96
---

# Todoist read-only integration

Hermes can read Todoist projects/tasks through Todoist's current Sync API. The built-in tools are intentionally read-only: they do not create, update, complete, move, or delete tasks.

## Setup

Store the Todoist token in Bitwarden Secrets Manager with the exact secret name:

```text
TODOIST_API_TOKEN
```

Do not paste the token into chat, Kanban comments, docs, or shell history. After adding/renaming the secret, verify from the Hermes host:

```bash
hermes secrets bitwarden sync
```

Then restart/reload Hermes so the runtime sees the token.

## Tools

### `todoist_read_only_probe`

Verifies the token and returns counts for projects, items, labels, and sections.

### `todoist_list_tasks`

Lists tasks read-only using the Sync API.

Arguments:

- `limit`: maximum tasks to return, 1-100.
- `include_completed`: include completed tasks when true.
- `project_name_contains`: optional case-insensitive project-name filter.

## API notes

Todoist's older REST v2 endpoint may return HTTP 410 Gone. Use the Sync API endpoint instead:

```text
https://api.todoist.com/api/v1/sync
```

with:

```text
sync_token=*
resource_types=["projects","items","labels","sections"]
```

## Safety policy

- Read/list is allowed.
- Creating, editing, completing, deleting, or moving tasks requires explicit Jonas approval and a separate write-capable workflow.
- Keep Todoist task state coordinated with Hermes Kanban: Kanban is the source of truth for agent improvement/backlog work; Todoist is for human tasks, reminders, errands, and personal/work next actions.
