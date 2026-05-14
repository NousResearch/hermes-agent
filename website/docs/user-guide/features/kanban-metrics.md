# Kanban metrics CLI

`hermes kanban metrics` reads a Kanban SQLite database in read-only mode and emits deterministic JSON plus concise markdown flow metrics for humans and shepherd cron jobs.

Example command:

```bash
hermes kanban metrics --db ~/.hermes/kanban.db --window 7d --format both \
  --json-out /tmp/kanban-metrics.json \
  --markdown-out /tmp/kanban-metrics.md
```

For copied fixture/snapshot databases, add `--immutable`:

```bash
hermes kanban metrics --db /tmp/kanban-copy.db --window 24h --immutable --format markdown
```

The command opens the database with SQLite URI `mode=ro` before reading `tasks`, `task_events`, `task_runs`, and `task_links`. Unlike normal Kanban commands, the metrics action is dispatched before `kanban_db.init_db()` so it does not create, migrate, or mutate the selected DB.

JSON output includes:

```json
{
  "generated_at": 1778793000,
  "window": {"seconds": 604800, "start": 1778188200, "end": 1778793000},
  "tasks": [
    {
      "task_id": "t_example",
      "title": "Implement example",
      "assignee": "backend-eng",
      "status": "done",
      "stage": "implementation",
      "durations": {
        "created_to_first_claim_seconds": 50,
        "ready_to_first_claim_seconds": 50,
        "active_seconds_total": 100,
        "blocked_seconds_total": 0,
        "dependency_wait_seconds": 0,
        "ready_queue_age_seconds": 0,
        "rework_active_seconds": 0,
        "open_age_seconds": 0
      }
    }
  ],
  "summaries": {
    "active_by_assignee": [],
    "active_by_stage": [],
    "blocked": {},
    "dependency_wait": {},
    "rework": {},
    "graph_cycle": {},
    "throughput": {},
    "wip": {},
    "reserve_backlog": {}
  },
  "oldest_waiting_gates": [],
  "shepherd_recommendations": []
}
```

Markdown output contains an executive summary and compact tables for active time by assignee/stage, WIP aging, oldest gates, and reserve coverage. Durations in JSON are always seconds; markdown humanizes them for readability.

Percentiles use nearest-rank calculation for deterministic p50/p90 values. Unknown event kinds and malformed event payloads are counted under `unknown_event_kinds` / `unknowns` and do not crash the report.
