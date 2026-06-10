# Memory Wiki

The Memory Wiki is a dashboard view that turns your local Hermes session history into browsable subject pages and daily logs.

## Launch

Start the dashboard:

```bash
hermes dashboard
```

Then open `/memory` from the dashboard sidebar.

## What it shows

- **Subjects** — deterministic topics extracted from session titles, previews, user messages, file paths, package names, tool calls, and slash commands.
- **Subject pages** — related sessions, message snippets, keywords, first-seen and last-touched metadata for a topic.
- **Daily logs** — sessions grouped by local calendar date, with subject chips and “what we did” bullets derived from messages and tool calls.
- **Recent activity** — recent sessions from the same local history database.

## Privacy model

Memory Wiki v1 is generated from your local Hermes profile database at `~/.hermes/state.db`. It does not require remote upload or a hosted index.

The dashboard API computes the data on demand from the existing `sessions` and `messages` tables. No schema migration or persistent memory-wiki cache is required for v1.

## Accuracy notes

Subject extraction is intentionally heuristic and local-only. It normalizes obvious aliases such as `Memory Wiki`, `memory wiki`, and `memory-wiki`, but the taxonomy may be imperfect. Future versions may add manual merge/rename controls or optional summarization jobs.

Daily-log work items are derived from user/assistant messages and tool calls. They are meant as navigation aids, not a canonical audit log.

## Manual QA checklist

1. Start the dashboard with `hermes dashboard`.
2. Open `/memory`.
3. Confirm subjects load from real local history.
4. Search a known topic.
5. Click a subject card and verify related sessions/snippets render.
6. Click a daily log and verify work items and sessions render.
7. Return to `/memory` using the back link.
