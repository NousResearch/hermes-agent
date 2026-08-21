---
title: "Agenda — Track prioritized, recurring, and one-off agenda items"
sidebar_label: "Agenda"
description: "Track prioritized, recurring, and one-off agenda items"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Agenda

Track prioritized, recurring, and one-off agenda items.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/agenda` |
| Version | `0.1.0` |
| Author | Thamer (taljeri), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Agenda`, `Goals`, `Task-Management`, `Priority`, `Productivity` |
| Related skills | [`weekly-review-planning`](/docs/user-guide/skills/bundled/productivity/productivity-weekly-review-planning), [`session-librarian`](/docs/user-guide/skills/bundled/productivity/productivity-session-librarian) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Agenda Skill

Manage prioritized, recurring, and one-off goal items with an SQLite-backed queue and CLI manager. The skill supports domain categorization, cooldown-based recurring schedules, post-mortem outcome logging, and spontaneous idea ("spark") tracking.

Zero external dependencies: uses Python standard library `sqlite3` and stores data in the profile-aware Hermes directory (`~/.hermes/agenda.db`).

## When to Use

- "Add a task to my agenda / goal list"
- "What is the next priority item on my agenda?"
- "Mark task #42 as done with outcome notes"
- "Show me my pending agenda items"
- "I have a quick idea / spark to log for later review"
- Setting up an autonomous recurring cron loop to surface high-priority goals

Don't use for:
- Session-level chat organization (use `session-librarian` instead)
- Calendar event scheduling (use `google-workspace` or native calendar tools)

## Prerequisites

- Python 3.9+ with standard library `sqlite3` (no third-party pip dependencies required).
- Database automatically initializes in `~/.hermes/agenda.db` on first execution (or customize via `HERMES_AGENDA_DB`).

## How to Run

Execute commands through the `terminal` tool pointing to the bundled script:

```bash
# Add a task
python3 skills/productivity/agenda/scripts/agenda.py add "Read research paper" --domain research --priority 1 --detail "Review section 3 methodology"

# Get next prioritized item and mark active
python3 skills/productivity/agenda/scripts/agenda.py next --n 1 --json

# Complete item with outcome notes
python3 skills/productivity/agenda/scripts/agenda.py done 1 --outcome "Identified key benchmark metrics"

# Capture a raw idea
python3 skills/productivity/agenda/scripts/agenda.py spark "Build automated benchmark harness" --domain research
```

## Quick Reference

| Task | Command |
|---|---|
| Surface next item(s) | `python3 skills/productivity/agenda/scripts/agenda.py next [--n N] [--domain D] [--json]` |
| Add new item | `python3 skills/productivity/agenda/scripts/agenda.py add "<title>" [--priority 1-5] [--cooldown N]` |
| List items | `python3 skills/productivity/agenda/scripts/agenda.py list [--status S] [--domain D] [--limit N]` |
| Complete item | `python3 skills/productivity/agenda/scripts/agenda.py done <id> [--outcome "<summary>"]` |
| Record spark idea | `python3 skills/productivity/agenda/scripts/agenda.py spark "<idea>" [--observation "<obs>"]` |
| List sparks | `python3 skills/productivity/agenda/scripts/agenda.py sparks [--status S]` |
| Summary status | `python3 skills/productivity/agenda/scripts/agenda.py status [--json]` |

## Architecture & Data Model

The database contains three interconnected tables:

1. **`agenda`**: Core task records.
   - `priority`: Integer scale (`1` = highest urgency, `5` = low urgency).
   - `status`: Lifecycle states: `pending` → `active` → `done` (or `recurring`).
   - `cooldown_days`: For recurring items; re-arms when completed.
2. **`log`**: Audit trail of completed items with timestamp and execution outcome.
3. **`sparks`**: Unstructured observations and nascent ideas evaluated for promotion into full agenda items.

Reference SQL schema: [references/schema.sql](file:///skills/productivity/agenda/references/schema.sql)

## Procedure

### 1. Adding & Categorizing Tasks
When a user expresses a goal or task:
1. Determine `title`, `domain` (e.g. `research`, `skill`, `dev`, `personal`), `kind` (`paper`, `experiment`, `bugfix`, `feature`), and `priority` (1-5, default 3).
2. For recurring tasks (e.g. weekly review), specify `--cooldown <days>`.
3. Call `terminal(command="python3 skills/productivity/agenda/scripts/agenda.py add ...")`.

### 2. Surfacing & Executing the Next Item
1. Retrieve the top item using `terminal(command="python3 skills/productivity/agenda/scripts/agenda.py next --n 1 --json")`.
2. Present the task details, priority, and steps to the user.
3. Perform the requested work or delegate subtasks.

### 3. Completing Tasks & Logging Outcomes
1. Once completed, invoke `terminal(command="python3 skills/productivity/agenda/scripts/agenda.py done <id> --outcome \"<summary>\"")`.
2. This records execution history in `log` and transitions recurring items cleanly.

### 4. Capturing Raw Sparks
When an unexpected finding or idea emerges during a session:
1. Log it with `terminal(command="python3 skills/productivity/agenda/scripts/agenda.py spark \"<idea>\" --observation \"<context>\"")`.
2. Review sparks during weekly planning sessions (`weekly-review-planning`).

## Autonomous Loop & Cron Integration

You can set up a recurring Hermes cronjob to periodically inspect and surface pending agenda items:

```bash
hermes cron create \
  --name "Agenda Surfacer" \
  --schedule "0 9 * * 1-5" \
  --prompt "Check the next high-priority agenda item using python3 skills/productivity/agenda/scripts/agenda.py next --n 1, and present a morning briefing."
```

## Pitfalls

- **Priority ordering:** Priority `1` is highest (top of queue), `5` is lowest. Do not invert the scale.
- **Atomic state transition:** `next` automatically sets status to `active` so subsequent concurrent queries do not double-dispatch the same task.
- **Recurring items:** Recurring tasks (`cooldown_days > 0`) transition back to `recurring` status upon `done`, preserving `times_done` counters.

## Verification

Verify functionality by running status and retrieval checks:
```bash
python3 skills/productivity/agenda/scripts/agenda.py status
python3 skills/productivity/agenda/scripts/agenda.py list --limit 5
```
