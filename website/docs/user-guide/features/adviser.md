---
sidebar_position: 12
title: "Adviser"
description: "A native second model that shadows the turn, verifies with read-only tools, and delivers nit/concern/blocker notes into the same session."
---

# Adviser

The adviser is a native port of oh-my-pi's advisor: one persistent second model
(the *reviewer*) watches the primary agent's incremental transcript, may verify
its suspicions with read-only tools, and sends terse advice back into the same
session. It reviews; it does not act — every note goes through the primary
agent's judgment.

Hermes already has Mixture of Agents (many reference models feed one aggregator)
and the post-turn background review (memory/skills). The adviser is different:
**one extra model, live, per turn**, whose only output is advice.

## Enable

```yaml
adviser:
  enabled: true
  # empty = inherit the primary route; pin one to control spend:
  model: anthropic/claude-sonnet-4
```

`model` is passed through verbatim — a slash is part of the model id (the id
above is an OpenRouter model), never a provider separator. To switch providers,
set `provider:` next to it (`provider: anthropic` + `model: claude-sonnet-4`);
otherwise the review runs on the primary agent's route.

The roster (who reviews, with which model/tools/instructions) lives in
`WATCHDOG.yml`, on the same search path as project context files:

- `$HERMES_HOME/WATCHDOG.yml` — user level
- `<dir>/WATCHDOG.yml` and `<dir>/.hermes/WATCHDOG.yml` — every directory from the
  session cwd up to the git root (or `$HOME`), ancestor first so the leaf file
  wins on a slug collision.

```yaml
# WATCHDOG.yml
instructions: |
  Prefer small diffs and cite AGENTS.md rules.
maxNotesPerUpdate: 3

advisors:
  - name: Security Reviewer
    model: anthropic/claude-sonnet-4
    tools: [read_file, search_files]   # default read-only set
    reviewMode: agent-end              # turn | agent-end
    reviewInterval: 2
    syncBacklog: strict
    instructions: |
      Focus on auth, injection, and secret handling.
  - name: Silent
    tools: []                          # no investigation tools
```

`WATCHDOG.md` files on the same path add `<attention>` blocks to every reviewer's
system prompt. Project context files (`AGENTS.md` and friends) are passed too, so
the reviewer can hold the agent to the user's standing instructions.

## Delivery

| Severity | Where it lands |
|---|---|
| `nit` | queued as an adviser row at the next safe boundary; never interrupts |
| `concern` | steered into the live turn (one standalone user row, `display_kind=adviser`) |
| `blocker` | steered **and** the running tool batch is interrupted |

At a terminal boundary a note at or above `continue_severity` (default
`concern`) wakes the model once more through the stop gate, bounded by
`max_continuations` (default 2). Notes arriving after the run are appended as a
durable adviser card instead of starting a turn.

Every delivered batch is framed for the model:

```
[adviser — advisory only]
- [concern · Security Reviewer] re-read the failing test
[end adviser]
```

## Admission

The reviewer cannot flood the turn:

- `maxNotesPerUpdate` (default 4, max 32) caps non-blockers per review; `blocker`
  is exempt.
- Notes are normalized for case/punctuation and deduplicated by rank: repeating
  the same text is dropped unless it is re-raised at a *higher* severity.
- Content-free chatter ("Stop.", "LGTM", "no issues") is suppressed.
- Notes emitted while the turn is still in progress are held and flushed at the
  completed boundary; only a `blocker` cuts through early.

## Cost and cadence

- `review_mode: turn` reviews every capture boundary; `agent-end` only terminal
  ones. `review_interval: N` reviews every Nth eligible boundary and folds the
  skipped deltas into the next review.
- Each review is one extra model conversation with its own append-only context
  (prompt caching applies across reviews). Stale `read_file` / `search_files`
  results from older reviews are elided before they are re-sent.
- `sync_backlog` pauses the primary at turn end until reviews catch up:
  `"1"/"3"/"5"` cap the wait at 30 s, `strict` waits without a wall-clock cap.
  A failing reviewer never parks the primary.
- Provider quota errors latch the reviewer until the session is reset; repeated
  failures drop the backlog and eventually halt it, so a broken adviser cannot
  burn tokens forever.

## Inspect

```python
agent.adviser.status()
# {'enabled': True, 'advisers': [{'name': 'Security Reviewer', 'status': 'running',
#   'backlog': 0, 'review_mode': 'turn', 'review_interval': 2, ...}], ...}
```

Per-reviewer transcripts are appended to
`$HERMES_HOME/adviser/<session_id>/__adviser.<slug>.jsonl` for cost and
post-mortem inspection.
