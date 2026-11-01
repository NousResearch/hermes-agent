---
title: "Proactive Memory — Surface the right memory before a long-horizon action"
sidebar_label: "Proactive Memory"
description: "Surface the right memory before a long-horizon action"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Proactive Memory

Surface the right memory before a long-horizon action.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/proactive-memory` |
| Path | `optional-skills/autonomous-ai-agents/proactive-memory` |
| Version | `1.0.0` |
| Author | Lexus2016 (Lexus2016), ljluestc (ljluestc), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `memory`, `long-horizon`, `behavioral-decay`, `reminders`, `cron`, `kanban`, `procedural`, `status` |
| Related skills | [`honcho`](../../optional/autonomous-ai-agents/autonomous-ai-agents-honcho.md), [`dynamic-workflow`](../../optional/autonomous-ai-agents/autonomous-ai-agents-dynamic-workflow.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Proactive Memory Skill

Fights *behavioral state decay* — the way a long-running agent stops acting on requirements and
lessons it already has as the trajectory grows (arXiv:2607.08716). It keeps a compact
status / knowledge / procedural memory bank on primitives Hermes already ships — `memory`,
`todo_list`, `session_search`, kanban, skills — and adds one helper, `scripts/memory_bank.py`,
that decides which reminder is relevant to the action you are about to take. It surfaces that
reminder at boundaries the agent controls; it never injects into a live prompt, because that would
break per-conversation caching and role alternation (see Pitfalls).

## When to Use

- A long session or a scheduled cron run keeps re-diagnosing an error it already solved, repeating
  a command that failed, or losing an earlier requirement.
- Before a committing or expensive action, you want a check against what past steps already learned.
- At session resume or the start of a cron/kanban run, you want the decision-relevant state pulled
  forward instead of re-derived.

Skip it for a short, single-step task: there is no horizon to decay over.

## Prerequisites

- `terminal` for `hermes ...` commands and `scripts/memory_bank.py` (Python 3.8+, stdlib only).
- The `memory` tool for durable cross-session facts; `todo_list` for the current task's steps;
  `session_search` to recall earlier attempts; the `kanban` toolset or `cronjob_manage` for
  long-horizon and scheduled work.
- Keep the bank under the profile's Hermes home, never a hardcoded home path: the local terminal
  exports `HERMES_HOME`, so use `"$HERMES_HOME/proactive-memory/<task>.jsonl"`.

## How to Run

Maintain a small bank as the trajectory produces durable state, then `check` it against the next
action at a boundary and act on the reminder. Report real tool output; never invent a reminder.

## Quick Reference

| Step | Primitive | Key calls |
|---|---|---|
| Capture status | `todo_list` + bank | `todo_list`, `memory_bank.py record --kind status` |
| Capture knowledge | `memory` + bank | `memory`, `memory_bank.py record --kind knowledge` |
| Capture procedure / pitfall | skills + bank | `skill_manage` (Pitfalls), `memory_bank.py record --kind procedural` |
| Recall earlier attempts | session DB | `session_search` |
| Gate a reminder | `scripts/memory_bank.py` | `memory_bank.py check --action "<next step>"` |
| Run it unattended | cron | `cronjob_manage(action="create", skills=["proactive-memory"])` |

## Procedure

### 1. Build a compact bank as durable state appears

1. Write an entry only when the trajectory produces state that would change a later action, and tag
   it with the tokens that make it relevant (`--trigger`). An entry with no trigger is rejected on
   purpose: a reminder must be grounded in something concrete, not generic advice.

   ```bash
   python scripts/memory_bank.py record "$HERMES_HOME/proactive-memory/nhs.jsonl" --kind procedural --text "uv sync fails behind the proxy unless UV_NATIVE_TLS=1" --trigger "uv sync,uv pip"
   ```

2. Kind picks the source of truth the entry mirrors: `status` = where the task is (also in
   `todo_list`), `knowledge` = a fact that holds across sessions (also worth a `memory` write),
   `procedural` = a working procedure or a failure to avoid (also a skill Pitfall via `skill_manage`).
3. When state changes, `record --supersedes <id>` retires the old entry so the bank never carries a
   stale "step 1 open" beside "step 1 done". The bank stays capped per kind (newest wins); a bank
   that grows without bound has already decayed, which is the failure this fights.

### 2. Gate the reminder against the next action

1. Before a committing or risky step, ask the bank what is relevant to *this* action:

   ```bash
   python scripts/memory_bank.py check "$HERMES_HOME/proactive-memory/nhs.jsonl" --action "run uv sync to install deps"
   ```

2. `check` returns only entries whose trigger appears in the action, procedural first. An unrelated
   action returns nothing — that silence is the point. Re-injecting the whole bank every step
   ("always inject") is noise; the gate surfaces a reminder when it would change the action.
3. Act on what comes back: adjust the command, re-open a dropped requirement, or skip a retry that
   already failed. If nothing matches, proceed — the check cost you one `terminal` call.

### 3. Recall before you rebuild

1. When starting a task that resembles a past one, `session_search` the error text or task name
   first, and seed the bank's `procedural` entries from what earlier sessions already solved.
2. This is where decay bites hardest in scheduled work: each cron run is a fresh session. Load the
   skill in the job so every run rebuilds its bank and checks against it:
   `cronjob_manage(action="create", schedule="every day 9am", prompt="Reconcile the deploy queue",
   skills=["proactive-memory"])`.

## Pitfalls

- **Never inject a reminder into the live prompt mid-turn.** Hermes keeps the system prompt
  byte-stable and forbids a synthetic user message injected mid-loop (`agent/AGENTS.md`): both
  break per-conversation caching and strict role alternation. The paper's mid-trajectory injection
  is exactly what Hermes will not do in core — this skill surfaces the reminder at a boundary the
  agent already controls (before an action, at cron-run start, at resume) instead.
- Don't create a second memory store. `status` lives in `todo_list`, `knowledge` belongs in
  `memory`, `procedural` belongs in a skill's Pitfalls; the bank is a compact working index over
  them for the relevance gate, not their replacement.
- Don't hardcode `~/.hermes`: profiles live elsewhere. Use `$HERMES_HOME`.
- An entry with a vague trigger (`the`, `it`) matches everything and recreates the always-inject
  noise. Trigger on the concrete token — the command, the filename, the requirement name.
- `memory_bank.py` rejects an unknown kind, empty text, a triggerless entry and an empty action,
  and writes nothing on bad input. Fix the input; don't bypass the check.

## Verification

- Relevance gate: `check --action` on a related action returns the matching entry; on an unrelated
  action returns `count: 0`. Both come from the same bank.
- Supersede: after `record --supersedes <id>`, `list` no longer shows the retired entry and `check`
  no longer surfaces it.
- Compactness: adding more than the per-kind cap leaves only the cap's worth active (`list --kind`).
- Grounding: `record` without `--trigger` exits 2 with a `memory_bank:` message and writes nothing.
