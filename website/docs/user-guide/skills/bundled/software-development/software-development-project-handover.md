---
title: "Project Handover — Record and resume where a project is at"
sidebar_label: "Project Handover"
description: "Record and resume where a project is at"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Project Handover

Record and resume where a project is at.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/project-handover` |
| Version | `1.0.0` |
| Author | Praggy (praggybuilds), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Projects`, `Handover`, `Resume`, `Status`, `Continuity` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Project Handover Skill

Keeps a short, per-project handover record (goal, now, next, blockers) with
`hermes project state`, so a new session, a restarted gateway or a delegated
worker can answer "where is this project at?" without the old transcript. It is
not a task tracker, a log or memory: one current record plus a bounded history,
never injected into prompts.

## When to Use

- Resuming work on a named project, or the user asks "where were we?".
- Ending a block of work on a project, before stopping or handing off.
- Before a long `delegate_task` run, so the worker and you share the same picture.
- The user asks to "note", "save" or "update" where a project stands.

## Prerequisites

- A Hermes project exists for the work (`hermes project list`; create one with
  `hermes project create <name> <folder>`). The record lives in the active
  profile's `projects.db`; other profiles never see it.
- `terminal` is available to run `hermes`.

## How to Run

Every command goes through `terminal`. Read with `--json`; write with `--set`
and always `--by agent`, which marks the record as yours rather than the user's.
`--by` is a self-declared label for filtering, not authentication; each saved
version carries one author, including carried-over fields.

```bash
hermes project list
hermes project show <project>
hermes project state <project> --json
hermes project state <project> --set --by agent --now "wiring the RPC" --next "tests"
```

## Quick Reference

```bash
hermes project state <project> --json
hermes project state <project> --set --by agent --goal "Ship v1 of the export flow"
hermes project state <project> --set --by agent --blockers ""
hermes project state <project> --history --limit 5 --json
```

| Field | Holds |
|---|---|
| `goal` | The outcome the project is driving at, in one line. |
| `now` | What is in progress at this moment. |
| `next` | The very next concrete step. |
| `blockers` | What is stopping progress, or empty. |

A flag you omit keeps its previous value; `--field ""` clears it. Each field is
capped at 2000 characters and only the 50 newest records per project are kept.

## Procedure

1. **Resolve the project.** Use the slug the user names. Otherwise run
   `hermes project list`, then `hermes project show <slug>` for the candidates,
   and pick the project whose folder is the most specific ancestor of the
   working directory. If none or several fit, ask.
2. **Read on resume.** Run `hermes project state <project> --json` before
   planning. A `null` state means nothing was recorded: say so and work from
   the user's instructions, never from a guess.
3. **Write at the boundaries:** at the end of a work block, before a long
   delegation, before you stop, and whenever the user asks. Update only the
   fields that changed.
4. **Keep it short and factual.** One or two sentences per field; link to files
   or PRs instead of pasting them. Record what is true now, not a diary.
5. **Never decide for the user.** A choice the user has not confirmed goes in
   `next` or `blockers` as an open question ("decide: Postgres or SQLite"),
   never in `goal` or `now` as settled.

## Pitfalls

- Omitting `--by agent` attributes your write to the user.
- Field flags or `--by` without `--set`, and `--limit` without `--history`, are
  refused (exit 2) instead of silently ignored.
- An archived project refuses writes; ask the user before restoring it.
- Records are per profile. Under `hermes -p other` you will not see them.
- Do not store secrets, tokens or personal data: the record is plain text.

## Verification

- `hermes project state <project> --json` shows your fields with
  `"updated_by": "agent"` and a fresh `updated_at`.
- `hermes project state <project> --history --limit 2` shows the previous record
  below the new one.
