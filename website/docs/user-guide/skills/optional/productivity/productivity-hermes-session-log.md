---
title: "Hermes-session-log — Lossless archive of Hermes sessions in an Obsidian vault."
sidebar_label: "Hermes-session-log"
description: "Lossless archive of Hermes sessions in an Obsidian vault."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}
# Hermes-session-log

Lossless archive of Hermes sessions in an Obsidian vault.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/hermes-session-log` |
| Path | `optional-skills/productivity/hermes-session-log` |
| Version | `1.0.0` |
| Author | lioxatugol |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Obsidian`, `Sessions`, `Export`, `Compaction`, `Memory` |
| Related skills | [`obsidian`](../../bundled/note-taking/note-taking-obsidian.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Hermes Session Log

A lossless, continuously-updated archive of Hermes sessions inside an Obsidian vault.
The source of truth is `state.db` (the `sessions` and `messages` tables): every
request of every session lands there, so the transcript of the *live* session is
always available. An idempotent export script turns each session into a readable
note + a raw JSONL transcript + a vault index.

The point: context compaction stops being a data loss. The agent can always
consult the uncompressed version of its own session and of any past session,
so one session can safely carry one project for as long as it needs.

## Archive layout

Default: `<vault>/Hermes Sessions/` (all paths come from `scripts/config.json`):

- `YYYY-MM-DD HH-MM <suffix>.md` — the full dialogue, no cuts: user/assistant
  messages verbatim with per-message timestamps, tool results clipped to ~300
  chars, tool calls shown as one-line `> tool: args` blocks. A `## 🗜 Context
  compaction` section marks compacted messages and carries the summary;
  frontmatter: title, session_id, session_start, source, model, profile,
  project (cwd), status (open / end reason), messages, compacted, raw.
- `<suffix>.jsonl` — the raw transcript: EVERY message verbatim
  (content, tool_calls, reasoning, compacted flag, timestamps).
- `Session Index.md` — table of all sessions (start time, title, project,
  status, message count, source).
- `.export-state.json` — export state (do not edit by hand).
- `suffix` = first 8 hex chars of the id for `local-<sha>` ids; for legacy
  `YYYYMMDD_HHMMSS[_hex]` ids — `MMDDHHMMSS[hex]`.

`request_dump_*.json` files under `<HERMES_HOME>/sessions` are error/retry
artifacts (they only exist for failed requests); the exporter uses them solely
as a legacy backfill for sessions absent from the database.

## Installation

1. Install the skill: `hermes skills install hermes-session-log`.
2. Copy `scripts/config.example.json` to `scripts/config.json` and set
   `vaultPath` / `folder` / `stateDb` to your layout. Defaults:
   vault = `$OBSIDIAN_VAULT_PATH` or `~/Documents/Obsidian Vault`;
   `state.db` auto-detected from `$HERMES_HOME`, `~/.hermes` or
   `%LOCALAPPDATA%\hermes` (Windows).
3. Manual export: `python <skill>/scripts/archive_hermes_sessions.py`
   (`--full` re-exports everything). The script is idempotent: only changed
   sessions are rewritten, and a no-op run prints nothing (cron watchdog).

## Live archive (recommended)

Schedule the export every few minutes as a no-agent Hermes cron job:

```bash
# 1. copy the wrapper next to the scheduler's script dir
cp <skill>/scripts/cron_wrapper.py "$HERMES_HOME/scripts/hermes-session-log-export.py"
# 2. create the job (equivalent CLI/hermes cron invocation, every 5 minutes)
#    name: hermes-session-log, schedule: 5m, no_agent: true,
#    script: hermes-session-log-export.py
```

The live session's note then lags the conversation by at most the interval.
If the machine crashes mid-session nothing is lost: `state.db` is the source
of truth and `--full` rebuilds the whole archive.

Optionally make the rules "always on": add a short note to your session-start
prefill (e.g. `System/session-log.md` in your vault) pointing at the archive —
the "After compaction" section below is a ready-made template.

## Current session (peering in during work)

1. `python <skill>/scripts/current_session.py` — prints sessions active in the
   last 30 minutes (id, source, open/closed, title, first user message) and
   the paths to their note and raw `.jsonl`.
2. If several are listed, yours is the one whose first user message belongs to
   this conversation; for older sessions search the index by date/suffix.

## After compaction

Treat the compaction summary as a table of contents, the archived note as the
source of truth:

- Grep the `.md` note for the needed fragments (requirements, decisions,
  commands, outputs, edits).
- Exact tool commands/outputs/edits: Grep the sibling `.jsonl` instead.
- Do not read the whole note — only the relevant fragments.

## Past sessions

1. Start from `Session Index.md` (project, date, status columns).
2. Grep only `*.md` files in the archive folder (raw files are noise): by
   project, keyword, or date.
3. Paraphrase only the relevant part back to the user, with a reference to
   the note.

## Rules

- Never modify or delete archive notes without an explicit user request.
- If the vault or the database moved, update `scripts/config.json`.
- This skill complements the one-shot `hermes session export` command: it
  automates it continuously and adds the vault index, the raw JSONL, and the
  compaction playbook.
