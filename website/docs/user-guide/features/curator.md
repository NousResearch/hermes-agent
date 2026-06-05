---
sidebar_position: 3
title: "Curator"
description: "Background maintenance for agent-created skills — usage tracking, staleness, archival, and LLM-driven review"
---

# Curator

The curator is a background maintenance pass for **agent-created skills**. It tracks how often each skill is viewed, used, and patched, moves long-unused skills through `active → stale → archived` states, and periodically spawns a short auxiliary-model review that proposes consolidations or patches drift.

It exists so that skills created via the [self-improvement loop](/user-guide/features/skills#agent-managed-skills-skill_manage-tool) don't pile up forever. Every time the agent solves a novel problem and saves a skill, that skill lands in `~/.hermes/skills/`. Without maintenance, you end up with dozens of narrow near-duplicates that pollute the catalog and waste tokens.

The curator **never touches** bundled skills (shipped with the repo) or hub-installed skills (from [agentskills.io](https://agentskills.io)). It only reviews skills the agent itself authored. It also **never auto-deletes** — the worst outcome is archival into `~/.hermes/skills/.archive/`, which is recoverable.

Tracks [issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816).

## How it runs

The curator is triggered by an inactivity check, not a cron daemon. On CLI session start, and on a recurring tick inside the gateway's cron-ticker thread, Hermes checks whether:

1. Enough time has passed since the last curator run (`interval_hours`, default **7 days**), and
2. The agent has been idle long enough (`min_idle_hours`, default **2 hours**).

If both are true, it spawns a background fork of `AIAgent` — the same pattern used by the memory/skill self-improvement nudges. The fork runs in its own prompt cache and never touches the active conversation.

:::info First-run behavior
On a brand-new install (or the first time a pre-curator install ticks after `hermes update`), the curator **does not run immediately**. The first observation seeds `last_run_at` to "now" and defers the first real pass by one full `interval_hours`. This gives you a full interval to review your skill library, pin anything important, or opt out entirely before the curator ever touches it.

If you want to see what the curator *would* do before it runs for real, run `hermes curator run --dry-run` — it produces the same review report without mutating the library.
:::

A run has two phases:

1. **Automatic transitions** (deterministic, no LLM). Skills unused for `stale_after_days` (30) become `stale`; skills unused for `archive_after_days` (90) are moved to `~/.hermes/skills/.archive/`.
2. **LLM review** (single aux-model pass, `max_iterations=8`). The forked agent surveys the agent-created skills, can read any of them with `skill_view`, and decides per-skill whether to keep, patch (via `skill_manage`), consolidate overlapping ones, or archive via the terminal tool. Consolidation treats a skill as a full package: if a skill has `references/`, `templates/`, `scripts/`, `assets/`, or relative links to those paths, the curator must either keep it standalone, re-home the needed support files and rewrite paths, or archive the entire package unchanged — not flatten only `SKILL.md` into another skill's `references/` file.

Pinned skills are off-limits to both the curator's auto-transitions and the agent's own `skill_manage` tool. See [Pinning a skill](#pinning-a-skill) below.

## Configuration

All settings live in `config.yaml` under `curator:` (not `.env` — this isn't a secret). Defaults:

```yaml
curator:
  enabled: true
  interval_hours: 168          # 7 days
  min_idle_hours: 2
  stale_after_days: 30
  archive_after_days: 90
```

To disable entirely, set `curator.enabled: false`.

### Running the review on a cheaper aux model

The curator's LLM review pass is a regular auxiliary task slot — `auxiliary.curator` — alongside Vision, Compression, Session Search, etc. "Auto" means "use my main chat model"; override the slot to pin a specific provider + model for the review pass instead.

**Easiest — `hermes model`:**

```bash
hermes model                   # → "Auxiliary models — side-task routing"
                               # → pick "Curator" → pick provider → pick model
```

The same picker is available in the web dashboard under the **Models** tab.

**Direct config.yaml (equivalent):**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # generous — reviews can take several minutes
```

Leaving `provider: auto` (the default) routes the review pass through whatever your main chat model is, matching the behavior of every other auxiliary task.

:::note Legacy config
Earlier releases used a one-off `curator.auxiliary.{provider,model}` block. That path still works but emits a deprecation log line — please migrate to `auxiliary.curator` above so the curator shares the same plumbing (`hermes model`, dashboard Models tab, `base_url`, `api_key`, `timeout`, `extra_body`) as every other aux task.
:::

## CLI

```bash
hermes curator status         # last run, counts, pinned list, LRU top 5
hermes curator run            # trigger a review now (blocks until the LLM pass finishes)
hermes curator run --background  # fire-and-forget: start the LLM pass in a background thread
hermes curator run --dry-run  # preview only — report without any mutations
hermes curator backup         # take a manual snapshot of ~/.hermes/skills/
hermes curator rollback       # restore from the newest snapshot
hermes curator rollback --list     # list available snapshots
hermes curator rollback --id <ts>  # restore a specific snapshot
hermes curator rollback -y         # skip the confirmation prompt
hermes curator pause          # stop runs until resumed
hermes curator resume
hermes curator pin <skill>    # never auto-transition this skill
hermes curator unpin <skill>
hermes curator restore <skill>  # move an archived skill back to active
```

## Backups and rollback

Before every real curator pass, Hermes takes a tar.gz snapshot of `~/.hermes/skills/` at `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz`. If a pass archives or consolidates something you didn't want touched, you can undo the whole run with one command:

```bash
hermes curator rollback        # restore newest snapshot (with confirmation)
hermes curator rollback -y     # skip the prompt
hermes curator rollback --list # see all snapshots with reason + size
```

The rollback itself is reversible: before replacing the skills tree, Hermes takes another snapshot tagged `pre-rollback to <target-id>`, so a mistaken rollback can be undone by rolling forward to that one with `--id`.

You can also take manual snapshots at any time with `hermes curator backup --reason "before-refactor"`. The `--reason` string lands in the snapshot's `manifest.json` and is shown in `--list`.

Snapshots are pruned to `curator.backup.keep` (default 5) to keep disk usage bounded:

```yaml
curator:
  backup:
    enabled: true
    keep: 5
```

Set `curator.backup.enabled: false` to disable automatic snapshotting. The manual `hermes curator backup` command still works when backups are disabled only if you set `enabled: true` first — the flag gates both paths symmetrically so there's no way to accidentally skip the pre-run snapshot on mutating runs.

`hermes curator status` also lists the five least-recently-used skills — a quick way to see what's likely to become stale next.

The same subcommands are available as the `/curator` slash command inside a running session (CLI or gateway platforms).

## What "agent-created" means

The curator only manages skills explicitly marked as **agent-created** in
`~/.hermes/skills/.usage.json`. A skill qualifies when ALL of the following
are true:

1. Its name is **not** in `~/.hermes/skills/.bundled_manifest` (bundled skills shipped with the repo).
2. Its name is **not** in `~/.hermes/skills/.hub/lock.json` (hub-installed skills).
3. Its `.usage.json` entry has `"created_by": "agent"` or `"agent_created": true`.

Currently, only the **background self-improvement review fork** sets this marker
— when it creates a new umbrella skill during its periodic review pass (~every 10
agent turns). The background fork runs with a write origin of `"background_review"`
(via `tools/skill_provenance.py`), which is the only path that triggers the
`mark_agent_created()` call in `skill_manage`.

Skills the foreground agent creates via `skill_manage(action="create")` during a
conversation are **not** marked as agent-created — they are considered
user-directed and the curator intentionally leaves them alone.

:::warning Your hand-written skills are NOT curated
If you manually created a `SKILL.md` or pointed Hermes at an external skill
directory, that skill will have a `.usage.json` entry with `created_by: null`
(or the field absent). The curator will not touch it. The same applies to
skills the foreground agent created at your request.

**To see which skills the curator actually manages**, run `hermes curator status`.
If the agent-created count is 0, no skills are currently in the curator's
jurisdiction — the LLM review pass is skipped and the report will show
`Model: (not resolved) via (not resolved)` with `Duration: 0s`.
:::

Skills that ARE agent-created follow the full lifecycle:

- `active` → (30d unused) `stale` → (90d unused) `archived`
- Pinned skills bypass all auto-transitions
- Archives are recoverable via `hermes curator restore <name>`

If you want to protect a specific skill from ever being touched — for example a
hand-authored skill you rely on — use `hermes curator pin <name>`. See the next
section.

## Pinning a skill

Pinning protects a skill from deletion — both the curator's automated archive passes and the agent's `skill_manage(action="delete")` tool call. Once a skill is pinned:

- The **curator** skips it during auto-transitions (`active → stale → archived`), and its LLM review pass is instructed to leave it alone.
- The **agent's `skill_manage` tool** refuses `delete` on it, pointing the user at `hermes curator unpin <name>`. Patches and edits still go through, so the agent can improve a pinned skill's content as pitfalls come up without a pin/unpin/re-pin dance.

Pin and unpin with:

```bash
hermes curator pin <skill>
hermes curator unpin <skill>
```

The flag is stored as `"pinned": true` on the skill's entry in `~/.hermes/skills/.usage.json`, so it survives across sessions.

Only **agent-created** skills can be pinned — bundled and hub-installed skills are never subject to curator mutation in the first place, and `hermes curator pin` will refuse with an explanatory message if you try.

If you want a stronger guarantee than "no deletion" — for instance, freezing a skill's content entirely while the agent still reads it — edit `~/.hermes/skills/<name>/SKILL.md` directly with your editor. The pin guards tool-driven deletion, not your own filesystem access.

## Usage telemetry

The curator maintains a sidecar at `~/.hermes/skills/.usage.json` with one entry per skill:

```json
{
  "my-skill": {
    "use_count": 12,
    "view_count": 34,
    "last_used_at": "2026-04-24T18:12:03Z",
    "last_viewed_at": "2026-04-23T09:44:17Z",
    "patch_count": 3,
    "last_patched_at": "2026-04-20T22:01:55Z",
    "created_at": "2026-03-01T14:20:00Z",
    "state": "active",
    "pinned": false,
    "archived_at": null
  }
}
```

Counters increment when:

- `view_count`: the agent calls `skill_view` on the skill.
- `use_count`: the skill is loaded into a conversation's prompt.
- `patch_count`: `skill_manage patch/edit/write_file/remove_file` runs on the skill.

Bundled and hub-installed skills are explicitly excluded from telemetry writes.

## Per-run reports

Every curator run writes a timestamped directory under `~/.hermes/logs/curator/`:

```
~/.hermes/logs/curator/
└── 20260429-111512/
    ├── run.json      # machine-readable: full fidelity, stats, LLM output
    └── REPORT.md     # human-readable summary
```

`REPORT.md` is a quick way to see what a given run did — which skills transitioned, what the LLM reviewer said, which skills it patched. Good for auditing without having to grep `agent.log`.

:::note No candidates? Report shows `(not resolved)`
When the curator has **no agent-created skills** to review, the LLM review pass
is skipped entirely. The report header will show
`Model: (not resolved) via (not resolved)` with `Duration: 0s` — this does **not**
indicate a configuration error or model resolution failure. It simply means there
were no candidates, so no model was ever invoked. The auto-transition phase still
runs and reports its counts normally.
:::

### Rename map in the summary

If a run consolidated multiple skills under an umbrella (or merged near-duplicates), the user-visible summary printed at the end of the run includes an explicit rename map showing every `old-name → new-name` pair the curator applied. This is in addition to per-skill transition lines, so when a wave of renames lands you can spot them at a glance without diffing the JSON report. The hint also surfaces under `hermes curator pin` so you can pin the umbrella name immediately if you want to lock the new label in.

## Restoring an archived skill

If the curator archived something you still want:

```bash
hermes curator restore <skill-name>
```

This moves the skill back from `~/.hermes/skills/.archive/` to the active tree and resets its state to `active`. The restore refuses if a bundled or hub-installed skill has since been installed under the same name (would shadow upstream).

## Disabling per environment

The curator is on by default. To turn it off:

- **For one profile only:** edit `~/.hermes/config.yaml` (or the active profile's config) and set `curator.enabled: false`.
- **For just one run:** `hermes curator pause` — the pause persists across sessions; use `resume` to re-enable.

The curator also refuses to run if `min_idle_hours` hasn't elapsed, so on an active dev machine it naturally only runs during quiet stretches.

## See also

- [Skills System](/user-guide/features/skills) — how skills work in general and the self-improvement loop that creates them
- [Memory](/user-guide/features/memory) — a parallel background review that maintains long-term memory
- [Bundled Skills Catalog](/reference/skills-catalog)
- [Issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816) — original proposal and design discussion
