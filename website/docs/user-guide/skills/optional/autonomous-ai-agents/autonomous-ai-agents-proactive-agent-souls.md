---
title: "Proactive Agent Souls — Give an agent a tiered proactivity contract in SOUL.md"
sidebar_label: "Proactive Agent Souls"
description: "Give an agent a tiered proactivity contract in SOUL.md"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Proactive Agent Souls

Give an agent a tiered proactivity contract in SOUL.md.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/proactive-agent-souls` |
| Path | `optional-skills/autonomous-ai-agents/proactive-agent-souls` |
| Version | `1.0.0` |
| Author | Pedro Relvas (PRELVAS94), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `proactivity`, `soul`, `identity`, `cron`, `heartbeat`, `notifications`, `quiet-hours`, `agent-design` |
| Related skills | [`hermes-agent`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md), [`dynamic-workflow`](../../optional/autonomous-ai-agents/autonomous-ai-agents-dynamic-workflow.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Proactive Agent Souls Skill

Put a standing proactivity contract in the agent's identity file, so an agent that runs
unattended ticks decides when to speak instead of reporting every tick. The contract is
three tiers — silent self-maintenance, record-but-don't-send, may-send-on-a-registered
trigger — plus a quiet-hours window, a send-suppression window, and an observations log for
everything that is not worth a message. This skill writes and validates that contract; it is
not a scheduler, a monitor, or a notification service — the ticks stay with `cron` and
`heartbeat`, delivery stays with the platform, and the trigger taste stays with the person
who owns the agent.

## When to Use

- An agent has unattended work (cron jobs, heartbeats, watch loops) and its messages should
  be decisions, not activity reports.
- A chatbot has started narrating progress, re-reporting the same finding, or pinging on
  quiet ticks — the fix is a contract in the identity file, not another prompt line.
- Several profiles run the same role (`hermes -p <profile>`), and you want one message
  discipline across all of them without repeating it in each task prompt.
- NOT for: a one-off instruction on a single task (that belongs in the task itself), project
  rules that only apply in one checkout (`AGENTS.md` / `.hermes.md`), or a temporary tone
  change for one session (`/personality`).

## Prerequisites

- Write access to the agent's home. Identity file: `$HERMES_HOME/SOUL.md` (default
  `~/.hermes/SOUL.md`); each profile home carries its own `SOUL.md` at
  `~/.hermes/profiles/<name>/SOUL.md`.
- One real delivery target, or "stay silent" has nothing to suppress: a `cron` job that
  delivers to a chat, or a heartbeat-enabled session.
- Python 3.10+ for the checker script (stdlib only, no install).

## How to Run

```bash
# 1. see the block the contract is made of
read_file ~/.hermes/skills/autonomous-ai-agents/proactive-agent-souls/templates/proactivity-section.md

# 2. append it to the identity file (approved write), then fill every <…> placeholder
#    (agents and editors: use the patch tool — never rewrite SOUL.md wholesale)

# 3. check the result
python3 ~/.hermes/skills/autonomous-ai-agents/proactive-agent-souls/scripts/check_proactivity_contract.py

# 4. check a profile home, or several at once, machine-readable
python3 ~/.hermes/skills/autonomous-ai-agents/proactive-agent-souls/scripts/check_proactivity_contract.py \
  ~/.hermes/profiles/work --json
```

Exit status is `0` when the contract is intact, `1` when a finding is reported, `2` on a
usage error — usable directly as a `cron` monitor script.

## Quick Reference

| Piece | Where it lives | What enforces it |
|---|---|---|
| Three tiers (silent / record / send) | `## Proactivity Protocol` in `SOUL.md` | the model, every session — including cron ticks |
| Trigger register | Tier 3 of the same section | you, by editing the file — nothing sends without one |
| Observations log | any path the section names, e.g. `~/notes/OBSERVATIONS.md` | the agent appends; never a message |
| Delivery silence | the job's prompt | `cron` and `heartbeat` suppress on `[SILENT]` (or empty output) |
| Structure check | `scripts/check_proactivity_contract.py` | exit code — CI, a monitor job, or a pre-edit habit |
| Sizes and precedence | `/context` | per-file listing: loaded, truncated, shadowed, blocked |

## Procedure

1. **Pick the identity file, not a task prompt.** `SOUL.md` in the agent's own home is
   injected as identity slot 1 in every session, and a cron session keeps that persona while
   skipping cwd instructions — that is why the contract belongs there. Project-scoped rules
   go to `AGENTS.md`/`.hermes.md`; a session-scoped voice change goes to `/personality`.

2. **Append the template block.** `read_file` the current `SOUL.md`, then `patch` the
   `templates/proactivity-section.md` content onto the end. Writes to `SOUL.md` are in the
   protected-instruction class: they need human approval even under yolo, so a headless
   agent cannot rewrite its own identity — ask the owner, or have them apply the block.
   Keep the file under `context_file_max_chars` (20K by default); past it the file is
   head+tail truncated and the contract's middle disappears. If the file is already large,
   put the block at the end so the tail survives truncation.

3. **Write a real trigger register.** Each Tier 3 entry is an event plus the decision it
   unlocks for the recipient. "Server is down" qualifies; "server was checked" does not. A
   trigger with no decision attached becomes noise within a week — leave it at Tier 2.

4. **Name the observations log and create it from `templates/observations-log.md`.** Give the
   path on the same line that mentions observations so the checker can see it. The log is
   where "seen twice, nothing to decide yet" accumulates; its status ladder (`open` →
   `confirmed` → `actioned`/`dismissed`) is what keeps the agent from re-investigating the
   same thing every tick.

5. **Wire the silence markers into the jobs.** In a `cron` job's prompt: "if nothing
   changed, reply with exactly `[SILENT]`". Empty output suppresses delivery too, and a
   heartbeat turn ending in `NO_REPLY`/`[SILENT]` posts nothing at all. Note that a job that
   *hard-fails* still delivers a `⚠️ Cron … failed` notice unless the job sets
   `--failure-deliver local` — that failure path is deliberate, not a contract breach.

6. **Validate.** Run the checker on every identity file you maintain; add it as a monitor
   job if the fleet is large. Then confirm the two behaviours by hand: a tick with nothing to
   report delivers nothing, and a tick with a triggered event delivers exactly one message.

7. **Re-check after upgrades.** In-place config migrations have rewritten `SOUL.md` before
   (the 40→41 migration stripped a legacy appended section from every profile's `SOUL.md`),
   and `hermes profile update` replaces a distribution-owned identity file outright. Re-run
   the checker after upgrading Hermes, and after any `hermes profile install`/`update`.

## Pitfalls

- **The contract guides, it does not enforce.** `SOUL.md` sets identity; nothing in the
  runtime checks that the agent obeyed a tier. Enforce mechanically where you can: `[SILENT]`
  in the job prompt, a script that only prints on change, a trigger register short enough to
  read.
- **An unattended agent cannot edit its own identity.** Protected-instruction writes always
  need human approval, so "the agent will tighten its own contract" is false — the loop is:
  the agent records observations, a human promotes a trigger.
- **Approval is not the only gate.** `SOUL.md` is scanned for prompt injection as
  user-authored content: a hit is warned about but the file still loads. A `SOUL.md` owned by
  a profile distribution (`distribution.yaml` lists it) is not user-authored, so a hit blocks
  the content instead — and it gets overwritten on the next distribution update.
- **Old sessions keep the old prompt.** Edits take effect cleanly in a new session; an
  already-running session may still be carrying the previous identity.
- **Do not restate the tiers in every task prompt.** Duplicating the contract in job prompts
  is how three slightly different escalation policies end up in one fleet; the job prompt
  should only carry the job-specific silence rule.
- **Never put project paths, commands, or credentials in the identity file.** Those belong in
  `AGENTS.md`, a skill, or `.env` — and a `SOUL.md` full of them is the file that gets
  truncated first.
- **A quiet agent is not an idle one.** Tier 1 work is invisible by design; if you cannot see
  what the agent did during a quiet week, read the observations log rather than un-muting the
  chat.
- `hermes --ignore-rules` skips `SOUL.md` injection entirely, so a contract test run with
  that flag proves nothing about the deployed identity.

## Verification

- `python3 <skill>/scripts/check_proactivity_contract.py ~/.hermes/SOUL.md` exits `0`, and
  reports `OK — proactivity contract intact` with the file's character count.
- The same command against an unfilled copy of the template fails with `placeholder`,
  `observations-log-missing`, and quiet-hours/suppression warnings — the template is a
  scaffold, not a contract.
- `/context` lists `SOUL.md` under Context files with a token estimate and a state of
  `loaded` (not `truncated`, `shadowed`, or `blocked`).
- End to end: one tick with nothing to report produces no delivery; a tick whose trigger
  fires produces one message naming the event and the decision. Two consecutive triggers
  inside the suppression window produce one message, not two.
