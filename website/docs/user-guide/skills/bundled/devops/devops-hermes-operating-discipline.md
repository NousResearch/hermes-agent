---
title: "Hermes Operating Discipline"
sidebar_label: "Hermes Operating Discipline"
description: "Use when modifying Hermes Agent runtime state, crons, skills, plugins, memory, backups, profiles, or Mission Control"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Operating Discipline

Use when modifying Hermes Agent runtime state, crons, skills, plugins, memory, backups, profiles, or Mission Control. Enforces plan-act-verify discipline and protects profile boundaries.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/devops/hermes-operating-discipline` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | macos, linux |
| Tags | `hermes`, `ops`, `crons`, `memory`, `skills`, `profiles`, `backups` |
| Related skills | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), `hermes-safe-local-backup`, `hermes-mission-control-dashboards` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Hermes Operating Discipline

## Overview

Use whenever a task changes Hermes itself: config, crons, skills, plugins, memory, profiles, backups, or dashboards. Runtime state is production state.

## When to Use

- Create/update/remove cron jobs.
- Add/patch/delete skills.
- Modify memory or user profile facts.
- Configure plugins/providers/tools.
- Touch another Hermes profile.
- Build Mission Control dashboards or health checks.

## Workflow

1. Load authoritative docs/skill. Use `hermes-agent` for Hermes setup/config questions.
2. Identify profile and blast radius: active profile, target profile, affected files/crons/memory, cross-profile authorization.
3. Back up or verify rollback path.
4. Act with tools: list crons before update/remove; view skills before patch/edit/delete; use memory only for durable facts.
5. Verify: re-list cron, re-read skill/file, run health scripts/tests where applicable, report exact id/path/status.

## Rules

- Memory: save durable facts only; no task progress, PR numbers, issue numbers, or completed-work logs.
- Cron: list before remove/update; future prompts self-contained; recurring jobs silent when no signal.
- Profile: never modify another profile's skills/plugins/cron/memory without explicit direction.
- Repo skills: create with files and git, not local `skill_manage(action=create)`.

## User Simulation Tests

- Stop a cron → list jobs, identify id, pause/remove exact job.
- Remember a correction → save compact declarative memory.
- Add skill in repo → file write + git.
- Hermes config question → load Hermes docs/skill first.
- Cross-profile edit without explicit scope → clarify.

## Common Pitfalls

1. Mutating memory with task progress.
2. Guessing cron ids.
3. Editing another profile by accident.
4. Creating local skills when PR needs repo files.

## Verification Checklist

- [ ] Relevant Hermes skill/docs loaded.
- [ ] Profile/scope identified.
- [ ] Backup/rollback considered.
- [ ] Change made with exact tool/file.
- [ ] Post-change verification done.
- [ ] User report includes id/path/status only.
