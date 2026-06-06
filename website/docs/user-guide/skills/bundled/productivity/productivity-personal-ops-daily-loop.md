---
title: "Personal Ops Daily Loop"
sidebar_label: "Personal Ops Daily Loop"
description: "Use when running Jaime's daily/weekly personal operations loop: morning briefing, evening close, calendar reminders, tasks, family logistics, and concise Tel..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Personal Ops Daily Loop

Use when running Jaime's daily/weekly personal operations loop: morning briefing, evening close, calendar reminders, tasks, family logistics, and concise Telegram-ready summaries.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/personal-ops-daily-loop` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | macos, linux |
| Tags | `calendar`, `reminders`, `daily-briefing`, `personal-ops` |
| Related skills | [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace), [`gmail-triage-jaime`](/docs/user-guide/skills/bundled/productivity/productivity-gmail-triage-jaime) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Personal Ops Daily Loop

## Overview

Consolidates calendar, reminders, Gmail highlights, family logistics, and weekly planning into short useful briefings. This is operational clarity, not a full dashboard dump.

## When to Use

- Morning briefing.
- Evening close.
- Weekly radar.
- Calendar event reminders.
- User asks what is today/tomorrow/this week.

## Workflow

1. Compute live date/time first. Relative dates are error-prone.
2. Collect sources: calendar today + next 7 days, reminders due/overdue/flagged, actionable unread Gmail, family/Zoé work items when present.
3. Prioritize time-bound events, required actions, then optional items.
4. Write naturally in Spanish by default: bullets, no tables, no robotic wording.
5. Close loops: suggest marking done when appropriate; do not complete reminders without confirmation.

## Briefing Shape

Morning: calendar today, near week, actionable unread mail, critical tasks.

Evening: pending items, prep for tomorrow, health/sleep nudge if relevant.

## User Simulation Tests

- User asks "mañana" near midnight → compute date live.
- Calendar empty today but busy week → say today is free and mention week.
- Gmail has 10 unread but one actionable → surface one, omit noise.
- User replies "leído" to briefing → acknowledge only.
- Family event with location → include time, person, place.

## Common Pitfalls

1. Off-by-one dates.
2. Dumping every email.
3. Overexplaining simple days.
4. Completing reminders without confirmation.

## Verification Checklist

- [ ] Current date/time checked.
- [ ] Calendar window correct.
- [ ] Gmail noise filtered.
- [ ] Reminders/tasks scoped.
- [ ] Output concise and Telegram-friendly.
