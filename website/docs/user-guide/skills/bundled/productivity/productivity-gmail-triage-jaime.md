---
title: "Gmail Triage Jaime"
sidebar_label: "Gmail Triage Jaime"
description: "Use when triaging Jaime's Gmail summaries, marking messages read, separating noise from actionable mail, and preserving account boundaries"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Gmail Triage Jaime

Use when triaging Jaime's Gmail summaries, marking messages read, separating noise from actionable mail, and preserving account boundaries. Never deletes or trashes messages without explicit confirmation.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/gmail-triage-jaime` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | macos, linux |
| Tags | `gmail`, `email`, `triage`, `unread`, `personal-ops` |
| Related skills | [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Gmail Triage for Jaime

## Overview

Handles the repeated Gmail loop: unread digests, "leído" replies, noise classification, and action extraction. Reduce cognitive load without losing important mail or mixing accounts.

## When to Use

- User replies "leído" to a Gmail digest.
- A cron reports new Gmail messages.
- User asks to silence recurring noise.
- A message may need follow-up, calendar, reminder, or FacOps handoff.

## Account Boundary Rule

Keep accounts separate: personal Gmail/calendar for normal Jaime ops; scoped invoice/task tracking account for FacOps only. Invoices/receipts for autónomo go to FacOps, not personal inbox logic.

## Workflow

1. Identify digest type: single alert, multi-message summary, noise warning, or actionable message.
2. Interpret reply: "Leído"/"Hecho" means acknowledge and mark only referenced messages read if tools are available; "silencia este" means propose filter/unsubscribe before durable suppression; "para Excel de autónomo" routes to FacOps.
3. Classify conservatively: important, normal, or noise.
4. Never delete/trash/archive/destructively filter without explicit confirmation.
5. Report briefly: simple "leído" gets "Hecho."; actionable mail gets next action.

## User Simulation Tests

- Patreon policy update + "Leído" → acknowledge, no lecture.
- Santander movement notice + "Leído" → acknowledge, do not delete.
- "Silencia este" to bank noise → propose narrow filter, ask before applying.
- Receipt + "para Excel" → route to FacOps.
- Mixed digest with security alert → do not mark everything noise.

## Common Pitfalls

1. Deleting obvious spam; Jaime forbids deletion without confirmation.
2. Mixing Gmail identities.
3. Over-talking simple acknowledgements.
4. Silencing finance/security too broadly.

## Verification Checklist

- [ ] Referenced messages only, not whole inbox.
- [ ] No destructive actions without confirmation.
- [ ] Account scope respected.
- [ ] Actionable messages surfaced.
- [ ] Reply is short unless there is a decision.
