---
title: "Recruiting Coordinator — Coordinate interview loops without making hiring decisions"
sidebar_label: "Recruiting Coordinator"
description: "Coordinate interview loops without making hiring decisions"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Recruiting Coordinator

Coordinate interview loops without making hiring decisions.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/bots/recruiting-coordinator` |
| Path | `optional-skills/bots/recruiting-coordinator` |
| Version | `0.1.0` |
| Author | Mark (unsupportedpastels), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `recruiting`, `interviews`, `scheduling`, `drafts` |
| Related skills | [`google-workspace`](../../bundled/productivity/productivity-google-workspace.md), [`meeting-action-items`](../../bundled/productivity/productivity-meeting-action-items.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Recruiting Coordinator Workflow Skill

Organize job-related interview logistics from a supplied tracker and draft the next operational action. It does not source, rank, recommend, reject, or decide about candidates.

## When to Use

- The user supplies an open role, panel, timezone, and loop tracker.
- The user wants stalled steps, prep packets, or scheduling drafts surfaced.
- Do not use for candidate ranking, hiring decisions, background inference, or candidate coaching.

## Prerequisites

The first useful task needs pasted or local role/loop data readable with `read_file`; private calendar or email login is not required. `google-workspace` may support connected calendars when configured, but skill installation does not confer access. Keep only job-related information.

## How to Run

1. Copy `templates/interview-loop-tracker.csv` into the owning profile workspace without replacing the original.
2. Use `references/sample-input.md` for a synthetic offline coordination pass or ingest user data.
3. Check the result with `references/expected-output-rubric.md`.

## Quick Reference

| Status | Coordinator action |
|---|---|
| Missing availability | Draft one availability request |
| Panel gap | Flag required interview competency coverage |
| Scorecard overdue | Draft a neutral reminder |
| Ready for debrief | Propose times; do not book |

A daily stalled-loop review may be recommended. Do not automatically create a scheduled job; explicit activation must define tracker source, cadence, retention, authorized viewers, and draft-only behavior.

## Procedure

### 1. Set role and privacy boundaries

Record role, stage definitions, timezone, panel, authorized data sources, and allowed actions. Exclude protected or sensitive personal characteristics. **Done when** scope is job-related and the default is draft-only.

### 2. Validate tracker facts

Reconcile names, stages, dates, availability, interviewers, and scorecard state against supplied records. Do not infer missing contact details or feedback. **Done when** conflicts and missing facts are visible.

### 3. Identify operational blockers

Flag absent availability, panel gaps, stale scorecards, timezone conflicts, and debrief dependencies. Do not rank candidates or convert logistics into a recommendation. **Done when** every stalled loop has one factual blocker and owner.

### 4. Draft the smallest next action

Prepare a scheduling note, reminder, or interviewer packet using only supplied facts. Label it DRAFT. Do not book or send, and do not state that a candidate should advance or be rejected. **Done when** each draft has a recipient placeholder, purpose, and facts-to-confirm list.

### 5. Update a safe tracker copy

Write a new tracker in the owning profile workspace and preserve the source. Apply `references/expected-output-rubric.md`. **Done when** no protected inference, ranking, booking, or outbound action occurred.

## Pitfalls

- Calendar availability is not consent to book.
- Missing feedback is not negative feedback.
- Interview logistics must not contain medical, family, age, ethnicity, disability, religion, or other protected inferences.
- Do not turn a recurring recommendation into an active routine without explicit approval.

## Verification

- [ ] Notes are job-related and contain no protected-characteristic inference.
- [ ] No candidate is ranked, recommended, advanced, or rejected.
- [ ] All messages and times remain drafts/proposals.
- [ ] Tracker source is preserved and output is access-appropriate.
