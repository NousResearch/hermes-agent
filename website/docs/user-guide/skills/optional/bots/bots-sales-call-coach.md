---
title: "Sales Call Coach — Score sales transcripts and coach one repeatable habit"
sidebar_label: "Sales Call Coach"
description: "Score sales transcripts and coach one repeatable habit"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sales Call Coach

Score sales transcripts and coach one repeatable habit.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/bots/sales-call-coach` |
| Path | `optional-skills/bots/sales-call-coach` |
| Version | `0.1.0` |
| Author | Mark (unsupportedpastels), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `sales`, `coaching`, `transcripts`, `scorecards` |
| Related skills | [`meeting-action-items`](../../bundled/productivity/productivity-meeting-action-items.md), [`document-to-action-items`](../../bundled/productivity/productivity-document-to-action-items.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Sales Call Coach Workflow Skill

Review a supplied transcript and produce an evidence-backed six-part scorecard plus one practice habit. This workflow does not join calls or assume audio can be transcribed.

## When to Use

- The user pastes or uploads a sales-call transcript.
- The user wants specific coaching grounded in quotes or timestamps.
- Do not use as a CRM, prospect contact tool, or employee performance decision system.

## Prerequisites

The first useful task needs a transcript readable with `read_file` or pasted in chat, plus what the rep sells and the call goal. For audio, do not promise transcription: use a configured transcription capability only if it is actually available, otherwise request a transcript.

## How to Run

1. Copy `templates/scorecard.md` to the owning profile workspace with a collision-safe output name.
2. Practice on `references/sample-input.md` or use a user-supplied transcript.
3. Validate evidence and coaching specificity with `references/expected-output-rubric.md`.

## Quick Reference

Score 1–5 only when evidence supports it: talk ratio, discovery, listening, objection handling, value framing, and next step. Mark a dimension insufficient rather than filling gaps.

A weekly coaching routine can be recommended after multiple comparable calls. Do not automatically create a scheduled job; explicit activation must define transcript source, consent, cadence, retention, and delivery.

## Procedure

### 1. Qualify the transcript

Record call goal, offering, participants, timestamp availability, and obvious gaps. Never claim to have heard tone or moments absent from the text. **Done when** source limitations and usable sections are listed.

### 2. Mark observable evidence

Tag rep questions, follow-ups, customer needs, objections, value statements, commitments, and next-step language. Calculate talk share or monologue length only when speaker turns and timing support it. **Done when** each metric is supported or marked unavailable.

### 3. Score six dimensions

Use consistent 1–5 anchors and attach at least one quote or timestamp to each score. Do not average away a missing dimension. **Done when** all six dimensions are scored or explicitly insufficient.

### 4. Choose one coaching habit

Select the highest-leverage repeatable behavior and give two or three concrete changes plus a short practice drill. Keep praise evidence-based. **Done when** the rep knows exactly what to try on the next call.

### 5. Write and verify the scorecard

Use `templates/scorecard.md`, preserve the transcript, and apply `references/expected-output-rubric.md`. **Done when** no claim exceeds the transcript and the output path is new.

## Pitfalls

- Transcript punctuation and speaker labels may be wrong.
- Do not infer emotion, intent, or vocal delivery from plain text.
- A score without a quote or timestamp is not auditable.
- Do not share coaching or establish a recurring routine without explicit approval.

## Verification

- [ ] Six dimensions use evidence or an insufficient-data label.
- [ ] Metrics are computed only when source structure supports them.
- [ ] Advice is limited to two or three changes and one habit.
- [ ] Transcript and generated scorecard remain separate files.
