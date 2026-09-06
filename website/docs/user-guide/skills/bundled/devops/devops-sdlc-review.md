---
title: "Sdlc Review — Review Kanban handoffs and route verified outcomes"
sidebar_label: "Sdlc Review"
description: "Review Kanban handoffs and route verified outcomes"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sdlc Review

Review Kanban handoffs and route verified outcomes.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/devops\sdlc-review` |
| Version | `1.2.0` |
| Author | Jakub Wolniewicz (@frizikk) + Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `kanban`, `review`, `quality`, `verification` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# SDLC Review Skill

Independently verify work handed from a Kanban implementation run to the review lane, then approve it, request changes, or escalate. Review the deliverable and its evidence; do not take over the implementer's work.

## When to Use

All must hold: the dispatcher spawned you for a task claimed from the `review` lane; an implementer submitted a `review_requested` handoff; the task needs an independent verdict before completion. Don't use it for a separate downstream review card — that is ordinary implementation work with a review-oriented spec and runs its own lifecycle.

## Prerequisites

- A Kanban worker context with the current task and run identifiers.
- Native tools: `kanban_show`, `kanban_comment`, `kanban_complete`, `kanban_request_changes`, `kanban_block`; plus `read_file`, `search_files`, and `terminal` for code deliverables.
- The task spec, acceptance criteria, handoff summary, and prior run history, all via `kanban_show`.

## How to Run

Loaded automatically by the review dispatcher. Always start with `kanban_show`; then read the latest `review_requested` handoff, inspect the actual deliverable, run relevant verification, choose exactly one verdict, and record concrete evidence in the terminal Kanban transition.

## Quick Reference

| Verdict | When | Terminal action |
|---|---|---|
| Approve | Acceptance criteria and verification pass | `kanban_complete` |
| Request changes | Correctable implementation defects remain | `kanban_comment`, then `kanban_request_changes` |
| Escalate | A human decision or external prerequisite is required | `kanban_block` |

A requested-changes transition returns the task to its original implementer; persisted reviewer provenance routes any re-review back to the same reviewer profile.

## Review Lenses

Vary the inspection each round — decorrelated lenses catch different defect classes, while repeating a lens mostly re-finds what it already found. Derive the round from history: count `changes_requested` entries in the "Prior attempts on this task" section (also visible in `kanban_show`); the round is that count plus one.

| Round | Lens | How to apply |
|---|---|---|
| 1 | Artifact | Read the diff cold, before the handoff summary; form an independent judgment, then investigate every mismatch against the narrative. |
| 2 | Execution | Check out and actually run it via `terminal` — build, test, exercise the reported behavior; verify each claim empirically. |
| 3+ | Contract | Re-read the ORIGINAL task and acceptance criteria, audit the deliverable strictly, and confirm every prior `kanban_request_changes` item landed. |

Procedure duties apply on every round; the lens only sets what you lead with.

## Procedure

1. **Orient** from `kanban_show`: original task body and acceptance criteria, latest implementation summary and metadata, changed files/commits/test evidence, prior comments and review findings. Treat the handoff as a claim to verify, not proof.
2. **Compare requested vs delivered.** Map every acceptance criterion to concrete evidence; note omissions, changed semantics, and scope drift.
   - *Code:* inspect changed paths and callers with `read_file`/`search_files`; via `terminal`, review the diff and run focused tests, lint, type checks, or build; exercise the failure path and one control path; check error handling, edge cases, concurrency, data preservation, security, and cross-platform behavior; confirm tests assert behavior, not source snapshots.
   - *Non-code:* inspect the full deliverable; check correctness, completeness, formatting, and provenance; validate referenced URLs or facts when they affect the verdict.
3. **Choose one verdict** and record evidence:
   - **Approve** only when criteria are met and evidence is sufficient: `kanban_complete` with a summary naming the exact checks that passed, plus a `review_outcome: approved` metadata field and any non-blocking caveat.
   - **Request changes** for specific, correctable defects: first `kanban_comment` with numbered findings (file/artifact + defect, where it is, how it reproduces, why it violates the task, the minimum fix), then `kanban_request_changes` with a concise reason.
   - **Escalate** only when a human decision or external prerequisite is required: `kanban_block` with reason `escalation: <decision or prerequisite>`.
4. **Preserve role separation.** Never edit the implementation as reviewer; request changes and independently verify the next candidate.

## Pitfalls

- **Rubber-stamping:** a passing handoff summary is not independent evidence.
- **Reviewer implementation:** editing the deliverable hides ownership and weakens re-review.
- **Vague findings:** "needs work" gives no reproducible correction target.
- **Style-only blocking:** don't request changes for preference nits when behavior and repo standards pass.
- **Skipping prior rounds:** re-review must confirm the corrections AND preserved passing behavior.
- **Blockers for rework:** correctable defects go to `kanban_request_changes`; reserve `kanban_block` for real external blockers or human decisions.
- **Approving without evidence:** every approval summary names the checks or artifacts actually inspected.

## Verification

Before submitting the verdict, confirm:

- [ ] `kanban_show` read for the current task and run; every acceptance criterion mapped to evidence.
- [ ] Actual deliverable inspected; relevant focused checks run, or an explicit reason recorded when execution was impossible.
- [ ] Prior requested changes re-tested; unrelated regressions and scope changes considered.
- [ ] Exactly one terminal action, carrying concrete non-secret evidence, with no reviewer edits to implementation files.
