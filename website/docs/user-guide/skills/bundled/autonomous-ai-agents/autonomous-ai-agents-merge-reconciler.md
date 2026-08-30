---
title: "Merge Reconciler — Neutral third-party resolution of agent merge conflicts"
sidebar_label: "Merge Reconciler"
description: "Neutral third-party resolution of agent merge conflicts"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Merge Reconciler

Neutral third-party resolution of agent merge conflicts.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/autonomous-ai-agents\merge-reconciler` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Multi-Agent`, `Git`, `Merge-Conflict`, `Kanban`, `Arbitration` |
| Related skills | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Merge Reconciler

Resolve a git merge conflict between two AGENTS' branches as an impartial third
party. Agents resolving conflicts against a peer's work reliably either
overwrite the peer or abandon their own change — they lack the peer's context
and are biased toward their own side. This skill is the fix: a neutral
reconciler that receives both diffs plus both sides' stated intents and
produces a merged result, like a merge-queue arbiter.

## When to Use

- Two agent branches/worktrees collide during a parallel campaign (kanban
  engineering pipeline, parallel-PR wave, multi-worktree refactor).
- `git merge` or `git rebase` halts on conflicts between two agents' work and
  neither original agent should self-adjudicate.
- Do NOT use for conflicts within a single agent's own work, or for trivial
  lockfile/generated-file conflicts (regenerate those instead).

## Prerequisites

- A repo checkout containing the halted merge, or the two branch names plus
  permission to run the merge yourself.
- Both sides' intent sources: kanban completion summaries (`terminal` running
  `hermes kanban show <task-id>`), PR bodies, or at minimum each branch's
  commit messages.
- The project's build/test command, if one exists.

## How to Run

**Standalone** — a human (or agent) invokes this skill inside the conflicted
repo: load the skill, then follow the Procedure top to bottom.

**Spawned neutral agent** — the preferred shape in multi-agent campaigns:

- `delegate_task`: spawn a subagent whose task message contains the repo path,
  both branch names, and both sides' intent summaries verbatim, plus an
  instruction to follow this skill.
- Kanban-native: create a reconciliation card assigned to a **third profile**
  (not either worker's profile) with BOTH conflicted cards linked as parents —
  `kanban_create(title="reconcile branch-a x branch-b", assignee="reconciler",
  parents=["t_a", "t_b"])`. The parent links carry both sides' completion
  summaries into the reconciler's context automatically; the card body should
  name the repo path and the two branches.

## Quick Reference

| Hunk class | Definition | Resolution |
|---|---|---|
| disjoint-intent | The two changes serve different goals and can coexist | Combine both |
| same-question-different-answer | Both sides answered one design question differently | Pick ONE per stated intents; surface the decision |
| superseded | One side's premise no longer holds after the other's change | Keep the surviving side; note why |

Impartiality contract: never favor the side that spawned you; touch ONLY
conflicted regions (no drive-by edits); every design-question pick must appear
explicitly in the hand-back summary.

## Procedure

### 1. Gather both sides

- Run via `terminal`: `git status` (confirm the conflicted state and list
  conflicted files), `git merge-base <A> <B>`, then for each side
  `git log --oneline <base>..<side>` and `git diff <base>..<side> -- <file>`
  for every conflicted file. In a halted merge, `HEAD` is one side and
  `MERGE_HEAD` is the other.
- Collect each side's intent: `hermes kanban show <task-id>` for completion
  summaries/metadata, or the PR body, or the commit messages from the log
  above. Write down one sentence of intent per side before touching any file.
- Done when: you can state both intents in your own words and have both diffs
  for every conflicted file.

### 2. Classify every conflicted hunk

- Open each conflicted file with `read_file` and locate each
  `<<<<<<<`/`=======`/`>>>>>>>` block.
- Assign each hunk exactly one class from the Quick Reference table, judging
  by the stated intents — not by which change looks nicer.
- If a single hunk contains multiple independent decisions (e.g., new logic
  that combines cleanly PLUS a styling/rounding choice both sides answered
  differently), decompose it into sub-decisions and classify each one.
- A single file often mixes classes: one hunk may be a design collision while
  a neighboring hunk is disjoint. Classify per hunk, not per file.
- Done when: every hunk has a written class and a one-line rationale.

### 3. Resolve under the impartiality contract

- Edit each hunk with `patch` (or `write_file` for whole-file rewrites):
  - disjoint-intent → merge both changes so each intent is fully served.
  - same-question-different-answer → pick the answer that best serves the
    STATED intents (e.g., an intent of "strict validation" beats "quick
    default" if the task required correctness). Never split the difference
    into a hybrid neither side asked for.
  - superseded → keep the surviving side; delete the dead premise.
- Never favor the side that spawned you. If intents genuinely tie, escalate
  (block the kanban card / report back) rather than guess.
- Change nothing outside conflict markers — no formatting, renames, or
  opportunistic fixes.
- `git add` each resolved file via `terminal`.
- Done when: `search_files` finds no `<<<<<<<` markers in the repo and every
  resolved file is staged.

### 4. Verify

- Run the project's build/tests via `terminal`; at minimum import/execute the
  touched modules. Both intents must be observable in the merged behavior
  (e.g., side A's new semantics AND side B's disjoint addition both present).
- Complete the merge: `git commit` (the default merge message plus a body
  listing hunk decisions is fine).
- Done when: verification passes and the merge commit exists.

### 5. Hand back

- Produce a completion summary naming EVERY hunk decision:
  `file:lines — class — which side(s) kept — rationale`. For every
  same-question-different-answer hunk, state the design question and the
  answer you picked so a human can veto it — never bury a design call.
- Kanban: `kanban_complete(summary=...)`. Standalone: print the summary.
- Done when: the summary is delivered and lists all hunks.

## Pitfalls

- **Self-favoring**: if you were spawned by one of the conflicting agents,
  you are structurally biased — state this and weigh the other side's intent
  deliberately. Prefer the third-profile shape so this never arises.
- **Splitting the difference** on a design collision produces a hybrid nobody
  designed; pick one answer and surface it.
- **Per-file classification**: files usually mix hunk classes; classifying a
  whole file as one class silently drops a disjoint change.
- **Drive-by edits** make the merge unreviewable and steal decisions from the
  original agents.
- **Missing intents**: commit messages alone can be thin; prefer kanban
  completion summaries or PR bodies. If neither side's intent is recoverable,
  escalate instead of guessing.
- **Repeat offenders**: repeated conflicts on the SAME file across rounds are
  a hotspot signal, not routine reconciliation work — flag it (e.g. a
  `hotspot: <path> — <reason>` kanban comment) so the orchestrator decomposes
  that file, rather than serially reconciling every new collision on it.

## Verification

- `git status` shows a clean tree on the target branch with a merge commit.
- No conflict markers remain (`search_files` pattern `<<<<<<<`).
- Build/tests pass; both sides' intents are demonstrably present or the
  dropped one is explicitly named in the summary.
- The hand-back summary enumerates every hunk with class and rationale.
