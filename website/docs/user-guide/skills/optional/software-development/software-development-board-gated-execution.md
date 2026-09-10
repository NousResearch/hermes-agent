---
title: "Board Gated Execution — Gate agent work on an issue board so tasks cannot drift"
sidebar_label: "Board Gated Execution"
description: "Gate agent work on an issue board so tasks cannot drift"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Board Gated Execution

Gate agent work on an issue board so tasks cannot drift.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/software-development/board-gated-execution` |
| Path | `optional-skills/software-development/board-gated-execution` |
| Version | `0.1.0` |
| Author | Nidhal Gharbi (NidhalxMRR), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Workflow`, `GitHub`, `Issues`, `Focus`, `Project-Management` |
| Related skills | [`github`](/docs/user-guide/skills/bundled/software-development/software-development-github), [`weekly-review-planning`](/docs/user-guide/skills/bundled/productivity/productivity-weekly-review-planning), [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Board-Gated Execution Skill

Long agent sessions lose work silently. A feature reaches "service running, route
written, UI missing, nothing committed", attention moves to an adjacent question,
and the half-built state is invisible until someone asks "did you finish that?".
This skill makes an issue board the source of truth for what to work on, and
makes consulting it a gate rather than a habit.

It does not replace the `github` skill (creating, triaging and closing issues,
and carrying one to a merged PR). It answers the question that comes before all
of that: *which* issue, and *may I start something new at all*.

## When to Use

- A session spans many turns and several topics, and work risks being left open
- The user asks "did you finish X?" about something you started earlier
- You are about to start a new task while another is still incomplete
- A project's open threads live only in chat history, not in a tracker

Don't use for: single-task sessions, repos where the user tracks work elsewhere
(Linear, Jira) without a mirror, or one-off questions with no follow-up work.

## Prerequisites

- `gh` authenticated with `repo` scope: `terminal(command="gh auth status")`
- A GitHub repo with Issues enabled
- Python 3.9+ for the helper scripts

The GitHub Projects **board view** needs `read:project`, which most tokens lack.
This skill deliberately uses **Issues plus labels** instead, so it works with the
default `repo` scope. Do not make Projects a hard dependency.

## Procedure

### 1. Create the label vocabulary

Run `scripts/setup-labels.sh`. It creates `area:*`, `blocked:human`,
`in-progress`, `needs-tests` and `P0`-`P3`, and is idempotent (create, else
edit). *Completion criterion:* `terminal(command="gh label list")` shows every
label the gate depends on — at minimum `in-progress`, `blocked:human`, `P0`-`P3`.

### 2. Seed the board from what exists only in conversation

Enumerate every open thread — half-built features, deferred decisions, missing
docs, things blocked on the user — and file one issue each. Use
`scripts/seed-issues.py` as the template: it embeds a `<!-- seed:<key> -->`
marker in each body and searches for that marker before creating, so re-runs
create nothing.

Every issue needs **acceptance criteria that can be settled by evidence**, not by
opinion. "Voice UI done" is not a criterion; "the user can hold a button, speak,
and hear a reply on the live site" is.

*Completion criterion:* running the seeder twice creates zero issues the second
time, and no open thread from the conversation is missing from the board.

### 3. Gate every start on the board

Before starting any work:

```
terminal(command="python3 scripts/next-task.py")
```

It prints the single issue to work on. Obey it. If what you were about to do is
not that issue, then either it is not the next thing, or it is not on the board —
and unboarded work is exactly how drift starts.

Selection order, enforced mechanically:

1. **WIP limit of one.** Two unblocked `in-progress` issues means one is
   drifting; the script exits 1 and refuses to name new work.
2. **Finishing beats starting.** An `in-progress` issue wins over a
   higher-priority fresh one.
3. **Priority otherwise:** P0 > P1 > P2 > P3.

`blocked:human` issues are **never selected**. They need a decision or a
credential the agent cannot supply, so "working" them means waiting, and waiting
is indistinguishable from drift. Report them to the user instead.

*Completion criterion:* the script exits 0 and names exactly one issue, or exits
1 with a refusal you then resolve.

### 4. Mark and close honestly

- Starting: `terminal(command="gh issue edit <n> --add-label in-progress")`
- Shipping: reference the issue in the commit body, then
  `terminal(command="gh issue close <n>")`

An issue left open after the work shipped teaches the board to lie; an issue
closed before the work is verified teaches it to lie faster. Close on evidence —
tests green, artifact exercised — never on intent.

*Completion criterion:* `gh issue list --state open` contains nothing already
delivered.

### 5. Prove the gate refuses

A gate that only ever says "go" is decoration. Run `scripts/gate-check.py`: it
marks a second issue `in-progress`, asserts the gate exits 1, names both
offenders and declines to name a task, then restores the label.

*Completion criterion:* all checks pass, and the board is left exactly as found.

## Quick Reference

```bash
bash scripts/setup-labels.sh                 # one-time label vocabulary
python3 scripts/seed-issues.py               # idempotent board seeding
python3 scripts/next-task.py                 # THE GATE — run before starting work
python3 scripts/gate-check.py                # prove the gate refuses
gh issue edit <n> --add-label in-progress    # mark started
gh issue close <n>                           # close on evidence
```

## Pitfalls

1. **A board you do not consult is just a nicer place to lose work.** The gate is
   worthless as an intention; run the script.
2. **Do not let the gate pick a `blocked:human` issue.** Surfacing a blocker to
   the user is progress; silently waiting on it is not.
3. **Acceptance criteria that need a judgment call cannot close an issue.** Write
   criteria a script or a human can settle by looking at evidence.
4. **Seeding without markers creates duplicates on every re-run.** The marker
   comment is what makes the seeder idempotent.
5. **GitHub Projects needs `read:project`.** Use Issues plus labels so the gate
   works with the `repo` scope alone; ask the user before requesting more.
6. **Priority labels only sort; they do not decide.** An `in-progress` P3 still
   outranks a fresh P1, because finishing beats starting.
7. **Closing an issue because the code exists.** Tests written after the fact
   encode what the code does, not what it should do — close on verified
   behaviour.

## Verification

- `python3 scripts/next-task.py` exits 0 and names one issue
- `python3 scripts/gate-check.py` passes every check and restores the board
- Running the seeder twice creates zero issues on the second run
- `gh issue list --state open` matches the work that is genuinely outstanding
