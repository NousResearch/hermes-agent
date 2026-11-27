---
title: "Agent Merge Conflict Arbiter — Safety-first arbiter for Git branch conflicts"
sidebar_label: "Agent Merge Conflict Arbiter"
description: "Safety-first arbiter for Git branch conflicts"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Agent Merge Conflict Arbiter

Safety-first arbiter for Git branch conflicts.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/agent-merge-conflict-arbiter` |
| Path | `optional-skills/autonomous-ai-agents/agent-merge-conflict-arbiter` |
| Version | `1.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Multi-Agent`, `Git`, `Merge-Conflict`, `Kanban`, `Arbitration` |
| Related skills | [`hermes-agent`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Agent Merge-Conflict Arbiter

Resolve a Git conflict between two branches or worktrees as an impartial third
party. Reconstruct each side's intent, classify every conflict hunk, and
produce a reviewable recommendation or an authorized resolution. This skill is
portable across orchestrators: branch names, worktree paths, pull requests,
kanban cards, and commit history are all valid sources of context.

## When to Use

- Two agent branches or worktrees collide during a parallel campaign.
- A `git merge` or `git rebase` is paused on conflicts and neither original
  agent should self-adjudicate.
- A maintainer wants a hunk-by-hunk reconciliation before changing repository
  state.
- Do not use this for conflicts entirely within one agent's own change, or for
  lockfiles/generated files that should be regenerated from their source.

## Safety and Operating Modes

Start and remain **read-only by default**. Inspection, classification, and a
written recommendation are complete work; never mutate the repository merely
because a conflict exists. Before a state-changing operation, obtain explicit
authorization for that operation. Use this authorization ladder, from least to
most consequential:

1. inspect files, history, metadata, and capabilities;
2. edit conflict markers or other conflicted regions;
3. stage resolved files with `git add`;
4. create a merge or rebase commit;
5. perform other branch operations, including merge, rebase, reset, restore,
   checkout, cleanup, or stash changes;
6. abort an in-progress merge or rebase.

Authorization for one level does not imply authorization for another. Never
silently run `git add`, `git commit`, `git merge`, `git rebase`, `git reset`,
`git checkout`, `git restore`, `git clean`, or destructive stash commands.
Never overwrite unrelated work or use destructive recovery as a shortcut.

## Prerequisites

Before analysis, use the native `terminal` tool for Git inspection and
capability probes. Use `read_file` to inspect conflicted content, `patch` for
an authorized narrow edit, and `delegate_task` only when a separately
authorized neutral worker is requested. Identify the repository, current
operation, and available context. Check capabilities before relying on them:

- Confirm the checkout/worktree and run `git status`; record whether a merge or
  rebase is paused and which files are conflicted.
- Confirm the two branch/worktree identities and compute their merge base. In a
  paused operation, `HEAD` and Git's status output identify the current side;
  inspect `.git/MERGE_HEAD` only when needed, and use Git's rebase status for a
  rebase.
- Check that both sides' intent sources are available: task/card summaries,
  pull request descriptions, branch logs, or commit messages. Do not pretend
  to have task-tracker or hosted-service access that is unavailable.
- Identify the project's build/test commands and whether generated-file
  instructions exist. If no reliable command or intent source is available,
  report that limitation and escalate rather than guessing.

## How to Run

**Standalone** — load this skill inside the conflicted repository and follow
Procedure top to bottom.

**Spawned neutral agent** — provide the repo path, both branch/worktree names,
and both sides' intent summaries verbatim. In an orchestrated campaign, assign
reconciliation to a third profile rather than either worker; link both parent
work items when the tracker supports it. Do not depend on a particular
orchestrator, delegate API, tracker, PR host, or shell.

## Quick Reference

| Hunk class | Definition | Resolution |
|---|---|---|
| disjoint-intent | The changes serve different goals and can coexist | Combine both |
| same-question-different-answer | Both sides answer one design question differently | Pick one based on stated intent and evidence |
| superseded | One side's premise no longer holds after the other's change | Keep the surviving side and explain why |
| unresolved-intent-tie | Available evidence cannot fairly distinguish the intents | Do not guess; escalate and leave state unchanged |

Impartiality contract: never favor the side that spawned you, the newer commit,
or the branch with the cleaner presentation. Touch only the authorized regions;
every design choice and every unresolved tie must appear in the hand-back
summary.

## Procedure

### 1. Establish a read-only baseline

- Run `git status`, identify the operation and conflicted files, and preserve
  the initial state in the working notes.
- Run `git merge-base <A> <B>`, then for each side inspect
  `git log --oneline <base>..<side>` and `git diff <base>..<side> -- <file>` for
  every conflicted file.
- Collect one plain-language intent statement per side from the available
  sources. Keep the source and distinguish facts from assumptions.
- Done when: the repository state, capabilities, both diffs, and both intents
  are recorded. If an intent is missing or contradictory, classify the affected
  decision as an unresolved-intent tie instead of inferring it.

### 2. Classify every hunk

- Open each conflicted file and locate every `<<<<<<<`/`=======`/`>>>>>>>`
  block. Inspect surrounding code and relevant tests, not just marker text.
- Assign each hunk exactly one class above, using stated intent and project
  requirements rather than code style or branch position.
- Decompose a hunk containing multiple independent decisions and classify each
  sub-decision. A file may contain several classes.
- Done when: every hunk has a class and a one-line evidence-based rationale.

### 3. Resolve only with authorization

For an inspection-only request, stop with the recommendation. If mutation is
explicitly authorized at the required ladder level:

- disjoint-intent → combine both changes so each intent is fully served;
- same-question-different-answer → choose the answer best supported by the
  stated intents and project evidence; never invent a compromise hybrid;
- superseded → keep the surviving side and document the obsolete premise;
- unresolved-intent-tie → do not edit; report the exact question and the
  missing evidence needed for a human decision.

Edit only the authorized conflict regions. Do not make formatting, cleanup,
renames, generated-file, or unrelated improvements. Stage only with separate
authorization. Commit only with separate authorization and include the hunk
decisions in the commit or hand-back summary.

### 4. Safe abort and recovery

Abort is a recovery action, not a cleanup shortcut. If an abort is requested:

1. Stop editing and report the current state and any uncommitted work.
2. Confirm which operation is active using `git status` and Git's operation
   metadata; never issue both abort commands.
3. Confirm explicit authorization to abandon that operation.
4. Run only the matching command: `git merge --abort` for a merge or
   `git rebase --abort` for a rebase.
5. Run `git status` afterward and report exactly what changed.

If the matching abort cannot restore the baseline, stop. Do not escalate to
reset, restore, clean, or stash destruction; preserve evidence and ask for a
recovery decision. If an authorized edit fails, leave the repository untouched
when possible and report the failure rather than improvising recovery.

### 5. Verify the result

Verification must match the requested operation and the project's capabilities:

- no conflict markers remain in the intended files;
- the diff is limited to authorized paths and regions;
- each retained intent is observable in the resolved artifact, or any dropped
  intent is explicitly named;
- project formatting, build, and focused tests pass when available;
- generated files are changed only when the repository's documented generator
  was run, and the generated diff contains no unrelated churn;
- the final `git status` and diff are reported. A clean tree with a merge commit
  is required only when commit authorization was granted; a recommendation may
  and should finish without staging or committing.

### 6. Hand back

Return a concise evidence register with:

`file:lines — class — side(s) kept — rationale — verification`

Include the initial state, capabilities unavailable, every same-question choice,
every unresolved intent tie, authorized mutations, test commands/results, and
whether a commit was created. For tracker-backed work, post the same summary to
the authorized card; otherwise print it for the requesting human or agent.

## Pitfalls

- **Self-favoring:** a resolver spawned by one worker is structurally biased;
  use a third profile and weigh both intent statements.
- **Splitting the difference:** a hybrid can satisfy neither design; choose one
  supported answer or escalate an intent tie.
- **Per-file classification:** neighboring hunks can have different classes;
  classify each hunk.
- **Drive-by edits:** unrelated cleanup makes the decision unreviewable.
- **Missing intent or capability:** state the gap and escalate; do not fabricate
  tracker context, tests, or authorization.
- **Generated files:** resolve source inputs first and use the documented
  generator; do not hand-edit generated output or treat regenerated noise as a
  conflict decision.
- **Repeat offenders:** repeated conflicts in one file are a hotspot signal;
  flag the path and reason so the orchestrator can decompose future work.

## Verification

- [ ] Initial repository state, operation, branches, and capabilities recorded.
- [ ] Both intents and the merge base are cited.
- [ ] Every hunk has a class, rationale, and explicit outcome.
- [ ] Unresolved intent ties are escalated rather than guessed.
- [ ] No unauthorized mutation or destructive recovery occurred.
- [ ] No conflict markers remain in the intended scope.
- [ ] Focused project checks pass, or unavailable checks are reported.
- [ ] Generated-file safeguards were followed.
- [ ] Final diff/status and hand-back summary are complete.
