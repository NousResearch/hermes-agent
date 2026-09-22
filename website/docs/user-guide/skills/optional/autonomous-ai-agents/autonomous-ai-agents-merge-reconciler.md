---
title: "Merge Reconciler — Use when reconciling conflicting Git branches safely"
sidebar_label: "Merge Reconciler"
description: "Use when reconciling conflicting Git branches safely"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Merge Reconciler

Use when reconciling conflicting Git branches safely.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/merge-reconciler` |
| Path | `optional-skills/autonomous-ai-agents/merge-reconciler` |
| Version | `1.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Git`, `Merge-Conflict`, `Arbitration`, `Reconciliation` |
| Related skills | [`hermes-agent`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Merge Reconciler

Act as a neutral third party for a Git conflict between two branches or
worktrees. Reconstruct both sides' intent, classify each conflict hunk, and
produce a reviewable resolution. Do not favor the branch that invoked you.

This is a standalone Git workflow. It does not require a task tracker, a
specific orchestrator, a delegate API, a PR host, or a particular shell.
Commit, staging, merge, rebase, and abort operations are **opt-in**; the
default workflow is inspection and recommendation only.

## When to Use

- Two branches or worktrees implement changes that collide.
- A merge or rebase is already paused on conflicts and an impartial resolver
  should assess the result.
- A human wants a hunk-by-hunk reconciliation before deciding whether to apply
  it.

Do not use this for a conflict entirely within one person's change, or for a
lockfile/generated file that should be regenerated from source.

## Safety and Operating Modes

Start in **read-only mode**. Before any state-changing command, ask for or
confirm explicit authorization for that operation. Treat these as separate
permissions:

- inspect files and history;
- edit conflict markers;
- stage resolved files with `git add`;
- create a merge/rebase commit;
- run `git merge --abort` or `git rebase --abort`.

Never silently run `git add`, `git commit`, `git merge`, `git rebase`, `git
reset`, `git checkout`, `git restore`, `git clean`, or destructive stash
commands. A recommendation can be complete without staging or committing.

### Safe abort

Abort is a recovery action, not a cleanup shortcut.

1. Stop editing and report the current state before aborting.
2. Confirm which operation is in progress (`git status`, and inspect
   `.git/MERGE_HEAD` or rebase state through Git's status output).
3. Confirm that the user explicitly wants the in-progress operation abandoned.
4. Use only the matching command: `git merge --abort` for a merge or `git
   rebase --abort` for a rebase.
5. Read `git status` afterward and report exactly what changed. If the abort
   fails, stop; do not substitute `reset --hard`, `clean`, or a guessed
   recovery.

If the tree contains unrelated local edits, do not stash, overwrite, or discard
them. Ask the user to provide a clean worktree, an isolated worktree, or an
explicitly scoped instruction for handling those edits.

## Capability Checks

Before analysis, verify the capabilities needed for the requested mode. Use
ordinary terminal commands and Hermes file tools; do not assume a Unix shell,
GNU utilities, a task tracker, or a hosted Git service.

- **Git:** run `git --version`; stop if Git is unavailable.
- **Repository:** run `git rev-parse --show-toplevel`; use that path as the
  repository root. Confirm the requested branches/worktrees exist with
  `git branch --list` and, when relevant, `git worktree list`.
- **State:** run `git status --short --branch`. Record whether a merge or rebase
  is already in progress and whether unrelated changes are present.
- **Read/write ability:** confirm that the relevant files can be read. If edits
  are requested, confirm the target worktree is writable before changing it.
- **Verification:** discover the project's documented build/test command from
  repository guidance (`README`, contributor docs, or project configuration).
  If no command is known, report that verification is unavailable rather than
  inventing one.
- **Platform limits:** if a requested shell command is unavailable on the
  current platform, use Git's cross-platform output or Hermes tools instead;
  report any capability that remains unavailable. Do not silently replace a
  missing check with an unverified assumption.

## Inputs and Intent

You need:

- the repository/worktree path;
- the two branch names, or the current paused merge/rebase state;
- one intent statement for each side, preferably from a PR description,
  design note, issue, or commit message;
- the project's verification command, if one exists.

Intent must be attributable to a source. If a side has only vague commit
messages, say so. If either intent cannot be recovered, do not infer it from
which code looks more polished: mark the decision unresolved and escalate it.

## Hunk Classification

Classify every conflict hunk independently. Split a hunk into sub-decisions
when it contains multiple independent questions.

| Class | Meaning | Default treatment |
|---|---|---|
| `disjoint-intent` | The changes serve different goals and can coexist | Combine both without weakening either intent |
| `same-question-different-answer` | Both sides answer one design question differently | Select one only when the stated intents clearly decide it |
| `superseded` | One side's premise no longer holds after the other change | Keep the surviving side and explain why |
| `unresolved-intent-tie` | The evidence does not justify choosing between materially different answers | Do not guess; stop at a recommendation and escalate |

## Procedure

### 1. Establish a safe baseline

Run, in read-only mode:

```text
git --version
git rev-parse --show-toplevel
git status --short --branch
git branch --list
git worktree list
git diff --name-only
```

If the current operation is a paused merge or rebase, identify it from
`git status` before inspecting files. Do not use `git diff` output as proof that
there are no untracked or unrelated changes; `git status --short` is required.

For two ordinary branches, find their common ancestor and inspect each side:

```text
git merge-base feature-a feature-b
git log --oneline BASE..feature-a
git log --oneline BASE..feature-b
git diff BASE..feature-a -- path/to/file
git diff BASE..feature-b -- path/to/file
```

Replace `BASE` with the actual merge-base value; do not type the literal word
`BASE`. In a halted merge, compare `HEAD` and `MERGE_HEAD` after obtaining the
other commit with `git rev-parse MERGE_HEAD`.

Record the branch names, commit IDs, conflicted paths, unrelated local edits,
and each side's intent before editing anything.

### 2. Inspect and classify

For each conflicted path, read the whole relevant file and locate every
`<<<<<<<`, `=======`, and `>>>>>>>` marker. Compare the marker regions with the
branch diffs and intent statements. Write a table before editing:

```text
path:lines | class | side(s) represented | intent evidence | rationale
```

Every hunk needs exactly one class and a rationale. A file may contain several
classes.

### 3. Reconcile without bias

- `disjoint-intent`: preserve both complete behaviors, integrating them only
  where the code can express both without changing either requirement.
- `same-question-different-answer`: choose one answer only if the intent
  evidence resolves the design question. State the question and selected side
  explicitly.
- `superseded`: retain the side whose premise still applies and explain the
  discarded premise.
- `unresolved-intent-tie`: leave the hunk unresolved (or provide clearly
  labeled alternatives in a report), identify the missing decision-maker or
  requirement, and stop. Never split the difference into an unrequested
  hybrid.

Touch only conflict regions. Do not reformat, rename, refactor, update
unrelated files, or “fix” nearby code. If edits are authorized, use the
smallest applicable patch and show the resulting diff.

### 4. Verify and optionally apply state changes

At minimum, inspect the proposed result and confirm that every intended hunk
has a recorded outcome. If an edit was authorized, run:

```text
git diff --check
git diff -- path/to/resolved-file
```

Then run the repository's documented build/test command. If it is absent or
fails for an environmental reason, report that plainly. Do not claim a merge
is verified merely because conflict markers disappeared.

Only after the user separately authorizes staging may you run:

```text
git add -- path/to/resolved-file
```

Re-check `git status --short` and `git diff --cached --check`. Only after a
separate explicit authorization may you complete the operation. For a paused
merge, that is ordinarily:

```text
git commit
```

For a paused rebase, follow the repository's intended rebase flow and ask
before each state-changing step. Do not create a commit as part of a
read-only reconciliation report.

### 5. Hand back a decision record

Report every hunk using:

```text
path:lines — class — side(s) kept or unresolved — rationale
```

For each design collision, include the design question, the evidence used, and
the selected answer. For every unresolved tie, name the missing decision and
state that no resolution was applied. Include verification results, whether
files were edited, whether anything was staged, and whether a commit or abort
was performed.

## Ordinary Git Examples

### Compare two branches without changing the repository

```text
git merge-base main feature-a
# Copy the returned commit ID as BASE.
git diff BASE..main -- src/config.py
git diff BASE..feature-a -- src/config.py
```

Run these commands from the repository without switching branches; they inspect
history and do not change the worktree.

### Inspect an existing conflict

```text
git status --short --branch
git diff --name-only --diff-filter=U
git diff -- path/to/conflicted-file
```

Resolving a file and staging it are separate actions. Staging is opt-in:

```text
git add -- path/to/conflicted-file
```

### Abort only with explicit authorization

```text
git status
# Confirm this is the intended operation and obtain approval.
git merge --abort
# or, if status identifies a rebase:
git rebase --abort
git status --short --branch
```

## Pitfalls

- **Self-favoring:** judge the stated requirements, not the invoking branch.
- **Missing intent:** escalate instead of treating commit order or code style as
  intent.
- **Unresolved ties:** do not guess and do not manufacture a compromise.
- **Per-file classification:** classify each hunk, not the whole file.
- **Drive-by edits:** keep the diff limited to conflict regions.
- **Premature mutation:** inspection, editing, staging, committing, and aborting
  each require their own authorization.
- **Unsafe cleanup:** never use `reset --hard`, `clean`, or destructive stash
  operations as a substitute for a safe abort.
- **Generated artifacts:** regenerate lockfiles or generated output from source
  when the project documents that workflow; do not hand-merge them by default.
- **Hotspots:** repeated conflicts in one path indicate a decomposition or
  ownership problem; report the hotspot rather than silently normalizing it.

## Completion Criteria

A read-only reconciliation is complete when:

- capabilities and repository state were checked;
- both intents and all conflicted hunks are documented;
- each hunk is resolved, or an unresolved intent tie is explicitly recorded;
- the proposed result and available verification are reported; and
- no state-changing Git action occurred without authorization.

A state-changing reconciliation is additionally complete only when the
explicitly authorized edits, staging, verification, commit, or abort have been
performed and confirmed with a fresh `git status`/diff readback.
