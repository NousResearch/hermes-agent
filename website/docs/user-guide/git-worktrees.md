---
sidebar_position: 3
sidebar_label: "Git Worktrees"
title: "Git Worktrees"
description: "Run Hermes agents safely with dedicated git worktrees, narrow ownership, and PR handoff"
---

# Git Worktrees

Hermes Agent is often used on large, long-lived repositories. When you want to
run multiple agents, protect the default branch, or keep a long-running change
reviewable, use a dedicated git worktree for each task.

A worktree is not just a convenience checkout. For agent-owned code changes it is
part of the safety boundary:

- The primary checkout stays a read-only snapshot of `origin/<default-branch>`.
- Each task gets its own branch and working directory.
- Agents stage, validate, commit, push, and open PRs only for their owned paths.
- The default branch reaches changes only through a merged pull request.

## Why Use Worktrees with Hermes?

Hermes treats the current working directory as the project root:

- CLI: the directory where you run `hermes` or `hermes chat`
- Messaging gateways: the directory set by `terminal.cwd` in `config.yaml`

If multiple agents share one checkout, their edits, generated files, staged
state, and test artifacts can collide. Worktrees give each session:

- Its own branch and working directory
- Its own Checkpoint Manager history for `/rollback`
- A clear ownership boundary for review and cleanup

See also: [Checkpoints and /rollback](./checkpoints-and-rollback.md).

## Default Agent Methodology

Before editing a repository, read its local context files (`AGENTS.md`,
`CLAUDE.md`, `.cursorrules`, or equivalent). If the repository provides its own
worktree/preflight command, use that command instead of generic git commands.
Project policy wins over this generic guide.

When there is no repo-owned helper, use this baseline flow from the primary
checkout:

```bash
cd /path/to/repo

git fetch origin --prune
default_branch="$(git remote show origin | sed -n 's/.*HEAD branch: //p')"
default_branch="${default_branch:-main}"

slug="fix-short-description"
repo_name="$(basename "$(git rev-parse --show-toplevel)")"
worktree_path="../${repo_name}-${slug}"
branch="agent/${slug}"

git worktree add -b "$branch" "$worktree_path" "origin/${default_branch}"
cd "$worktree_path"
```

Then install or bootstrap the environment inside that worktree if the project
uses per-checkout editable installs, generated config, or a managed virtualenv.
Do not blindly reuse another checkout's `.venv`; many Python and Node projects
record absolute paths or editable-install metadata per checkout.

## Scope and Ownership Before Editing

Declare the slice you own in your prompt or handoff notes. Before editing,
validation, staging, committing, or pushing, inspect both shared state and the
owned paths:

```bash
git status --short --branch --untracked-files=all
git diff --name-status HEAD -- path/to/owned/file path/to/owned/dir
git diff --check HEAD -- path/to/owned/file path/to/owned/dir
```

Rules for agents:

- Treat unrelated dirty or staged files as another agent's work.
- Do not stage, unstage, reset, revert, summarize, or clean unrelated files.
- If an owned file overlaps unrelated work, stop and ask the maintainer.
- Stage only intentional paths; do not use `git add .` or `git add -A`.
- Keep each branch narrow and short-lived.

## Running Multiple Agents in Parallel

Create one worktree per task, with non-overlapping owned paths:

```bash
cd /path/to/repo

git fetch origin --prune
git worktree add -b agent/task-a ../repo-task-a origin/main
git worktree add -b agent/task-b ../repo-task-b origin/main
```

Start each Hermes process from the worktree it owns:

```bash
# Terminal 1
cd ../repo-task-a
hermes

# Terminal 2
cd ../repo-task-b
hermes
```

Avoid assigning the same file or generated artifact to two active worktrees.
Overlapping paths should serialize through one owning worktree instead of racing.
If a repository has a claim/preflight system, use it before each edit and before
handoff.

## Validation, Commit, Push, and PR Handoff

Run the repository's scoped validation for the paths you changed. Prefer project
wrappers when they exist because they know local environment, generated files,
and CI parity requirements.

Before committing:

```bash
git status --short --branch --untracked-files=all
git diff --name-status HEAD -- path/to/owned/file path/to/owned/dir
git diff --check HEAD -- path/to/owned/file path/to/owned/dir
```

Commit and push only the owned paths:

```bash
git add -- path/to/owned/file path/to/owned/dir
git diff --cached --name-status
git commit -m "fix: concise description"
git push -u origin HEAD
```

Then open a pull request with the validation evidence in the body. Do not push
directly to the default branch. Spawned subagents should stay PR-free; the
launching agent owns final review, push, and PR creation.

If your feature branch already exists and you need to update it against the
latest default branch, fetch and rebase that feature branch, then push it with
`--force-with-lease`. Never force-push the default branch or a branch you do not
solely own.

## Cleaning Up Worktrees Safely

After the PR merges, clean up only the worktree and branch you own:

```bash
cd /path/to/primary-checkout

git fetch origin --prune
git worktree remove ../repo-task-a
git branch -d agent/task-a
```

Use `git worktree list` to inspect existing worktrees. Do not remove another
agent's worktree unless you have explicit confirmation that it is stale and safe
to delete.

Hermes checkpoint data under `~/.hermes/checkpoints/` is not automatically pruned
when you remove a worktree, but it is usually small.

## Using `hermes -w` (Automatic Worktree Mode)

Hermes has a built-in `-w` / `--worktree` flag that creates a disposable worktree
for a CLI session:

```bash
cd /path/to/repo
hermes -w
```

Hermes will:

- Create a worktree under `.worktrees/` in the current repository.
- Create a branch like `hermes/hermes-<hash>` from the current `HEAD`.
- Run the CLI session from that worktree.
- Copy paths listed in `.worktreeinclude` when present.

Important limits:

- `hermes -w` does not fetch `origin` or branch from `origin/<default-branch>`.
- It does not run project-specific preflight, claim, bootstrap, or validation
  commands.
- On exit, Hermes removes the worktree and branch unless the worktree has commits
  that are not reachable from any remote branch. Uncommitted-only edits are not a
  durable handoff.

Use `hermes -w` for quick isolation and disposable experiments. For durable PR
work, production repositories, or projects with their own worktree method, prefer
the explicit/repo-owned worktree flow above.

## `.worktreeinclude`

You can list gitignored files to copy into automatic `hermes -w` worktrees:

```text
# .worktreeinclude
.env.local
config/local.yaml
```

Keep this list small and intentional. Avoid copying virtualenvs, dependency
folders, caches, secrets, or generated artifacts unless the repository explicitly
says doing so is safe. Many projects require each worktree to run its own
bootstrap so editable installs and generated config point at the correct
checkout.

## Putting It All Together

Use this checklist for agent-owned repo changes:

1. Read the repo's local agent instructions.
2. Create or enter a dedicated worktree from freshly fetched `origin/<default-branch>`.
3. Bootstrap that worktree's environment if needed.
4. Own a narrow path set; avoid overlap with other worktrees.
5. Validate, stage, commit, and push only owned paths.
6. Open a PR; never push the default branch directly.
7. Remove only your worktree after the PR merges.

This combination gives you isolated edits, reproducible validation evidence, and
clean pull-request handoff without letting concurrent agents step on each other.
