---
sidebar_position: 3
sidebar_label: "Git Worktrees"
title: "Git Worktrees"
description: "Run multiple Hermes agents safely on the same repository using git worktrees and isolated checkouts"
---

# Git Worktrees

Hermes Agent is often used on large, long‑lived repositories. When you want to:

- Run **multiple agents in parallel** on the same project, or
- Keep experimental refactors isolated from your main branch,

Git **worktrees** are the safest way to give each agent its own checkout without duplicating the entire repository.

This page shows how to combine worktrees with Hermes so each session has a clean, isolated working directory.

## Why Use Worktrees with Hermes?

Hermes treats the **current working directory** as the project root:

- CLI: the directory where you run `hermes` or `hermes chat`
- Messaging gateways: the directory set by `terminal.cwd` in `~/.hermes/config.yaml`

If you run multiple agents in the **same checkout**, their changes can interfere with each other:

- One agent may delete or rewrite files the other is using.
- It becomes harder to understand which changes belong to which experiment.

With worktrees, each agent gets:

- Its **own branch and working directory**
- Its **own Checkpoint Manager history** for `/rollback`

See also: [Checkpoints and /rollback](./checkpoints-and-rollback.md).

## Quick Start: Creating a Worktree

### From inside a session: `/worktree new`

The fastest path (inspired by Copilot CLI's `/worktree new`): from an
interactive CLI session, run

```
/worktree new my-experiment
```

Hermes creates `.worktrees/my-experiment/` inside the repo (branch
`hermes/my-experiment`, based on the freshly-fetched remote tip unless
`worktree_sync: false`), and retargets the session's terminal and file tools
into it — no restart needed. Omit the name to get a random `hermes-<id>`
tree. `/worktree` alone shows the active tree; `/worktree list` lists all of
them. On exit a clean, pushed tree is removed; a tree with uncommitted changes
or unpushed commits triggers the same keep / archive / remove question as
`hermes -w`.

### Manually with git

From your main repository (containing `.git/`), create a new worktree for a feature branch:

```bash
# From the main repo root
cd /path/to/your/repo

# Create a new branch and worktree in ../repo-feature
git worktree add ../repo-feature feature/hermes-experiment
```

This creates:

- A new directory: `../repo-feature`
- A new branch: `feature/hermes-experiment` checked out in that directory

Now you can `cd` into the new worktree and run Hermes there:

```bash
cd ../repo-feature

# Start Hermes in the worktree
hermes
```

Hermes will:

- See `../repo-feature` as the project root.
- Use that directory for context files, code edits, and tools.
- Use a **separate checkpoint history** for `/rollback` scoped to this worktree.

## Running Multiple Agents in Parallel

You can create multiple worktrees, each with its own branch:

```bash
cd /path/to/your/repo

git worktree add ../repo-experiment-a feature/hermes-a
git worktree add ../repo-experiment-b feature/hermes-b
```

In separate terminals:

```bash
# Terminal 1
cd ../repo-experiment-a
hermes

# Terminal 2
cd ../repo-experiment-b
hermes
```

Each Hermes process:

- Works on its own branch (`feature/hermes-a` vs `feature/hermes-b`).
- Writes checkpoints under a different shadow repo hash (derived from the worktree path).
- Can use `/rollback` independently without affecting the other.

This is especially useful when:

- Running batch refactors.
- Trying different approaches to the same task.
- Pairing CLI + gateway sessions against the same upstream repo.

## Cleaning Up Worktrees Safely

When you are done with an experiment:

1. Decide whether to keep or discard the work.
2. If you want to keep it:
   - Merge the branch into your main branch as usual.
3. Remove the worktree:

```bash
cd /path/to/your/repo

# Remove the worktree directory and its reference
git worktree remove ../repo-feature
```

Notes:

- `git worktree remove` will refuse to remove a worktree with uncommitted changes unless you force it.
- Removing a worktree does **not** automatically delete the branch; you can delete or keep the branch using normal `git branch` commands.
- Hermes checkpoint data under `~/.hermes/checkpoints/` is not automatically pruned when you remove a worktree, but it is usually very small.

## Best Practices

- **One worktree per Hermes experiment**
  - Create a dedicated branch/worktree for each substantial change.
  - This keeps diffs focused and PRs small and reviewable.
- **Name branches after the experiment**
  - e.g. `feature/hermes-checkpoints-docs`, `feature/hermes-refactor-tests`.
- **Commit frequently**
  - Use git commits for high‑level milestones.
  - Use [checkpoints and /rollback](./checkpoints-and-rollback.md) as a safety net for tool‑driven edits in between.
- **Avoid running Hermes from the bare repo root when using worktrees**
  - Prefer the worktree directories instead, so each agent has a clear scope.

## Using `hermes -w` (Automatic Worktree Mode)

Hermes has a built‑in `-w` flag that **automatically creates a disposable git worktree** with its own branch. You don't need to set up worktrees manually — just `cd` into your repo and run:

```bash
cd /path/to/your/repo
hermes -w
```

Hermes will:

- Create a temporary worktree under `.worktrees/` inside your repo.
- Check out an isolated branch (e.g. `hermes/hermes-<hash>`).
- Run the full CLI session inside that worktree.

Hermes checks the worktree out without running the repository's own git configuration: its hooks, `core.fsmonitor` and clean/smudge filters are off for that checkout, as they are for the worktrees Hermes creates for kanban tasks and subagents. A repository that stores files through Git LFS gets pointer files there; run `git lfs pull` inside the worktree if the session needs their contents.

This is the easiest way to get worktree isolation. You can also combine it with a single query:

```bash
hermes -w -z "Fix issue #123"
```

For parallel agents, open multiple terminals and run `hermes -w` in each — every invocation gets its own worktree and branch automatically.

### Leaving a session with unfinished work

When an interactive `hermes -w` session (classic CLI or `--tui`) exits and its
worktree has uncommitted changes or unpushed commits, Hermes asks:

- **Keep the worktree** (default, also what Enter/Esc picks) — nothing changes.
- **Archive and remove it** — the full working state is archived (see below)
  and the tree is removed before the session finishes exiting, so a large tree
  is never cut short; Hermes prints the `hermes worktree restore` command. If
  the archive fails, the tree is kept.
- **Remove it and discard the changes** — only when you pick it explicitly.

One-shot runs (`-q`, `-z`), cron jobs, gateway sessions and any exit without a
terminal never ask: the tree is kept and the stale-worktree sweep handles it
later. A clean tree whose commits are pushed is removed without asking.

## Stale worktree sweep and restore

Every repository Hermes creates a worktree in (`hermes -w`, `/worktree new`,
subagent worktree isolation, the desktop app's git pane) is recorded in
`<HERMES_HOME>/worktree_repos.json` for the active profile. At `hermes -w`
startup and every 6 hours from the cron ticker, Hermes sweeps each of those
repositories' `.worktrees/`:

| Tree | Decision |
|---|---|
| Idle past its tier (`hermes-*`/`subagent-*` 24 h, named trees 72 h), clean and merged or pushed | removed |
| Idle past its tier with uncommitted changes or unpushed commits | **archived, then removed** |
| A live hermes lock owner, or any live process whose cwd (Linux: or an open file) is inside | never touched |
| File, `HEAD` or reflog activity younger than its tier (a deleted file counts through its folder) | never touched |
| Kanban task trees (`t_<hex>`) | never touched (the Kanban GC owns them) |
| An initialized submodule with uncommitted edits, a different commit than recorded, or commits on no remote | kept (`keep-submodule-work`: an archive records only the submodule pointer) |

An archive is a commit stored at `refs/hermes/archive/<tree>` holding the
tree's `HEAD`, staged changes, unstaged changes and untracked files (files your
`.gitignore` excludes, such as `node_modules/` or `.venv/`, are not captured).
It is built with a temporary index, so the tree itself is never modified; if
the archive cannot be built, the tree is kept. If the tree is entered or its
staged changes move (`git add`, `git apply --cached`) while the archive is
being built, the archive is saved and the tree is kept too. File bytes are stored and
restored exactly: no `.gitattributes`, `info/attributes`, `core.attributesFile`
line-ending or filter rule is applied. Untracked files larger than
`worktrees.archive_max_file_mb` are left out and listed in the archive. A
branch that exists on a remote (an open PR) is kept; otherwise the archive
holds its commits and the local branch is deleted.

```bash
hermes worktree list                 # trees plus archives (name, age, branch, reason)
hermes worktree prune --dry-run      # every archive/remove/skip decision and why; changes nothing
hermes worktree restore lane-x       # recreate .worktrees/lane-x with its branch, staged and untracked files
hermes worktree restore lane-x --path ../lane-x-review
```

Restore puts the branch back when it is missing or still points at the
archived `HEAD`; otherwise the tree comes back detached at that commit.
Archives older than `worktrees.archive_retention_days` are deleted by the
sweep.

```yaml
worktrees:
  archive_dirty: true          # false = never remove dirty/unpushed trees (the old behaviour)
  archive_max_file_mb: 50      # untracked files above this are listed, not archived
  archive_retention_days: 30   # 0 keeps archives forever
  max_total_gb: 0              # optional budget for idle trees; oldest are archived + removed first
```

There is no limit on how many trees may exist: hundreds of parallel subagents
each keep their tree for as long as they work in it. `max_total_gb` only ever
evicts idle trees past their tier.

Subagent worktrees (`delegation.worktree_isolation: true`) follow the same
rule when the child finishes: a clean tree is removed, and a tree with commits
or edits is archived and removed with its branch kept for the parent to merge.

## Putting It All Together

- Use **git worktrees** to give each Hermes session its own clean checkout.
- Use **branches** to capture the high‑level history of your experiments.
- Use **checkpoints + `/rollback`** to recover from mistakes inside each worktree.

This combination gives you:

- Strong guarantees that different agents and experiments do not step on each other.
- Fast iteration cycles with easy recovery from bad edits.
- Clean, reviewable pull requests.

## Developing the UI surfaces across worktrees

The TypeScript surfaces (`ui-tui/`, `apps/desktop/`) each need a `node_modules`, which a fresh `npm ci` per worktree duplicates across every branch. If you hack on the TUI or desktop app from multiple worktrees, see [TUI & Desktop from Worktrees](../developer-guide/worktree-ui-dev.md) for the `htui` / `hgui` helpers that share one install by symlink.
