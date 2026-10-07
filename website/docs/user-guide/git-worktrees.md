---
sidebar_position: 3
sidebar_label: "Git Worktrees"
title: "Git Worktrees"
description: "Run multiple Rabbit agents safely on the same repository using git worktrees and isolated checkouts"
---

# Git Worktrees

Rabbit Agent is often used on large, long‑lived repositories. When you want to:

- Run **multiple agents in parallel** on the same project, or
- Keep experimental refactors isolated from your main branch,

Git **worktrees** are the safest way to give each agent its own checkout without duplicating the entire repository.

This page shows how to combine worktrees with Rabbit so each session has a clean, isolated working directory.

## Why Use Worktrees with Rabbit?

Rabbit treats the **current working directory** as the project root:

- CLI: the directory where you run `rabbit` or `rabbit chat`
- Messaging gateways: the directory set by `terminal.cwd` in `~/.rabbit/config.yaml`

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

Rabbit creates `.worktrees/my-experiment/` inside the repo (branch
`rabbit/my-experiment`, based on the freshly-fetched remote tip unless
`worktree_sync: false`), and retargets the session's terminal and file tools
into it — no restart needed. Omit the name to get a random `rabbit-<id>`
tree. `/worktree` alone shows the active tree; `/worktree list` lists all of
them. On exit the tree is kept only if it has unpushed commits, exactly like
`rabbit -w`.

### Manually with git

From your main repository (containing `.git/`), create a new worktree for a feature branch:

```bash
# From the main repo root
cd /path/to/your/repo

# Create a new branch and worktree in ../repo-feature
git worktree add ../repo-feature feature/rabbit-experiment
```

This creates:

- A new directory: `../repo-feature`
- A new branch: `feature/rabbit-experiment` checked out in that directory

Now you can `cd` into the new worktree and run Rabbit there:

```bash
cd ../repo-feature

# Start Rabbit in the worktree
rabbit
```

Rabbit will:

- See `../repo-feature` as the project root.
- Use that directory for context files, code edits, and tools.
- Use a **separate checkpoint history** for `/rollback` scoped to this worktree.

## Running Multiple Agents in Parallel

You can create multiple worktrees, each with its own branch:

```bash
cd /path/to/your/repo

git worktree add ../repo-experiment-a feature/rabbit-a
git worktree add ../repo-experiment-b feature/rabbit-b
```

In separate terminals:

```bash
# Terminal 1
cd ../repo-experiment-a
rabbit

# Terminal 2
cd ../repo-experiment-b
rabbit
```

Each Rabbit process:

- Works on its own branch (`feature/rabbit-a` vs `feature/rabbit-b`).
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
- Rabbit checkpoint data under `~/.rabbit/checkpoints/` is not automatically pruned when you remove a worktree, but it is usually very small.

## Best Practices

- **One worktree per Rabbit experiment**
  - Create a dedicated branch/worktree for each substantial change.
  - This keeps diffs focused and PRs small and reviewable.
- **Name branches after the experiment**
  - e.g. `feature/rabbit-checkpoints-docs`, `feature/rabbit-refactor-tests`.
- **Commit frequently**
  - Use git commits for high‑level milestones.
  - Use [checkpoints and /rollback](./checkpoints-and-rollback.md) as a safety net for tool‑driven edits in between.
- **Avoid running Rabbit from the bare repo root when using worktrees**
  - Prefer the worktree directories instead, so each agent has a clear scope.

## Using `rabbit -w` (Automatic Worktree Mode)

Rabbit has a built‑in `-w` flag that **automatically creates a disposable git worktree** with its own branch. You don't need to set up worktrees manually — just `cd` into your repo and run:

```bash
cd /path/to/your/repo
rabbit -w
```

Rabbit will:

- Create a temporary worktree under `.worktrees/` inside your repo.
- Check out an isolated branch (e.g. `rabbit/rabbit-<hash>`).
- Run the full CLI session inside that worktree.

Rabbit checks the worktree out without running the repository's own git configuration: its hooks, `core.fsmonitor` and clean/smudge filters are off for that checkout, as they are for the worktrees Rabbit creates for kanban tasks and subagents. A repository that stores files through Git LFS gets pointer files there; run `git lfs pull` inside the worktree if the session needs their contents.

This is the easiest way to get worktree isolation. You can also combine it with a single query:

```bash
rabbit -w -z "Fix issue #123"
```

For parallel agents, open multiple terminals and run `rabbit -w` in each — every invocation gets its own worktree and branch automatically.

## Putting It All Together

- Use **git worktrees** to give each Rabbit session its own clean checkout.
- Use **branches** to capture the high‑level history of your experiments.
- Use **checkpoints + `/rollback`** to recover from mistakes inside each worktree.

This combination gives you:

- Strong guarantees that different agents and experiments do not step on each other.
- Fast iteration cycles with easy recovery from bad edits.
- Clean, reviewable pull requests.

## Developing the UI surfaces across worktrees

The TypeScript surfaces (`ui-tui/`, `apps/desktop/`) each need a `node_modules`, which a fresh `npm ci` per worktree duplicates across every branch. If you hack on the TUI or desktop app from multiple worktrees, see [TUI & Desktop from Worktrees](../developer-guide/worktree-ui-dev.md) for the `htui` / `hgui` helpers that share one install by symlink.
