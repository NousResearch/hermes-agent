---
sidebar_position: 8
sidebar_label: "Checkpoints & Rollback"
title: "Checkpoints and /rollback"
description: "Filesystem safety nets for destructive operations using shadow git repos and automatic snapshots"
---

# Checkpoints and `/rollback`

Rabbit Agent can automatically snapshot your project before **destructive operations** and restore it with a single command. Checkpoints are **opt-in** as of v2 — most users never use `/rollback`, and the shadow-store storage is non-trivial over time, so the default is off.

Enable checkpoints per-session with `--checkpoints`:

```bash
rabbit chat --checkpoints
```

Or enable globally in `~/.rabbit/config.yaml`:

```yaml
checkpoints:
  enabled: true
```

This safety net is powered by an internal **Checkpoint Manager** that keeps a single shared shadow git repository under `~/.rabbit/checkpoints/store/` — your real project `.git` is never touched. Every project the agent works in shares the same store, so git's content-addressable object DB deduplicates across projects and across turns.

## What Triggers a Checkpoint

Checkpoints are taken automatically before:

- **File tools** — `write_file` and `patch`
- **Destructive terminal commands** — `rm`, `rmdir`, `cp`, `install`, `mv`, `sed -i`, `truncate`, `dd`, `shred`, output redirects (`>`), and `git reset`/`clean`/`checkout`

The agent creates **at most one checkpoint per directory per turn**, so long-running sessions don't spam snapshots.

## Quick Reference

In-session slash commands:

| Command | Description |
|---------|-------------|
| `/rollback` | List all checkpoints with change stats |
| `/rollback <N>` | Restore to checkpoint N, keeping your hand-edits (also undoes last chat turn) |
| `/rollback <N> --all` | Full restore — overwrites your hand-edits too |
| `/rollback diff <N>` | Preview diff between checkpoint N and current state |
| `/rollback <N> <file>` | Restore a single file from checkpoint N |

CLI for inspecting and managing the store outside a session:

| Command | Description |
|---------|-------------|
| `rabbit checkpoints` | Show total size, project count, per-project breakdown |
| `rabbit checkpoints status` | Same as bare `checkpoints` |
| `rabbit checkpoints list` | Alias for `status` |
| `rabbit checkpoints prune` | Force a sweep: delete orphans/stale, GC, enforce size cap |
| `rabbit checkpoints clear` | Nuke the entire checkpoint base (asks first) |
| `rabbit checkpoints clear-legacy` | Delete only the `legacy-*` archives from v1 migration |

## How Checkpoints Work

At a high level:

- Rabbit detects when tools are about to **modify files** in your working tree.
- Once per conversation turn (per directory), it:
  - Resolves a reasonable project root for the file.
  - Initialises or reuses the **single shared shadow store** at `~/.rabbit/checkpoints/store/`.
  - Stages into a per-project index, builds a tree, and commits to a per-project ref (`refs/rabbit/<project-hash>`).
- These per-project refs form a checkpoint history that you can inspect and restore via `/rollback`.

```mermaid
flowchart LR
  user["User command\n(rabbit, gateway)"]
  agent["AIAgent\n(run_agent.py)"]
  tools["File & terminal tools"]
  cpMgr["CheckpointManager"]
  store["Shared shadow store\n~/.rabbit/checkpoints/store/"]

  user --> agent
  agent -->|"tool call"| tools
  tools -->|"before mutate\nensure_checkpoint()"| cpMgr
  cpMgr -->|"git add/commit-tree/update-ref"| store
  cpMgr -->|"OK / skipped"| tools
  tools -->|"apply changes"| agent
```

## Configuration

Configure in `~/.rabbit/config.yaml`:

```yaml
checkpoints:
  enabled: false              # master switch (default: false — opt-in)
  max_snapshots: 20           # max checkpoints per project (enforced via ref rewrite + gc)
  max_total_size_mb: 500      # hard cap on total store size; oldest commits dropped
  max_file_size_mb: 10        # skip any single file larger than this

  # Auto-maintenance (on by default): sweep ~/.rabbit/checkpoints/ in the
  # background — the CLI on a helper thread right after launch, the gateway
  # on its housekeeping tick — and delete project entries whose last_touch is
  # older than retention_days. Runs at most once per min_interval_hours,
  # tracked via a .last_prune marker. It never blocks the prompt or gateway
  # startup: the `git gc` that reclaims space can take tens of seconds on a
  # large store. This sweep never deletes "orphan" entries (working directory
  # not found) — a missing workdir is ambiguous (deleted project vs. an
  # unmounted external volume / network share / VPN not yet up), so orphan
  # cleanup is only ever done via the explicit `rabbit checkpoints prune`
  # command below, with a confirmation prompt.
  auto_prune: true
  retention_days: 7
  min_interval_hours: 24
```

To disable everything:

```yaml
checkpoints:
  enabled: false
  auto_prune: false
```

When `enabled: false`, the Checkpoint Manager is a no-op and never attempts git operations. When `auto_prune: false`, the store grows until you run `rabbit checkpoints prune` manually.

## Listing Checkpoints

From a CLI session:

```
/rollback
```

Rabbit responds with a formatted list showing change statistics:

```text
📸 Checkpoints for /path/to/project:

  1. 4270a8c  2026-03-16 04:36  before patch  (1 file, +1/-0)
  2. eaf4c1f  2026-03-16 04:35  before write_file
  3. b3f9d2e  2026-03-16 04:34  before terminal: sed -i s/old/new/ config.py  (1 file, +1/-1)

  /rollback <N>             restore to checkpoint N (keeps your hand-edits)
  /rollback <N> --all       full restore, overwriting your hand-edits too
  /rollback diff <N>        preview changes since checkpoint N
  /rollback <N> <file>      restore a single file from checkpoint N
```

## Inspecting the Store from the Shell

```bash
rabbit checkpoints
```

Sample output:

```text
Checkpoint base: /home/you/.rabbit/checkpoints
Total size:      142.3 MB
  store/         138.1 MB
  legacy-*       4.2 MB
Projects:        12

  WORKDIR                                                       COMMITS    LAST TOUCH  STATE
  /home/you/code/rabbit-agent                                        20       2h ago  live
  /home/you/code/experiments/rl-runner                                8       1d ago  live
  /home/you/code/old-prototype                                        3       9d ago  orphan
  ...

Legacy archives (1):
  legacy-20260506-050616                           4.2 MB

Clear with: rabbit checkpoints clear-legacy
```

Force a full sweep (ignores the 24h idempotency marker):

```bash
rabbit checkpoints prune --retention-days 3 --max-size-mb 200
```

## Previewing Changes with `/rollback diff`

Before committing to a restore, preview what has changed since a checkpoint:

```
/rollback diff 1
```

This shows a git diff stat summary followed by the actual diff.

## Restoring with `/rollback`

```
/rollback 1
```

Behind the scenes, Rabbit:

1. Verifies the target commit exists in the shadow store.
2. Takes a **pre-rollback snapshot** of the current state so you can "undo the undo" later.
3. Restores tracked files in your working directory — **preserving your hand-edits** (see below).
4. **Undoes the last conversation turn** so the agent's context matches the restored filesystem state.

### User hand-edits are preserved by default

`/rollback <N>` restores only the files Rabbit itself changed. Every successful
`write_file` / `patch` records the file's content hash in an **agent-write
ledger**; at restore time, any file whose current contents no longer match what
Rabbit last wrote (you edited it afterwards, or Rabbit never touched it) is
**skipped** instead of overwritten, and listed in the output:

```
✅ Restored to checkpoint a1b2c3d4: before write_file
↷ Kept your hand-edits: src/config.py, notes.md
Use /rollback <N> --all to restore those too.
```

To force the classic full restore that reverts everything — including your own
edits — add `--all`:

```
/rollback 1 --all
```

If the ledger is empty (a store created before this feature, or Rabbit hasn't
written any files in the project yet), `/rollback` falls back to the full
restore automatically.

## Single-File Restore

Restore just one file from a checkpoint without affecting the rest of the directory:

```
/rollback 1 src/broken_file.py
```

## Safety and Performance Guards

### Nested Git repositories

A checkpoint of a parent directory can store a nested repository as a Git
**gitlink** (a commit reference), not a copy of its files. Recursive capture of
nested repositories is not supported: their uncommitted edits and untracked
files are not recoverable from that parent checkpoint. Checkpoints taken by this
version or later are labelled in `/rollback` listings, for example
`before write_file: app.py [nested git repos not captured: tool]`.

If the selected checkpoint contains gitlinks, a full rollback (including
`--all`) is refused before changing files or creating a pre-rollback snapshot.
Selecting a nested repository, a file below it, or a Git pathspec matching it
also refuses the restore rather than reporting success for uncaptured files.
You can still restore unrelated captured files, for example
`/rollback 1 notes.txt`. Keep separate backups or checkpoints taken directly
from the nested repository's own working directory.

### Container Backends

With a container terminal backend (`docker`, `singularity`, `modal`, `daytona`, `vercel_sandbox`, or a container plugin), file paths belong to the sandbox rather than the host. Rabbit therefore does not take checkpoints or record the agent-write ledger for those paths, and `/rollback` explains the limitation: it still lists existing host checkpoints but refuses diff and restore, on the CLI and in messaging-gateway chats alike; `/diff session` answers with the same reason. The TUI and Desktop behave the same: `/rollback list` still works while `/rollback diff` and `/rollback <N>` are refused with that reason. Local and SSH backends are unaffected. To point `terminal.cwd` at the container-side view of a mounted directory see [`terminal.docker_mount_cwd_to_workspace`](./configuration.md).

- **Git availability** — if `git` is not found on `PATH`, checkpoints are transparently disabled.
- **Directory scope** — Rabbit skips overly broad directories (root `/`, home `$HOME`).
- **Repository size** — directories with more than 50,000 files are skipped.
- **Per-file size cap** — files larger than `max_file_size_mb` (default 10 MB) are excluded from the snapshot. Prevents accidentally swallowing datasets, model weights, or generated media.
- **Total store size cap** — when the store exceeds `max_total_size_mb` (default 500 MB), the oldest commit per project is dropped round-robin. Each drop is reclaimed before deciding whether another is necessary. Every project keeps at least one snapshot. A failed Git operation stops pruning and is reported; maintenance never discards more history to compensate for failed reclamation.
- **Real pruning, off the hot path** — `max_snapshots` and the size cap are enforced by rewriting the per-project ref at checkpoint time (cheap); the store is then marked `.gc-pending` and the periodic prune runs `git gc --prune=now` once, so loose objects don't accumulate and a tool call never waits on a full repack.
- **Concurrent operations** — snapshots, restores, diffs and maintenance use one process-shared store lock. An operation reports a busy store rather than running GC over another process's unpublished objects. A restore applies its selected tree before pruning the safety snapshot's history.
- **No-change snapshots** — if there are no changes since the last snapshot, the checkpoint is skipped.
- **Non-fatal errors** — snapshot failures do not block your tools. Pruning failures are logged as warnings; explicit maintenance reports an error count.

## Where Checkpoints Live

```text
~/.rabbit/checkpoints/
  ├── store/                 # single shared bare git repo
  │   ├── HEAD, objects/     # git internals (shared across projects)
  │   ├── refs/rabbit/<hash> # per-project branch tip
  │   ├── indexes/<hash>     # per-project git index
  │   ├── projects/<hash>.json  # workdir + created_at + last_touch
  │   ├── .gc-pending        # refs rewritten since the last gc; cleared by the next prune
  │   └── info/exclude
  ├── .last_prune            # auto-prune idempotency marker
  └── legacy-<ts>/           # archived pre-v2 per-project shadow repos
```

Each `<hash>` is derived from the absolute path of the working directory. You normally never need to touch these manually — use `rabbit checkpoints status` / `prune` / `clear` instead.

The sibling `.checkpoints.lock` coordinates processes and survives a store clear. Do not remove it while Rabbit is running.

### Migration from v1

Before the v2 rewrite, each working directory got its own complete shadow git repo directly under `~/.rabbit/checkpoints/<hash>/`. That layout couldn't dedup objects across projects and had a documented no-op pruner — the store would grow without bound.

On first v2 run, any pre-v2 shadow repos are moved into `~/.rabbit/checkpoints/legacy-<timestamp>/` so the new single-store layout starts clean. Old `/rollback` history is still reachable by manually inspecting the legacy archive with `git`; once you're confident you don't need it, run:

```bash
rabbit checkpoints clear-legacy
```

to reclaim the space. Legacy archives are also swept by `auto_prune` after `retention_days`.

## Best Practices

- **Enable checkpoints only when you need them** — `rabbit chat --checkpoints` or per-profile `enabled: true`.
- **Use `/rollback diff` before restoring** — preview what will change to pick the right checkpoint.
- **Use `/rollback` instead of `git reset`** when you want to undo agent-driven changes only.
- **Check `rabbit checkpoints status` occasionally** if you use checkpoints regularly — shows which projects are active and what the store costs you.
- **Combine with Git worktrees** for maximum safety — keep each Rabbit session in its own worktree/branch, with checkpoints as an extra layer.

For running multiple agents in parallel on the same repo, see the guide on [Git worktrees](./git-worktrees.md).
