---
sidebar_position: 6
title: "Worktree Conflict Notifications"
description: "Notify peer agents when one edits a file the other has recently read (jcode feature adoption)."
---

# Worktree Conflict Notifications

When two Hermes agents run in parallel worktrees on the same repo, Agent A may read a file that Agent B later edits. Agent A's view of that file is now stale — but it doesn't know, so it might make conflicting edits.

Worktree conflict notifications close this gap: when Agent B edits a file Agent A has read, Agent A receives a system message reminding it to re-read the file before relying on cached content.

Adopted from [jcode](https://github.com/1jehuang/jcode)'s multi-agent conflict notification pattern.

## Enable

Set the config flag and start a worktree session:

```bash
hermes config set agent.worktree_conflict_notifications true
hermes -w "fix the bug in auth.py"
```

Or pass the flag directly:

```bash
hermes -w --conflict-notify "fix the bug in auth.py"
```

## How it works

1. Every `read_file` call records the path in a process-wide TTL'd watched set (10 minute window).
2. A background thread (`GitIndexWatcher`) polls `git status --porcelain` on the worktree's repo root every 2 seconds.
3. When the watcher detects a changed file that's in any agent's watched set, it looks up peer sessions whose `cwd` is under the same repo in `~/.hermes/state.db`.
4. For each peer session, the notifier adds a comment to the peer's active kanban task (if any), or falls back to a log entry that the operator can pick up.

The notification message looks like:

```
[worktree-conflict-watch] Files you recently read have changed: auth.py, utils.py.
Re-read before relying on cached content.
```

## Caveats

- **Opt-in only.** The feature is disabled by default — turning it on means peers will see system messages you didn't directly send. Users running multiple agents in the same repo on purpose are the target audience.
- **Kanban-comment delivery requires an active kanban task.** If the peer session isn't a kanban worker, the notification falls back to a log line. A richer push channel (gateway webhook, CLI status event) is a v2 addition.
- **Same-host only.** Peer discovery uses the local `state.db`. Agents running on different machines can't notify each other yet.
- **No false-positive guarantee.** If you read `auth.py` and someone on the same repo edits `auth.py.bak`, you'll get notified. The watcher doesn't filter by extension or gitignore.

## Inspecting

```bash
hermes doctor
```

The doctor output includes the worktree watcher status (enabled/disabled, repo root, peer count) alongside memory and recall.

## Disabling

```bash
hermes config set agent.worktree_conflict_notifications false
```

## See also

- [Git Worktrees](./git-worktrees) (todo: link) — the `-w` flag that creates isolated worktrees for parallel agents
- [Kanban](./kanban) — the underlying task / comment surface the notifications use
