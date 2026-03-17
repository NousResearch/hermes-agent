---
sidebar_position: 3
title: "Config Backup"
description: "Version-control your Hermes config with git — local snapshots or remote push to GitHub/GitLab"
---

# Config Backup

`hermes config backup` keeps your `~/.hermes/` config under git version control. Every change to
`config.yaml`, `SOUL.md`, skills, cron jobs, and memories is automatically committed — hourly or on
demand — with full history and optional remote push to GitHub, GitLab, or any git host.

Your secrets (`.env`, `auth.json`) are **always excluded** and never committed.

---

## Quick start

```bash
# 1. Initialize a local backup repo
hermes config backup init

# 2. Enable hourly auto-backup
hermes config backup auto on

# 3. Check status any time
hermes config backup status
```

That's it. From this point every config change is automatically captured once per hour.

---

## Commands

### `hermes config backup init [remote]`

Initializes a git repository in `~/.hermes/`, creates a `.gitignore` that excludes secrets, and
takes the first snapshot. Optionally adds a remote for off-machine backup.

```bash
# Local only
hermes config backup init

# With remote (GitHub, GitLab, self-hosted, etc.)
hermes config backup init git@github.com:you/hermes-config.git
hermes config backup init https://github.com/you/hermes-config
```

Safe to run multiple times — re-running on an existing repo is a no-op.

---

### `hermes config backup push`

Commits any changed tracked files and pushes to remote (if configured). Does nothing if config
is unchanged.

```bash
hermes config backup push
```

```
  ✓ Committed snapshot (2026-03-17 18:30)
  ✓ Pushed to https://github.com/you/hermes-config
```

---

### `hermes config backup pull`

Pulls the latest config from your remote. Useful when switching machines or restoring after a
reinstall.

```bash
hermes config backup pull
```

Requires a remote to be configured (`hermes config backup init <url>`).

---

### `hermes config backup status`

Shows a snapshot of your backup state at a glance.

```bash
hermes config backup status
```

```
  Config Backup Status

  Last commit:  a87640c auto: config snapshot 2026-03-17 18:30 (2 hours ago)
  Remote:       https://github.com/you/hermes-config
  Auto-backup:  on (hourly)
  Changes:      none (clean)
```

If there are uncommitted changes:

```
  Uncommitted changes:
     M config.yaml
     M memories/MEMORY.md
```

---

### `hermes config backup auto on|off`

Enables or disables an hourly cron job that runs `hermes config backup push` automatically.

```bash
hermes config backup auto on    # commits every hour if anything changed
hermes config backup auto off   # stops auto-backup
```

Logs are written to `~/.hermes/logs/backup.log`.

---

## What gets tracked

| Path | Tracked |
|---|---|
| `config.yaml` | ✅ |
| `SOUL.md` | ✅ |
| `skills/` | ✅ |
| `cron/` | ✅ |
| `memories/` | ✅ |
| `.env` | ❌ never (API keys) |
| `auth.json` | ❌ never (OAuth tokens) |
| `logs/` | ❌ |
| `sessions/` | ❌ |
| `state.db` | ❌ |

---

## Setting up a remote

Create a **private** repository on GitHub (or any git host), then:

```bash
hermes config backup init git@github.com:you/hermes-config.git
hermes config backup push
hermes config backup auto on
```

To change the remote URL later, just run `init` again with the new URL:

```bash
hermes config backup init git@gitlab.com:you/hermes-config.git
```

---

## Restoring config on a new machine

```bash
# 1. Install Hermes
# 2. Clone your config repo into ~/.hermes/
git clone git@github.com:you/hermes-config.git ~/.hermes

# 3. Re-add your secrets (.env is not in the repo)
hermes setup

# 4. Re-enable auto-backup
hermes config backup auto on
```

---

## Viewing history

Since `~/.hermes/` is a standard git repo, all git commands work:

```bash
# See full history
git -C ~/.hermes log --oneline

# See what changed in the last commit
git -C ~/.hermes show

# Diff current state vs yesterday
git -C ~/.hermes diff HEAD~5

# Restore a previous config.yaml
git -C ~/.hermes checkout HEAD~2 -- config.yaml
```

---

## Security

- `.env` and `auth.json` are in `.gitignore` and will never be staged or committed
- If using a remote, use a **private** repository
- The `.gitignore` is written on `init` and covers all known secrets and runtime files
