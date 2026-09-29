---
sidebar_position: 6
title: "State Replication"
description: "Sync skills, memories, and session history across multiple Hermes instances"
---

# State Replication

When running Hermes on multiple machines, each instance starts as an isolated "self" with separate skills, memories, and session history. State replication synchronizes this state so all instances share the same learned knowledge.

## What Gets Synced

| Component | Location | Method | Priority |
|-----------|----------|--------|----------|
| **Skills** | `~/.hermes/skills/` | Git push/pull (cron every 5 min) | Required |
| **Memories** | `~/.hermes/memories/` | Git push/pull (cron every 5 min) | Required |
| **Session history** | `~/.hermes/state.db` | SQLite snapshot via shared store | Required |
| **Cron tasks** | `~/.hermes/hermes.db` | Cold backup + restore | Recommended |
| **Config** | `~/.hermes/config.yaml` | Independent per machine | Optional |
| **Cache** | `~/.hermes/cache/` | Do not sync (disposable) | Never |

## Quick Start: 3-Step Setup

### Step 1: Initialize Git for skills and memories

Run this on every server that should share state:

```bash
cd ~/.hermes
git init
git add skills/ memories/
git commit -m "init: state baseline"
git remote add origin git@github.com:your-org/hermes-state.git

cat > .gitignore << 'EOF'
state.db
state.db-*
hermes.db
config.yaml
cache/
audio_cache/
image_cache/
auth.lock
logs/
pastes/
sandboxes/
*.tar.gz
EOF

git push -u origin main
```

### Step 2: Install sync cron job

Create the sync script:

```bash
cat > ~/.hermes/scripts/sync-state.sh << 'SCRIPT'
#!/bin/bash
cd ~/.hermes
git pull --rebase origin main 2>/dev/null || true
git add skills/ memories/
git diff --cached --quiet || (git commit -m "auto-sync $(date -u +%Y%m%dT%H%M%SZ)" && git push origin main)
SCRIPT
chmod +x ~/.hermes/scripts/sync-state.sh
```

Install the cron job:

```bash
(crontab -l 2>/dev/null; echo "*/5 * * * * cd ~/.hermes && bash scripts/sync-state.sh >> ~/.hermes/logs/sync.log 2>&1") | crontab -
```

### Step 3: Sync the session database

:::warning
Do not `cp` or `rsync` directly onto a running Hermes instance's `state.db` — Hermes holds an open file descriptor, and replacing the file under it causes data loss.
:::

**From primary server** (the server that will stay writable), create a consistent snapshot:

```bash
sqlite3 ~/.hermes/state.db "VACUUM INTO '/tmp/hermes-state-sync.db';"
scp /tmp/hermes-state-sync.db user@replica-server:/tmp/hermes-state-incoming.db
```

**On the replica server**, verify and apply the snapshot:

```bash
# Stop Hermes first
hermes stop

# Verify integrity
sqlite3 /tmp/hermes-state-incoming.db "PRAGMA integrity_check;"
sqlite3 /tmp/hermes-state-incoming.db "PRAGMA foreign_key_check;"

# If both pass, replace the database
cp /tmp/hermes-state-incoming.db ~/.hermes/state.db
rm -f ~/.hermes/state.db-shm ~/.hermes/state.db-wal

# Restart Hermes
hermes start
```

For automated syncing, run this as a cron job on the primary server every 5 minutes:

```bash
#!/bin/bash
# ~/.hermes/scripts/sync-state-db.sh
set -euo pipefail
cd ~/.hermes
sqlite3 state.db "VACUUM INTO '/tmp/hermes-state-sync.db';"
scp /tmp/hermes-state-sync.db user@replica:/tmp/hermes-state-incoming.db
ssh user@replica "
  sqlite3 /tmp/hermes-state-incoming.db 'PRAGMA integrity_check;' && \
  sqlite3 /tmp/hermes-state-incoming.db 'PRAGMA foreign_key_check;' && \
  hermes stop && \
  cp /tmp/hermes-state-incoming.db ~/.hermes/state.db && \
  rm -f ~/.hermes/state.db-shm ~/.hermes/state.db-wal && \
  rm -f /tmp/hermes-state-incoming.db && \
  hermes start
"
rm -f /tmp/hermes-state-sync.db
```

## Architecture

```
                  Git Repository
                 (skills + memories)
                ┌─────────┬─────────┐
                │         │         │
          push/pull   push/pull  push/pull
                │         │         │
         ┌──────┴──┐ ┌───┴───┐ ┌───┴──────┐
         │Primary  │ │Replica│ │Replica   │
         │Server A │ │Server B│ │Server C  │
         │state.db │ │state.db│ │state.db  │
         │  (RW)   │ │  (RO)  │ │  (RO)    │
         └────┬─────┘ └───────┘ └─────┬────┘
              │                     │
              └───── Shared Store ──┘
                  (S3 / NFS / rsync)
```

Two independent sync channels:
- **Git** for skills and memories (text files with linear history)
- **SQLite snapshots** via shared store for session history (binary database)

## Design Properties

- **Skills + Memories** use Git for conflict resolution — ideal for text files with infrequent edits by a single user
- **State.db** uses `VACUUM INTO` which creates a transactionally consistent snapshot without locking the live database — safer than `cp` or `wal_checkpoint`
- **FTS5 search indexes** are maintained by SQLite triggers. When the database file is replicated, the indexes come with it — no rebuild needed
- **Session ID format** (`YYYYMMDD_HHMMSS_<random>`) is collision-resistant across servers
- **Config stays independent** per server — API keys and proxy settings are machine-specific

## Advanced: SQLite Replication Technologies

For setups that need lower latency or don't want to stop/start Hermes:

| Tool | Mechanism | Latency | Code Change | Best For |
|------|-----------|---------|-------------|----------|
| **Litestream** | WAL changes streamed to S3 | ~1s | None | Cloud servers with S3 access |
| **LiteFS** | WAL frames shipped via FUSE | <10ms | None | Primary-replica on same network |
| **rqlite** | Raft consensus over HTTP | <50ms | Needs HTTP API client | True distributed SQLite |
| **LibSQL (Turso)** | Embedded replicas via gRPC | <100ms | Needs libSQL client | Multi-writer eventual consistency |

## Limitations

1. **Replicas are read-only** — only the primary server can write to `state.db`. All writes are eventually synced to replicas (1-5 minute lag)
2. **Hermes must stop briefly** on replicas during snapshot replacement (a few seconds for database swap)
3. **Large databases** (>1GB) should use LiteFS instead of file-level snapshots to avoid transferring the full database each cycle
4. **SQLite version compatibility** — keep all servers on the same SQLite version
