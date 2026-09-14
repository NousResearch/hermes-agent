---
sidebar_position: 8
title: "Session Storage Recovery"
description: "Recovering session storage and resolving database lockouts after deleted-WAL generation guards fire"
---

# Session Storage Recovery

Hermes Agent isolates and protects your conversation history in `~/.hermes/state.db`. If an external event replaces the SQLite database underneath a running process (for example, during an automated update, backup restore, or disk synchronization), SQLite's write-ahead log (WAL) generation check trips.

When this occurs, Hermes halts active writes on that handle to prevent database corruption, buffers unsaved messages locally, and surfaces in-product recovery options.

---

## The Deleted-WAL Guard

### Why the Guard Trips

SQLite WAL mode associates an active database handle with a specific WAL generation number. If:
1. Another Hermes process, update script, or restore command replaces `state.db`,
2. An orphaned process holds an old WAL lock descriptor, or
3. A background task rolls over database sidecars,

the running process detects a generation mismatch (`DeletedWalGenerationError`). Rather than writing pages into an orphaned WAL that could corrupt the active database, Hermes engages a strict write quarantine.

### Zero Data Loss Invariant

When the deleted-WAL guard fires:
- **No data is discarded.** Any in-flight turns or pending transcript updates that cannot be committed to SQLite are automatically spooled to plain JSONL files under `~/.hermes/sessions/<id>.jsonl` and the `~/.hermes/pending_messages/` spool.
- **Sidecar evidence is captured.** The replaced WAL and its shm sidecars are preserved in timestamped capture directories: `~/.hermes/state.db.retired-wal-<timestamp>-<pid>/` along with a `manifest.json`.
- **Pre-update backups remain intact.** Any pre-update emergency backups (`~/.hermes/state.db.pre-update-emergency-*.bak`) remain available in your Hermes home directory.

---

## In-Product Recovery

### Desktop App (One-Click Recovery)

When Hermes detects a `deleted_wal` persistence failure during a chat turn, Desktop displays a notification banner:

> **Hermes paused saving this chat** because another Hermes process replaced its session database. Nothing is lost.

Click the **Recover** action button directly in the banner.

Desktop triggers an authenticated repair request to `/api/ops/doctor` with `fix: true`. Hermes:
1. Detects and gracefully terminates any lingering background holders or orphaned gateway processes holding the old WAL descriptor.
2. Re-verifies `state.db` health, FTS5 triggers, and metadata markers.
3. Restores database write readiness so your conversation resumes saving seamlessly.

You can also trigger this repair at any time from **Command Center &rarr; Maintenance &rarr; Run doctor --fix**.

---

### Command Line (`hermes doctor --fix`)

If you are using the CLI or a headless server, resolve the condition using `hermes doctor`:

```bash
# 1. Check database state and identify conflicting processes
hermes doctor

# 2. Automatically stop orphaned holders and recover access
hermes doctor --fix
```

#### What `hermes doctor --fix` Does:
- **Detects Holders:** Identifies any running processes (`hermes gateway`, GUI backends, or subagents) holding deleted SQLite sidecars (`iter_deleted_sqlite_sidecar_holders`).
- **Terminates Orphaned Holders:** Safely sends termination signals to stale PIDs to release file descriptor locks.
- **Re-opens & Re-checks:** Re-opens `SessionDB` to confirm standard WAL access and validates FTS5 index integrity.
- **Surfaces Artifacts:** If retired WAL captures exist, doctor prints the exact inspection command:
  ```bash
  hermes sessions recover --source ~/.hermes/state.db.retired-wal-<id>/state.db-wal --inspect-only
  ```

---

## Inspecting and Restoring Captured Artifacts

If you want to review transactions captured in a retired WAL directory before archiving them:

```bash
# Inspect contents without modifying your main state.db
hermes sessions recover \
  --source "$HOME/.hermes/state.db.retired-wal-20260913-120000-1234/state.db-wal" \
  --inspect-only

# Recover records into a separate database file
hermes sessions recover \
  --source "$HOME/.hermes/state.db.retired-wal-20260913-120000-1234/state.db-wal" \
  --output "$HOME/recovered-sessions.db"
```

Once verified, you may delete the `state.db.retired-wal-*` directory to free up disk space.

---

## Related Documentation

- [Sessions Guide](./sessions.md) — Managing sessions, titles, resumes, and compression.
- [State DB Developer Guide](../developer-guide/state-db-recovery.md) — Internal recovery mechanics and FTS repair details.
