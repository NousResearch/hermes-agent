# state.db Corruption Recovery

When the default/global `state.db` becomes unreadable, the gateway degrades
to a JSONL routing fallback: `SessionStore._db` is `None`, routing keys still
resolve via `sessions.json`, and (for multiplexed profile routes) each
AIAgent lazily opens the *profile's own* `state.db` for transcript storage.
That degraded mode is survivable — but a corrupt global DB can make Telegram
chats appear to "forget" prior turns if the routing identity was never
persisted. This page documents how to recover safely without data loss.

## Symptoms

- `file is not a database` when opening the default `state.db`
  (`$HERMES_HOME/state.db`).
- Timestamped backups appear beside it:
  `state.db.malformed-backup-YYYYMMDD_HHMMSS`.
- A `state.db.repair.lock` file exists. The lock is `flock`-based: a stale
  zero-byte lock file is harmless. It only blocks while a live process holds
  the flock; the kernel releases it automatically when that process exits.
- Telegram sessions that used to resume now start with no prior context.
- Profile DBs (e.g. `profiles/<name>/state.db`) report integrity `ok` and
  contain messages, but the session row has NULL `session_key`/`chat_id`.

## How the pieces interact

1. At gateway startup, `SessionStore.__init__` opens `SessionDB()`. If the
   default DB is corrupt, `_db = None` and routing falls back to
   `sessions.json` (the JSONL mirror).
2. A multiplexed profile route runs inside `_profile_runtime_scope`, which
   redirects `get_hermes_home()` to the profile home. The AIAgent's lazy
   `SessionDB()` therefore opens the *profile's* `state.db`.
3. Before the fix shipped with this page, that lazy creation wrote the row
   WITHOUT `session_key`/`chat_id`/`chat_type`/`thread_id`/`origin_json`,
   and the gateway's `record_gateway_session_peer` self-heal never ran
   because `_db` was None — so the row stayed identity-less and
   `find_latest_gateway_session_for_peer` could not recover it. The fixed
   `_ensure_db_session` writes the full routing identity on first creation,
   and `load_transcript` falls back to the scoped profile DB so prior turns
   are readable even while the global DB is down.

## Recovery steps (no data loss)

### 1. Verify the corruption and back it up

```bash
HERMES_HOME=${HERMES_HOME:-~/.hermes}
DB="$HERMES_HOME/state.db"
python3 - <<'PY'
import sqlite3, sys
try:
    con = sqlite3.connect(sys.argv[1])
    con.execute("PRAGMA integrity_check").fetchone()
    print("ok")
except Exception as exc:
    print(f"corrupt: {exc}")
PY "$DB"
```

If it fails, copy the file (and any `-wal`/`-shm` sidecars) to a timestamped
backup before touching anything:

```bash
cp -a "$DB" "${DB}.manual-backup-$(date +%Y%m%d_%H%M%S)"
[ -f "$DB-wal" ] && cp -a "$DB-wal" "${DB}-wal.manual-backup-$(date +%Y%m%d_%H%M%S)"
```

### 2. Clear the stale repair lock (safe)

The lock is advisory; removing the file is safe when no Hermes process is
repairing. Stop the gateway first, then:

```bash
rm -f "$DB.repair.lock"
```

### 3. Restore a healthy DB from a known-good backup (preferred)

If any of the `malformed-backup-*`/`backup-before-fts-rebuild-*` files open
cleanly, restore the most recent one:

```bash
ls -la "$HERMES_HOME"/state.db.malformed-backup-* "$HERMES_HOME"/state.db.backup-before-fts-rebuild-* 2>/dev/null
# pick the newest that passes the integrity check above, then:
cp -a "$BACKUP" "$DB"
rm -f "$DB-wal" "$DB-shm"
```

### 4. Otherwise: start fresh, keep sessions.json

The gateway's JSONL routing index (`$HERMES_HOME/sessions/sessions.json`)
holds the `session_key -> session_id` map and survives a DB reset. Move the
corrupt file aside and let Hermes create a new DB:

```bash
mv "$DB" "${DB}.corrupt-$(date +%Y%m%d_%H%M%S)"
rm -f "$DB-wal" "$DB-shm"
```

New sessions get fresh DB rows with full routing identity (post-fix). The
routing keys in `sessions.json` keep resuming the same session ids, so the
conversation continuity is preserved even though the old transcript rows are
gone from the global DB.

### 5. Backfill identity on existing identity-less rows

If a profile DB already contains Telegram session rows with NULL
`session_key`/`chat_id` (pre-fix), you can repair them from the JSONL routing
index. The mapping is `session_key -> session_id`; the row id equals the
session id. Example (run with the gateway stopped, profile-scoped):

```bash
python3 - <<'PY'
import json, sqlite3, sys
from pathlib import Path

home = Path(sys.argv[1])            # profile home, e.g. ~/.hermes/profiles/orion
routing = json.loads((home / "sessions" / "sessions.json").read_text())
db_path = home / "state.db"
con = sqlite3.connect(db_path)
cur = con.cursor()
# key format: agent:<profile>:<platform>:<chat_type>:<chat_id>:<user_id>
for key, entry in routing.items():
    if not isinstance(entry, dict) or "session_id" not in entry:
        continue
    sid = entry["session_id"]
    parts = key.split(":")
    if len(parts) >= 5 and parts[0] == "agent":
        platform = parts[2]
        chat_type = parts[3]
        chat_id = parts[4]
        user_id = parts[5] if len(parts) > 5 else None
        cur.execute(
            """UPDATE sessions
               SET session_key = COALESCE(session_key, ?),
                   chat_id = COALESCE(chat_id, ?),
                   chat_type = COALESCE(chat_type, ?),
                   thread_id = COALESCE(thread_id, ?),
                   user_id = COALESCE(user_id, ?)
             WHERE id = ? AND session_key IS NULL""",
            (key, chat_id, chat_type, None, user_id, sid),
        )
con.commit()
print(f"backfilled {cur.rowcount} rows")
con.close()
PY "$HOME/.hermes/profiles/orion"
```

Verify before/after with the peer lookup:

```bash
python3 - <<'PY'
import os, sqlite3, sys
os.environ.setdefault("HERMES_HOME", sys.argv[1])
from hermes_state import SessionDB
db = SessionDB(db_path=sys.argv[2])
print(db.find_latest_gateway_session_for_peer(
    session_key="agent:orion:telegram:group:-5287315359:8148316720",
    source="telegram",
))
PY
```

### 6. Restart and verify

```bash
hermes gateway restart
hermes gateway status
# send a follow-up message in the affected Telegram chat and confirm the
# agent resumes the same conversation (it references prior turns)
```

## Prevention

- The fix in `run_agent.py::AIAgent._ensure_db_session` persists the full
  routing identity (`session_key`, `chat_id`, `chat_type`, `thread_id`,
  `user_id`, `display_name`, `origin_json`) at first row creation, so rows
  created by the agent (including multiplexed profile routes) are never
  identity-less again.
- `gateway/session.py::SessionStore.load_transcript` now falls back to the
  scoped profile DB when `_db` is None, so prior turns remain readable while
  the global DB is unavailable.
- Keep backups of `state.db` (and the profile DBs) — see
  [Session Storage](session-storage.md) for the canonical store layout.
