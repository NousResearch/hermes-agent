# Session Storage

Hermes Agent uses SQLite by default (`~/.hermes/state.db`) to persist session
metadata, full message history, and model configuration across CLI and gateway
sessions. PostgreSQL 17 or later is an optional backend for conversation state.

Source files: `hermes_state.py` (facade) plus the `hermes_state_*.py` siblings (schema, fts, search, compression, portability, gateway, ...)

## Hermes home and profile isolation

`get_hermes_home()` is the authoritative filesystem resolver for state and
configuration. It uses a context-local override first, then the `HERMES_HOME`
environment variable, and finally the platform default (`~/.hermes` on macOS
and Linux; `%LOCALAPPDATA%/hermes` on Windows). Consequently, the default
database is always `get_hermes_home() / "state.db"`, not a path that callers
should hard-code as `~/.hermes/state.db`.

Named profiles are isolated directories: a profile named `coder`, for example,
uses `<default Hermes root>/profiles/coder/` and therefore has its own
`state.db`, configuration, logs, and other profile-scoped state. A process that
creates a database, reads configuration, or starts a child process for a
profile must retain or pass that profile's `HERMES_HOME`; falling back to the
default root mixes the wrong profile's state into the operation.

The CLI bootstrap calls `_apply_profile_override()` before importing the rest
of Hermes. An explicit `--profile`/`-p` resolves that profile and writes the
resolved directory to `HERMES_HOME`. Without an explicit selector, a
profile-specific `HERMES_HOME` is preserved; otherwise the bootstrap can use
the default root's active-profile selection. `HOME` only determines the
platform default used when no context override or `HERMES_HOME` is available.
Changing `HOME` is not a safe way to select a named profile. In particular, a
subprocess that drops `HERMES_HOME` can fall back to the default profile even
when another profile is active, so subprocess spawners should pass
`HERMES_HOME` explicitly.

Use `display_hermes_home()` only for user-facing text. It formats the resolved
home relative to the user's home directory when possible (for example,
`~/.hermes/profiles/coder`); it does not provide a separate resolution rule.

### Test isolation guard

Tests must use a temporary `HERMES_HOME` or an explicit temporary database
path. The live-system guard raises before a test-context process opens a
production `state.db` under the real default Hermes root or a real named
profile, preventing fixture data or SQLite side effects from reaching a live
installation.

`HERMES_STATE_DB_GUARD_BYPASS=1` is a test-only escape hatch for a spawned
child process that genuinely must access the live database. The equivalent
in-process escape hatch is `@pytest.mark.live_system_guard_bypass`. Do not set
either bypass in normal Hermes commands, development shells, or application
configuration: it disables the guard (a hard `RuntimeError`) that protects live
session history, and a shell that exports it hands the bypass to every later
pytest run.

## PostgreSQL conversation state

Install the optional driver in your Hermes environment:

```bash
uv pip install -e '.[postgres]'
```

Configure the profile's `config.yaml`:

```yaml
database:
  backend: postgres
  schema: hermes_personal
```

Put the connection string in that profile's `.env`:

```dotenv
HERMES_DATABASE_URL=postgresql://hermes:password@localhost:5432/hermes
```

Create the database and role beforehand. The role needs permission to create its
schema and objects within it. Hermes initializes those objects on the first
writable open. Read-only opens require an initialized schema. PostgreSQL errors
are reported without falling back to SQLite.

Each profile resolves its own configuration and credentials. Process-level
`HERMES_DATABASE_URL` overrides apply only to the process's home profile. Set a
distinct `database.schema` for each independent profile. If omitted, Hermes
derives it from the profile name and absolute database path; set an explicit
schema before moving a profile to another machine or directory. Restart Hermes
after changing backend settings so existing shared handles can close.

This backend stores sessions, messages, model usage, conversation generations,
routing, compression/turn leases, and optional Telegram topic bindings. History,
resume, search, analytics, and profile session readers use the same `SessionDB`
API and existing field names. Cron, Kanban, delivery/delegation ledgers, and other
application stores retain their existing SQLite storage. Local profile files are
still required; this setting does not make an entire deployment stateless.

The PostgreSQL implementation uses native SQL and a bounded connection pool.
Each write callback runs in one transaction under a per-schema advisory lock,
preserving the existing read/check/write ordering across clients. Reads use
independent read-only transactions. Remote leases expire by TTL; a PID on one
machine cannot establish that a holder on another machine has died.

Search uses a GIN index over bounded text vectors. Oversized messages are also
checked through an overflow index so text beyond the vector limit remains
searchable; stored message content is not truncated. SQLite's existing search
matcher supplies the result semantics and snippets. PostgreSQL autovacuum manages
physical storage. Use PostgreSQL backup tools such as `pg_dump` for this backend;
SQLite repair and file backup procedures below apply only to SQLite.

### Copy existing SQLite history

Stop writers for the source profile and the destination profile. Configure an
empty PostgreSQL schema as above, then run from the Hermes checkout:

```bash
python scripts/migrate_sqlite_to_postgres.py /path/to/source/state.db \
  --profile-home /path/to/destination/profile
```

The command opens SQLite read-only, checks integrity and foreign keys, then copies
conversation tables in batches. Message IDs, archived/rewound messages, usage,
provenance, routing, and optional Telegram topic bindings are retained. Generated
search/display fields are rebuilt. Table counts are checked before one final
commit; a failed copy rolls back, and a populated destination is refused. The
SQLite source remains unchanged. Active leases, heartbeats, and application
ledgers are not copied. Restart the destination profile after a successful copy.

### Test the backend

The normal suite needs no PostgreSQL server. Backend integration tests opt in
with `HERMES_TEST_POSTGRES=1`; CI supplies a disposable PostgreSQL 17 service at
`127.0.0.1:55433`, database/user `hermes_test`/`hermes`, password `hermes-test`.
Tests create and drop uniquely named schemas. They fail if opted-in PostgreSQL
or its driver is unavailable.

```bash
HERMES_TEST_POSTGRES=1 scripts/run_tests.sh tests/hermes_state/test_postgres_*.py
```

### Desktop profile isolation and compaction generations

Each named profile stores its transcript in its own `$HERMES_HOME/state.db`,
including when one `hermes serve` process serves several profiles. In-session
agent rebuilds (Bot Chat capability refresh and `tools.configure`) must retain
that session's database handle and bind its profile home during construction.
Releasing the outgoing agent must not close the handle inherited by its replacement.
`tools.configure` resolves configuration from the live session's `profile_home`,
even when the client supplies only `session_id`. Rebuilds prepare model configuration
before allocating a replacement, then install the agent and transfer ownership
together; preparation failure leaves the existing agent responsible for teardown.
Explicit profiles that cannot be resolved or whose directory has disappeared fail
before accessing launch configuration or history. A stale `tools.configure`
session ID likewise returns `session not found` without changing configuration;
omitting the session ID still supports the global settings operation.

In-place compaction archives old rows with `active=0` and inserts the retained
context as `active=1` rows. A protected message can therefore legitimately appear
in both generations with identical content and timestamp. Do not delete these
archive rows as duplicates. Diagnose duplicate *live* writes using `active=1`,
and check the database's profile as well as the session ID when investigating
history that appears to revert.



## Codex app-server input ownership

The agent persists an accepted user input before starting its Codex turn. Codex
then projects that input as a leading `userMessage` notification. At the runtime
splice boundary, Hermes excludes only that leading item when it exactly matches
the text serialized into `turn/start`, including rich-input coercion. Later or
nonmatching user events remain intact, as do separately accepted identical turns.
This also applies to synthetic/keyless input; it does not depend on a platform
message ID. Existing historical duplicates are not rewritten. The gateway skips
its transcript write when the agent reports that it owns persistence.

## Gateway exception-path input ownership

A gateway exception can occur before agent construction or after its input reaches
SQLite. The gateway gives the accepted input an owner marker in the existing
`display_metadata` sidecar and passes it through the agent's normal persistence
path. Provider messages never contain this metadata. Platform markers namespace
the inbound message ID by platform, profile, scope, chat, and thread; the original
`platform_message_id` remains unchanged for quote/reply resolution. Keyless turns
receive a fresh marker, even for identical text and timestamps.

The exception writer probes only for that marker, following the published reroute
and canonical live compression successor, then compression ancestors. Active rows
and compaction archives count; undone rows, observed input, and unrelated writers
do not. An unrelated process writing the same session cannot suppress this turn.
No whole-history baseline or archived message-body allocation is needed. Failed
ownership reads do not authorize a speculative append; ordinary history-read
failures retain the existing history-unavailable response.

Normal agent-owned persistence is unchanged. This is failure-writer arbitration,
not universal exactly-once delivery, content deduplication, or a schema migration.
Historical rows are not rewritten; unmarked historical inputs cannot establish
ownership for a redelivered event.

## Architecture Overview

```
~/.hermes/state.db (SQLite, WAL mode)
├── sessions              — Session metadata, token counts, billing
├── messages              — Full message history per session
├── session_model_usage   — Per-model/per-task usage attribution rows
├── messages_fts          — FTS5 virtual table (content + tool_name + tool_calls)
├── messages_fts_trigram  — FTS5 virtual table with trigram tokenizer (CJK / substring search)
├── messages_fts_cjk      — FTS5 virtual table with cjk_unicode61 tokenizer
├── state_meta            — Key/value metadata table
├── gateway_routing       — Gateway routing metadata
├── compression_locks     — Cross-process compression locking
├── async_delegations     — Async delegation bookkeeping
├── delivery_obligations  — Gateway outbox (owed replies); created lazily by gateway/delivery_ledger.py
└── schema_version        — Single-row table tracking migration state
```

`hermes sessions recover` copies the row-bearing tables above into the
recovered database (FTS indexes and `schema_version` are regenerated), including
the lazily-created `delivery_obligations` ledger when the source has one — its
row count is verified like `sessions`/`messages`.

Key design decisions:
- **WAL mode** for concurrent readers + one writer (gateway multi-platform)
- **FTS5 virtual table** for fast text search across all session messages
- **Session lineage** via `parent_session_id` chains (compression-triggered splits)
- **Source tagging** (`cli`, `telegram`, `discord`, etc.) for platform filtering
- Batch runner and RL trajectories are NOT stored here (separate systems)


## SQLite Schema

### Sessions Table

Abridged — see `SCHEMA_SQL` in `hermes_state_common.py` (applied by `hermes_state_schema.py`) for the full current column list
(which also includes gateway routing metadata such as `session_key`, `chat_id`,
`chat_type`, `thread_id`, `display_name`, `origin_json`, `expiry_finalized`,
workspace fields `cwd` / `git_branch` / `git_repo_root`, handoff and
compression-failure fields, `profile_name`, `transport_profile` (the multiplex
bot that received the lane, nullable), `rewind_count`, `archived`, and
`pinned`):

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    -- ... additional gateway/workspace/handoff/compression columns ...
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique
    ON sessions(title) WHERE title IS NOT NULL;
```

`user_id` is the principal on the other end of the session: messaging adapters
store the platform sender id, and `desktop` / dashboard sessions opened through
an authenticated `hermes serve` (OAuth or the basic username/password provider)
store the login as `<provider>:<user id>` (for example `basic:alice`). Sessions
with nobody behind them — anonymous loopback use, `subagent`, `cron`, `kanban`
— keep it empty. The value is set when the row is created and never inferred
later.

### Messages Table

Abridged — the full schema also includes `effect_disposition`,
`platform_message_id`, `observed`, `active`, `compacted`, `api_content`,
`display_kind`, and `display_metadata`:

```sql
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT
    -- ... additional display/compaction columns ...
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
```

Notes:
- `tool_calls` is stored as a JSON string (serialized list of tool call objects)
- `reasoning_details`, `codex_reasoning_items`, and `codex_message_items` are stored as JSON strings
- `reasoning_details` is always kept in history; on the chat-completions wire it is replayed only to OpenRouter and the Nous Portal (every other chat-completions route gets a copy without it, since strict schemas reject the field)
- Desktop history hydration retains assistant sidecars in both REST and JSON-RPC (`session.resume`, `session.activate`, `session.history`) projections, including rows with reasoning and tool calls. REST may return the SQLite JSON string while RPC returns decoded items; Desktop accepts both. A final Responses reply may live only in `codex_message_items` while `content` is empty. Canonical content still takes precedence, and analysis/commentary items are not promoted to reply text.
- `reasoning` stores the raw reasoning text for providers that expose it
- A reasoning-only clean stop (empty `content`, `finish_reason=stop`, reasoning present) is answered with the reasoning text, but the assistant row is never written with that text as `content`: `content` stays empty, the text lives in `reasoning`/`reasoning_content`, and `api_content` carries it so the next request replays the answer byte-identically. History surfaces therefore show it as reasoning, not as a reply.
- `api_content` is a byte-fidelity sidecar: the exact content string sent to the API for this message when it differs from `content` (ephemeral memory/plugin injections, persist overrides). It preserves the wire bytes for prompt-cache-stable replay — stored as sent, except lone surrogates, which sqlite3 cannot bind and which the conversation loop scrubs from every outgoing payload anyway. `NULL` means `content` was sent verbatim.
- Timestamps are Unix epoch floats (`time.time()`)

### FTS5 Full-Text Search

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id'
);
```

The FTS5 table is kept in sync via three triggers that fire on INSERT, UPDATE,
and DELETE of the `messages` table. The current triggers are gated on the
`fts_rebuild_high_water` / `fts_rebuild_progress` markers in `state_meta` (so a
background FTS rebuild can proceed without double-indexing) and cover all three
indexed columns — see `SCHEMA_SQL` in `hermes_state_common.py` for the exact SQL.


## Schema Version and Migrations

Current schema version: **23**

The `schema_version` table stores a single integer. Simple column additions are handled declaratively by `_reconcile_columns()` (which diffs live columns against `SCHEMA_SQL` and ADDs any missing ones). The version-gated chain is reserved for data migrations and index/FTS changes that can't be expressed declaratively:

| Version | Change |
|---------|--------|
| 1 | Initial schema (sessions, messages, FTS5) |
| 2 | Add `finish_reason` column to messages |
| 3 | Add `title` column to sessions |
| 4 | Add unique index on `title` (NULLs allowed, non-NULL must be unique) |
| 5 | Add billing columns: `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`, `billing_provider`, `billing_base_url`, `billing_mode`, `estimated_cost_usd`, `actual_cost_usd`, `cost_status`, `cost_source`, `pricing_version` |
| 6 | Add reasoning columns to messages: `reasoning`, `reasoning_details`, `codex_reasoning_items` |
| 7 | Add `reasoning_content` column to messages |
| 8 | Add `api_call_count` column to sessions |
| 9 | Add `codex_message_items` column to messages for Codex Responses message id/phase replay |
| 10 | Add `messages_fts_trigram` virtual table (trigram tokenizer for CJK / substring search) and backfill existing rows |
| 11 | Re-index `messages_fts` and `messages_fts_trigram` to cover `tool_name` + `tool_calls` and switch from external-content to inline mode; drop old triggers and backfill every message row |
| 16 | Tag delegate subagent rows in `model_config` (`$._delegate_from`) so session pickers stay clean after parent deletes orphan them |
| 18 | Gateway metadata consolidation — backfill `display_name` / `origin_json` / `expiry_finalized` from `sessions.json` |
| 20 | Per-model usage attribution — seed `session_model_usage` rows from historical per-session aggregate totals |
| 22 | Task-dimension usage attribution — rebuild `session_model_usage` so the `task` column participates in the PRIMARY KEY |
| 23 | FTS storage redesign — external-content FTS tables replacing the v11 inline-mode copies (opt-in transition for existing DBs) |
| 29 | Cron sessions leave the trigram (substring/CJK) index; `messages_fts_trigram_src` view + triggers filter on `sessions.source`, one-time rebuild purges historical rows |
| 30 | Delegate-child (subagent) sessions leave the trigram index too — `source='subagent'` or the `$._delegate_from` marker (`FTS_TRIGRAM_SESSION_SQL`). Rows stay in `messages` and the standard `messages_fts` word index, so `session_search` still finds them; only the ~2.6× trigram shadow tables shrink. Same one-time rebuild as v29 |

Versions not listed above were declarative column additions handled by `_reconcile_columns()` (version bump only, no data migration).

Declarative column adds use `ALTER TABLE ADD COLUMN` wrapped in try/except to handle the column-already-exists case (idempotent). The version number is bumped after each successful migration block.


## Write Contention Handling

Multiple hermes processes (gateway + CLI sessions + worktree agents) share one
`state.db`. The `SessionDB` class handles write contention with:

- **Short SQLite timeout** (1 second) instead of the default 30s
- **Time-budgeted application-level retry** with random jitter (20-150ms for the
  first 2s, then 250ms-1s): 20s for routine writes, 60s for transcript writes
  (their failure aborts the turn), 0.5s for observation-only activity writes
- **BEGIN IMMEDIATE** transactions to surface lock contention at transaction start
- **Periodic WAL checkpoints** every 50 successful writes (PASSIVE mode)

This avoids the "convoy effect" where SQLite's deterministic internal backoff
causes all competing writers to retry at the same intervals.

```
_WRITE_PATIENCE_S, _TRANSCRIPT_WRITE_PATIENCE_S, _ACTIVITY_WRITE_PATIENCE_S = 20.0, 60.0, 0.5
_WRITE_RETRY_MIN_S, _WRITE_RETRY_MAX_S = 0.020, 0.150
_WRITE_RETRY_SLOW_MIN_S, _WRITE_RETRY_SLOW_MAX_S = 0.250, 1.000
_CHECKPOINT_EVERY_N_WRITES = 50
```

When a writer exhausts its budget the turn ends with
`session_persistence_failed:locked` and, on Linux, `hermes_state_lockowners`
logs a WARNING naming the process that held the lock at that moment
(`PID 594094 (hermes --worktree --yolo) holds WAL write lock on state.db-shm`),
read from `/proc/locks` — SQLite's byte-range `fcntl` locks encode the lock kind
in their offset (`state.db-shm` byte 120 = WAL write, 121 = checkpoint,
123-127 = read slots; the 1 GiB pending-byte page on `state.db` = rollback-journal
PENDING/RESERVED/SHARED). The open-descriptor scan cannot make this distinction
because every Hermes process has the DB open. Look for that line in
`~/.hermes/logs/errors.log` next to the `database is locked` failure.

Lock contention is recognised by SQLite result code (`SQLITE_BUSY` /
`SQLITE_LOCKED`, `hermes_state_errors.is_sqlite_lock_error`), not by message
text. In rollback-journal (`delete`) mode a lock lost inside FTS5's table
constructor arrives as `SQLITE_BUSY` with the text `vtable constructor failed:
messages_fts`; it is treated like `database is locked`. Opening a writable
`SessionDB` waits up to `_WRITE_PATIENCE_S`; a read-only open waits its
`_READ_BUSY_TIMEOUT_S` (5 s) read budget once. If the lock outlasts that, the dashboard
answers 503 (busy), not 500.


## Common Operations

### Initialize

```python
from hermes_state import SessionDB

db = SessionDB()                           # Default: ~/.hermes/state.db
db = SessionDB(db_path=Path("~/.hermes/cache/scratch/test.db").expanduser())  # Custom path
```

### Create and Manage Sessions

```python
# Create a new session
db.create_session(
    session_id="sess_abc123",
    source="cli",
    model="anthropic/claude-sonnet-4.6",
    user_id="user_1",
    parent_session_id=None,  # or previous session ID for lineage
)

# End a session
db.end_session("sess_abc123", end_reason="user_exit")

# Reopen a session (clear ended_at/end_reason)
db.reopen_session("sess_abc123")
```

### Store Messages

```python
msg_id = db.append_message(
    session_id="sess_abc123",
    role="assistant",
    content="Here's the answer...",
    tool_calls=[{"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}],
    token_count=150,
    finish_reason="stop",
    reasoning="Let me think about this...",
)
```

### Retrieve Messages

```python
# Raw messages with all metadata
messages = db.get_messages("sess_abc123")

# OpenAI conversation format (for API replay)
conversation = db.get_messages_as_conversation("sess_abc123")
# Returns: [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
```

### Session Titles

```python
# Set a title (must be unique among non-NULL titles)
db.set_session_title("sess_abc123", "Fix Docker Build")

# Resolve by title (returns most recent in lineage)
session_id = db.resolve_session_by_title("Fix Docker Build")

# Auto-generate next title in lineage
next_title = db.get_next_title_in_lineage("Fix Docker Build")
# Returns: "Fix Docker Build #2"
```


## Full-Text Search

The `search_messages()` method supports FTS5 query syntax with automatic
sanitization of user input.

### Basic Search

```python
results = db.search_messages("docker deployment")
```

### FTS5 Query Syntax

| Syntax | Example | Meaning |
|--------|---------|---------|
| Keywords | `docker deployment` | Both terms (implicit AND) |
| Quoted phrase | `"exact phrase"` | Exact phrase match |
| Boolean OR | `docker OR kubernetes` | Either term |
| Boolean NOT | `python NOT java` | Exclude term |
| Prefix | `deploy*` | Prefix match |

### Filtered Search

```python
# Search only CLI sessions
results = db.search_messages("error", source_filter=["cli"])

# Exclude gateway sessions
results = db.search_messages("bug", exclude_sources=["telegram", "discord"])

# Search only user messages
results = db.search_messages("help", role_filter=["user"])
```

### Search Results Format

Each result includes:
- `id`, `session_id`, `role`, `timestamp`
- `snippet` — FTS5-generated snippet with `>>>match<<<` markers
- `context` — 1 message before and after the match (content truncated to 200 chars)
- `source`, `model`, `session_started` — from the parent session

The `_sanitize_fts5_query()` method handles edge cases:
- Strips unmatched quotes and special characters
- Wraps hyphenated terms in quotes (`chat-send` → `"chat-send"`)
- Removes dangling boolean operators (`hello AND` → `hello`)


## Session Lineage

Sessions can form chains via `parent_session_id`. This happens when context
compression triggers a session split in the gateway.

### Query: Find Session Lineage

```sql
-- Find all ancestors of a session
WITH RECURSIVE lineage AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN lineage l ON s.id = l.parent_session_id
)
SELECT id, title, started_at, parent_session_id FROM lineage;

-- Find all descendants of a session
WITH RECURSIVE descendants AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN descendants d ON s.parent_session_id = d.id
)
SELECT id, title, started_at FROM descendants;
```

### Query: Recent Sessions with Preview

```sql
SELECT s.*,
    COALESCE(
        (SELECT SUBSTR(m.content, 1, 63)
         FROM messages m
         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
         ORDER BY m.timestamp, m.id LIMIT 1),
        ''
    ) AS preview,
    COALESCE(
        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
        s.started_at
    ) AS last_active
FROM sessions s
ORDER BY s.started_at DESC
LIMIT 20;
```

### Query: Token Usage Statistics

```sql
-- Total tokens by model
SELECT model,
       COUNT(*) as session_count,
       SUM(input_tokens) as total_input,
       SUM(output_tokens) as total_output,
       SUM(estimated_cost_usd) as total_cost
FROM sessions
WHERE model IS NOT NULL
GROUP BY model
ORDER BY total_cost DESC;

-- Sessions with highest token usage
SELECT id, title, model, input_tokens + output_tokens AS total_tokens,
       estimated_cost_usd
FROM sessions
ORDER BY total_tokens DESC
LIMIT 10;
```


## Export and Cleanup

```python
# Export a single session with messages
data = db.export_session("sess_abc123")

# Export all sessions (with messages) as list of dicts
all_data = db.export_all(source="cli")

# Delete old sessions (only ended sessions)
deleted_count = db.prune_sessions(older_than_days=90)
deleted_count = db.prune_sessions(older_than_days=30, source="telegram")

# Clear messages but keep the session record
db.clear_messages("sess_abc123")

# Delete session and all messages
db.delete_session("sess_abc123")
```


## Database Location

Default path: `get_hermes_home() / "state.db"` — `~/.hermes/state.db` for the
default profile, `~/.hermes/profiles/<name>/state.db` for a named profile, or
wherever `HERMES_HOME` points (see [Hermes home and profile isolation](#hermes-home-and-profile-isolation)).

The database file, WAL file (`state.db-wal`), and shared-memory file
(`state.db-shm`) are all created in the same directory.
