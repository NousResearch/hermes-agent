---
sidebar_position: 18
title: "PostgreSQL State Backend"
description: "Optional PostgreSQL backend for session and state storage, in place of the default single-file SQLite database"
---

# PostgreSQL State Backend

Hermes stores sessions, messages, and agent state in a single SQLite file
(`~/.hermes/state.db`) by default. That is the right choice for a personal
install: zero setup, zero moving parts, and it comfortably handles tens of
thousands of messages.

Some deployments outgrow it. If you run Hermes across several hosts against
shared state, on a container filesystem where SQLite's locking semantics do not
hold, or under an operational policy that requires a managed database, you can
point session storage at an external PostgreSQL server instead.

This backend is **opt-in and off by default**. Installs that do not configure it
never load the driver and never pay for it.

## When to use it

Consider PostgreSQL when any of these apply:

- **Multiple hosts share one state store.** Several Hermes processes (gateway,
  dashboard, cron) running on different machines against the same sessions.
- **The filesystem is unsuitable for SQLite.** Network filesystems and some
  container volumes do not provide the locking or `mmap` guarantees SQLite needs.
- **Operational policy requires a managed database.** Backups, point-in-time
  restore, failover, and monitoring handled by existing database infrastructure.

Stay on the default SQLite backend otherwise. It is faster for a single-host
install, needs no server, and is the more thoroughly exercised path.

## Requirements

- **PostgreSQL 16 or newer.**
- **The `postgres` extra**, which installs the `psycopg` driver:

  ```bash
  pip install 'hermes-agent[postgres]'
  # or, from a source checkout:
  uv sync --extra postgres
  ```

  Hermes will also attempt to install the driver on first use if it is missing,
  so an existing install that flips the config key generally does not need a
  manual step. Environments that block outbound PyPI access should install the
  extra ahead of time. The published Docker image ships with it already baked in.

- **The `pg_trgm` extension** is recommended but not required. Hermes tries to
  create it on first connect (`CREATE EXTENSION IF NOT EXISTS pg_trgm`); it
  backs the GIN trigram indexes that accelerate substring search.

  If the connecting role may not create extensions, Hermes logs a warning and
  continues — search still works, because the `ILIKE` path is plain SQL and the
  full-text path uses core PostgreSQL `tsvector`. Only the trigram acceleration
  is lost, which matters once the message table is large. Hermes re-attempts the
  extension on every connect, so enabling it later is picked up automatically
  with no migration to run by hand.

  On self-managed PostgreSQL, a superuser can install it once per database:
  `CREATE EXTENSION IF NOT EXISTS pg_trgm;`

  See [Enabling `pg_trgm` on managed PostgreSQL](#enabling-pg_trgm-on-managed-postgresql)
  below when the provider restricts extensions.

The database user needs `CREATE` on the target database — Hermes manages its own
tables, indexes, and migrations.

### Enabling `pg_trgm` on managed PostgreSQL

Hosted providers only let non-superusers create extensions from an allow-list.
Hermes runs fine without `pg_trgm`, but enabling it is recommended before the
message table grows, because it is what keeps substring search off a sequential
scan.

**Azure Database for PostgreSQL — Flexible Server.** The `azure.extensions`
server parameter is empty by default, and `CREATE EXTENSION` fails with:

```
ERROR: extension "pg_trgm" is not allow-listed for users in
       Azure Database for PostgreSQL
```

Add it to the allow-list, preserving any extensions already listed — the value
is a comma-separated list and setting it replaces the whole thing:

```bash
# Inspect the current value first so you don't drop existing entries.
az postgres flexible-server parameter show \
  --resource-group <rg> --server-name <server> \
  --name azure.extensions --query value -o tsv

az postgres flexible-server parameter set \
  --resource-group <rg> --server-name <server> \
  --name azure.extensions --value pg_trgm
```

In the portal the parameter is under **Server parameters**; search for
`azure.extensions` and pick `pg_trgm` from the dropdown rather than the "All"
tab, where it is not listed.

`azure.extensions` is a dynamic parameter, so the change generally applies
without a restart — confirm with:

```bash
az postgres flexible-server parameter show \
  --resource-group <rg> --server-name <server> \
  --name azure.extensions \
  --query "{value:value,dynamic:isDynamicConfig,restartPending:isConfigPendingRestart}"
```

If `restartPending` is true, or `CREATE EXTENSION` still reports the extension
as not allow-listed, restart the server and retry:

```bash
az postgres flexible-server restart --resource-group <rg> --name <server>
```

Allow-listing makes the extension *available*; it does not install it.
Extensions are per-database, so each database needs its own `CREATE EXTENSION`.
Hermes does this itself on the next connect — restart the agent, or just let it
reconnect, and it will install `pg_trgm` and build the trigram indexes with no
further action. To confirm:

```sql
SELECT count(*) FROM pg_extension WHERE extname = 'pg_trgm';   -- expect 1
SELECT count(*) FROM pg_indexes WHERE indexname LIKE '%\_trgm';  -- expect 3
```

**Other managed providers.** Amazon RDS and Aurora ship `pg_trgm` in the
default `rds.allowed_extensions`, so `CREATE EXTENSION` normally just works.
Google Cloud SQL supports it without an allow-list. In both cases the role
still needs sufficient privilege on the target database.

## Enabling it

Select the backend under `sessions:` in your active profile's `config.yaml`
(`~/.hermes/config.yaml` for the default profile):

```yaml
sessions:
  state_backend: postgres
```

Or via the CLI:

```bash
hermes config set sessions.state_backend postgres
```

Supply the credential-bearing DSN in the profile's `.env` file or your
orchestrator's secret store:

```bash
HERMES_STATE_DATABASE_URL="postgresql://hermes:secret@db.example.com:5432/hermes?sslmode=require"
```

The DSN is passed to the driver **unchanged**, so TLS mode, host, port, and
credentials are yours to specify. Use `sslmode=require` (or stricter) for
connections crossing a network.

### DSN environment variables

These credential variables take precedence over `sessions.postgres_dsn` in
`config.yaml`:

| Variable | Purpose |
|---|---|
| `HERMES_STATE_DATABASE_URL` | PostgreSQL DSN |
| `HERMES_STATE_POSTGRES_DSN` | Alternate name for the same value |

Resolution order is `HERMES_STATE_DATABASE_URL` →
`HERMES_STATE_POSTGRES_DSN` → `sessions.postgres_dsn`. Keep backend selection
in `sessions.state_backend`; setting a DSN alone does not switch backends.

## Migrating existing sessions

The `hermes migrate state-to-postgres` subcommand performs a one-shot copy of
an existing SQLite state database into PostgreSQL:

```bash
hermes migrate state-to-postgres [--dsn 'postgresql://...'] [--sqlite-path PATH]
```

Running without `--dsn` resolves the target from the
`HERMES_STATE_DATABASE_URL` / `HERMES_STATE_POSTGRES_DSN` environment
variables, or from `sessions.postgres_dsn` in `config.yaml` when
`sessions.state_backend: postgres` is already set. `--sqlite-path` defaults
to `state.db` under your Hermes home.

Use `-y` / `--yes` to skip the confirmation prompt in scripts or CI:

```bash
hermes migrate state-to-postgres --dsn 'postgresql://...' --yes
```

The equivalent direct invocation (documented for scripting or for users who
prefer to run the module without the CLI) is:

```bash
python -m migrate_state_to_postgres --dsn 'postgresql://...' [--sqlite-path PATH]
```

The migration reads raw database rows from one read-only SQLite snapshot and
checks the destination against that same snapshot. It preserves:

- All sessions, including child/delegate sessions, and their parent links and
  cached system prompts.
- All messages, including rewound and compacted rows, with their ids,
  timestamps, reasoning, display metadata and display order. Structured content
  is decoded and re-encoded to replace the legacy NUL marker. Derived display
  identities are marked for rebuilding when stored content changes.
- Per-model usage, gateway routing and hygiene state, conversation generation
  counters, and durable metadata such as session goals, loops and scheduled
  heartbeats.
- Telegram topic settings and bindings when those optional tables exist in the
  source. Migrating a store without topic tables does not enable topic mode.

**Retries preserve existing target history.** Missing rows are inserted without
duplicating keys. Existing session/message rows are not refreshed or overwritten;
a differing value makes verification fail, even when row counts match. Peer
conversation generations are the exception: the larger source or target value
wins so a retired prompt-cache identity cannot be reused. Use a fresh, dedicated
target database; keep any populated target intact while investigating a mismatch.

**Some state stays local.** SQLite outboxes such as `async_delegations` continue
to use the source file. Process heartbeats, compression locks and turn leases
are not copied; restarted processes establish their own liveness and ownership.
SQLite FTS progress, file generation/provenance and vacuum bookkeeping are not
portable and are excluded. The migration does not move profile files or make
SQLite-local queues shared across hosts.

**Verification covers every copied row and durable column**, including message
content, prompt references and auxiliary tables. Session/message counts show
matching primary keys; they do not prove that values match. Only a successful
`Migration complete` result (or `OK` from the standalone module) means the
whole verification passed. A mismatch exits with status 1 and leaves the SQLite
source untouched.

Recommended sequence:

1. Stop every Hermes process using the source or target and take a SQLite
   backup. Use a SQLite-aware backup if WAL sidecars are still present. Keep
   writers stopped through verification and the backend switch; later source
   writes are outside the migration snapshot.
2. Run the migration into a fresh, dedicated PostgreSQL database. Confirm it
   reports `Migration complete`, rather than relying only on equal counts.
3. Set `sessions.state_backend: postgres` in the active profile's `config.yaml`.
4. Start Hermes and confirm `/resume` and `session_search` behave. Follow the
   search-backfill guidance below if imported history still needs indexing.
5. Keep the SQLite file and backup for recovery and any remaining local outboxes.

## Behavioral notes

**Failure is loud, not silent.** If the DSN is wrong or the server is
unreachable, opening the session store raises. Hermes does **not** quietly fall
back to SQLite — a silent fallback would write to a different database than the
one you configured and split your history across two stores.

**Search uses native full-text indexing, with an `ILIKE` fallback.** SQLite's
FTS5 index has no direct PostgreSQL equivalent, so the backend builds its own:
a `tsvector` column (`messages.fts_content`) with a GIN index. Hermes attempts
to populate it for each new message. Queries use `fts_content @@ tsquery` with the
`simple` dictionary (lowercasing, no stemming — the right choice for a corpus
mixing code identifiers, proper nouns, and multiple languages), which gives
tokenized multi-word AND search.

:::note When search backfill is needed

`fts_content IS NULL` marks a row that needs indexing. Imported or older history
and failed indexing attempts can leave these rows. A later transcript repair
that changes message content, tool names or tool calls also clears the affected
search vector, preventing searches from using stale text.

PostgreSQL migration 27 installs triggers that maintain display and search
data, and invalidates existing derived values once. The display index rebuilds
when a writable Hermes process reads a conversation; restore search vectors
with the backfill below.

:::

While any rows still need indexing, search uses the `ILIKE` fallback across all
rows. Multi-word queries use substring matching during this period, with
`pg_trgm` indexes providing acceleration when available.

Run or repeat this backfill whenever rows have `fts_content IS NULL`. It updates
only those rows and is safe to re-run:

```sql
UPDATE messages
   SET fts_content = to_tsvector(
       'simple', coalesce(content, '') || ' ' ||
       coalesce(tool_name, '') || ' ' || coalesce(tool_calls, ''))
 WHERE fts_content IS NULL;
```

Search switches to the FTS path automatically once no `NULL` rows remain.
Later content repairs can require another backfill, including on a database
that has always used PostgreSQL.

**Two independent schema version counters.** `SCHEMA_VERSION` in
`hermes_state_common.py` governs the shared and SQLite schema and is recorded
in `schema_version`. PostgreSQL schema setup and its own migration list live
in `hermes_state_pg_schema.py`, with a separate counter in
`pg_migration_version`. The two numbers are deliberately unrelated — do not
expect them to match, and do not merge the tables. (They shared one table
originally, which meant that once the shared version climbed past the highest
Postgres-only migration number, every Postgres migration looked already-applied
and was skipped.)

Postgres-only migrations are idempotent and retryable. Writable startup applies
missing ledger entries, including gaps left by optional migrations. A step may
create missing objects, replace trigger definitions or invalidate stale derived
data. Completed steps are recorded and skipped on later opens; a database with
no `pg_migration_version` entries replays the migration list.

**Read-only opens read PostgreSQL, but cannot write to it.** Read-only callers
— the dashboard's status and session listing, cron history, usage analytics,
and resume lookup — are served from the same physical store as the live write
path. Routing them to a local `state.db` instead would report on a different
database than the one being written to.

Because those callers are not the owner of the store, a read-only open is
restricted in three ways:

- It runs no DDL. Schema is created and migrated only by writable opens.
- It checks both the shared version in `schema_version` and the PostgreSQL-only
  ledger in `pg_migration_version`. If the schema is absent or either version is
  below this build's requirement, the open fails. Start Hermes normally
  against the database once to apply pending changes, then retry the read-only
  operation. Newer store versions are accepted to support rolling upgrades.
- The connection sets `default_transaction_read_only`, so the server itself
  rejects any write with SQLSTATE `25006`. The prohibition is enforced by
  PostgreSQL, not by convention.

**Structured message content uses a `U+0001` sentinel prefix.** PostgreSQL's
`text` type cannot store `NUL`, so the marker distinguishing JSON-encoded
multimodal content from plain strings uses `U+0001` on write. The legacy
`NUL` prefix is still accepted on read, so rows written by older versions decode
correctly on both backends.

## Verifying

After enabling, confirm the backend is actually engaged:

```bash
# Sessions should appear in the target database, not in state.db
psql "$HERMES_STATE_DATABASE_URL" -c 'SELECT count(*) FROM sessions;'
psql "$HERMES_STATE_DATABASE_URL" -c 'SELECT count(*) FROM messages;'
```

Then start a session, send a message, and re-run the message count — it should
increase. A count that stays flat while Hermes appears healthy means the backend
did not engage and writes are still going to SQLite; check that
`sessions.state_backend` is `postgres` in the active profile and that the DSN
points to the database you are querying.

## Cleaning stale tool-call markers

You can inspect affected rows in PostgreSQL without changing them:

```bash
hermes sessions clean-markers --dry-run
```

The command's automatic backup step supports SQLite only. Before cleaning a
PostgreSQL store, take a native PostgreSQL backup with `pg_dump` or your managed
provider's backup facility, then use the existing `--no-backup` flag:

```bash
hermes sessions clean-markers --no-backup
```

Without that flag, a cleanup that would change PostgreSQL rows stops with an
explanation before modifying them. The flag skips only the automatic SQLite
backup step; it does not create a PostgreSQL backup.

## Troubleshooting

**`PostgreSQL state backend requires psycopg`** — the `postgres` extra is not
installed and the automatic install did not succeed. Install it explicitly:
`pip install 'hermes-agent[postgres]'`.

**`permission denied to create extension "pg_trgm"`** or **`extension "pg_trgm"
is not allow-listed`** — the connecting role cannot create extensions. This is
**not fatal**: Hermes logs a warning, skips the extension and its trigram
indexes, and runs with search intact but unaccelerated. Fix it at your leisure —
on self-managed PostgreSQL have a superuser run
`CREATE EXTENSION IF NOT EXISTS pg_trgm;` against the target database; on a
managed provider see
[Enabling `pg_trgm` on managed PostgreSQL](#enabling-pg_trgm-on-managed-postgresql).
Hermes retries on the next connect and installs it once permitted.

**Search feels slow on a large database** — check whether the trigram indexes
exist (`SELECT count(*) FROM pg_indexes WHERE indexname LIKE '%\_trgm';`,
expect 3). Zero means `pg_trgm` was never installed, so substring search is
sequential-scanning. Enable the extension as above.

**Turn errors mentioning connection failures** — a managed server restarting for
maintenance surfaces as transient contention and is reported as a retryable
condition rather than storage damage. Retry the message; if it persists, check
server availability and connection limits.

**Search returns nothing** — confirm the backend engaged (see *Verifying*
above). Zero results with a healthy-looking gateway is the classic symptom of
writes landing in one store while reads come from another.
