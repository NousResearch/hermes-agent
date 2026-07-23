"""Versioned canonical Facebook schema."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 2


OUTBOX_SCHEMA_SQL = r"""
CREATE TABLE facebook_write_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,
    action TEXT NOT NULL,
    recipient_key TEXT NOT NULL,
    payload TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    permission_tier INTEGER NOT NULL CHECK (permission_tier IN (1, 2)),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'sent', 'uncertain', 'failed', 'cancelled')),
    available_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    authorized_at TEXT NOT NULL,
    claimed_by TEXT,
    claim_token TEXT UNIQUE,
    claimed_at TEXT,
    lease_expires_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    reconciliation_attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    external_result_key TEXT,
    postcondition_json TEXT,
    reconciled_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS facebook_write_outbox_claim_idx
ON facebook_write_outbox(status, available_at, id);

CREATE TABLE facebook_write_outbox_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    outbox_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    details_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (outbox_id) REFERENCES facebook_write_outbox(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS facebook_write_outbox_events_idx
ON facebook_write_outbox_events(outbox_id, id);

ALTER TABLE birthday_wishes
ADD COLUMN outbox_id INTEGER REFERENCES facebook_write_outbox(id);

CREATE INDEX IF NOT EXISTS birthday_wishes_outbox_idx ON birthday_wishes(outbox_id);
"""


SCHEMA_SQL = r"""
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE facebook_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE friends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_key TEXT,
    name TEXT NOT NULL,
    profile_url TEXT,
    thread_url TEXT,
    gender TEXT,
    status TEXT DEFAULT 'active',
    is_friend INTEGER NOT NULL DEFAULT 1 CHECK (is_friend IN (0, 1)),
    is_business INTEGER NOT NULL DEFAULT 0 CHECK (is_business IN (0, 1)),
    is_favorite INTEGER NOT NULL DEFAULT 0 CHECK (is_favorite IN (0, 1)),
    is_fake INTEGER NOT NULL DEFAULT 0 CHECK (is_fake IN (0, 1)),
    permission_tier INTEGER NOT NULL DEFAULT 1 CHECK (permission_tier IN (0, 1, 2)),
    profile_analysis TEXT,
    timeline_data TEXT,
    llm_analysis TEXT,
    relationship_tier TEXT,
    relationship_status TEXT,
    added_at TEXT,
    updated_at TEXT
);

CREATE UNIQUE INDEX friends_canonical_key_uq
ON friends(canonical_key)
WHERE canonical_key IS NOT NULL;

CREATE INDEX friends_profile_url_idx ON friends(profile_url);
CREATE INDEX friends_name_idx ON friends(name);

CREATE TRIGGER friends_fill_canonical_key_after_insert
AFTER INSERT ON friends
WHEN NEW.canonical_key IS NULL
 AND NEW.profile_url IS NOT NULL
 AND trim(NEW.profile_url) <> ''
BEGIN
    UPDATE friends
       SET canonical_key = 'facebook:url:' || lower(rtrim(trim(NEW.profile_url), '/'))
     WHERE id = NEW.id;
END;

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_key TEXT,
    friend_id INTEGER NOT NULL,
    sender_name TEXT,
    message_text TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    direction TEXT CHECK (direction IS NULL OR direction IN ('sent', 'received')),
    text TEXT,
    sent_at TEXT,
    message_id TEXT,
    source_system TEXT NOT NULL DEFAULT 'runtime',
    source_record_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (friend_id) REFERENCES friends(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX messages_message_key_uq
ON messages(message_key)
WHERE message_key IS NOT NULL;

CREATE UNIQUE INDEX messages_external_id_uq
ON messages(message_id)
WHERE message_id IS NOT NULL;

CREATE INDEX messages_friend_timestamp_idx ON messages(friend_id, timestamp);

CREATE TRIGGER messages_fill_compat_after_insert
AFTER INSERT ON messages
WHEN NEW.message_key IS NULL
  OR NEW.sender_name IS NULL
  OR NEW.message_text IS NULL
  OR NEW.direction IS NULL
  OR NEW.text IS NULL
  OR NEW.sent_at IS NULL
BEGIN
    UPDATE messages
       SET sender_name = COALESCE(
               NEW.sender_name,
               CASE
                   WHEN NEW.direction = 'sent' THEN 'Me'
                   ELSE (SELECT name FROM friends WHERE id = NEW.friend_id)
               END
           ),
           message_text = COALESCE(NEW.message_text, NEW.text, ''),
           timestamp = CASE
               WHEN NEW.sender_name IS NULL
                AND NEW.message_text IS NULL
                AND NEW.sent_at IS NOT NULL
                   THEN NEW.sent_at
               ELSE COALESCE(NEW.timestamp, NEW.sent_at, CURRENT_TIMESTAMP)
           END,
           direction = COALESCE(
               NEW.direction,
               CASE WHEN NEW.sender_name = 'Me' THEN 'sent' ELSE 'received' END
           ),
           text = COALESCE(NEW.text, NEW.message_text, ''),
           sent_at = COALESCE(NEW.sent_at, NEW.timestamp, CURRENT_TIMESTAMP),
           message_key = COALESCE(
               NEW.message_key,
               CASE
                   WHEN NEW.message_id IS NOT NULL
                       THEN 'facebook:message-id:' || NEW.message_id
                   ELSE 'facebook:runtime:' || NEW.id
               END
           )
     WHERE id = NEW.id;
END;

CREATE TABLE interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_key TEXT,
    friend_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    details TEXT,
    interacted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_system TEXT NOT NULL DEFAULT 'runtime',
    source_record_id TEXT,
    FOREIGN KEY (friend_id) REFERENCES friends(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX interactions_key_uq
ON interactions(interaction_key)
WHERE interaction_key IS NOT NULL;

CREATE TABLE online_activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    friend_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    online_status TEXT,
    FOREIGN KEY (friend_id) REFERENCES friends(id) ON DELETE CASCADE
);

CREATE INDEX online_activity_friend_timestamp_idx
ON online_activity_log(friend_id, timestamp);

CREATE TABLE outreach_campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    friend_id INTEGER NOT NULL UNIQUE,
    goal TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    strategy_summary TEXT,
    milestones_achieved TEXT,
    milestones_pending TEXT,
    redflags_triggered TEXT,
    state_history TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_read_status TEXT DEFAULT 'unknown',
    last_online_status TEXT DEFAULT 'unknown',
    pending_draft TEXT,
    FOREIGN KEY (friend_id) REFERENCES friends(id) ON DELETE CASCADE
);

CREATE TABLE outreach_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT,
    campaign_id INTEGER,
    friend_id INTEGER,
    goal TEXT,
    rel_type TEXT,
    custom_text TEXT,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (campaign_id) REFERENCES outreach_campaigns(id) ON DELETE SET NULL,
    FOREIGN KEY (friend_id) REFERENCES friends(id) ON DELETE SET NULL
);

CREATE TABLE birthday_wishes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    friend_name TEXT,
    profile_url TEXT,
    age_info TEXT,
    proposed_text TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT,
    sent_at TEXT,
    repair_instruction TEXT
);

CREATE TABLE write_approvals (
    token_hash TEXT PRIMARY KEY,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    recipient_key TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_at TEXT,
    metadata_json TEXT
);

CREATE INDEX write_approvals_pending_idx
ON write_approvals(action, recipient_key, expires_at)
WHERE consumed_at IS NULL;

""" + OUTBOX_SCHEMA_SQL + r"""

CREATE TABLE migration_runs (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    sources_json TEXT NOT NULL,
    source_hashes_json TEXT NOT NULL,
    report_json TEXT
);

CREATE TABLE migration_reconciliation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('inserted', 'merged', 'conflict', 'unresolved')),
    target_record_id TEXT,
    reason TEXT,
    details_json TEXT,
    FOREIGN KEY (migration_id) REFERENCES migration_runs(id) ON DELETE CASCADE
);

CREATE INDEX migration_reconciliation_summary_idx
ON migration_reconciliation(migration_id, entity_type, status);
"""


def _user_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def create_schema(connection: sqlite3.Connection, *, applied_at: str) -> None:
    existing = _user_tables(connection)
    if existing:
        raise RuntimeError(
            "Refusing to create the canonical schema over existing tables; "
            "run the explicit migration"
        )
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        "INSERT INTO schema_migrations(version, applied_at, description) VALUES (?, ?, ?)",
        (SCHEMA_VERSION, applied_at, "Canonical Facebook schema v2"),
    )
    connection.executemany(
        "INSERT INTO facebook_settings(key, value, updated_at) VALUES (?, ?, ?)",
        [
            ("schema_version", str(SCHEMA_VERSION), applied_at),
            ("write_actions_enabled", "0", applied_at),
        ],
    )
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()


def ensure_schema(connection: sqlite3.Connection, *, applied_at: str) -> None:
    tables = _user_tables(connection)
    if not tables:
        create_schema(connection, applied_at=applied_at)
        return

    version = connection.execute("PRAGMA user_version").fetchone()[0]
    has_migrations = "schema_migrations" in tables
    if version != SCHEMA_VERSION or not has_migrations:
        raise RuntimeError(
            f"Facebook DB schema is unversioned or unsupported (user_version={version}); "
            "run facebook_migrate_v1.py for legacy databases or "
            "facebook_migrate_v2.py for canonical v1"
        )


def migrate_v1_to_v2(connection: sqlite3.Connection, *, applied_at: str) -> None:
    """Add the fail-closed durable write outbox to a canonical v1 database."""
    tables = _user_tables(connection)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION and {
        "facebook_write_outbox",
        "facebook_write_outbox_events",
    }.issubset(tables):
        return
    if version != 1 or "schema_migrations" not in tables:
        raise RuntimeError(
            f"Expected canonical Facebook schema v1, found user_version={version}"
        )

    statements = [
        statement.strip()
        for statement in OUTBOX_SCHEMA_SQL.split(";")
        if statement.strip()
    ]
    try:
        connection.execute("BEGIN IMMEDIATE")
        for statement in statements:
            if statement.startswith("ALTER TABLE birthday_wishes"):
                birthday_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(birthday_wishes)")
                }
                if "outbox_id" in birthday_columns:
                    continue
            connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at, description) "
            "VALUES (?, ?, ?)",
            (SCHEMA_VERSION, applied_at, "Durable fail-closed Facebook write outbox"),
        )
        connection.execute(
            "UPDATE facebook_settings SET value = ?, updated_at = ? "
            "WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION), applied_at),
        )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
