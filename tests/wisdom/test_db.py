from __future__ import annotations

from wisdom.apply import create_application_proposals
from wisdom.capture import capture_text
from wisdom.classify import classify_capture
from wisdom.db import WisdomDB
from wisdom.interpret import interpret_capture


def test_initializes_tables_and_journal_mode(wisdom_db):
    tables = {
        row["name"]
        for row in wisdom_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        ).fetchall()
    }
    assert {"schema_version", "raw_events", "captures", "interpretations", "applications", "settings"} <= tables
    mode = wisdom_db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert mode in {"wal", "delete"}


def test_migrations_are_idempotent(wisdom_config):
    db = WisdomDB(wisdom_config.db_path)
    db.init()
    db.init()
    version = db.conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    assert version == 2
    columns = {row["name"] for row in db.conn.execute("PRAGMA table_info(captures)").fetchall()}
    assert {"review_status", "reviewed_at", "accepted_at", "dismissed_at", "applied_at"} <= columns
    db.close()


def test_insert_capture_interpretation_application_and_fts_search(wisdom_db, wisdom_config):
    outcome = capture_text(
        "Remember this: clients buy peace of mind, not alpha.",
        config=wisdom_config,
        db=wisdom_db,
        session_key="chat-1",
        message_ref="msg-1",
    )
    assert outcome.status == "captured"
    capture = outcome.capture
    assert capture is not None
    assert capture.original_text == "Remember this: clients buy peace of mind, not alpha."
    assert capture.review_status == "unreviewed"

    interpretation = interpret_capture(wisdom_db, capture.id)
    assert interpretation is not None
    assert interpretation.method == "deterministic"

    apps = create_application_proposals(wisdom_db, capture.id)
    assert apps
    assert all(app.status == "proposed" for app in apps)

    assert wisdom_db.search("peace of mind", limit=5)[0].id == capture.id
    assert wisdom_db.search("client-facing", limit=5)[0].id == capture.id


def test_like_fallback_search(wisdom_config):
    db = WisdomDB(wisdom_config.db_path, force_no_fts=True)
    db.init()
    record = db.create_capture(
        original_text="Note this: fallback search should find this rareword",
        cleaned_text="fallback search should find this rareword",
        classification=classify_capture("fallback search should find this rareword"),
        channel="test",
        source_kind="text",
        session_key_hash=None,
        message_ref_hash=None,
    )
    results = db.search("rareword", limit=5)
    assert [r.id for r in results] == [record.id]
    db.close()
