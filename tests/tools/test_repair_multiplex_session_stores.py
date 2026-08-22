from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tools.repair_multiplex_session_stores import (
    RepairBlocked,
    _preflight_move,
    build_plan,
    execute_plan,
    main,
    plan_fingerprint,
)


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE system_prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    system_prompt_hash TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    profile_name TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT,
    timestamp REAL NOT NULL
);
CREATE TABLE session_model_usage (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    api_call_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, model)
);
CREATE TABLE compression_locks (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    holder TEXT NOT NULL
);
CREATE TABLE gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
);
CREATE TABLE gateway_hygiene_state (
    session_key TEXT PRIMARY KEY,
    failure_streak INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def make_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA user_version = 26")
    conn.commit()
    return conn


def seed_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    session_key: str | None = None,
    profile_name: str | None = None,
    origin_profile: str | None = None,
    parent_session_id: str | None = None,
    prompt_hash: str | None = None,
    started_at: float = 1.0,
) -> None:
    origin = None
    if origin_profile is not None:
        origin = json.dumps(
            {
                "platform": "telegram",
                "chat_id": "chat",
                "profile": origin_profile,
            }
        )
    conn.execute(
        """INSERT INTO sessions
           (id, source, user_id, session_key, chat_id, chat_type, thread_id,
            display_name, origin_json, system_prompt_hash, parent_session_id,
            started_at, profile_name)
           VALUES (?, 'telegram', 'user', ?, 'chat', 'dm', NULL,
                   'Chat', ?, ?, ?, ?, ?)""",
        (
            session_id,
            session_key,
            origin,
            prompt_hash,
            parent_session_id,
            started_at,
            profile_name,
        ),
    )


def ids(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute("SELECT id FROM sessions")}
    finally:
        conn.close()


@pytest.fixture
def homes(tmp_path: Path):
    home = tmp_path / ".hermes"
    root = home / "state.db"
    finance = home / "profiles" / "finance" / "state.db"
    root_conn = make_store(root)
    finance_conn = make_store(finance)
    yield home, root, finance, root_conn, finance_conn
    root_conn.close()
    finance_conn.close()


def test_dry_run_is_read_only_and_keeps_keyless_rows(homes):
    home, root, finance, root_conn, finance_conn = homes
    seed_session(root_conn, "local", session_key=None, profile_name=None)
    seed_session(
        root_conn,
        "misplaced",
        session_key="agent:finance:telegram:dm:chat",
        profile_name="finance",
        origin_profile="finance",
    )
    root_conn.commit()

    plan = build_plan(home)
    assert plan.safe
    assert plan.session_count == 1
    manifest = execute_plan(plan, apply=False)

    assert manifest["dry_run"] is True
    assert ids(root) == {"local", "misplaced"}
    assert ids(finance) == set()


def test_apply_moves_lineage_related_rows_routing_and_remaps_message_ids(homes, tmp_path):
    home, root, finance, root_conn, finance_conn = homes
    root_conn.execute("INSERT INTO system_prompts VALUES ('prompt-hash', 'system')")
    finance_conn.execute("INSERT INTO system_prompts VALUES ('other', 'other')")

    # Existing destination message id=1 forces the moving message rowids to be
    # remapped. Its session is unrelated and must remain untouched.
    seed_session(
        finance_conn,
        "existing",
        session_key="agent:finance:telegram:dm:other",
        profile_name="finance",
    )
    finance_conn.execute(
        "INSERT INTO messages(id, session_id, role, content, timestamp) "
        "VALUES (1, 'existing', 'user', 'existing', 1.0)"
    )

    # Parent has no explicit profile; one unambiguous child claim propagates to
    # the whole lineage component so the FK remains intact.
    seed_session(root_conn, "parent", prompt_hash="prompt-hash")
    seed_session(
        root_conn,
        "child",
        session_key="agent:finance:telegram:dm:chat",
        profile_name="finance",
        origin_profile="finance",
        parent_session_id="parent",
        prompt_hash="prompt-hash",
        started_at=2.0,
    )
    root_conn.execute(
        "INSERT INTO messages(id, session_id, role, content, timestamp) "
        "VALUES (1, 'parent', 'user', 'parent message', 1.0)"
    )
    root_conn.execute(
        "INSERT INTO messages(id, session_id, role, content, timestamp) "
        "VALUES (2, 'child', 'assistant', 'child message', 2.0)"
    )
    root_conn.execute(
        "INSERT INTO session_model_usage VALUES ('child', 'model', 3)"
    )
    root_conn.execute(
        "INSERT INTO compression_locks VALUES ('child', 'stale-holder')"
    )
    root_conn.execute(
        "INSERT INTO gateway_routing VALUES (?, ?, ?, 2.0)",
        (
            "",
            "agent:finance:telegram:dm:chat",
            json.dumps(
                {
                    "session_key": "agent:finance:telegram:dm:chat",
                    "session_id": "child",
                }
            ),
        ),
    )
    root_conn.execute(
        "INSERT INTO gateway_hygiene_state VALUES "
        "('agent:finance:telegram:dm:chat', 4)"
    )
    root_conn.commit()
    finance_conn.commit()
    root_conn.close()
    finance_conn.close()

    plan = build_plan(home)
    assert plan.safe, plan.blocked
    assert len(plan.moves) == 1
    assert plan.moves[0].session_ids == ["child", "parent"]

    manifest = execute_plan(
        plan,
        apply=True,
        backup_dir=tmp_path / "backups",
    )
    assert manifest["moves"][0]["status"] == "completed"
    assert len(manifest["backups"]) == 2
    assert all(Path(info["path"]).exists() for info in manifest["backups"].values())

    assert ids(root) == set()
    assert ids(finance) == {"existing", "parent", "child"}

    conn = sqlite3.connect(finance)
    conn.row_factory = sqlite3.Row
    try:
        child = conn.execute("SELECT * FROM sessions WHERE id='child'").fetchone()
        assert child["parent_session_id"] == "parent"
        assert conn.execute(
            "SELECT prompt FROM system_prompts WHERE hash='prompt-hash'"
        ).fetchone()[0] == "system"
        moved_messages = conn.execute(
            "SELECT id, session_id, content FROM messages "
            "WHERE session_id IN ('parent','child') ORDER BY id"
        ).fetchall()
        assert {row["content"] for row in moved_messages} == {
            "parent message",
            "child message",
        }
        assert all(row["id"] != 1 for row in moved_messages)
        assert conn.execute(
            "SELECT api_call_count FROM session_model_usage "
            "WHERE session_id='child'"
        ).fetchone()[0] == 3
        assert conn.execute(
            "SELECT holder FROM compression_locks WHERE session_id='child'"
        ).fetchone()[0] == "stale-holder"
        assert conn.execute("SELECT COUNT(*) FROM gateway_routing").fetchone()[0] == 1
        assert conn.execute(
            "SELECT failure_streak FROM gateway_hygiene_state"
        ).fetchone()[0] == 4
    finally:
        conn.close()

    # The repair is idempotent after the copy/delete sequence completes.
    second = build_plan(home)
    assert second.safe
    assert second.moves == []


def test_conflicting_profile_evidence_blocks(homes):
    home, root, finance, root_conn, finance_conn = homes
    seed_session(
        root_conn,
        "conflict",
        session_key="agent:finance:telegram:dm:chat",
        profile_name="career-ops",
    )
    root_conn.commit()

    plan = build_plan(home)
    assert not plan.safe
    assert any("conflicting profile evidence" in item for item in plan.blocked)


def test_lineage_with_two_profile_claims_blocks(tmp_path: Path):
    home = tmp_path / ".hermes"
    root_conn = make_store(home / "state.db")
    finance_conn = make_store(home / "profiles" / "finance" / "state.db")
    career_conn = make_store(home / "profiles" / "career" / "state.db")
    try:
        seed_session(
            root_conn,
            "parent",
            session_key="agent:finance:telegram:dm:one",
            profile_name="finance",
        )
        seed_session(
            root_conn,
            "child",
            session_key="agent:career:telegram:dm:two",
            profile_name="career",
            parent_session_id="parent",
        )
        root_conn.commit()
        plan = build_plan(home)
        assert not plan.safe
        assert any("lineage component" in item for item in plan.blocked)
    finally:
        root_conn.close()
        finance_conn.close()
        career_conn.close()


def test_missing_destination_store_blocks(tmp_path: Path):
    home = tmp_path / ".hermes"
    root_conn = make_store(home / "state.db")
    try:
        seed_session(
            root_conn,
            "misplaced",
            session_key="agent:finance:telegram:dm:chat",
            profile_name="finance",
        )
        root_conn.commit()
        plan = build_plan(home)
        assert not plan.safe
        assert any("does not exist" in item for item in plan.blocked)
    finally:
        root_conn.close()


def test_conflicting_destination_duplicate_blocks_before_write(homes):
    home, root, finance, root_conn, finance_conn = homes
    seed_session(
        root_conn,
        "dup",
        session_key="agent:finance:telegram:dm:chat",
        profile_name="finance",
    )
    seed_session(
        finance_conn,
        "dup",
        session_key="agent:finance:telegram:dm:other",
        profile_name="finance",
    )
    root_conn.commit()
    finance_conn.commit()
    root_conn.close()
    finance_conn.close()

    plan = build_plan(home)
    assert plan.safe, plan.blocked
    with pytest.raises(RepairBlocked, match="conflicting session id"):
        _preflight_move(plan.moves[0])
    assert ids(root) == {"dup"}
    assert ids(finance) == {"dup"}


def test_identical_destination_duplicate_finishes_interrupted_copy(homes, tmp_path):
    home, root, finance, root_conn, finance_conn = homes
    kwargs = dict(
        session_key="agent:finance:telegram:dm:chat",
        profile_name="finance",
        origin_profile="finance",
    )
    seed_session(root_conn, "dup", **kwargs)
    seed_session(finance_conn, "dup", **kwargs)
    root_conn.execute(
        "INSERT INTO messages(id, session_id, role, content, timestamp) "
        "VALUES (1, 'dup', 'user', 'same', 1.0)"
    )
    finance_conn.execute(
        "INSERT INTO messages(id, session_id, role, content, timestamp) "
        "VALUES (9, 'dup', 'user', 'same', 1.0)"
    )
    root_conn.commit()
    finance_conn.commit()
    root_conn.close()
    finance_conn.close()

    plan = build_plan(home)
    assert plan.safe
    execute_plan(plan, apply=True, backup_dir=tmp_path / "backups")
    assert ids(root) == set()
    assert ids(finance) == {"dup"}
    conn = sqlite3.connect(finance)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='dup'"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_discovers_origin_session_id_without_declared_foreign_key(homes, tmp_path):
    """Durable async-delegation wake targets move with their session.

    ``async_delegations.origin_session_id`` was added without a SQLite FK, so
    name-based discovery is required; otherwise a repaired profile loses the
    recovered completion record even though the transcript moved.
    """

    home, root, finance, root_conn, finance_conn = homes
    for conn in (root_conn, finance_conn):
        conn.execute(
            """CREATE TABLE async_delegations (
                   delegation_id TEXT PRIMARY KEY,
                   parent_session_id TEXT,
                   origin_session_id TEXT NOT NULL DEFAULT '',
                   state TEXT NOT NULL
               )"""
        )
    seed_session(
        root_conn,
        "moving",
        session_key="agent:finance:telegram:dm:chat",
        profile_name="finance",
        origin_profile="finance",
    )
    root_conn.execute(
        "INSERT INTO async_delegations VALUES "
        "('delegation-1', NULL, 'moving', 'completed')"
    )
    root_conn.commit()
    finance_conn.commit()
    root_conn.close()
    finance_conn.close()

    plan = build_plan(home)
    assert plan.safe, plan.blocked
    execute_plan(plan, apply=True, backup_dir=tmp_path / "backups-origin")

    source = sqlite3.connect(root)
    destination = sqlite3.connect(finance)
    try:
        assert source.execute("SELECT COUNT(*) FROM async_delegations").fetchone()[0] == 0
        assert destination.execute(
            "SELECT origin_session_id, state FROM async_delegations "
            "WHERE delegation_id='delegation-1'"
        ).fetchone() == ("moving", "completed")
    finally:
        source.close()
        destination.close()


def test_plan_fingerprint_is_stable_and_changes_with_plan(homes):
    home, _root, _finance, root_conn, _finance_conn = homes
    seed_session(
        root_conn,
        "first",
        session_key="agent:finance:telegram:dm:first",
        profile_name="finance",
    )
    root_conn.commit()
    first = build_plan(home)
    assert plan_fingerprint(first) == plan_fingerprint(build_plan(home))

    seed_session(
        root_conn,
        "second",
        session_key="agent:finance:telegram:dm:second",
        profile_name="finance",
    )
    root_conn.commit()
    second = build_plan(home)
    assert plan_fingerprint(first) != plan_fingerprint(second)


def test_cli_enforces_reviewed_plan_hash_and_gateway_stop(tmp_path: Path):
    home = tmp_path / ".hermes"
    root = home / "state.db"
    finance = home / "profiles" / "finance" / "state.db"
    root_conn = make_store(root)
    finance_conn = make_store(finance)
    seed_session(
        root_conn,
        "moving",
        session_key="agent:finance:telegram:dm:chat",
        profile_name="finance",
    )
    root_conn.commit()
    root_conn.close()
    finance_conn.close()

    dry_manifest = tmp_path / "dry.json"
    assert main(
        [
            "--hermes-home",
            str(home),
            "--manifest",
            str(dry_manifest),
        ]
    ) == 0
    reviewed = json.loads(dry_manifest.read_text())
    fingerprint = reviewed["plan_sha256"]

    blocked_manifest = tmp_path / "blocked.json"
    assert main(
        [
            "--hermes-home",
            str(home),
            "--apply",
            "--yes",
            "--plan-sha256",
            fingerprint,
            "--manifest",
            str(blocked_manifest),
        ]
    ) == 2
    assert "--apply requires --gateway-stopped" in json.loads(
        blocked_manifest.read_text()
    )["blocked"]
    assert ids(root) == {"moving"}

    applied_manifest = tmp_path / "applied.json"
    assert main(
        [
            "--hermes-home",
            str(home),
            "--apply",
            "--yes",
            "--gateway-stopped",
            "--plan-sha256",
            fingerprint,
            "--backup-dir",
            str(tmp_path / "cli-backups"),
            "--manifest",
            str(applied_manifest),
        ]
    ) == 0
    applied = json.loads(applied_manifest.read_text())
    assert applied["plan_sha256"] == fingerprint
    assert applied["moves"][0]["status"] == "completed"
    assert ids(root) == set()
    assert ids(finance) == {"moving"}


def test_reviewed_plan_hash_blocks_same_move_after_database_bytes_change(tmp_path: Path):
    home = tmp_path / ".hermes"
    root = home / "state.db"
    finance = home / "profiles" / "finance" / "state.db"
    root_conn = make_store(root)
    finance_conn = make_store(finance)
    seed_session(
        root_conn,
        "moving",
        session_key="agent:finance:telegram:dm:chat",
        profile_name="finance",
    )
    root_conn.commit()
    root_conn.close()
    finance_conn.close()

    dry_manifest = tmp_path / "dry-byte-pinned.json"
    assert main(
        ["--hermes-home", str(home), "--manifest", str(dry_manifest)]
    ) == 0
    reviewed = json.loads(dry_manifest.read_text())

    # The move IDs and profile claims are unchanged, but the exact reviewed
    # session bytes are not. Apply must require a new dry-run review.
    conn = sqlite3.connect(root)
    conn.execute("UPDATE sessions SET display_name='Changed after review' WHERE id='moving'")
    conn.commit()
    conn.close()

    blocked_manifest = tmp_path / "blocked-byte-pinned.json"
    assert main(
        [
            "--hermes-home",
            str(home),
            "--apply",
            "--yes",
            "--gateway-stopped",
            "--plan-sha256",
            reviewed["plan_sha256"],
            "--manifest",
            str(blocked_manifest),
        ]
    ) == 2
    blocked = json.loads(blocked_manifest.read_text())
    assert any("does not match the current plan" in item for item in blocked["blocked"])
    assert ids(root) == {"moving"}
    assert ids(finance) == set()


def test_dry_run_surfaces_unkeyed_session_table_as_blocked(homes):
    home, _root, _finance, root_conn, finance_conn = homes
    for conn in (root_conn, finance_conn):
        conn.execute(
            "CREATE TABLE unkeyed_events (session_id TEXT, payload TEXT)"
        )
    seed_session(
        root_conn,
        "moving-unkeyed",
        session_key="agent:finance:telegram:dm:unkeyed",
        profile_name="finance",
    )
    root_conn.execute(
        "INSERT INTO unkeyed_events VALUES ('moving-unkeyed', 'event')"
    )
    root_conn.commit()
    finance_conn.commit()

    plan = build_plan(home)
    assert plan.safe
    manifest = execute_plan(plan, apply=False)
    assert manifest["safe"] is False
    assert any("no primary key" in item for item in manifest["blocked"])
    assert manifest["moves"][0]["status"] == "blocked"
