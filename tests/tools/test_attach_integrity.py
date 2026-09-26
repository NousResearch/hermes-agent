"""Attachment corruption regression: fabricated inline attachments.

A worker attached a 16,986 B report to its task via ``kanban_attach`` but
passed a hallucinated placeholder as ``content_base64`` — the string ``L2FuZHJvaWQvLi4v``, which validly decodes to
the 12 bytes ``/android/../``. The handler decoded it, stored the 12-byte dud,
recorded ``size=12`` with a NULL ``content_type``, and returned ``ok:true``:
silent garbage. The card read complete while its only board-side artifact was
a lie.

Fix contract proven here:

  1. ``kanban_attach`` refuses payloads that decode to filesystem-path-looking
     fragments (the incident family) and errors loudly instead of storing
     under ``ok:true``; legitimate small attaches still succeed.
  2. ``content_type`` is derived from the filename when the caller omits it
     (parity with the CLI surface, ``hermes_cli/kanban.py:_cmd_attach``), and
     the stored row + tool result carry the payload's sha256 so callers can
     verify byte-identity after the call.
  3. ``kanban_attach_file`` exists: the worker passes an absolute source path
     and Hermes reads the bytes off disk — the model never touches the bytes
     (same trust level as kernel-side ``kanban_complete(artifacts=...)``).
  4. ``store_attachment_bytes`` (the single write path shared by agent tool,
     CLI, and dashboard) verifies what it wrote — size re-read from disk and
     sha256 against an expected digest — and fails loudly on mismatch;
     silent garbage at the storage layer is impossible.
  5. Legacy boards get the ``sha256`` column via the additive migration.

The incident invocation — ``content_base64='L2FuZHJvaWQvLi4v'``,
``filename='audit-usermd-2026-09-05.md'``, no ``content_type`` — must be a
clean rejection on the new code. On the pre-fix code every test in sections
1/2/3/4 fails (payload stored under ok:true, NULL content_type, no
kanban_attach_file tool, no guard, no column).
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B64_INCIDENT = "L2FuZHJvaWQvLi4v"  # decodes to b'/android/../' (12 bytes)


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Simulate being a worker: HERMES_HOME isolated, HERMES_KANBAN_TASK set
    after we've created the task. Mirrors the fixture in
    test_kanban_tools.py (kept local so this regression file is
    self-contained)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialized kanban DB. Mirrors the
    fixture in tests/plugins/test_kanban_attachments.py."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _row_by_id(conn, attachment_id: int):
    return conn.execute(
        "SELECT id, size, content_type, sha256, stored_path, filename "
        "FROM task_attachments WHERE id = ?",
        (attachment_id,),
    ).fetchone()


def _all_rows(conn, tid: str):
    return conn.execute(
        "SELECT id, size, content_type, sha256 FROM task_attachments "
        "WHERE task_id = ?",
        (tid,),
    ).fetchall()


def _attachments_dir(tid: str) -> Path:
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    return kb.task_attachments_dir(tid)


def _assert_nothing_stored(tid: str) -> None:
    """No task_attachments row and no blob under the attachments dir."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    conn = connect()
    try:
        assert _all_rows(conn, tid) == []
    finally:
        conn.close()
    att_dir = _attachments_dir(tid)
    blobs = list(att_dir.iterdir()) if att_dir.exists() else []
    assert blobs == [], f"orphan blobs left behind: {blobs}"


# ---------------------------------------------------------------------------
# 1. Handler guard: the incident payload family is rejected loudly
# ---------------------------------------------------------------------------

def test_attach_rejects_incident_payload(worker_env):
    """The exact incident invocation shape must be a clean rejection.

    Old behaviour: {ok:true, size:12} + 12-byte '/android/../' blob + row
    with size=12 and NULL content_type.
    """
    from tools import kanban_tools as kt

    tid = worker_env
    out = kt._handle_attach({
        "task_id": tid,
        "filename": "audit-usermd-2026-09-05.md",
        "content_base64": B64_INCIDENT,
    })
    d = json.loads(out)
    assert "error" in d, out
    assert "integrity" in d["error"].lower(), out
    assert "12 bytes" in d["error"], out
    _assert_nothing_stored(tid)


@pytest.mark.parametrize(
    "payload",
    [
        b"/android/../",           # the incident bytes
        b"/Users/worker/report.md",  # an absolute path stored as content
        b"~/notes.md",             # tilde path fragment
        b"../escape",              # relative traversal fragment
    ],
)
def test_attach_rejects_path_looking_payloads(worker_env, payload):
    """Any payload that decodes to a filesystem-path fragment is refused —
    the model emitted a placeholder instead of encoded bytes."""
    from tools import kanban_tools as kt

    out = kt._handle_attach({
        "task_id": worker_env,
        "filename": "artifact.md",
        "content_base64": base64.b64encode(payload).decode(),
    })
    d = json.loads(out)
    assert "error" in d, out
    assert "integrity" in d["error"].lower(), out
    _assert_nothing_stored(worker_env)


def test_attach_legit_small_text_file_still_allowed(worker_env):
    """The guard must not nuke legitimate small attaches: a genuinely
    encoded 38-byte text file sails through (size re-echoed)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from tools import kanban_tools as kt

    source = b"x" * 38
    out = kt._handle_attach({
        "task_id": worker_env,
        "filename": "notes.txt",
        "content_base64": base64.b64encode(source).decode(),
    })
    d = json.loads(out)
    assert d.get("ok") is True, out
    assert d["size"] == 38
    conn = connect()
    try:
        rows = _all_rows(conn, worker_env)
        assert len(rows) == 1
        assert rows[0]["size"] == 38
    finally:
        conn.close()
    stored = _attachments_dir(worker_env) / "notes.txt"
    assert stored.read_bytes() == source


# ---------------------------------------------------------------------------
# 2. content_type derivation + sha256 recording/echo
# ---------------------------------------------------------------------------

def _attach_real(worker_env, filename: str, source: bytes, **extra) -> dict:
    from tools import kanban_tools as kt

    args = {
        "task_id": worker_env,
        "filename": filename,
        "content_base64": base64.b64encode(source).decode(),
    }
    args.update(extra)
    return json.loads(kt._handle_attach(args))


def test_attach_derives_content_type_markdown(worker_env):
    """content_type omitted → derived from the filename (acceptance
    example: 'text/markdown'). Old behaviour stored NULL."""
    source = b"# audit\n\n" + b"word " * 20
    d = _attach_real(worker_env, "audit-usermd-2026-09-05.md", source)
    assert d.get("ok") is True, d

    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    conn = connect()
    try:
        row = _row_by_id(conn, d["attachment_id"])
        assert row["content_type"] == "text/markdown"
        assert row["sha256"] == hashlib.sha256(source).hexdigest()
    finally:
        conn.close()


def test_attach_derives_content_type_json_and_binary(worker_env):
    """.json → application/json; unknown binary ext → the conventional
    non-empty application/octet-stream (never NULL)."""
    d1 = _attach_real(worker_env, "results.json", b'{"ok": true}')
    d2 = _attach_real(worker_env, "blob.dat", bytes(range(256)))
    assert d1.get("ok") is True, d1
    assert d2.get("ok") is True, d2

    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    conn = connect()
    try:
        r1 = _row_by_id(conn, d1["attachment_id"])
        r2 = _row_by_id(conn, d2["attachment_id"])
        assert r1["content_type"] == "application/json"
        assert r2["content_type"] == "application/octet-stream"
    finally:
        conn.close()


def test_attach_result_echoes_sha256(worker_env):
    """The tool result carries the stored sha256 so the worker can verify
    byte-identity in the same breath as the size echo."""
    source = b"payload for sha echo\n"
    d = _attach_real(worker_env, "echo.bin", source)
    assert d.get("ok") is True, d
    assert d["sha256"] == hashlib.sha256(source).hexdigest()
    assert d["content_type"] == "application/octet-stream"


def test_attach_explicit_content_type_wins(worker_env):
    """An explicit content_type overrides the derived one."""
    d = _attach_real(
        worker_env, "weird.bin", b"data", content_type="application/x-weird"
    )
    assert d.get("ok") is True, d
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    conn = connect()
    try:
        row = _row_by_id(conn, d["attachment_id"])
        assert row["content_type"] == "application/x-weird"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. kanban_attach_file — path-based attach (bytes never pass the model)
# ---------------------------------------------------------------------------

def _make_source(tid: str, size: int = 16_986) -> Path:
    """A worktree-shaped source path: .worktrees/<task>/ + spaces + unicode."""
    src_dir = _attachments_dir(tid).parent / f".worktrees/{tid}"/ "sub dir"
    src_dir.mkdir(parents=True, exist_ok=True)
    source = bytes((i * 7 + 13) % 256 for i in range(size))
    p = src_dir / "audit-réport 2026-09-05.md"
    p.write_bytes(source)
    return p


def test_attach_file_happy_path_worktree_shape(worker_env):
    """The failing invocation's shape — a git-worktree path to a 16,986 B
    markdown file — attaches byte-exact with derived content_type."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from tools import kanban_tools as kt

    src = _make_source(worker_env)
    source = src.read_bytes()
    assert len(source) == 16_986

    out = kt._handle_attach_file({"task_id": worker_env, "path": str(src)})
    d = json.loads(out)
    assert d.get("ok") is True, out
    assert d["size"] == 16_986
    assert d["sha256"] == hashlib.sha256(source).hexdigest()
    assert d["content_type"] == "text/markdown"

    conn = connect()
    try:
        row = _row_by_id(conn, d["attachment_id"])
        assert row["size"] == 16_986
        assert row["sha256"] == hashlib.sha256(source).hexdigest()
        stored = Path(row["stored_path"])
        assert stored.read_bytes() == source  # byte-equal
    finally:
        conn.close()


def test_attach_file_missing_path_clean_error(worker_env):
    from tools import kanban_tools as kt

    out = kt._handle_attach_file({
        "task_id": worker_env,
        "path": "/nonexistent/path/report.md",
    })
    d = json.loads(out)
    assert "error" in d, out
    _assert_nothing_stored(worker_env)


def test_attach_file_directory_rejected(worker_env):
    from tools import kanban_tools as kt

    out = kt._handle_attach_file({"task_id": worker_env, "path": str(Path.home())})
    d = json.loads(out)
    assert "error" in d, out
    _assert_nothing_stored(worker_env)


def test_attach_file_defaults_filename_to_source_leaf(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from tools import kanban_tools as kt

    src = _make_source(worker_env, size=64)
    out = kt._handle_attach_file({"task_id": worker_env, "path": str(src)})
    d = json.loads(out)
    assert d.get("ok") is True, out
    conn = connect()
    try:
        row = _row_by_id(conn, d["attachment_id"])
        assert row["filename"] == src.name  # unicode leaf preserved
    finally:
        conn.close()


def test_attach_file_explicit_content_type_override(worker_env):
    from tools import kanban_tools as kt

    src = _make_source(worker_env, size=32)
    out = kt._handle_attach_file({
        "task_id": worker_env,
        "path": str(src),
        "content_type": "application/x-custom",
    })
    d = json.loads(out)
    assert d.get("ok") is True, out
    assert d["content_type"] == "application/x-custom"


# ---------------------------------------------------------------------------
# 4. store_attachment_bytes integrity guard (the single shared write path)
# ---------------------------------------------------------------------------

def test_store_guard_wrong_expected_sha_fails_loudly(kanban_home, tmp_path):
    """expected_sha256 that does not match the payload →
    AttachmentIntegrityError, no row, no blob."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    conn = connect()
    try:
        tid = kb.create_task(conn, title="guard-test", assignee="t")
        with pytest.raises(kb.AttachmentIntegrityError):
            kb.store_attachment_bytes(
                conn, tid, "f.txt", b"real bytes",
                expected_sha256="0" * 64,
            )
        assert _all_rows(conn, tid) == []
    finally:
        conn.close()
    att_dir = kb.task_attachments_dir(tid)
    assert not (att_dir / "f.txt").exists()


def test_store_guard_detects_post_write_disk_corruption(kanban_home):
    """The guard re-reads what landed on disk: a poisoned write (disk bytes
    differ from the buffer) is caught before the row is recorded."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    real_write = Path.write_bytes

    def poisoned(self, data):
        real_write(self, b"corrupted!!")

    conn = connect()
    try:
        tid = kb.create_task(conn, title="guard-disk", assignee="t")
        import unittest.mock as mock

        with mock.patch.object(Path, "write_bytes", poisoned):
            with pytest.raises(kb.AttachmentIntegrityError):
                kb.store_attachment_bytes(conn, tid, "f.txt", b"genuine bytes")
        assert _all_rows(conn, tid) == []
        # The corrupted blob itself was reaped with the row.
        att_dir = kb.task_attachments_dir(tid)
        assert not (att_dir / "f.txt").exists()
    finally:
        conn.close()


def test_store_guard_expected_size_mismatch_fails(kanban_home):
    """expected_size must match the payload exactly."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    conn = connect()
    try:
        tid = kb.create_task(conn, title="guard-size", assignee="t")
        with pytest.raises(kb.AttachmentIntegrityError):
            kb.store_attachment_bytes(
                conn, tid, "f.txt", b"100 bytes?",
                expected_size=999,
            )
        assert _all_rows(conn, tid) == []
    finally:
        conn.close()


def test_store_guard_verify_false_is_legacy_escape(kanban_home):
    """verify=False documents the legacy behaviour (no self-check) for
    callers that must bypass — not used by any Hermes surface by default."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    conn = connect()
    try:
        tid = kb.create_task(conn, title="guard-off", assignee="t")
        att_id = kb.store_attachment_bytes(
            conn, tid, "f.txt", b"bytes",
            expected_sha256="0" * 64,
            verify=False,
        )
        row = _row_by_id(conn, att_id)
        assert row["size"] == len(b"bytes")
    finally:
        conn.close()


def test_store_guard_records_sha256_and_rechecks(kanban_home):
    """With expected_sha256 given, the row records that digest and the
    post-write re-read must agree with it."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    data = b"sha-recorded bytes"
    digest = hashlib.sha256(data).hexdigest()
    conn = connect()
    try:
        tid = kb.create_task(conn, title="guard-sha", assignee="t")
        att_id = kb.store_attachment_bytes(
            conn, tid, "f.txt", data, expected_sha256=digest
        )
        row = _row_by_id(conn, att_id)
        assert row["sha256"] == digest
        assert Path(row["stored_path"]).read_bytes() == data
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 5. Legacy-board migration: task_attachments gains sha256
# ---------------------------------------------------------------------------

def test_migration_adds_sha256_to_legacy_attachments(tmp_path, monkeypatch):
    """A board created before the column exists must migrate additively:
    create the old-shape table first, then init_db — SCHEMA_SQL's
    IF NOT EXISTS keeps the legacy table, and the migration adds sha256."""
    import sqlite3

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    # Default board's DB lives at <HERMES_HOME>/kanban.db (back-compat path;
    # see kanban_db.kanban_db_path). Simulate a legacy board: everything is
    # modern EXCEPT task_attachments, which predates the sha256 column.
    kb.init_db()
    db_path = home / "kanban.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE task_attachments")
        conn.execute(
            """
            CREATE TABLE task_attachments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id      TEXT NOT NULL,
                filename     TEXT NOT NULL,
                stored_path  TEXT NOT NULL,
                content_type TEXT,
                size         INTEGER NOT NULL DEFAULT 0,
                uploaded_by  TEXT,
                created_at   INTEGER NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(task_attachments)")}
    finally:
        conn.close()
    assert "sha256" in cols
