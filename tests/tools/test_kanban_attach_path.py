"""kanban_attach(path=...): the bytes never pass through the model.

A model re-emitting a file as base64 transcribes it token by token and
alters long payloads (a 7,408-byte evidence file came back 7,407 bytes with
a character changed inside a digest). Reading the file server-side is the
only byte-exact route; content_base64 stays as the fallback.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """A claimed task with HERMES_HOME isolated and the worker env pinned."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="attach-path", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def _stored(task_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        return kb.list_attachments(conn, task_id)
    finally:
        conn.close()


def _attach(args):
    from tools import kanban_tools as kt

    return json.loads(kt._handle_attach(args))


@pytest.mark.parametrize("payload", [
    pytest.param(b"column     aligned       value\n" * 350, id="spaces"),
    pytest.param(bytes(range(256)) * 32, id="binary"),
    pytest.param(("5440c9aeaad7  — café\n" * 600).encode("utf-8"), id="utf8-hex"),
])
def test_attach_path_is_byte_exact(worker_env, tmp_path, payload):
    src = tmp_path / "evidence.bin"
    src.write_bytes(payload)
    d = _attach({"path": str(src)})
    assert d.get("ok") is True, d
    assert d["size"] == len(payload)
    atts = _stored(worker_env)
    assert [a.filename for a in atts] == ["evidence.bin"]
    assert Path(atts[0].stored_path).read_bytes() == payload


def test_attach_path_honours_explicit_filename(worker_env, tmp_path):
    src = tmp_path / "raw.txt"
    src.write_bytes(b"abc")
    d = _attach({"path": str(src), "filename": "report.txt"})
    assert d.get("ok") is True, d
    assert [a.filename for a in _stored(worker_env)] == ["report.txt"]


@pytest.mark.parametrize("args", [
    pytest.param({}, id="neither"),
    pytest.param({"content_base64": "YWJj"}, id="both"),
])
def test_attach_requires_exactly_one_source(worker_env, tmp_path, args):
    src = tmp_path / "a.txt"
    src.write_bytes(b"abc")
    if args:
        args = dict(args, path=str(src))
    d = _attach(dict(args, filename="a.txt"))
    assert "exactly one of path" in d["error"]
    assert _stored(worker_env) == []


@pytest.mark.parametrize("make", [
    pytest.param(lambda d: "relative/a.txt", id="relative"),
    pytest.param(lambda d: str(d), id="directory"),
    pytest.param(lambda d: str(d / "missing.txt"), id="missing"),
])
def test_attach_path_rejects_unreadable_sources(worker_env, tmp_path, make):
    d = _attach({"path": make(tmp_path)})
    assert "cannot read path" in d["error"]
    assert _stored(worker_env) == []


def test_attach_path_enforces_size_cap_without_storing(worker_env, tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "KANBAN_ATTACHMENT_MAX_BYTES", 16)
    src = tmp_path / "big.bin"
    src.write_bytes(b"x" * 17)
    d = _attach({"path": str(src)})
    assert "limit" in d["error"]
    assert _stored(worker_env) == []


def test_attach_base64_fallback_unchanged(worker_env):
    d = _attach({"content_base64": "YWJj"})
    assert "filename is required" in d["error"]
    d = _attach({"content_base64": "YWJj", "filename": "a.txt"})
    assert d.get("ok") is True, d
    assert Path(_stored(worker_env)[0].stored_path).read_bytes() == b"abc"


def test_attach_schema_offers_path_and_requires_no_inline_bytes():
    from tools import kanban_tools as kt

    params = kt.KANBAN_ATTACH_SCHEMA["parameters"]
    assert "path" in params["properties"]
    assert "content_base64" not in params["required"]
    assert "filename" not in params["required"]
