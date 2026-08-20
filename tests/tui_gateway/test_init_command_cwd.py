"""Tests for /init in tui_gateway — session cwd targeting.

The TUI routes ``/init`` through ``command.dispatch`` (same as ``/learn`` and
``/undo``). Regression coverage for the desktop bug where /init resolved the
project root from the *process* cwd (the desktop launcher's home directory)
instead of the session's real workspace.

The desktop starts a session from a sidebar project via ``session.create``
with a ``cwd`` param; that value lands in ``_sessions[sid]["cwd"]`` and must
be the project root /init writes AGENTS.md into.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    yield mod
    # Restore in place instead of clear+reload: importlib.reload
    # re-registers atexit hooks (duplicate ThreadPoolExecutor shutdowns
    # race the stderr buffer at interpreter exit — same class as PR #34217)
    # and re-captures module-level paths like _hermes_home against this
    # test's soon-deleted tmpdir, breaking later files in the same process.
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._db = None


@pytest.fixture()
def db(hermes_home):
    return SessionDB(db_path=hermes_home / "state.db")


def _call(server, method, **params):
    return server._methods[method](1, params)


def _make_session(server, db, sid, session_key, cwd):
    """Minimal live session row with a pinned workspace cwd."""
    db.create_session(session_key, source="tui")
    history = db.get_messages_as_conversation(session_key)
    agent = MagicMock()
    agent._memory_manager = MagicMock()
    agent._last_flushed_db_idx = len(history)
    s = {
        "session_key": session_key,
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": agent,
        "attached_images": [],
        "cols": 120,
        "cwd": str(cwd),
    }
    server._sessions[sid] = s
    server._db = db
    return sid, session_key, s, agent


def test_init_uses_session_cwd_not_process_cwd(server, db, tmp_path, monkeypatch):
    """The /init prompt must target the session's pinned workspace.

    Regression for the desktop bug: the gateway process launches from the
    user's home, and sessions started from a sidebar project carry the real
    project root in ``_sessions[sid][\"cwd\"]``. Falling back to the process
    cwd wrote AGENTS.md into the home directory instead of the repo.
    """
    project = tmp_path / "project"
    project.mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)
    monkeypatch.chdir(launcher)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    sid, _, _, _ = _make_session(server, db, "sid-init-cwd", "tui-init-cwd", project)

    try:
        resp = _call(server, "command.dispatch", session_id=sid, name="init", arg="")
    finally:
        server._sessions.pop(sid, None)

    result = resp["result"]
    assert result["type"] == "send"
    assert f"project at: {project}" in result["message"]
    assert f"{project}/AGENTS.md" in result["message"]


def test_init_without_session_falls_back_to_runtime_cwd(
    server, db, tmp_path, monkeypatch
):
    """No pinned workspace? /init should fall back to the runtime carrier
    (TERMINAL_CWD), still not the process launch dir."""
    project = tmp_path / "project"
    project.mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)
    monkeypatch.chdir(launcher)
    monkeypatch.setenv("TERMINAL_CWD", str(project))

    try:
        resp = _call(server, "command.dispatch", session_id="nope", name="init", arg="")
    finally:
        server._sessions.pop("nope", None)

    result = resp["result"]
    assert result["type"] == "send"
    assert f"project at: {project}" in result["message"]
