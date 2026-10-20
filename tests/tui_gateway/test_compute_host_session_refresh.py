"""Host frame refresh preserves workspace provenance and refuses replaced records."""
import io
import threading
from contextlib import contextmanager

import pytest

from tui_gateway import server
from tui_gateway.compute_host import ComputeHost


def test_existing_host_session_refresh_keeps_workspace_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    owner = "basic:alice"
    session = {"session_key": "picked", "history_lock": threading.RLock(),
               "cwd": str(tmp_path), "explicit_cwd": False, "source": "desktop",
               "profile_home": None, "auth_user_id": owner}
    monkeypatch.setitem(server._sessions, "refresh", session)
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        updated = host._ensure_server_session(server, {
            "sid": "refresh", "cwd": str(project), "explicit_cwd": True})
        assert updated is session
        assert server._terminal_task_cwd_with_source(updated) == (str(project), "session")
        assert updated["auth_user_id"] == owner
        assert server._session_transport_contains(updated, host._transport)
        host._ensure_server_session(server, {"sid": "refresh", "cols": 100})
        assert updated["explicit_cwd"] is True and updated["cols"] == 100
        host._ensure_server_session(server, {"sid": "refresh", "explicit_cwd": False})
        assert updated["explicit_cwd"] is False
        assert server._terminal_task_cwd_with_source(updated) == ("", "process")
    finally:
        host.close()


def test_existing_host_session_replaced_before_lock_is_not_refreshed(monkeypatch):
    successor = {"cwd": "successor", "explicit_cwd": False}

    @contextmanager
    def replace_before_lock():
        monkeypatch.setitem(server._sessions, "refresh", successor)
        yield

    original = {"cwd": "original", "explicit_cwd": False,
                "history_lock": replace_before_lock()}
    monkeypatch.setitem(server._sessions, "refresh", original)
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        with pytest.raises(ValueError, match="host runtime replaced"):
            host._ensure_server_session(server, {
                "sid": "refresh", "cwd": "new", "explicit_cwd": True})
        assert original["cwd"] == "original" and original["explicit_cwd"] is False
        assert successor == {"cwd": "successor", "explicit_cwd": False}
        assert "transport" not in original
    finally:
        host.close()
