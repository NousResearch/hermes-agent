"""session.title must tolerate non-string ``title`` on the inline reader path."""

from __future__ import annotations

import contextlib
import threading

from tui_gateway import server


def _session(**extra):
    ready = threading.Event()
    ready.set()
    sess = {
        "agent": object(),
        "agent_ready": ready,
        "session_key": "session-key",
        "pending_title": None,
        "history_lock": threading.Lock(),
    }
    sess.update(extra)
    return sess


def test_session_title_rejects_null_after_coerce(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return None

        def get_session(self, _key):
            return None

        def set_session_title(self, _key, _title):
            raise AssertionError("must not persist empty title")

    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield _FakeDB()

    sid = "title-null"
    server._sessions[sid] = _session()
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    try:
        resp = server.dispatch(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": sid, "title": None},
            }
        )
        assert resp["error"]["code"] == 4021
    finally:
        server._sessions.pop(sid, None)


def test_session_title_coerces_int_title(monkeypatch):
    state = {"title": None}

    class _FakeDB:
        def get_session_title(self, _key):
            return state["title"]

        def get_session(self, _key):
            return {"id": "session-key", "title": state["title"]} if state["title"] else None

        def set_session_title(self, _key, title):
            state["title"] = title
            return True

    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield _FakeDB()

    sid = "title-int"
    server._sessions[sid] = _session()
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    monkeypatch.setattr(server, "_emit_session_info_for_session", lambda *a, **k: None)
    try:
        resp = server.dispatch(
            {
                "id": "2",
                "method": "session.title",
                "params": {"session_id": sid, "title": 42},
            }
        )
        assert "error" not in resp, resp
        assert resp["result"]["title"] == "42"
        assert state["title"] == "42"
    finally:
        server._sessions.pop(sid, None)


def test_session_title_list_title_does_not_crash(monkeypatch):
    """List title must not AttributeError on the inline reader path."""
    state = {"title": None}

    class _FakeDB:
        def get_session_title(self, _key):
            return state["title"]

        def get_session(self, _key):
            return {"id": "session-key", "title": state["title"]} if state["title"] else None

        def set_session_title(self, _key, title):
            state["title"] = title
            return True

    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield _FakeDB()

    sid = "title-list"
    server._sessions[sid] = _session()
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    monkeypatch.setattr(server, "_emit_session_info_for_session", lambda *a, **k: None)
    try:
        resp = server.dispatch(
            {
                "id": "3",
                "method": "session.title",
                "params": {"session_id": sid, "title": ["hello", "world"]},
            }
        )
        # Coerced str(list) is non-empty — accepted; the contract is no crash.
        assert "error" not in resp, resp
        assert isinstance(resp["result"]["title"], str)
        assert state["title"] == "['hello', 'world']"
    finally:
        server._sessions.pop(sid, None)
