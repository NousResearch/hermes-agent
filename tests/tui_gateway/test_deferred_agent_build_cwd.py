"""Regression coverage for deferred TUI/Desktop session context."""

from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from types import SimpleNamespace

from tui_gateway import server


def test_deferred_agent_build_threads_session_cwd(monkeypatch, tmp_path):
    """A cold Desktop build must not fall back to the serve process cwd."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured = {}
    built = threading.Event()
    ready = threading.Event()
    sid = f"cwd-build-{uuid.uuid4().hex[:8]}"
    session = {
        "agent_ready": ready,
        "session_key": f"cwd-key-{uuid.uuid4().hex[:8]}",
        "cwd": str(workspace),
        "pending_title": "User-selected title",
    }

    class TitleDB:
        title = None

        @staticmethod
        def sanitize_title(title):
            from hermes_state import SessionDB
            return SessionDB.sanitize_title(title)

        @staticmethod
        def create_session(key, **_kwargs):
            return key

        def set_session_title(self, _key, title):
            self.title = self.sanitize_title(title)
            return True

        def get_session_title(self, _key):
            return self.title

    title_db = TitleDB()

    def fake_set_session_context(key, cwd=None):
        captured["context_key"] = key
        captured["context_cwd"] = cwd
        return []

    def fake_make_agent(*args, **kwargs):
        captured["agent_cwd"] = kwargs.get("cwd_override")
        captured["session_title_hint"] = kwargs.get("session_title_hint")
        captured["session_title_source"] = kwargs.get("session_title_source")
        built.set()
        return SimpleNamespace(model="test", session_id=session["session_key"])

    monkeypatch.setattr(server, "_set_session_context", fake_set_session_context)
    monkeypatch.setattr(server, "_get_db", lambda: title_db)
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(
        "tui_gateway.entry.ensure_mcp_discovery_started", lambda: None
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert built.wait(timeout=15), "agent build thread never called _make_agent"
        assert ready.wait(timeout=5), "agent_ready never set after build"
    finally:
        server._sessions.pop(sid, None)
        from tools.approval import unregister_gateway_notify

        unregister_gateway_notify(session["session_key"])

    assert captured["context_cwd"] == str(workspace)
    assert captured["agent_cwd"] == str(workspace)
    assert captured["session_title_hint"] == "User-selected title"
    assert captured["session_title_source"] == "user"


def test_deferred_build_drops_overlong_pending_title(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session = {
        "session_key": "new-session",
        "pending_title": "x" * (SessionDB.MAX_TITLE_LENGTH + 1),
    }
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")

    kwargs = server._deferred_build_agent_kwargs(session, db)

    assert kwargs["session_title_hint"] is None
    assert kwargs["session_title_source"] is None
    assert session["pending_title"] is None
    db.close()


def test_two_deferred_drafts_atomically_claim_duplicate_title(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    sessions = [
        {"session_key": "draft-a", "pending_title": "Shared title"},
        {"session_key": "draft-b", "pending_title": "Shared title"},
    ]
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    barrier = threading.Barrier(2)

    def build(session):
        barrier.wait()
        return server._deferred_build_agent_kwargs(session, db)

    with ThreadPoolExecutor(max_workers=2) as pool:
        kwargs = list(pool.map(build, sessions))

    assert [item["session_title_hint"] for item in kwargs].count("Shared title") == 1
    assert [item["session_title_hint"] for item in kwargs].count(None) == 1
    assert all(session["pending_title"] is None for session in sessions)
    db.close()


def test_deferred_build_keeps_pending_title_after_transient_store_failure(monkeypatch):
    def fail_write(*_args):
        raise OSError("locked")

    session = {"session_key": "new-session", "pending_title": "Retry me"}
    db = SimpleNamespace(set_session_title=fail_write)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)

    kwargs = server._deferred_build_agent_kwargs(session, db)

    assert kwargs["session_title_hint"] is None
    assert session["pending_title"] == "Retry me"


def test_pending_title_draft_skips_background_prebuild(monkeypatch):
    started = []

    class ImmediateTimer:
        daemon = True

        def __init__(self, _delay, callback):
            self.callback = callback

        def start(self):
            self.callback()

    session = {"session_key": "draft", "pending_title": "Still drafting"}
    server._sessions["sid"] = session
    monkeypatch.setattr(server.threading, "Timer", ImmediateTimer)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: started.append(args))
    try:
        server._schedule_agent_build("sid")
    finally:
        server._sessions.pop("sid", None)

    assert started == []


def test_live_tui_title_rename_refreshes_agent_title_context(monkeypatch):
    class TitleDB:
        title = "Old title"

        def get_session_title(self, _key):
            return self.title

        def set_session_title(self, _key, title):
            self.title = title
            return True

    db = TitleDB()
    agent = SimpleNamespace(_session_title_hint="Old title", _session_title_source="llm")
    session = {"session_key": "session-key", "pending_title": None, "agent": agent}
    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    try:
        response = server.handle_request({
            "id": "1",
            "method": "session.title",
            "params": {"session_id": "sid", "title": "New title"},
        })
    finally:
        server._sessions.pop("sid", None)

    assert response["result"]["title"] == "New title"
    from tools.bot_mode_dm import _session_title
    assert _session_title(agent) == "New title"
