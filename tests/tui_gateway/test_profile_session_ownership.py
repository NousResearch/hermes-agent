from __future__ import annotations

import threading
import types
from pathlib import Path
from typing import Any

import hermes_state
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from tui_gateway import server


def test_init_session_publishes_profile_and_lease_before_worker(
    tmp_path: Path, monkeypatch
) -> None:
    sid = "profile-runtime"
    profile_home = tmp_path / "profile-home"
    lease = object()
    observed: dict[str, object] = {}
    agent = types.SimpleNamespace(model="test/model")

    def _worker(_key, _model, profile_home=None):
        observed.update(server._sessions[sid])
        observed["worker_profile_home"] = profile_home
        return types.SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_SlashWorker", _worker)
    monkeypatch.setattr(server, "_attach_worker", lambda *_args: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_args: None)
    monkeypatch.setattr(
        server, "_start_notification_poller", lambda *_args: threading.Event()
    )
    monkeypatch.setattr(
        server, "_notify_session_boundary", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})

    try:
        server._init_session(
            sid,
            "profile-session",
            agent,
            [],
            cwd=str(tmp_path),
            source="desktop",
            profile_home=profile_home,
            active_session_lease=lease,
        )

        assert observed["profile_home"] == str(profile_home)
        assert observed["active_session_lease"] is lease
        assert observed["worker_profile_home"] == str(profile_home)
    finally:
        server._sessions.pop(sid, None)


def test_same_durable_key_resumes_independently_per_profile(
    tmp_path: Path, monkeypatch
) -> None:
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    db_a = SessionDB(db_path=profile_a / "state.db")
    db_b = SessionDB(db_path=profile_b / "state.db")
    key = "shared-durable-key"
    db_a.create_session(key, source="desktop", model="test/a")
    db_b.create_session(key, source="desktop", model="test/b")
    homes = {"a": profile_a, "b": profile_b}
    transports = {"current": object()}
    created_leases: list[object] = []

    def _db_factory(*, db_path: str | Path | None = None, **_kwargs: Any):
        assert db_path is not None
        path = Path(db_path).resolve()
        if path == (profile_a / "state.db").resolve():
            return db_a
        if path == (profile_b / "state.db").resolve():
            return db_b
        raise AssertionError(f"unexpected profile db: {path}")

    def _claim(*_args, **_kwargs):
        lease = types.SimpleNamespace(enabled=False, released=False)
        created_leases.append(lease)
        return lease, None

    monkeypatch.setattr(hermes_state, "SessionDB", _db_factory)
    monkeypatch.setattr(server, "_profile_home", lambda name: homes.get(name))
    monkeypatch.setattr(server, "_claim_active_session_slot", _claim)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(server, "current_transport", lambda: transports["current"])

    created_sids: list[str] = []
    try:
        result_a = server._methods["session.resume"](
            "resume-a",
            {"session_id": key, "profile": "a", "source": "desktop"},
        )["result"]
        sid_a = result_a["session_id"]
        created_sids.append(sid_a)
        transport_a = server._sessions[sid_a]["transport"]

        transports["current"] = object()
        result_b = server._methods["session.resume"](
            "resume-b",
            {"session_id": key, "profile": "b", "source": "desktop"},
        )["result"]
        sid_b = result_b["session_id"]
        created_sids.append(sid_b)

        assert sid_b != sid_a
        assert server._sessions[sid_a]["profile_home"] == str(profile_a)
        assert server._sessions[sid_b]["profile_home"] == str(profile_b)
        assert server._sessions[sid_b]["transport"] is transports["current"]
        assert server._sessions[sid_b]["transport"] is not transport_a
        assert server._find_live_session_by_key(key, profile_home=profile_a) == (
            sid_a,
            server._sessions[sid_a],
        )
        assert server._find_live_session_by_key(key, profile_home=profile_b) == (
            sid_b,
            server._sessions[sid_b],
        )

        transports["current"] = object()
        reused_a = server._methods["session.resume"](
            "resume-a-again",
            {"session_id": key, "profile": "a", "source": "desktop"},
        )["result"]
        assert reused_a["session_id"] == sid_a
        assert len(created_leases) == 2
    finally:
        for sid in created_sids:
            server._sessions.pop(sid, None)
        db_a.close()
        db_b.close()


def test_child_mirror_routes_same_key_by_profile(tmp_path, monkeypatch):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    key = "shared-child-key"
    sid_a = "child-live-a"
    sid_b = "child-live-b"
    server._sessions[sid_a] = {
        "session_key": key,
        "profile_home": str(profile_a),
        "agent": None,
        "_finalized": False,
    }
    server._sessions[sid_b] = {
        "session_key": key,
        "profile_home": str(profile_b),
        "agent": None,
        "_finalized": False,
    }
    emitted: list[tuple[str, str, dict | None]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload)),
    )
    server._child_mirrors.clear()
    server._active_child_runs.clear()

    try:
        server._mirror_subagent_to_child(
            "subagent.text",
            {"child_session_id": key, "text": "profile-a-token"},
            profile_home=profile_a,
        )
        server._mirror_subagent_to_child(
            "subagent.text",
            {"child_session_id": key, "text": "profile-b-token"},
            profile_home=profile_b,
        )

        deltas = [item for item in emitted if item[0] == "message.delta"]
        assert deltas == [
            ("message.delta", sid_a, {"text": "profile-a-token"}),
            ("message.delta", sid_b, {"text": "profile-b-token"}),
        ]
        assert server._child_run_active(key, profile_a)
        assert server._child_run_active(key, profile_b)
        assert len(server._child_mirrors) == 2

        server._mirror_subagent_to_child(
            "subagent.complete",
            {"child_session_id": key, "summary": "done-a"},
            profile_home=profile_a,
        )
        assert not server._child_run_active(key, profile_a)
        assert server._child_run_active(key, profile_b)
        server._mirror_subagent_to_child(
            "subagent.complete",
            {"child_session_id": key, "summary": "done-b"},
            profile_home=profile_b,
        )
        assert not server._child_run_active(key, profile_b)
        assert server._child_mirrors == {}
    finally:
        server._sessions.pop(sid_a, None)
        server._sessions.pop(sid_b, None)
        server._child_mirrors.clear()
        server._active_child_runs.clear()


def test_profile_live_metadata_uses_profile_db(tmp_path: Path, monkeypatch) -> None:
    launch_home = tmp_path / "launch-home"
    profile_home = tmp_path / "profile-home"
    launch_db = SessionDB(db_path=launch_home / "state.db")
    profile_db = SessionDB(db_path=profile_home / "state.db")
    sid = "profile-live-metadata"
    key = "shared-metadata-key"
    launch_db.create_session(key, source="desktop", model="test/launch")
    profile_db.create_session(key, source="desktop", model="test/profile")
    launch_db.set_session_title(key, "Launch title")
    profile_db.set_session_title(key, "Profile title")
    launch_db.append_message(key, "user", "launch message")
    profile_db.append_message(key, "user", "profile message")
    session = {
        "agent": types.SimpleNamespace(model="test/profile", provider="test"),
        "created_at": 1.0,
        "history": [{"role": "user", "content": "memory fallback"}],
        "history_lock": threading.Lock(),
        "pending_title": None,
        "profile_home": str(profile_home),
        "running": False,
        "session_key": key,
    }
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {"total": 0})
    monkeypatch.setattr(
        server, "_emit_session_info_for_session", lambda *_args, **_kwargs: None
    )

    try:
        assert server._session_live_title(session, key) == "Profile title"

        title = server._methods["session.title"]("title-get", {"session_id": sid})[
            "result"
        ]
        assert title["title"] == "Profile title"

        history = server._methods["session.history"]("history", {"session_id": sid})[
            "result"
        ]
        assert history["messages"] == [{"role": "user", "text": "profile message"}]

        status = server._methods["session.status"]("status", {"session_id": sid})[
            "result"
        ]["output"]
        assert "Title: Profile title" in status
        assert "profile-home" in status
        assert "launch-home" not in status

        renamed = server._methods["session.title"](
            "title-set", {"session_id": sid, "title": "Profile renamed"}
        )["result"]
        assert renamed == {"pending": False, "title": "Profile renamed"}
        assert profile_db.get_session_title(key) == "Profile renamed"
        assert launch_db.get_session_title(key) == "Launch title"
    finally:
        server._sessions.pop(sid, None)
        profile_db.close()
        launch_db.close()


def test_profile_retry_truncate_and_undo_use_profile_db(
    tmp_path: Path, monkeypatch
) -> None:
    launch_home = tmp_path / "launch-home"
    profile_home = tmp_path / "profile-home"
    launch_db = SessionDB(db_path=launch_home / "state.db")
    profile_db = SessionDB(db_path=profile_home / "state.db")
    sid = "profile-history-mutation"
    key = "shared-history-key"
    for db, prefix in ((launch_db, "launch"), (profile_db, "profile")):
        db.create_session(key, source="desktop", model="test/model")
        for turn in range(1, 4):
            db.append_message(key, "user", f"{prefix} question {turn}")
            db.append_message(key, "assistant", f"{prefix} answer {turn}")
    profile_history = profile_db.get_messages_as_conversation(key)
    session = {
        "agent": types.SimpleNamespace(model="test/model", provider="test"),
        "attached_images": [],
        "cols": 100,
        "cwd": str(tmp_path),
        "history": list(profile_history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "profile_home": str(profile_home),
        "running": False,
        "session_key": key,
        "source": "desktop",
    }
    server._sessions[sid] = session

    class _NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server.threading, "Thread", _NoopThread)
    monkeypatch.setattr(server, "current_transport", lambda: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    try:
        submit = server._methods["prompt.submit"](
            "retry",
            {
                "session_id": sid,
                "text": "replacement",
                "truncate_before_user_ordinal": 1,
            },
        )
        assert "error" not in submit, submit
        assert len(profile_db.get_messages_as_conversation(key)) == 2
        assert len(launch_db.get_messages_as_conversation(key)) == 6

        session["running"] = False
        undo = server._methods["command.dispatch"](
            "undo", {"session_id": sid, "name": "undo", "arg": ""}
        )
        assert "error" not in undo, undo
        assert undo["result"]["message"] == "profile question 1"
        assert profile_db.get_messages_as_conversation(key) == []
        launch_history = launch_db.get_messages_as_conversation(key)
        assert len(launch_history) == 6
        assert launch_history[0]["content"] == "launch question 1"
    finally:
        server._sessions.pop(sid, None)
        profile_db.close()
        launch_db.close()


def test_profile_branch_uses_profile_db_home_and_atomic_session_fields(
    tmp_path: Path, monkeypatch
) -> None:
    launch_home = tmp_path / "launch-home"
    profile_home = tmp_path / "profile-home"
    launch_db = SessionDB(db_path=launch_home / "state.db")
    profile_db = SessionDB(db_path=profile_home / "state.db")
    parent_sid = "profile-parent-runtime"
    parent_key = "profile-parent"
    child_key = "profile-child"
    fake_lease = object()
    captured: dict[str, Any] = {}

    profile_db.create_session(
        parent_key,
        source="desktop",
        model="test/model",
        cwd=str(tmp_path),
    )
    profile_db.set_session_title(parent_key, "Parent")
    server._sessions[parent_sid] = {
        "agent": types.SimpleNamespace(model="test/model", session_id=parent_key),
        "cols": 100,
        "cwd": str(tmp_path),
        "history": [{"role": "user", "content": "hello"}],
        "history_lock": threading.Lock(),
        "profile_home": str(profile_home),
        "session_key": parent_key,
        "source": "desktop",
    }

    def _claim(_key, **kwargs):
        captured["claim_profile_home"] = kwargs.get("profile_home")
        return fake_lease, None

    def _make_agent(_sid, _key, **kwargs):
        captured["agent_db"] = kwargs.get("session_db")
        captured["agent_home"] = Path(get_hermes_home())
        return types.SimpleNamespace(model="test/model", session_id=child_key)

    def _init(_sid, _key, _agent, _history, **kwargs):
        captured["init"] = kwargs

    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    monkeypatch.setattr(server, "_new_session_key", lambda: child_key)
    monkeypatch.setattr(server, "_claim_active_session_slot", _claim)
    monkeypatch.setattr(server, "_make_agent", _make_agent)
    monkeypatch.setattr(server, "_init_session", _init)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(server, "_set_session_context", lambda *_args: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_args: None)

    owned_db = None
    try:
        response = server.handle_request({
            "id": "branch-profile",
            "method": "session.branch",
            "params": {"session_id": parent_sid},
        })

        assert response is not None
        assert "error" not in response, response
        assert profile_db.get_session(child_key) is not None
        assert launch_db.get_session(child_key) is None
        assert captured["claim_profile_home"] == str(profile_home)
        assert captured["agent_home"] == profile_home
        owned_db = captured["agent_db"]
        assert owned_db is not launch_db
        assert owned_db.get_session(child_key) is not None
        init_kwargs = captured["init"]
        assert init_kwargs["session_db"] is owned_db
        assert init_kwargs["profile_home"] == str(profile_home)
        assert init_kwargs["active_session_lease"] is fake_lease
    finally:
        server._sessions.pop(parent_sid, None)
        if owned_db is not None:
            owned_db.close()
        profile_db.close()
        launch_db.close()
