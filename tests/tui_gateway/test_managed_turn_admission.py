"""Managed TUI turns do not queue or silently replay a lost admission response."""
from contextlib import contextmanager
import threading
import sqlite3

import pytest

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.managed_turn_admission import preflight


def _rpc(method: str, rid: str, params: dict) -> dict:
    return server.handle_request({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})


@contextmanager
def _db_context(db):
    yield db


def test_managed_lookup_is_read_only_and_scoped_to_own_session(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("owned", source="desktop")
        db.create_session("other", source="desktop")
        key = "a" * 32
        row_id = db.append_message("owned", "user", content="read a public report", managed_turn_key=key)
        session = {"session_key": "owned", "profile_home": str(tmp_path), "history_lock": threading.Lock()}
        monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (
            {**session, "session_key": params["session_id"]}, None))
        monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))
        assert _rpc("prompt.managed_turn.get", "request-1", {"session_id": "owned", "managed_turn_key": key})["result"] == {
            "found": True, "user_row_id": row_id}
        assert _rpc("prompt.managed_turn.get", "request-2", {"session_id": "other", "managed_turn_key": key})["result"] == {
            "found": False}
        assert _rpc("prompt.managed_turn.get", "request-3", {"session_id": "owned", "managed_turn_key": "b" * 32})["result"] == {
            "found": False}
        assert _rpc("prompt.managed_turn.get", "bad-params", {"session_id": "owned", "managed_turn_key": key,
                                                                  "extra": "not permitted"})["error"]["code"] == 4000
        monkeypatch.setattr(server, "_profile_home", lambda _name: tmp_path / "foreign-profile")
        foreign = _rpc("prompt.managed_turn.get", "foreign", {
            "session_id": "owned", "managed_turn_key": key, "profile": "foreign"})
        assert foreign["error"]["code"] == 4031, foreign
        assert len(db.get_messages("owned")) == 1
        db.close()
        db = SessionDB(tmp_path / "state.db")
        assert _rpc("prompt.managed_turn.get", "request-4", {"session_id": "owned", "managed_turn_key": key})["result"] == {
            "found": True, "user_row_id": row_id}
    finally:
        db.close()


def test_managed_submit_persists_receipt_before_worker_start_and_never_reposts(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    starts = []
    session = {"session_key": "owned", "profile_home": str(tmp_path),
               "history_lock": threading.Lock(), "running": False, "agent": None}
    try:
        db.create_session("owned", source="desktop")
        monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))
        monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))
        monkeypatch.setattr(server, "_legacy_group_fence_error", lambda *args: None)
        monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: None)
        monkeypatch.setattr(server, "_reattach_refusal", lambda *args: None)
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda *args: True)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda *args: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
        def claim(*args):
            session["running"] = True
            return None, {}
        monkeypatch.setattr(server, "_lock_in_submit_turn", claim)
        class NoWorker:
            def __init__(self, target, daemon):
                self.target = target
            def start(self):
                starts.append(db.get_managed_turn("owned", "a" * 32))
        monkeypatch.setattr(server.threading, "Thread", NoWorker)
        request = {"session_id": "owned", "text": "report", "managed_turn_key": "a" * 32}
        response = _rpc("prompt.submit", "req1", request)
        assert response["result"]["status"] == "streaming", response
        row_id = response["result"]["user_row_id"]
        assert starts == [{"user_row_id": row_id}]
        session["running"] = False  # the response was lost; a duplicate cannot start a second worker
        duplicate = _rpc("prompt.submit", "req2", request)
        assert duplicate["error"]["code"] == 4090, duplicate
        conflict = _rpc("prompt.submit", "req3", {**request, "text": "different task"})
        assert conflict["error"]["code"] == 4090, conflict
        assert len(starts) == 1
        assert len(db.get_messages("owned")) == 1
    finally:
        db.close()


def test_managed_submit_refuses_uncommitted_receipt_without_starting_worker(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    session = {"session_key": "owned", "profile_home": str(tmp_path),
               "history_lock": threading.Lock(), "running": False, "agent": None}
    started = []
    try:
        db.create_session("owned", source="desktop")
        raw = sqlite3.connect(tmp_path / "state.db")
        try:
            raw.execute("""CREATE TRIGGER deny_managed BEFORE INSERT ON managed_turn_submissions
                           BEGIN SELECT RAISE(ABORT, 'fixture disk failure'); END""")
            raw.commit()
        finally:
            raw.close()
        monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))
        monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))
        monkeypatch.setattr(server, "_legacy_group_fence_error", lambda *args: None)
        monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: None)
        monkeypatch.setattr(server, "_reattach_refusal", lambda *args: None)
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda *args: True)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda *args: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda *args: started.append("build"))
        def claim(*args):
            session["running"] = True
            return None, {}
        monkeypatch.setattr(server, "_lock_in_submit_turn", claim)
        response = _rpc("prompt.submit", "r1", {
            "session_id": "owned", "text": "do not launch", "managed_turn_key": "a" * 32})
        assert response["error"]["code"] == 5031, response
        assert started == []
        assert session["running"] is False
        assert db.get_messages("owned") == []
        assert db.get_managed_turn("owned", "a" * 32) is None
    finally:
        db.close()


def test_managed_claim_rechecks_busy_under_admission_lock(monkeypatch):
    session = {"session_key": "owned", "history_lock": threading.Lock(), "running": True,
               "agent": None, "history": [], "attached_images": []}
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *args, **kw: (_ for _ in ()).throw(
        AssertionError("must not start a second turn")))
    error, _ = server._lock_in_submit_turn("r1", "sid", session, "second",
                                            {"managed_turn_key": "a" * 32}, False, None, None, None)
    assert error["error"]["code"] == 4009
    assert session["running"] is True


def test_idle_probe_losing_race_reenters_normal_busy_policy(monkeypatch):
    session = {"session_key": "owned", "history_lock": threading.Lock(), "running": False,
               "agent": None, "history": [], "attached_images": []}
    original_probe = server._resolve_busy_prompt
    probes = []
    def race_probe(*args, **kwargs):
        probes.append(True)
        result = original_probe(*args, **kwargs)
        if len(probes) == 1:
            session["running"] = True
        return result
    monkeypatch.setattr(server, "_resolve_busy_prompt", race_probe)
    monkeypatch.setattr(server, "_handle_busy_submit", lambda *a, **k: {
        "jsonrpc": "2.0", "id": "r1", "result": {"status": "queued"}})
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not start a second turn")))
    response, _ = server._claim_idle_submit("r1", "sid", session, "second", {},
        internal_hosted_submit=False, managed_key=None, has_truncation=False, transport=None,
        turn_author=None, display_kind=None, requested_rebind_ids=None, hosted_task=None)
    assert response["result"]["status"] == "queued"
    assert len(probes) == 2


@pytest.mark.parametrize("mixed", [False, True])
def test_two_claims_after_idle_probe_start_only_one_worker(monkeypatch, mixed):
    session = {"session_key": "owned", "history_lock": threading.Lock(), "running": False,
               "agent": None, "history": [], "attached_images": []}
    original_probe = server._resolve_busy_prompt
    barrier = threading.Barrier(2)
    seen = threading.local()
    starts = []
    def simultaneous_probe(*args, **kwargs):
        result = original_probe(*args, **kwargs)
        if result is None and not getattr(seen, "first_probe", False):
            seen.first_probe = True
            barrier.wait(timeout=5)
        return result
    monkeypatch.setattr(server, "_resolve_busy_prompt", simultaneous_probe)
    monkeypatch.setattr(server, "_handle_busy_submit", lambda *a, **k: {
        "jsonrpc": "2.0", "id": "r", "result": {"status": "queued"}})
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *a, **k: starts.append(a[1]))
    responses = []
    def submit(key):
        try:
            params = {"managed_turn_key": key} if key is not None else {}
            responses.append(server._claim_idle_submit("r", "owned", session, "second",
                params, internal_hosted_submit=False, managed_key=key,
                has_truncation=False, transport=None, turn_author=None, display_kind=None,
                requested_rebind_ids=None, hosted_task=None)[0])
        except Exception as exc:
            responses.append(exc)
    keys = ("a" * 32, None if mixed else "b" * 32)
    threads = [threading.Thread(target=submit, args=(key,)) for key in keys]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert len(starts) == 1
    assert sum(response is None for response in responses) == 1, responses
    assert sum(isinstance(response, dict) and (
        response.get("error", {}).get("code") == 4009 or
        response.get("result", {}).get("status") == "queued")
        for response in responses) == 1, responses


def test_managed_submit_cannot_trigger_typed_voice_stop(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: True)
    monkeypatch.setattr(server, "_typed_stop_phrase_response", lambda rid, text: (
        calls.append(text), {"jsonrpc": "2.0", "id": rid, "result": {"voice_stopped": True}})[1])
    response = _rpc("prompt.submit", "r1", {"session_id": "owned", "text": "stop",
                                                    "managed_turn_key": "a" * 32})
    assert response["error"]["code"] == 4002
    assert calls == []


def test_managed_submit_rejects_explicit_foreign_profile(tmp_path, monkeypatch):
    session = {"history_lock": threading.Lock(), "running": False, "profile_home": str(tmp_path)}
    monkeypatch.setattr(server, "_profile_home", lambda _name: tmp_path / "foreign-profile")
    response = preflight("r1", {"managed_turn_key": "a" * 32, "profile": "foreign"},
                         session, "report", ())
    assert response["error"]["code"] == 4031
    monkeypatch.setattr(server, "_profile_home", lambda _name: tmp_path)
    assert preflight("r2", {"managed_turn_key": "a" * 32, "profile": "owned"},
                     session, "report", ()) is None


def test_managed_submission_refuses_internal_relay_author():
    session = {"history_lock": threading.Lock(), "running": False}
    response = preflight("r1", {"managed_turn_key": "a" * 32, "_turn_author": object()},
                         session, "report", ())
    assert response["error"]["code"] == 4002


def test_managed_busy_submit_never_queues_or_steers(monkeypatch):
    session = {"session_key": "owned", "history_lock": threading.Lock(), "running": True,
               "attached_images": [], "agent": None}
    monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: None)
    monkeypatch.setattr(server, "_legacy_group_fence_error", lambda *args: None)
    monkeypatch.setattr(server, "_reattach_refusal", lambda *args: None)
    response = _rpc("prompt.submit", "request-1", {
        "session_id": "owned", "text": "new task", "managed_turn_key": "a" * 32})
    assert response["error"]["code"] == 4009, response
    assert session.get("queued_prompt") is None
    assert session.get("queued_prompts") is None
