"""The real in-process prompt path records and resumes an exact private session."""

from contextlib import contextmanager
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms as rooms
from gateway import hosted_room_local_sessions as bindings
from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC, HostedRoomSessionError
from tests.gateway.test_hosted_room_local_sessions import running_turn


@pytest.fixture(params=["default", "ops"])
def context(request, tmp_path, monkeypatch):
    profile = request.param
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "origin")
    db_path = home / "state.db" if profile == "default" else home / "profiles" / profile / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=db_path)
    turn = running_turn(home / "state.db", profile=profile)
    record = {
        "source": "bot_room", "profile_home": None if profile == "default" else str(db_path.parent),
        "session_key": "original-private-context", "history_lock": threading.Lock(), "history": [],
        "running": False, "history_version": 0, "room_plumbing": True, "pending_hidden": True,
        "pending_title": "Group: room", "title": "Group: room", "inflight_turn": None,
        "parent_session_id": None, "agent_ready": threading.Event(), "transport": None,
    }

    @contextmanager
    def owner_db(*args):
        yield db

    monkeypatch.setattr(server, "_workdir_owner_db", owner_db)
    monkeypatch.setattr(server, "_session_db", owner_db)
    monkeypatch.setattr(server, "_profile_db", owner_db)
    monkeypatch.setattr(server, "_response_profile_name", lambda selected: selected)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(server, "_workdir_row_model_config", lambda _session: ("test-model", {}))
    monkeypatch.setattr(server, "_persisted_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_sess_nowait", lambda _params, _rid: (record, None))
    monkeypatch.setattr(server, "_typed_stop_phrase_response", lambda *_args: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_args: True)
    started = []

    class DeferredTurn:
        def __init__(self, **kwargs):
            self.target = kwargs["target"]

        def start(self):
            started.append(record["session_key"])

    monkeypatch.setattr(server, "threading", SimpleNamespace(Thread=DeferredTurn))
    rpc = HostedRoomServerRPC(server, db_path=turn.path)
    yield SimpleNamespace(profile=profile, db=db, record=record, turn=turn, rpc=rpc, started=started)
    db.close()


def submit(ctx):
    return ctx.rpc.submit(profile=ctx.profile, session_id="runtime", prompt="Review the plan", source="bot_room",
        task=ctx.turn.task, execution_generation=ctx.turn.attempt.execution_generation, on_terminal=lambda _r: None)


def seed(ctx, session_id=None):
    key = session_id or ctx.record["session_key"]
    ctx.db.create_session(key, source="bot_room", profile_name=ctx.profile)
    ctx.db.set_session_title(key, "Group: room")
    ctx.db.set_session_hidden(key, True)
    ctx.db.append_message(key, "user", "Keep the private prior decision.")
    ctx.db.append_message(key, "assistant", "The prior decision is retained.")
    return key


def test_fresh_session_is_persisted_only_by_real_submit_and_bound_before_turn_thread(context):
    ctx = context
    assert ctx.db.get_session(ctx.record["session_key"]) is None
    assert bindings.lookup_binding(ctx.turn.path, room_id="room", profile=ctx.profile) is None
    assert submit(ctx)["status"] == "streaming"
    binding = bindings.lookup_binding(ctx.turn.path, room_id="room", profile=ctx.profile)
    assert binding["session_id"] == ctx.record["session_key"]
    assert ctx.started == [binding["session_id"]]
    assert ctx.db.get_session(binding["session_id"])["source"] == "bot_room"


def test_existing_private_context_is_unchanged_and_lookup_no_longer_uses_title(context, monkeypatch):
    ctx = context
    key = seed(ctx)
    before = ctx.db.get_messages(key)
    submit(ctx)
    monkeypatch.setitem(server._methods, "session.list", lambda *_args: pytest.fail("title lookup must not choose bound context"))
    restarted = HostedRoomServerRPC(server, db_path=ctx.turn.path)
    assert restarted.resolve_exact(profile=ctx.profile, title="Group: room", source="bot_room")["session_id"] == key
    assert ctx.db.get_messages(key) == before


def test_missing_bound_context_is_not_recreated_or_replaced(context):
    ctx = context
    key = seed(ctx)
    submit(ctx)
    ctx.record["running"] = False
    ctx.db.delete_session(key)
    with pytest.raises(HostedRoomSessionError):
        submit(ctx)
    assert ctx.db.get_session(key) is None
    assert ctx.started == [key]
    assert bindings.lookup_binding(ctx.turn.path, room_id="room", profile=ctx.profile)["session_id"] == key


def test_unrecorded_missing_resumed_context_is_not_created(context):
    ctx = context
    ctx.record["lazy"] = True
    with pytest.raises(HostedRoomSessionError):
        submit(ctx)
    assert ctx.db.get_session(ctx.record["session_key"]) is None
    assert not ctx.started


def test_compression_continuation_keeps_the_recorded_anchor(context):
    ctx = context
    anchor = seed(ctx)
    submit(ctx)
    ctx.record["running"] = False
    ctx.db.end_session(anchor, end_reason="compression")
    ctx.db.create_session("compressed-context", source="bot_room", parent_session_id=anchor, profile_name=ctx.profile)
    ctx.db.append_message("compressed-context", "user", "A real compressed continuation.")
    assert ctx.db.get_compression_tip(anchor) == "compressed-context"
    ctx.record["session_key"] = "compressed-context"
    submit(ctx)
    assert bindings.lookup_binding(ctx.turn.path, room_id="room", profile=ctx.profile)["session_id"] == anchor
    assert ctx.started == [anchor, "compressed-context"]


def test_different_context_cannot_replace_a_recorded_session(context):
    ctx = context
    anchor = seed(ctx)
    submit(ctx)
    ctx.record["running"] = False
    ctx.db.create_session("another-context", source="bot_room", profile_name=ctx.profile)
    ctx.record["session_key"] = "another-context"
    with pytest.raises(HostedRoomSessionError):
        submit(ctx)
    assert ctx.started == [anchor]


def test_recreated_session_identity_is_rejected_even_with_the_same_key(context):
    ctx = context
    anchor = seed(ctx)
    submit(ctx)
    ctx.record["running"] = False
    ctx.db._execute_write(lambda conn: conn.execute("UPDATE sessions SET started_at=started_at+1 WHERE id=?", (anchor,)))
    with pytest.raises(HostedRoomSessionError):
        submit(ctx)
    assert ctx.started == [anchor]


def test_another_profile_store_is_not_a_fallback(context, tmp_path, monkeypatch):
    ctx = context
    seed(ctx)
    other = SessionDB(db_path=tmp_path / "other-profile.db")

    @contextmanager
    def expected(_params):
        yield other

    monkeypatch.setattr(server, "_profile_db", expected)
    try:
        with pytest.raises(HostedRoomSessionError):
            submit(ctx)
        assert not ctx.started
        assert bindings.lookup_binding(ctx.turn.path, room_id="room", profile=ctx.profile) is None
    finally:
        other.close()


def test_json_cannot_supply_the_internal_guard_or_start_a_turn(context):
    ctx = context
    result = server._methods["prompt.submit"]("untrusted", {
        "session_id": "runtime", "text": "hello", "_hosted_session_guard": "forged",
    })
    assert result["error"]["code"] == 4120
    assert not ctx.started
    assert ctx.db.get_session(ctx.record["session_key"]) is None


def test_busy_context_is_not_queued_without_its_internal_guard(context, monkeypatch):
    ctx = context
    ctx.record["running"] = True
    ctx.record["inflight_turn"] = {"text": "Existing work"}
    monkeypatch.setattr(server, "_handle_busy_submit", lambda *_a, **_kw: pytest.fail("must not drop task proof into a text queue"))
    with pytest.raises(HostedRoomSessionError) as error:
        submit(ctx)
    assert error.value.not_admitted is True
    assert ctx.record["running"] is True
    assert ctx.record["inflight_turn"] == {"text": "Existing work"}
    assert not ctx.started
    assert bindings.lookup_binding(ctx.turn.path, room_id="room", profile=ctx.profile) is None
