"""Real profile routing, compression publication and cold/live room recovery."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms as rooms
from gateway import hosted_room_local_sessions as bindings
from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.hosted_room_driver import HostedRoomRuntime, HostedRoomBinding
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC
from tests.gateway.test_hosted_room_local_sessions import running_turn


@pytest.fixture(params=[("default", "default"), ("default", "ops"), ("ops", "default"), ("ops", "ops")])
def context(request, tmp_path, monkeypatch):
    launch, profile = request.param
    root = tmp_path / "home" / ".hermes"
    (root / "profiles" / "ops").mkdir(parents=True)
    launch_home = root if launch == "default" else root / "profiles" / "ops"
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    launch_db = SessionDB(db_path=launch_home / "state.db")
    monkeypatch.setattr(server, "_db", launch_db)
    monkeypatch.setattr(server, "_db_error", None)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "origin")
    # Keep installed profile/session/compression resolution; disable only model
    # work and unrelated asynchronous UI setup in this deterministic test.
    for name in ("_schedule_agent_build", "_schedule_session_cap_enforcement", "_enable_gateway_prompts",
                 "_register_session_cwd", "_ensure_active_session_slot", "_typed_stop_phrase_response",
                 "_project_info_for_cwd", "_maybe_schedule_auto_continue"):
        monkeypatch.setattr(server, name, lambda *a, **kw: None)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a: False)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *a: True)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-no-model")
    starts = []

    class DeferredThread:
        def __init__(self, **kwargs):
            self.target = kwargs["target"]

        def start(self):
            starts.append(self.target)

        def is_alive(self):
            return False

    monkeypatch.setattr(server, "threading", SimpleNamespace(**{**vars(threading), "Thread": DeferredThread}))
    turn = running_turn(root / "state.db", profile=profile)
    rpc = HostedRoomServerRPC(server, db_path=turn.path)
    sid = rpc.create(profile=profile, title="Group: room", source="bot_room")["session_id"]
    record = server._sessions[sid]
    with server._session_db(record) as db:
        yield SimpleNamespace(profile=profile, turn=turn, rpc=rpc, sid=sid, record=record, db=db, starts=starts)
    server._sessions.clear()
    launch_db.close()


def submit(ctx, sid=None):
    return ctx.rpc.submit(profile=ctx.profile, session_id=sid or ctx.sid, prompt="Review the plan",
        source="bot_room", task=ctx.turn.task, execution_generation=ctx.turn.attempt.execution_generation,
        on_terminal=lambda _result: None)


def compress(ctx):
    anchor = ctx.record["session_key"]
    submit(ctx)
    ctx.db.set_session_title(anchor, "Group: room")
    ctx.db.append_message(anchor, "user", "Original decision")
    ctx.db.append_message(anchor, "assistant", "Original answer")
    assert ctx.db.try_acquire_compression_lock(anchor, "test-holder")
    ctx.db.publish_compression_child(parent_session_id=anchor, child_session_id="compressed-tip",
        source="bot_room", profile_name=ctx.profile, compression_lock_holder="test-holder",
        messages=[{"role": "user", "content": "New decision after compression"},
                  {"role": "assistant", "content": "New answer after compression"}])
    ctx.db.release_compression_lock(anchor, "test-holder")
    ctx.record["session_key"] = "compressed-tip"
    ctx.record["history"] = ctx.db.get_messages_as_conversation("compressed-tip")
    return anchor


def test_live_compressed_attempt_prevents_premature_worker_recovery(context):
    ctx = context
    compress(ctx)
    binding = HostedRoomBinding("room", "origin", 1)
    runtime = HostedRoomRuntime(db_path=ctx.turn.path, rooms=[binding], turn_lock=lambda _p: nullcontext(),
        rpc=ctx.rpc, clock=lambda: ctx.turn.lease.expires_at + 1, process_generation="replacement-worker")
    assert runtime._inspect_local_recovery_session(driver.get_task(ctx.turn.path, ctx.turn.task)).active
    with pytest.raises(driver.LeaseHeldError):
        runtime._process_room(binding)
    assert ctx.record["running"] is True
    assert driver.get_task(ctx.turn.path, ctx.turn.task)["status"] == "running"


@pytest.mark.parametrize("damage", ["deleted", "recreated"])
def test_lost_admitted_descendant_is_refused_before_cold_resume(context, damage):
    ctx = context
    anchor = compress(ctx)
    ctx.record["running"] = False
    submit(ctx)
    old = bindings.lookup_binding(ctx.turn.path, room_id="room", profile=ctx.profile)
    assert old["session_id"] == anchor
    assert old["last_session_id"] == "compressed-tip"
    server._sessions.clear()
    assert ctx.db.delete_session("compressed-tip")
    if damage == "recreated":
        ctx.db.create_session("compressed-tip", source="bot_room", parent_session_id=anchor, profile_name=ctx.profile)
        assert ctx.db.get_session("compressed-tip")["started_at"] != old["last_session_started_at"]
    with pytest.raises(bindings.LocalSessionBindingError):
        ctx.rpc.resolve_exact(profile=ctx.profile, title="Group: room", source="bot_room")
    assert not server._sessions
    assert len(ctx.starts) == 2
    assert bindings.lookup_binding(ctx.turn.path, room_id="room", profile=ctx.profile) == old


def test_intact_descendant_survives_cold_resume_without_losing_anchor(context):
    ctx = context
    anchor = compress(ctx)
    ctx.record["running"] = False
    submit(ctx)
    expected_history = ctx.db.get_messages_as_conversation("compressed-tip", include_row_ids=True)
    server._sessions.clear()
    resolved = ctx.rpc.resolve_exact(profile=ctx.profile, title="Group: room", source="bot_room")
    assert resolved["session_id"] == "compressed-tip"
    resumed = ctx.rpc.resume(profile=ctx.profile, session_id=resolved["session_id"], source="bot_room")
    assert server._sessions[resumed["session_id"]]["history"] == expected_history
    assert submit(ctx, resumed["session_id"])["status"] == "streaming"
    saved = bindings.lookup_binding(ctx.turn.path, room_id="room", profile=ctx.profile)
    assert saved["session_id"] == anchor
    assert saved["last_session_id"] == "compressed-tip"
