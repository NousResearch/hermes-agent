"""Completed Desktop turns retain writer ownership without consuming host execution slots."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hermes_cli.active_sessions import active_session_registry_snapshot
from hermes_cli.execution_budget import execution_budget_snapshot, try_acquire_execution_permit
from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def prompt_budget_env(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("max_concurrent_sessions: 2\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_concurrent_sessions": 2})
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    calls, events, sessions = [], [], []
    for name in ("_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd",
                 "_sync_session_key_after_compress", "_emit_settled_session_info"):
        monkeypatch.setattr(server, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: events.append(args))
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_goal_followup_after_turn", lambda *args: None)

    def make(sid):
        db.create_session(sid, source="desktop")
        ready = threading.Event()
        ready.set()

        def provider(*args, **kwargs):
            calls.append(sid)
            return {"final_response": "synthetic done", "messages": []}

        agent = SimpleNamespace(session_id=sid, clear_interrupt=lambda: None, run_conversation=provider)
        session = {"agent": agent, "agent_ready": ready, "session_key": sid,
                   "source": "desktop", "history": [], "history_lock": threading.RLock(),
                   "running": True, "transport": None, "history_version": 0,
                   "attached_images": [], "image_counter": 0, "cols": 80, "slash_worker": None,
                   "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None}
        server._sessions[sid] = session
        sessions.append(session)
        return session

    yield make, calls, events
    for session in sessions:
        if worker := session.get("_run_thread"):
            worker.join(15)
            assert not worker.is_alive()
        server._release_active_session_slot(session)
    db.close()


def _finish_prompt(sid, session):
    session["running"] = True
    assert server._run_prompt_submit("turn", sid, session, "synthetic prompt") is True
    worker = session["_run_thread"]
    worker.join(15)
    assert not worker.is_alive()
    assert session["running"] is False
    assert session.get("inflight_turn") is None


def test_completed_desktop_turns_leave_cloud_execution_capacity(prompt_budget_env, tmp_path):
    from gateway.config import Platform
    from gateway.execution_admission import run_claimed_message

    make, calls, _events = prompt_budget_env
    for sid in ("desktop-one", "desktop-two"):
        _finish_prompt(sid, make(sid))
    assert calls == ["desktop-one", "desktop-two"]
    assert len(active_session_registry_snapshot()) == 2
    assert execution_budget_snapshot() == (2, [])

    async def cloud_provider(*args):
        assert len(execution_budget_snapshot()[1]) == 1
        return "synthetic cloud done"

    runner = SimpleNamespace(
        _resolve_profile_home_for_source=lambda source: tmp_path,
        _hm_post_admission_consume=AsyncMock(return_value=(False, None)),
        _handle_message_with_agent=AsyncMock(side_effect=cloud_provider),
        _run_post_turn_hooks=AsyncMock(), _restore_pending_one_turn_model_override=Mock(),
        _clear_durable_active_turn=AsyncMock(), _release_running_agent_state=Mock(),
        _release_turn_lease=Mock())
    result = asyncio.run(run_claimed_message(
        runner, SimpleNamespace(), SimpleNamespace(platform=Platform.TELEGRAM), "cloud-route", 1, False))
    assert result == "synthetic cloud done"
    runner._handle_message_with_agent.assert_awaited_once()
    assert len(active_session_registry_snapshot()) == 2
    assert execution_budget_snapshot() == (2, [])


def test_real_prompt_refuses_at_two_executions_then_retries_after_release(prompt_budget_env):
    make, calls, events = prompt_budget_env
    held = []
    try:
        for sid in ("cloud-one", "cloud-two"):
            permit, refusal = try_acquire_execution_permit(execution_id=sid, surface="gateway:telegram")
            assert refusal is None
            held.append(permit)
        session = make("desktop")
        session["attached_images"] = ["synthetic.png"]
        assert server._run_prompt_submit("refused", "desktop", session, "synthetic prompt") is False
        assert calls == []
        assert session.get("_run_thread") is None
        assert session["running"] is False
        assert any("MAX_CONCURRENT_SESSIONS" in str(event) for event in events)
        assert len(execution_budget_snapshot()[1]) == 2
        assert session["attached_images"] == ["synthetic.png"]
        session["attached_images"] = []
        held.pop().release()
        _finish_prompt("desktop", session)
        assert calls == ["desktop"]
        assert len(execution_budget_snapshot()[1]) == 1
        assert len(active_session_registry_snapshot()) == 1
    finally:
        for permit in held:
            permit.release()


def test_hosted_capacity_preflight_is_proven_not_admitted(prompt_budget_env, tmp_path, monkeypatch):
    from gateway.hosted_room_driver import TaskIdentity
    from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC, HostedRoomSessionError

    make, calls, _events = prompt_budget_env
    session = make("hosted-preflight")
    session.update(source="bot_room", running=False)
    monkeypatch.setattr(server, "_profile_home", lambda profile: tmp_path)
    held = [try_acquire_execution_permit(execution_id=f"busy-{i}", surface="cloud")[0] for i in range(2)]
    callback = Mock()
    try:
        with pytest.raises(HostedRoomSessionError) as caught:
            HostedRoomServerRPC(server).submit(
                profile="default", session_id="hosted-preflight", prompt="synthetic", source="bot_room",
                task=TaskIdentity("room", "task", "thread", "turn"), execution_generation=1,
                on_terminal=callback, member_id="member")
        assert caught.value.not_admitted is True
        assert caught.value.code == 4090
        assert "execution limit" in str(caught.value)
        assert calls == [] and session["running"] is False
        callback.assert_not_called()
        assert session.get("active_session_lease") is None
    finally:
        for permit in held:
            permit.release()


def test_hosted_late_capacity_refusal_commits_terminal_receipt(prompt_budget_env, tmp_path):
    from gateway import hosted_room_driver as state, hosted_rooms
    from tests.tui_gateway.test_hosted_room_driver_runtime import (
        BINDING, PROFILE, ROOM_ID, FakeSessionRPC, _admit, _identity, _runtime,
    )

    make, calls, _events = prompt_budget_env
    room_db = tmp_path / "room.db"
    hosted_rooms.create_room(room_db, room_id=ROOM_ID, name="Synthetic room",
                             members=[{"profile": PROFILE, "handle": PROFILE}],
                             authority_gateway_id=BINDING.gateway_id, now=time.time())
    identity = _identity()
    _admit(room_db, identity)
    runtime = _runtime(room_db, FakeSessionRPC(), lease_ttl_seconds=30)
    attempt = state.start_task(room_db, identity, runtime._ensure_lease(BINDING),
                               expected_cancel_generation=0, clock=time.time)
    session = make("hosted-race")
    session.update(source="bot_room", _hosted_room_task={"task_id": identity.task_id})
    held = [try_acquire_execution_permit(execution_id=f"busy-{i}", surface="cloud")[0] for i in range(2)]
    try:
        assert server._run_prompt_submit(
            "race", "hosted-race", session, "synthetic",
            terminal_callback=lambda receipt: runtime._on_terminal(BINDING, attempt, receipt)) is False
        result = state.get_task(room_db, identity)
        assert result["status"] == "failed"
        assert "No work started" in result["result"]["error"]
        assert calls == [] and session["running"] is False
        assert session.get("_hosted_room_task") is None
        assert session.get("inflight_turn") is None
        assert session.get("active_session_lease") is None
        assert len(execution_budget_snapshot()[1]) == 2
    finally:
        for permit in held:
            permit.release()
