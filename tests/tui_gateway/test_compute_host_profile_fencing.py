"""Behavioral fences for profile-routed compute-host callbacks and transport frames.

These controls exercise the real bridge and :class:`HostSupervisor` objects with only their
transport boundaries held by deterministic test events.  They do not inspect implementation
source or start a provider/model process.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor


def _session(home: Path, sid: str) -> dict:
    return {
        "session_key": sid,
        "profile_home": str(home),
        "history": [],
        "history_lock": threading.RLock(),
    }


def test_profile_routed_frame_cannot_write_or_mirror_another_home(monkeypatch, tmp_path: Path) -> None:
    """A child bound to A cannot relay a request addressed to B's session."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    sid_b = "session-b"
    session_b = _session(home_b, sid_b)
    monkeypatch.setattr(server, "_sessions", {sid_b: session_b})
    written: list[dict] = []
    monkeypatch.setattr(server, "write_json", lambda frame: written.append(frame) or True)

    frame = {
        "jsonrpc": "2.0",
        "id": "request-b",
        "method": "clarify",
        "params": {"session_id": sid_b, "prompt": "profile-scoped request"},
    }

    server._relay_compute_host_rpc(frame, target_key=str(home_a.resolve()))

    assert written == []
    assert "_compute_host_open_request" not in session_b


@pytest.mark.parametrize("frame_kind", ["turn", "late-control"])
def test_host_supervisor_keeps_pending_work_during_callback_handoff(
    monkeypatch: pytest.MonkeyPatch, frame_kind: str,
) -> None:
    """Popping a reply cannot create a false idle window before its callback starts."""
    supervisor = HostSupervisor(argv=[sys.executable, "-c", ""], autostart=False)
    monkeypatch.setattr(supervisor, "start", lambda: None)
    monkeypatch.setattr(supervisor, "_send_frame", lambda _frame: None)
    callback_frames: list[dict] = []
    handoff = threading.Event()
    release = threading.Event()

    def callback(frame: dict) -> None:
        callback_frames.append(frame)

    if frame_kind == "turn":
        request_id = supervisor.submit_turn(
            {"sid": "session-a", "request_id": "turn-1", "text": "hello"},
            on_complete=callback,
        )
        frame = {"type": "turn.end", "sid": "session-a", "request_id": request_id}
    else:
        request_id = "late-1"
        supervisor._register_late_control_handler(request_id, callback)
        frame = {"type": "control.ack", "request_id": request_id, "result": {"ok": True}}

    original_invoke = supervisor._invoke_callback

    def gated_invoke(*args, **kwargs) -> None:
        handoff.set()
        assert release.wait(timeout=2)
        original_invoke(*args, **kwargs)

    monkeypatch.setattr(supervisor, "_invoke_callback", gated_invoke)
    worker = threading.Thread(target=supervisor._handle_host_frame, args=(frame,), daemon=True)
    worker.start()
    try:
        assert handoff.wait(timeout=2)
        assert supervisor.has_pending_work(), frame_kind
    finally:
        release.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert callback_frames == [frame]
    assert not supervisor.has_pending_work()


def test_profile_target_never_falls_back_to_legacy_singleton(monkeypatch, tmp_path: Path) -> None:
    """A live target, then its replaced/unknown session, cannot use the old singleton seam."""
    home = tmp_path / "profile"
    home.mkdir()
    sid = "session-replaced"
    live = _session(home, sid)
    sessions = {sid: live}
    routed_supervisor = HostSupervisor(argv=[sys.executable, "-c", ""], autostart=False)
    legacy_supervisor = object()
    monkeypatch.setattr(server, "_sessions", sessions)
    monkeypatch.setattr(server, "_compute_host_supervisor", legacy_supervisor, raising=False)
    monkeypatch.setattr(server, "_compute_host_supervisors", {str(home.resolve()): routed_supervisor})

    selected = server._compute_host_supervisor_for_session(
        sid, live, {"turn_isolation": True})
    assert selected is routed_supervisor
    assert selected is not legacy_supervisor

    replacement = _session(home, sid)
    sessions[sid] = replacement
    with pytest.raises(RuntimeError, match="compute-host session is unavailable"):
        server._compute_host_supervisor_for_session(sid, live, {"turn_isolation": True})

    sessions.pop(sid)
    with pytest.raises(RuntimeError, match="compute-host session is unavailable"):
        server._compute_host_supervisor_for_session(sid, replacement, {"turn_isolation": True})


def test_failure_drain_does_not_consume_callback_reservation_for_no_callback_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash row without a callback cannot make another blocked callback look idle."""
    supervisor = HostSupervisor(argv=[sys.executable, "-c", ""], autostart=False)
    monkeypatch.setattr(supervisor, "start", lambda: None)
    monkeypatch.setattr(supervisor, "_send_frame", lambda _frame: None)
    callback_entered = threading.Event()
    release_callback = threading.Event()
    callback_frames: list[dict] = []

    def callback(frame: dict) -> None:
        callback_entered.set()
        release_callback.wait(timeout=2)
        callback_frames.append(frame)

    # Preserve insertion order: the no-callback turn is drained before the callback reservation.
    supervisor.submit_turn({"sid": "none", "request_id": "none"})
    supervisor.submit_turn({"sid": "callback", "request_id": "callback"}, on_complete=callback)
    drain = threading.Thread(
        target=supervisor._fail_pending_turns,
        kwargs={"reason": "crash", "message": "child exited"},
        daemon=True,
    )
    drain.start()
    try:
        assert callback_entered.wait(timeout=2)
        assert supervisor.has_pending_work()
    finally:
        release_callback.set()
        drain.join(timeout=2)

    assert not drain.is_alive()
    assert len(callback_frames) == 1
    assert callback_frames[0]["request_id"] == "callback"
    assert not supervisor.has_pending_work()


class _BlockingTransport:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.frames: list[dict] = []

    def write(self, frame: dict) -> bool:
        self.entered.set()
        if not self.release.wait(timeout=2):
            return False
        self.frames.append(frame)
        return True


class _RecordingTransport:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def write(self, frame: dict) -> bool:
        self.frames.append(frame)
        return True


def test_target_relay_captures_transport_before_replacement_without_holding_sessions_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A same-sid replacement cannot reroute a frame captured by the old target."""
    home = tmp_path / "profile"
    home.mkdir()
    sid = "session-replaced-during-write"
    old_transport = _BlockingTransport()
    replacement_transport = _RecordingTransport()
    old_session = _session(home, sid)
    old_session["transport"] = old_transport
    replacement = _session(home, sid)
    replacement["transport"] = replacement_transport
    sessions = {sid: old_session}
    monkeypatch.setattr(server, "_sessions", sessions)
    target_key = str(home.resolve())
    frame = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"session_id": sid, "type": "profile.event", "payload": {"value": 1}},
    }

    relay = threading.Thread(
        target=server._relay_compute_host_rpc,
        args=(frame,),
        kwargs={"target_key": target_key},
        daemon=True,
    )
    relay.start()
    try:
        assert old_transport.entered.wait(timeout=2)
        replaced = threading.Event()

        def replace_session() -> None:
            with server._sessions_lock:
                sessions[sid] = replacement
                replaced.set()

        replacement_thread = threading.Thread(target=replace_session, daemon=True)
        replacement_thread.start()
        assert replaced.wait(timeout=2)
    finally:
        old_transport.release.set()
        relay.join(timeout=2)
        if "replacement_thread" in locals():
            replacement_thread.join(timeout=2)

    assert not relay.is_alive()
    assert not replacement_thread.is_alive()
    assert len(old_transport.frames) == 1
    assert old_transport.frames[0]["params"]["session_id"] == sid
    assert replacement_transport.frames == []


def test_target_relay_projection_captures_session_before_replacement_without_holding_sessions_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A blocked hosted-room projection keeps the original session and transport target."""
    from tui_gateway import hosted_room_member_activity

    home = tmp_path / "profile"
    home.mkdir()
    sid = "session-projection-replaced"
    old_transport = _BlockingTransport()
    replacement_transport = _RecordingTransport()
    old_session = _session(home, sid)
    old_session["transport"] = old_transport
    hosted_task_a = {
        "room_id": "room-a", "task_id": "task-a", "thread_id": "thread-a",
        "turn_id": "turn-a", "member_id": "member-a", "execution_generation": 1,
    }
    old_session["_hosted_room_task"] = hosted_task_a
    replacement = _session(home, sid)
    replacement["transport"] = replacement_transport
    replacement["_hosted_room_task"] = dict(hosted_task_a)
    sessions = {sid: old_session}
    monkeypatch.setattr(server, "_sessions", sessions)

    projection_entered = threading.Event()
    release_projection = threading.Event()
    projected_sessions: list[dict | None] = []
    projected_tasks: list[dict] = []
    original_projection = hosted_room_member_activity.project_room_member_activity

    def capture_activity(hosted_task: dict, **_kwargs: object) -> bool:
        projected_tasks.append(dict(hosted_task))
        return True

    monkeypatch.setattr(hosted_room_member_activity, "emit_room_member_activity", capture_activity)

    def blocking_projection(frame: dict, projection_map: dict) -> bool:
        projection_entered.set()
        release_projection.wait()
        projected_sessions.append(projection_map.get(sid))
        return original_projection(frame, projection_map)

    monkeypatch.setattr(
        hosted_room_member_activity,
        "project_room_member_activity",
        blocking_projection,
    )
    frame = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "session_id": sid,
            "type": "message.delta",
            "payload": {"text": "projected"},
        },
    }

    relay = threading.Thread(
        target=server._relay_compute_host_rpc,
        args=(frame,),
        kwargs={"target_key": str(home.resolve())},
        daemon=True,
    )
    replacement_thread: threading.Thread | None = None
    replaced = threading.Event()

    def replace_session() -> None:
        with server._sessions_lock:
            sessions[sid] = replacement
            replaced.set()

    relay.start()
    try:
        assert projection_entered.wait(timeout=2)
        replacement_thread = threading.Thread(target=replace_session, daemon=True)
        replacement_thread.start()
        assert replaced.wait(timeout=2)
        release_projection.set()
        assert old_transport.entered.wait(timeout=2)
    finally:
        release_projection.set()
        old_transport.release.set()
        relay.join(timeout=2)
        if replacement_thread is not None:
            replacement_thread.join(timeout=2)

    assert not relay.is_alive()
    assert replacement_thread is not None and not replacement_thread.is_alive()
    assert len(projected_sessions) == 1
    assert projected_tasks == [hosted_task_a]
    assert len(old_transport.frames) == 1
    assert old_transport.frames[0]["params"]["session_id"] == sid
    assert replacement_transport.frames == []


def test_target_relay_projection_captures_hosted_task_before_history_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A blocked projection keeps task coordinates captured before the live task is cleared."""
    from tui_gateway import hosted_room_member_activity

    home = tmp_path / "profile"
    home.mkdir()
    sid = "session-projection-task-mutated"
    old_transport = _BlockingTransport()
    old_session = _session(home, sid)
    old_session["transport"] = old_transport
    hosted_task_a = {
        "room_id": "room-a", "task_id": "task-a", "thread_id": "thread-a",
        "turn_id": "turn-a", "member_id": "member-a", "execution_generation": 1,
    }
    old_session["_hosted_room_task"] = hosted_task_a
    monkeypatch.setattr(server, "_sessions", {sid: old_session})

    projection_entered = threading.Event()
    release_projection = threading.Event()
    task_mutated = threading.Event()
    projected_tasks: list[dict] = []
    original_projection = hosted_room_member_activity.project_room_member_activity

    def capture_activity(hosted_task: dict, **_kwargs: object) -> bool:
        projected_tasks.append(dict(hosted_task))
        return True

    monkeypatch.setattr(hosted_room_member_activity, "emit_room_member_activity", capture_activity)

    def blocking_projection(frame: dict, projection_map: dict) -> bool:
        projection_entered.set()
        release_projection.wait()
        return original_projection(frame, projection_map)

    monkeypatch.setattr(
        hosted_room_member_activity,
        "project_room_member_activity",
        blocking_projection,
    )
    frame = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "session_id": sid,
            "type": "message.delta",
            "payload": {"text": "projected"},
        },
    }

    relay = threading.Thread(
        target=server._relay_compute_host_rpc,
        args=(frame,),
        kwargs={"target_key": str(home.resolve())},
        daemon=True,
    )

    def clear_hosted_task() -> None:
        with old_session["history_lock"]:
            old_session.pop("_hosted_room_task", None)
            task_mutated.set()

    mutation_thread = threading.Thread(target=clear_hosted_task, daemon=True)
    mutation_started = False
    relay.start()
    try:
        assert projection_entered.wait(timeout=2)
        mutation_started = True
        mutation_thread.start()
        assert task_mutated.wait(timeout=2)
        release_projection.set()
        assert old_transport.entered.wait(timeout=2)
    finally:
        release_projection.set()
        old_transport.release.set()
        relay.join(timeout=2)
        if mutation_started:
            mutation_thread.join(timeout=2)

    assert not relay.is_alive()
    assert not mutation_thread.is_alive()
    assert old_session.get("_hosted_room_task") is None
    assert projected_tasks == [hosted_task_a]
    assert len(old_transport.frames) == 1
    assert old_transport.frames[0]["params"]["session_id"] == sid


def test_target_sessionless_event_reaches_only_matching_profile_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A target-scoped child global event cannot fan out to another profile or an unknown target."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    transport_a = _RecordingTransport()
    transport_b = _RecordingTransport()
    session_a = _session(home_a, "session-a")
    session_a["transport"] = transport_a
    session_b = _session(home_b, "session-b")
    session_b["transport"] = transport_b
    monkeypatch.setattr(server, "_sessions", {"session-a": session_a, "session-b": session_b})
    monkeypatch.setattr(server, "_live_transports", {transport_a, transport_b})
    frame = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": "skin.changed", "payload": {"name": "profile-a"}},
    }

    assert server._relay_compute_host_rpc(frame, target_key=str(home_a.resolve()))
    assert len(transport_a.frames) == 1
    assert transport_b.frames == []

    assert not server._relay_compute_host_rpc(frame, target_key=str((tmp_path / "missing").resolve()))
    assert len(transport_a.frames) == 1
    assert transport_b.frames == []


def test_detached_close_does_not_start_retired_captured_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A close captured before retirement must not revive its stopped profile child.

    The child and supervisor are real; only the child command is the existing provider-free
    fixture.  The event at the captured-owner seam creates the production interleaving: pop
    captures the owner, retirement shuts it down and removes its registry, then close releases
    the captured interrupt.  A same-SID replacement is installed after retirement to prove the
    stale close does not resolve or route through the replacement.
    """
    home = tmp_path / "profile"
    home.mkdir()
    sid = "session-detached-retired"
    (home / "profile-owner-session.txt").write_text(sid, encoding="utf-8")
    (home / "profile-owner-marker.txt").write_text("retired", encoding="utf-8")
    registry = home / "state" / "dashboard-compute-host.json"
    fixture_child = Path(__file__).with_name("profile_owner_child.py")
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, str(fixture_child)],
        cwd=repo,
        env=env,
        expected_hermes_home=str(home),
        respawn_max=0,
        autostart=False,
    )
    target_key = str(home.resolve())
    session = _session(home, sid)
    session.update({
        "agent": None,
        "agent_ready": object(),
        "running": True,
        "transport": server._detached_ws_transport,
        "source": "desktop",
        "_compute_host_active": True,
        "_compute_host_turn_id": "turn-detached",
    })
    sessions = {sid: session}
    supervisors = {target_key: supervisor}
    monkeypatch.setattr(server, "_sessions", sessions)
    monkeypatch.setattr(server, "_compute_host_supervisors", supervisors)
    monkeypatch.setattr(server, "_compute_host_supervisor_retiring", set())
    monkeypatch.setattr(
        server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})

    supervisor.start()
    initial_pid = supervisor.pid
    assert initial_pid > 0
    assert registry.exists()
    spawn_reasons: list[str] = []
    original_spawn = supervisor._spawn_locked

    def record_spawn(*args, **kwargs):
        spawn_reasons.append(str(kwargs.get("reason") or ""))
        return original_spawn(*args, **kwargs)

    monkeypatch.setattr(supervisor, "_spawn_locked", record_spawn)
    owner_checked = threading.Event()
    release_close = threading.Event()
    original_owner = server._compute_host_close_owner

    def hold_captured_owner(closing_session: dict, closing_sid: str):
        owner = original_owner(closing_session, closing_sid)
        assert owner is supervisor
        owner_checked.set()
        release_close.wait()
        return owner

    monkeypatch.setattr(server, "_compute_host_close_owner", hold_captured_owner)
    monkeypatch.setattr(
        server, "_compute_host_supervisor_for_session",
        lambda *_args, **_kwargs: pytest.fail("detached close resolved a replacement supervisor"),
    )
    teardown_calls: list[dict] = []
    monkeypatch.setattr(server, "_teardown_session", lambda value, **_kw: teardown_calls.append(value))

    popped = server._pop_session_by_id(sid)
    assert popped is session
    close_errors: list[Exception] = []

    def close_detached() -> None:
        try:
            server._teardown_popped_session(session, end_reason="tui_close")
        except Exception as exc:  # preserve the real close-path failure for the assertion below
            close_errors.append(exc)

    close_thread = threading.Thread(target=close_detached, name="detached-close-test")
    close_thread.start()
    replacement = _session(home, sid)
    replacement["transport"] = _RecordingTransport()
    try:
        assert owner_checked.wait(timeout=2)
        with session["history_lock"]:
            session.pop("_compute_host_turn_id")
        server._retire_compute_host_target(sid, session)
        assert not supervisor.is_running()
        assert target_key not in supervisors
        assert not registry.exists()
        sessions[sid] = replacement
        release_close.set()
        close_thread.join(timeout=2)
        assert not close_thread.is_alive()
        assert not supervisor.is_running()
        assert not registry.exists()
    finally:
        release_close.set()
        supervisor.shutdown()
        close_thread.join(timeout=2)

    assert not close_thread.is_alive()
    assert close_errors == []
    assert teardown_calls == [session]
    assert spawn_reasons == []
    assert supervisors == {}
    assert sessions[sid] is replacement
    assert replacement.get("_turn_cancel_requested") is None
