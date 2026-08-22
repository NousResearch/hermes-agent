import io
import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tui_gateway import compute_host, server
from tui_gateway.compute_host import ComputeHost, _default_workers
from tui_gateway.host_supervisor import (
    MUTATOR_ROUTE_TABLE,
    HostSupervisor,
    append_log_record,
)


def _isolated_ocr_probe(frame: dict, result_queue) -> None:
    """Run the real prompt worker in a clean spawned child process."""
    from tui_gateway import server as child_server
    from tui_gateway.compute_host import ComputeHost as ChildComputeHost

    agent_runs = []
    persisted = []
    events = []

    class _DB:
        def append_messages_batch(self, session_id, messages, **_kwargs):
            persisted.append((session_id, list(messages)))
            return len(messages)

    class _Agent:
        session_id = "stored-ios"
        _session_db = _DB()

        def clear_interrupt(self):
            return None

        def run_conversation(self, *_args, **_kwargs):
            agent_runs.append(True)
            return {"final_response": "agent-ran", "messages": []}

    fake_server = SimpleNamespace(
        _sessions={},
        _make_agent=lambda *_args, **_kwargs: _Agent(),
        _transfer_db_to_agent=lambda *_args: False,
        _load_show_reasoning=lambda: False,
        _load_tool_progress_mode=lambda: "all",
        _sanitize_client_source=lambda value: value,
    )

    def _init_session(sid, key, agent, history, **kwargs):
        fake_server._sessions[sid] = {
            "agent": agent,
            "session_key": key,
            "history": list(history),
            "history_lock": threading.Lock(),
            "history_version": 0,
            "inflight_turn": None,
            "running": True,
            "attached_images": [],
            "image_counter": 0,
            "cwd": frame.get("cwd") or os.getcwd(),
            "cols": 80,
            "slash_worker": None,
            "show_reasoning": False,
            "tool_progress_mode": "all",
            "source": frame.get("source") or "ios",
            "transport": None,
            "required_prompt_handler": kwargs.get("required_prompt_handler"),
        }

    fake_server._init_session = _init_session
    host = ChildComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        child_session = host._ensure_server_session(fake_server, frame)
        child_server._sessions = {frame["sid"]: child_session}
        child_server._emit = (
            lambda name, sid, payload=None: events.append((name, sid, payload))
        )
        child_server._wire_callbacks = lambda _sid: None
        child_server._apply_pending_model_switch = lambda *_args: None
        child_server._sync_agent_model_with_config = lambda *_args: None
        child_server._sync_bot_capabilities = lambda *_args: None
        child_server._session_cwd = lambda _session: frame.get("cwd") or os.getcwd()
        child_server._register_session_cwd = lambda _session: None
        child_server._emit_settled_session_info = lambda *_args: None
        child_server.record_turn_start = lambda *_args, **_kwargs: None
        child_server._retire_turn_marker = lambda *_args: None
        child_server.render_message = lambda *_args: None

        child_server._run_prompt_submit(
            frame["request_id"],
            frame["sid"],
            child_session,
            frame["text"],
            image_paths=list(frame.get("attached_images") or []),
        )
        child_session["_run_thread"].join(timeout=5)

        cleared = host._ensure_server_session(
            fake_server, {**frame, "required_prompt_handler": None}
        )
        clear_decision = child_server.invoke_pre_prompt_dispatch(
            session_id=frame["sid"],
            session_key=frame["session_key"],
            source=frame["source"],
            text="cleared",
            attached_images=frame["attached_images"],
            required_prompt_handler=cleared.get("required_prompt_handler"),
        )
        if clear_decision.action == "allow":
            cleared["agent"].run_conversation("cleared")
        cleared_handler = cleared.get("required_prompt_handler")

        reasserted = host._ensure_server_session(fake_server, frame)
        reassert_decision = child_server.invoke_pre_prompt_dispatch(
            session_id=frame["sid"],
            session_key=frame["session_key"],
            source=frame["source"],
            text="reasserted",
            attached_images=frame["attached_images"],
            required_prompt_handler=reasserted.get("required_prompt_handler"),
        )
        if reassert_decision.action == "allow":
            reasserted["agent"].run_conversation("reasserted")

        result_queue.put(
            {
                "initial_handler": frame.get("required_prompt_handler"),
                "cleared_handler": cleared_handler,
                "reasserted_handler": reasserted.get("required_prompt_handler"),
                "clear_action": clear_decision.action,
                "reassert_action": reassert_decision.action,
                "agent_runs": len(agent_runs),
                "history": child_session.get("history"),
                "persisted": persisted,
                "event_types": [event[0] for event in events],
            }
        )
    finally:
        host.close()


def _json_lines(out: io.StringIO) -> list[dict]:
    frames = []
    for line in out.getvalue().splitlines():
        if line.strip():
            frames.append(json.loads(line))
    return frames


def _wait_for_frame(out: io.StringIO, predicate, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in _json_lines(out):
            if predicate(frame):
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for frame; saw={_json_lines(out)}")


def test_compute_host_workers_inherit_tui_pool_env_or_8(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_RPC_POOL_WORKERS", raising=False)
    monkeypatch.delenv("HERMES_COMPUTE_HOST_WORKERS", raising=False)
    assert _default_workers() == 8

    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "11")
    assert _default_workers() == 11

    # Dead-RC tombstone: malformed env falls back to 8, not the old except-branch 4.
    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "not-an-int")
    assert _default_workers() == 8


def test_mutator_route_table_matches_prd_inventory():
    assert MUTATOR_ROUTE_TABLE == {
        "prompt.submit": "turn-path",
        "session.interrupt": "turn-path",
        "reload.mcp": "run-concurrent",
        "session.save": "run-concurrent",
        "session.compress": "idle-gated",
        "prompt.submit.truncate": "idle-gated",
        "slash.model": "idle-gated",
        "slash.personality": "idle-gated",
        "slash.prompt": "idle-gated",
        "slash.compress": "idle-gated",
        "session.reset": "idle-gated",
        "session.history.reload": "idle-gated",
        "slash.retry": "idle-gated",
    }


def test_append_log_record_single_write_lines(tmp_path):
    path = tmp_path / "agent.log"

    def writer(i: int) -> None:
        append_log_record(path, f"line-{i:03d}-" + ("x" * 2000))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    assert sorted(line.split("-", 2)[1] for line in lines) == [f"{i:03d}" for i in range(32)]
    assert all(line.endswith("x" * 2000) for line in lines)


def test_supervisor_startup_reconcile_pid_reuse_guard(tmp_path, monkeypatch):
    registry = tmp_path / "dashboard-compute-host.json"
    registry.write_text(json.dumps({"host_pid": os.getpid(), "boot_id": "stale"}), encoding="utf-8")

    killed: list[int] = []
    supervisor = HostSupervisor(registry_path=registry, argv=[sys.executable, "-c", ""], autostart=False)
    monkeypatch.setattr(supervisor, "_pid_matches_compute_host", lambda _pid: False)
    monkeypatch.setattr(supervisor, "_terminate_pid", lambda pid, **_kw: killed.append(pid))

    result = supervisor.reconcile_startup_orphan()

    assert result == "pid-reuse-ignored"
    assert killed == []
    assert not registry.exists()


def _make_compress_host_session(events: list) -> dict:
    class _Agent:
        model = "host-model"
        provider = "host-provider"
        tools = []
        _cached_system_prompt = ""
        session_input_tokens = 1
        session_output_tokens = 1
        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2
        session_api_calls = 1
        session_id = "rotated-id"

    agent = _Agent()
    agent.context_compressor = type("ContextEngineStub", (), {})()
    agent.context_compressor.on_session_start = (
        lambda *_args, **_kwargs: events.append("notify")
    )
    return {
        "agent": agent,
        "session_key": "before-key",
        "history": [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "before"},
        ],
        "history_lock": threading.Lock(),
        "history_version": 2,
        "running": False,
        "manual_compression_lock": threading.Lock(),
    }


def _record_finalize(monkeypatch, events: list[str], *sids: str) -> None:
    """Give ``flush_all_sessions`` sessions and record which ones finalize."""
    keys = sids or ("s1",)
    monkeypatch.setattr(
        server,
        "_sessions",
        {sid: {"session_key": sid} for sid in keys},
        raising=False,
    )
    monkeypatch.setattr(
        server,
        "_finalize_session",
        lambda _session, end_reason="tui_close": events.append(
            f"finalize:{_session['session_key']}:{end_reason}"
        ),
        raising=False,
    )


def _register_turn(host: ComputeHost, fn, sid: str = "s1") -> None:
    """Submit a turn exactly the way ``_handle_turn_start`` does."""
    host._track_turn_future(host._executor.submit(fn), sid)


def test_compute_host_refreshes_and_clears_handler_on_existing_child_session():
    child_session = {
        "required_prompt_handler": None,
        "history_lock": threading.Lock(),
    }
    fake_server = SimpleNamespace(_sessions={"ios-ui": child_session})
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        first = host._ensure_server_session(
            fake_server,
            {
                "sid": "ios-ui",
                "required_prompt_handler": "hoppe_ocr_approval",
            },
        )
        assert first["required_prompt_handler"] == "hoppe_ocr_approval"

        second = host._ensure_server_session(
            fake_server,
            {"sid": "ios-ui", "required_prompt_handler": None},
        )
        assert second["required_prompt_handler"] is None
    finally:
        host.close()


def test_isolated_required_image_is_blocked_before_agent_execution(tmp_path):
    """The serialized turn must retain fail-closed OCR policy in the child."""
    image_path = tmp_path / "protected.png"
    image_path.write_bytes(b"image")
    parent_session = {
        "session_key": "stored-ios",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "attached_images": [str(image_path)],
        "cols": 80,
        "cwd": str(tmp_path),
        "source": "ios",
        "required_prompt_handler": "hoppe_ocr_approval",
    }
    frame = server._compute_host_turn_frame(
        "isolated-image", "ios-ui", parent_session, "Bitte prüfen"
    )
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_isolated_ocr_probe, args=(frame, result_queue))
    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("isolated OCR probe timed out")

    assert process.exitcode == 0
    result = result_queue.get(timeout=2)
    assert result["initial_handler"] == "hoppe_ocr_approval"
    assert result["cleared_handler"] is None
    assert result["reasserted_handler"] == "hoppe_ocr_approval"
    assert result["clear_action"] == "allow"
    assert result["reassert_action"] == "block"
    assert result["agent_runs"] == 1
    assert result["history"][0]["role"] == "user"
    assert result["history"][1]["role"] == "assistant"
    assert result["persisted"]
    assert result["event_types"] == [
        "message.start",
        "message.delta",
        "message.complete",
    ]


@pytest.mark.parametrize("init_fails", [False, True], ids=["normal-init", "fallback"])
def test_compute_host_new_child_session_retains_required_handler(init_fails):
    init_calls = []
    fake_server = SimpleNamespace(
        _sessions={},
        _make_agent=lambda *_args, **_kwargs: SimpleNamespace(),
        _transfer_db_to_agent=lambda *_args: False,
        _load_show_reasoning=lambda: False,
        _load_tool_progress_mode=lambda: "all",
        _sanitize_client_source=lambda value: value,
    )

    def init_session(sid, key, agent, history, **kwargs):
        init_calls.append(kwargs)
        if init_fails:
            raise RuntimeError("force fallback")
        fake_server._sessions[sid] = {
            "agent": agent,
            "session_key": key,
            "history": history,
            "history_lock": threading.Lock(),
            "required_prompt_handler": kwargs.get("required_prompt_handler"),
        }

    fake_server._init_session = init_session
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        session = host._ensure_server_session(
            fake_server,
            {
                "sid": "ios-ui",
                "session_key": "stored-ios",
                "required_prompt_handler": "hoppe_ocr_approval",
                "attached_images": ["/tmp/protected.png"],
                "source": "ios",
            },
        )
    finally:
        host.close()

    assert session["required_prompt_handler"] == "hoppe_ocr_approval"
    if not init_fails:
        assert init_calls[0]["required_prompt_handler"] == "hoppe_ocr_approval"


def test_shutdown_drains_in_flight_turn_before_finalizing_sessions(monkeypatch):
    events: list[str] = []
    _record_finalize(monkeypatch, events)

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    running = threading.Event()

    def _turn() -> None:
        running.set()
        time.sleep(0.3)
        events.append("turn_end")

    _register_turn(host, _turn, sid="s1")
    assert running.wait(timeout=5.0)

    host.shutdown(reason="sigterm", wait=3.0)

    # ``_finalize_session`` latches on ``session["_finalized"]``, so its single
    # run has to observe the finished turn or the tail is unpersistable. A turn
    # that *did* drain must still finalize — the live-turn skip must not
    # over-reach into sessions whose work is done.
    assert events == ["turn_end", "finalize:s1:compute_host_sigterm"]

    # The done-callback still has to remove the entry now that the container is
    # a dict: ``set.discard`` was a valid bare callback, ``dict.pop`` is not.
    deadline = time.monotonic() + 2.0
    while host._turn_futures and time.monotonic() < deadline:
        time.sleep(0.01)
    assert host._turn_futures == {}, "in-flight turns must not accumulate"


def test_shutdown_retains_a_live_turns_session_when_the_drain_deadline_expires(monkeypatch):
    wait = 1.0
    events: list[str] = []
    _record_finalize(monkeypatch, events, "live", "idle")

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    release = threading.Event()
    running = threading.Event()

    def _stuck_turn() -> None:
        running.set()
        release.wait(timeout=30.0)

    _register_turn(host, _stuck_turn, sid="live")
    assert running.wait(timeout=5.0)

    try:
        started = time.monotonic()
        host.shutdown(reason="sigterm", wait=wait)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    # ``_finalize_session`` is one-shot, and the ``shutdown(wait=False)`` that
    # follows does not join the turn. Spending "live"'s single latch mid-turn
    # would leave it permanently un-finalizable and release its active-session
    # lease out from under running work — the same lifecycle race the drain
    # exists to close, just moved past the deadline. It is retained unfinalized
    # for recovery instead. A turn outliving the window must not cost the flush
    # for anyone else, so "idle" still finalizes in the same pass.
    assert events == ["finalize:idle:compute_host_sigterm"]
    assert elapsed < wait


def test_shutdown_retains_live_sessions_within_the_stdin_closed_budget(monkeypatch):
    """The tightest real budget any caller uses is ``wait=2.0``.

    ``run_host`` finalizes through ``host.shutdown(reason="stdin_closed",
    wait=2.0)``, which is where the reserve — ``wait`` minus
    ``min(_FLUSH_RESERVE_SECS, wait / 2)`` — has the least room to work with.
    The retain-live-sessions rule must hold there without costing the flush for
    idle sessions and without pushing the call past the budget the supervisor's
    kill escalation is timed against.
    """
    wait = 2.0
    drain_budget = wait - min(compute_host._FLUSH_RESERVE_SECS, wait / 2.0)

    events: list[str] = []
    _record_finalize(monkeypatch, events, "live", "idle")

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    release = threading.Event()
    running = threading.Event()

    def _stuck_turn() -> None:
        running.set()
        release.wait(timeout=30.0)

    _register_turn(host, _stuck_turn, sid="live")
    assert running.wait(timeout=5.0)

    try:
        started = time.monotonic()
        host.shutdown(reason="stdin_closed", wait=wait)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert events == ["finalize:idle:compute_host_stdin_closed"]
    assert elapsed >= drain_budget - 1e-6, "the drain must use its full window"
    assert elapsed < wait


def test_shutdown_drain_sleep_never_overshoots_the_reserve(monkeypatch):
    """The drain's per-tick sleep must be bounded by the time left to it.

    A flat tick overshoots the drain deadline by up to one tick, eating the
    reserve held back for ``flush_all_sessions``; for a small ``wait`` that is
    the whole reserve. Asserting on the *requested* sleep totals rather than on
    wall-clock keeps this deterministic: each sleep is clamped to the remaining
    time, so the sum can never exceed the drain budget however the scheduler
    interleaves.
    """
    wait = 0.34
    drain_budget = wait - min(compute_host._FLUSH_RESERVE_SECS, wait / 2.0)

    events: list[str] = []
    _record_finalize(monkeypatch, events, "idle")

    slept: list[float] = []
    real_sleep = time.sleep

    def _recording_sleep(seconds: float) -> None:
        slept.append(seconds)
        real_sleep(seconds)

    monkeypatch.setattr(compute_host.time, "sleep", _recording_sleep)

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    release = threading.Event()
    running = threading.Event()

    def _stuck_turn() -> None:
        running.set()
        release.wait(timeout=30.0)

    _register_turn(host, _stuck_turn, sid="live")
    assert running.wait(timeout=5.0)

    try:
        host.shutdown(reason="sigterm", wait=wait)
    finally:
        release.set()

    assert events == ["finalize:idle:compute_host_sigterm"]
    assert slept, "the drain loop should have ticked at least once"
    assert sum(slept) <= drain_budget + 1e-6
