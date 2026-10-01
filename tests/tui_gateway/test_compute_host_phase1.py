import io
import json
import os
import signal
import sys
import threading
import time


from tui_gateway import compute_host, host_supervisor as hs, server
from tui_gateway.compute_host import ComputeHost, _default_workers
from tui_gateway.host_supervisor import (
    HostSupervisor,
    append_log_record,
)


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


def test_compute_host_workers_inherit_tui_pool_env(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_RPC_POOL_WORKERS", raising=False)
    monkeypatch.delenv("HERMES_COMPUTE_HOST_WORKERS", raising=False)
    default = _default_workers()

    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "11")
    assert _default_workers() == 11

    # Malformed env falls back to the same default as unset.
    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "not-an-int")
    assert _default_workers() == default


def test_compute_host_routes_relayed_response_and_lock_to_its_open_request(monkeypatch):
    """The child owns the server request's wait: a relayed client response frame resolves it in-process,
    and a relayed ``clarify.lock`` is answered with that method's result for the parent to ack."""
    from tui_gateway import server_requests
    out = io.StringIO()
    host = ComputeHost(stdout=out, heartbeat_secs=0)
    sid = "host-clarify"
    server._sessions[sid] = {"history_lock": threading.Lock()}
    req = server_requests.ServerRequest(sid, "clarify", {"question": "?"})
    with server_requests._lock:
        server_requests._open[req.id] = req
    locks = []
    monkeypatch.setitem(server._methods, "clarify.lock",
                        lambda rid, params: locks.append((rid, dict(params))) or {"result": {"status": "ok", "remaining": []}})

    try:
        host._handle_respond({"sid": sid, "request_id": "relay-lock",
                              "params": {"lock": {"request_id": req.id, "question_id": "q0", "answer": "a"}}})
        assert locks == [("relay-lock", {"request_id": req.id, "question_id": "q0", "answer": "a"})]
        assert _json_lines(out)[-1]["response"] == {"result": {"status": "ok", "remaining": []}}

        host._handle_respond({"sid": sid, "request_id": "relay-response",
                              "params": {"frame": {"jsonrpc": "2.0", "id": req.id, "result": {"answer": "yes"}}}})
        assert req.answered and req.result == {"answer": "yes"} and req.event.is_set()
        frame = _json_lines(out)[-1]
        assert frame["type"] == "respond.ack" and frame["response"]["result"] == {"status": "ok"}
    finally:
        server._sessions.pop(sid, None)
        server_requests.reset_for_tests()
        host.close()




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


class _ExitedProc:
    pid = 12345

    def __init__(self, code: int) -> None:
        self.code = code

    def wait(self) -> int:
        return self.code


def test_supervisor_reports_confirmed_cgroup_oom_as_memory_failure(tmp_path, monkeypatch):
    supervisor = HostSupervisor(registry_path=tmp_path / "host.json", argv=[sys.executable, "-c", ""], autostart=False)
    proc = _ExitedProc(-signal.SIGKILL)
    supervisor._proc = proc
    supervisor._spawn_memory_state = {"oom_kill": 4, "current": 100, "max": 1000, "path": "/slice"}
    monkeypatch.setattr(hs, "_read_cgroup_memory_state", lambda: {"oom_kill": 5, "current": 900, "max": 1000, "path": "/slice"}, raising=False)
    monkeypatch.setattr(supervisor, "_remove_registry", lambda: None)
    failures: list[tuple[str, str]] = []
    respawns: list[dict] = []
    monkeypatch.setattr(supervisor, "_fail_pending_turns", lambda *, reason, message: failures.append((reason, message)))
    monkeypatch.setattr(supervisor, "_maybe_respawn_after_crash", lambda **kw: respawns.append(kw))

    supervisor._wait_for_exit(proc)

    assert failures and failures[0][0] == "oom_kill"
    assert "exited with code -9" in failures[0][1]
    assert "oom_kill delta=1" in failures[0][1]
    assert "memory.current=900" in failures[0][1]
    assert respawns == [{"exit_reason": "oom_kill", "exit_info": {"exit_code": -9, "oom_kill_delta": 1, "memory": {"oom_kill": 5, "current": 900, "max": 1000, "path": "/slice"}}}]


def test_supervisor_plain_sigkill_without_oom_delta_stays_generic_crash(tmp_path, monkeypatch):
    supervisor = HostSupervisor(registry_path=tmp_path / "host.json", argv=[sys.executable, "-c", ""], autostart=False)
    proc = _ExitedProc(-signal.SIGKILL)
    supervisor._proc = proc
    supervisor._spawn_memory_state = {"oom_kill": 4, "current": 100, "max": 1000, "path": "/slice"}
    monkeypatch.setattr(hs, "_read_cgroup_memory_state", lambda: {"oom_kill": 4, "current": 900, "max": 1000, "path": "/slice"}, raising=False)
    monkeypatch.setattr(supervisor, "_remove_registry", lambda: None)
    failures: list[tuple[str, str]] = []
    respawns: list[dict] = []
    monkeypatch.setattr(supervisor, "_fail_pending_turns", lambda *, reason, message: failures.append((reason, message)))
    monkeypatch.setattr(supervisor, "_maybe_respawn_after_crash", lambda **kw: respawns.append(kw))

    supervisor._wait_for_exit(proc)

    assert failures == [("crash", "compute host exited with code -9")]
    assert respawns == [{"exit_reason": "crash", "exit_info": {"exit_code": -9}}]


def test_supervisor_suppresses_oom_respawn_while_cgroup_still_above_threshold(tmp_path, monkeypatch):
    supervisor = HostSupervisor(registry_path=tmp_path / "host.json", argv=[sys.executable, "-c", ""], autostart=False)
    spawns: list[dict] = []
    monkeypatch.setattr(supervisor, "_spawn_locked", lambda **kw: spawns.append(kw))
    monkeypatch.setattr(supervisor, "_wait_until_oom_pressure_recedes", lambda _info: False, raising=False)

    supervisor._maybe_respawn_after_crash(exit_reason="oom_kill", exit_info={"memory": {"current": 900, "max": 1000, "path": "/slice"}})

    assert spawns == []
    assert supervisor._stopped_respawning is True


def test_supervisor_respawns_after_confirmed_oom_only_after_pressure_recedes(tmp_path, monkeypatch):
    supervisor = HostSupervisor(registry_path=tmp_path / "host.json", argv=[sys.executable, "-c", ""], autostart=False)
    spawns: list[dict] = []
    monkeypatch.setattr(supervisor, "_spawn_locked", lambda **kw: spawns.append(kw))
    monkeypatch.setattr(supervisor, "_wait_until_oom_pressure_recedes", lambda _info: True, raising=False)

    supervisor._maybe_respawn_after_crash(exit_reason="oom_kill", exit_info={"memory": {"current": 100, "max": 1000, "path": "/slice"}})

    assert spawns == [{"reason": "oom_recovery"}]
    assert supervisor._stopped_respawning is False


def test_supervisor_oom_respawn_has_separate_finite_budget(tmp_path, monkeypatch):
    supervisor = HostSupervisor(registry_path=tmp_path / "host.json", argv=[sys.executable, "-c", ""], autostart=False)
    supervisor._oom_restart_times = [time.monotonic()]
    spawns: list[dict] = []
    monkeypatch.setattr(supervisor, "_spawn_locked", lambda **kw: spawns.append(kw))
    monkeypatch.setattr(supervisor, "_wait_until_oom_pressure_recedes", lambda _info: True, raising=False)

    supervisor._maybe_respawn_after_crash(exit_reason="oom_kill", exit_info={"memory": {"current": 100, "max": 1000}})

    assert spawns == []
    assert supervisor._stopped_respawning is True


def test_supervisor_refuses_manual_start_while_oom_recovery_is_pending(tmp_path, monkeypatch):
    supervisor = HostSupervisor(registry_path=tmp_path / "host.json", argv=[sys.executable, "-c", ""], autostart=False)
    supervisor._oom_respawn_pending = True
    spawns: list[dict] = []
    monkeypatch.setattr(supervisor, "_spawn_locked", lambda **kw: spawns.append(kw))

    try:
        supervisor.start()
    except RuntimeError as exc:
        assert "deferred after a cgroup OOM" in str(exc)
    else:
        raise AssertionError("start() must not bypass a pending OOM recovery gate")

    assert spawns == []


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
    clock = [100.0]
    monkeypatch.setattr(compute_host.time, "monotonic", lambda: clock[0])

    def _recording_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

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
