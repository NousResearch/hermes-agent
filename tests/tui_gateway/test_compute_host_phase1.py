import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from tui_gateway import compute_host, server
from tui_gateway.compute_host import ComputeHost, _default_workers
from tui_gateway.host_supervisor import (
    MUTATOR_ROUTE_TABLE,
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


def test_supervisor_spawn_keeps_final_child_env_sanitized(tmp_path, monkeypatch):
    import tui_gateway.host_supervisor as supervisor_mod
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / ".env").write_text(
        "OPENAI_API_KEY=source-provider\nCUSTOM_CAP=source-cap\n",
        encoding="utf-8",
    )
    (target / ".env").write_text(
        "OPENAI_API_KEY=target-provider\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(source))
    for key, value in {
        "OPENAI_API_KEY": "source-provider",
        "CUSTOM_CAP": "source-cap",
        "GH_TOKEN": "source-github",
        "DB_PASSWORD": "source-password",
        "BWS_ACCESS_TOKEN": "source-bws",
        "OP_SERVICE_ACCOUNT_TOKEN": "source-op",
        "APPTAINERENV_SINGULARITYENV_GH_TOKEN": "source-wrapped",
    }.items():
        monkeypatch.setenv(key, value)

    captured: dict = {}

    class _Proc:
        pid = 4242
        stdin = io.StringIO()
        stdout = io.StringIO()
        stderr = io.StringIO()

    class _NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    supervisor = HostSupervisor(
        registry_path=tmp_path / "host.json",
        argv=[sys.executable, "-c", ""],
        env={"GH_TOKEN": "unsafe-constructor-override", "SAFE_OVERRIDE": "keep"},
        expected_hermes_home=str(target),
        autostart=False,
    )

    def _fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        supervisor._hello = {"boot_id": "test"}
        supervisor._hello_event.set()
        return _Proc()

    monkeypatch.setattr(supervisor_mod, "_Thread", _NoopThread)
    monkeypatch.setattr(supervisor_mod.subprocess, "Popen", _fake_popen)

    set_multiplex_active(True)
    token = set_hermes_home_override(target)
    try:
        supervisor._spawn_locked(reason="test")
    finally:
        reset_hermes_home_override(token)
        set_multiplex_active(False)

    child_env = captured["env"]
    assert child_env["OPENAI_API_KEY"] == "target-provider"
    assert child_env["SAFE_OVERRIDE"] == "keep"
    for key in (
        "CUSTOM_CAP",
        "GH_TOKEN",
        "DB_PASSWORD",
        "BWS_ACCESS_TOKEN",
        "OP_SERVICE_ACCOUNT_TOKEN",
        "APPTAINERENV_SINGULARITYENV_GH_TOKEN",
    ):
        assert key not in child_env


def test_supervisor_contextless_respawn_uses_captured_profile_owner(
    tmp_path, monkeypatch
):
    """PROC-001: crash recovery must not depend on caller ContextVars."""
    import tui_gateway.host_supervisor as supervisor_mod
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / ".env").write_text(
        "OPENAI_API_KEY=source-provider\nCUSTOM_CAP=source-cap\n",
        encoding="utf-8",
    )
    (target / ".env").write_text(
        "OPENAI_API_KEY=target-provider\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv("OPENAI_API_KEY", "source-provider")
    monkeypatch.setenv("CUSTOM_CAP", "source-cap")

    captured: dict = {}

    class _Proc:
        pid = 4243
        stdin = io.StringIO()
        stdout = io.StringIO()
        stderr = io.StringIO()

    class _NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    set_multiplex_active(True)
    target_token = set_hermes_home_override(target)
    try:
        supervisor = HostSupervisor(
            registry_path=tmp_path / "host-respawn.json",
            argv=[sys.executable, "-c", ""],
            expected_hermes_home=str(target),
            autostart=False,
        )
    finally:
        reset_hermes_home_override(target_token)

    def _fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        supervisor._hello = {
            "boot_id": "test",
            "hermes_home": str(target),
        }
        supervisor._hello_event.set()
        return _Proc()

    monkeypatch.setattr(supervisor_mod, "_Thread", _NoopThread)
    monkeypatch.setattr(supervisor_mod.subprocess, "Popen", _fake_popen)
    try:
        # Rotate the target credential after construction. The respawn must
        # rebuild temporal authority from the captured owner identities rather
        # than resurrecting the constructor's old value snapshot.
        (target / ".env").write_text(
            "OPENAI_API_KEY=rotated-target-provider\n",
            encoding="utf-8",
        )
        # Models the wait thread after a crash: multiplex mode remains active,
        # but the profile-home ContextVar was not copied into the new thread.
        supervisor._spawn_locked(reason="contextless-respawn")
    finally:
        set_multiplex_active(False)

    child_env = captured["env"]
    assert child_env["HERMES_HOME"] == str(target)
    assert child_env["OPENAI_API_KEY"] == "rotated-target-provider"
    assert "CUSTOM_CAP" not in child_env


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
