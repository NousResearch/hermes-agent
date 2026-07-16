import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


def _stdout_queue(proc: subprocess.Popen) -> queue.Queue[dict]:
    out: queue.Queue[dict] = queue.Queue()
    assert proc.stdout is not None

    def drain() -> None:
        for line in proc.stdout or []:
            out.put(json.loads(line))

    threading.Thread(target=drain, daemon=True).start()
    return out


def _read_json_line(out: queue.Queue[dict], timeout: float = 2.0) -> dict:
    try:
        return out.get(timeout=timeout)
    except queue.Empty as exc:
        raise AssertionError("timed out waiting for compute host JSON") from exc


def test_compute_host_line_json_seed_turn_interrupt():
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tui_gateway.compute_host"],
        cwd=str(repo),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    out = _stdout_queue(proc)
    try:
        hello = _read_json_line(out)
        assert hello["type"] == "hello"
        assert hello["host_pid"] == proc.pid

        proc.stdin.write(json.dumps({"type": "session.seed", "sid": "s1", "request_id": "seed"}) + "\n")
        proc.stdin.flush()
        assert _read_json_line(out)["type"] == "session.seeded"

        proc.stdin.write(
            json.dumps(
                {
                    "type": "turn.start",
                    "sid": "s1",
                    "request_id": "turn",
                    "prompt": "hello",
                    "delta_count": 3,
                    "delay_s": 0,
                }
            )
            + "\n"
        )
        proc.stdin.flush()

        seen = []
        while True:
            frame = _read_json_line(out)
            seen.append(frame["type"])
            if frame["type"] == "turn.end":
                assert frame["history_version"] == 1
                assert frame["message_count"] == 2
                break
        assert seen.count("delta") == 3

        proc.stdin.write(json.dumps({"type": "shutdown", "request_id": "stop"}) + "\n")
        proc.stdin.flush()
        assert _read_json_line(out)["type"] == "shutdown.ack"
        proc.wait(timeout=2)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_shutdown_drains_pending_observer_deliveries(monkeypatch):
    """The compute host owns live agent turns, so its durability shutdown
    (SIGTERM/orphan) must give accepted final-result observations a bounded
    chance to deliver before daemon workers die with the process."""
    import io
    import time

    import hermes_cli.plugins as plugins_module
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tui_gateway.compute_host import ComputeHost

    manager = PluginManager()
    received: list[dict] = []
    done = threading.Event()

    def _slow_listener(*, event):
        time.sleep(0.2)
        received.append(event)
        done.set()

    context = PluginContext(PluginManifest(name="compute-host-observer"), manager)
    context.register_hook("post_agent_result", _slow_listener)
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    assert manager.emit_observer_hook(
        "post_agent_result", {"event": "agent.result", "sequence": 1}
    )
    host.shutdown(reason="sigterm", wait=0.1)

    # Drain completed inside shutdown: the slow callback already ran and the
    # listener generation is retired, so nothing lingers past the flush.
    assert done.is_set()
    assert received and received[0]["sequence"] == 1
    assert manager.has_active_observer_hook("post_agent_result") is False


def test_shutdown_without_plugin_state_is_a_noop(monkeypatch):
    """Observer drain must fail open: no plugin manager, no effect."""
    import io

    import hermes_cli.plugins as plugins_module
    from tui_gateway.compute_host import ComputeHost

    monkeypatch.setattr(plugins_module, "_plugin_manager", None)
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    host.shutdown(reason="sigterm", wait=0.1)
    assert plugins_module._plugin_manager is None


def test_shutdown_frame_drains_observers_before_reader_hard_exit(monkeypatch):
    """The explicit supervisor `shutdown` frame hard-exits via os._exit in the
    reader — bypassing shutdown()/atexit — so the frame path itself must own a
    bounded drain, ordered ack → drain → `_closed` → hard exit.

    Ownership matters: if `_closed` were signaled before the drain, run_host's
    main loop would wake and race a second drain; the non-owning side (main
    return or the reader's os._exit) could then kill the process while the
    owner was still delivering. The listener therefore records, at each
    delivery, that `_closed` was still unsignaled and the hard exit had not
    fired — proving the frame-owned drain (not the main-thread finally) is
    what delivered."""
    import io
    import time

    import hermes_cli.plugins as plugins_module
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tui_gateway import compute_host

    manager = PluginManager()
    stdout = io.StringIO()
    done = threading.Event()
    ack_written = threading.Event()
    deliveries: list[dict] = []
    hosts: list[compute_host.ComputeHost] = []
    exit_calls: list[int] = []

    class _CapturingHost(compute_host.ComputeHost):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            hosts.append(self)

        def emit(self, frame):
            super().emit(frame)
            if frame.get("type") == "shutdown.ack":
                ack_written.set()

    def _listener(*, event):
        # Delivery of sequence 1 may begin the instant it is emitted, before
        # the shutdown frame exists — so it gates deterministically on the
        # host writing the ack instead of a scheduler sleep. Sequence 2 stays
        # queued behind it, forcing the drain to wait for queued AND
        # in-flight work. All ordering facts are sampled at callback
        # COMPLETION and asserted outside (observer exceptions are
        # intentionally swallowed by the worker).
        sequence = event.get("sequence")
        ack_wait_ok = True
        if sequence == 1:
            ack_wait_ok = ack_written.wait(timeout=2)
        host = hosts[0] if hosts else None
        deliveries.append(
            {
                "sequence": sequence,
                "ack_wait_ok": ack_wait_ok,
                "closed_at_completion": bool(host and host._closed.is_set()),
                "exit_called_at_completion": bool(exit_calls),
                "ack_visible_at_completion": "shutdown.ack" in stdout.getvalue(),
                "done_ns": time.perf_counter_ns(),
            }
        )
        if sequence == 2:
            done.set()

    context = PluginContext(PluginManifest(name="frame-shutdown-observer"), manager)
    context.register_hook("post_agent_result", _listener)
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)

    real_exit = os._exit

    def _fake_exit(code: int) -> None:
        # Record and return (no SystemExit): the reader's next `_closed`
        # check breaks its loop cleanly, so no unhandled-thread-exception
        # warning muddies the run.
        exit_calls.append(time.perf_counter_ns())

    monkeypatch.setattr(compute_host.os, "_exit", _fake_exit)
    monkeypatch.setattr(compute_host, "ComputeHost", _CapturingHost)
    try:

        def _frames():
            # Emitted on the reader thread immediately before the shutdown
            # frame: sequence 1 goes in flight, sequence 2 stays queued.
            assert manager.emit_observer_hook(
                "post_agent_result", {"event": "agent.result", "sequence": 1}
            )
            assert manager.emit_observer_hook(
                "post_agent_result", {"event": "agent.result", "sequence": 2}
            )
            yield json.dumps({"type": "shutdown", "request_id": "frame-shutdown"}) + "\n"

        compute_host.run_host(stdin=_frames(), stdout=stdout)
    finally:
        monkeypatch.setattr(compute_host.os, "_exit", real_exit)

    assert done.is_set(), "queued observation was not delivered by the drain"
    assert [d["sequence"] for d in deliveries] == [1, 2]
    # Frame-owned ordering, sampled at each callback completion: the ack was
    # already written, `_closed` was still unsignaled (so run_host's main
    # loop — and its finally-shutdown — had not woken), and the reader's
    # hard exit had not fired. The frame-owned drain, not the main-thread
    # finally, is what delivered.
    assert all(d["ack_wait_ok"] for d in deliveries)
    assert all(d["ack_visible_at_completion"] for d in deliveries)
    assert all(not d["closed_at_completion"] for d in deliveries)
    assert all(not d["exit_called_at_completion"] for d in deliveries)
    assert exit_calls, "reader hard-exit seam was not reached"
    assert exit_calls[0] >= deliveries[-1]["done_ns"]


def test_concurrent_shutdown_cannot_return_before_owning_drain_delivers(monkeypatch):
    """Orphan-like race: the parent-guard daemon calls shutdown() (then
    os._exit) while shutdown()'s `_closed.set()` has already woken run_host's
    main finally into a concurrent shutdown(). Only one caller can own the
    runtime's detach-and-wait; the other must boundedly wait for the owner —
    otherwise the hard-exit side returns instantly and os._exit kills the
    owning drain mid-delivery. Proven with a gate-controlled listener: the
    second shutdown caller's return implies the accepted callback already
    completed."""
    import io
    import time

    import hermes_cli.plugins as plugins_module
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tui_gateway.compute_host import ComputeHost

    manager = PluginManager()
    gate = threading.Event()
    delivered = threading.Event()

    def _gated_listener(*, event):
        gate.wait(timeout=2)
        delivered.set()

    context = PluginContext(PluginManifest(name="concurrent-shutdown-observer"), manager)
    context.register_hook("post_agent_result", _gated_listener)
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    assert manager.emit_observer_hook(
        "post_agent_result", {"event": "agent.result", "sequence": 1}
    )

    records: dict[str, bool] = {}
    waiter_entered = threading.Event()
    guard_returned = threading.Event()

    # Instrument the host's done-event wait: only the NON-owner branch of
    # _drain_result_observers calls it, so waiter_entered proves the guard
    # genuinely entered the bounded wait (a non-waiting regression could
    # otherwise pass whenever the scheduler delayed the guard past delivery).
    real_done_wait = host._observer_drain_done.wait

    def _instrumented_wait(timeout=None):
        waiter_entered.set()
        return real_done_wait(timeout)

    monkeypatch.setattr(host._observer_drain_done, "wait", _instrumented_wait)

    def _owner_shutdown():  # run_host main-finally analogue
        host.shutdown(reason="stdin_closed", wait=0.1)
        records["owner_saw_delivery"] = delivered.is_set()

    def _guard_shutdown():  # parent-guard hard-exit analogue
        host.shutdown(reason="orphan", wait=0.1)
        records["guard_saw_delivery"] = delivered.is_set()
        guard_returned.set()

    owner = threading.Thread(target=_owner_shutdown, name="owner-shutdown")
    guard = threading.Thread(target=_guard_shutdown, name="guard-shutdown")

    owner.start()
    # Deterministically let the owner claim the drain before the guard races
    # in: the drain lock is held only while the owning drain runs.
    deadline = time.monotonic() + 2.0
    while not host._observer_drain_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert host._observer_drain_lock.locked(), "owner never claimed the drain"

    guard.start()
    # The guard must be observably inside the non-owner wait — and must not
    # have returned — while the listener gate is still closed.
    assert waiter_entered.wait(timeout=2), "guard never entered the non-owner wait"
    assert not guard_returned.is_set()
    assert not delivered.is_set()
    # Release the listener quickly — well inside the owner's bounded drain
    # window — and let both shutdown paths finish.
    gate.set()
    owner.join(timeout=5)
    guard.join(timeout=5)
    assert not owner.is_alive() and not guard.is_alive()

    # The contract under test: NEITHER shutdown caller — in particular the
    # hard-exit-bound guard — returned before the accepted callback completed.
    assert delivered.is_set()
    assert records.get("owner_saw_delivery") is True
    assert records.get("guard_saw_delivery") is True
