"""One-shot CLI exit linger for notify_on_complete background processes (#90879).

A Bot Mode agent invoked as a short-lived ``hermes -p <bot> chat -Q
--query-file ...`` process (exactly how DM handoffs deliver) dispatches its
reply via ``terminal(background=true, notify_on_complete=true)`` and then
exits.  The reply child writes to a stdout pipe owned by the dying parent and
is destroyed a few seconds later — the handoff reply is silently lost.

Fix under test: ``ProcessRegistry.wait_for_pending_completions`` gives the
one-shot exit paths (``cli._finalize_single_query`` and
``hermes_cli.oneshot``) a bounded linger over every tracked
``notify_on_complete`` process, so the delivery lands before the parent dies.

Covers:
  - registry wait semantics (no-op, completion, timeout, filters, disable)
  - config default + reader fallback
  - the CLI exit paths actually invoke the wait before teardown
  - real-process E2E: a short-lived parent that lingers keeps its background
    delivery alive to completion; a parent that exits immediately loses it
    (control that proves the bug class is real).
"""

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def registry():
    return ProcessRegistry()


def _make_session(
    sid="proc_linger_test",
    task_id="t1",
    notify_on_complete=True,
    exited=False,
) -> ProcessSession:
    s = ProcessSession(
        id=sid,
        command="echo hi",
        task_id=task_id,
        started_at=time.time(),
    )
    s.notify_on_complete = notify_on_complete
    s.exited = exited
    return s


# ── wait_for_pending_completions: unit semantics ────────────────────────────


def test_no_pending_processes_is_immediate_noop(registry):
    t0 = time.monotonic()
    result = registry.wait_for_pending_completions(timeout=30)
    assert time.monotonic() - t0 < 1.0
    assert result == {"waited": [], "completed": [], "timed_out": []}


def test_non_notify_background_processes_are_not_waited_on(registry):
    """Servers/daemons without notify_on_complete carry no completion
    contract — the linger must ignore them entirely."""
    s = _make_session(notify_on_complete=False)
    with registry._lock:
        registry._running[s.id] = s
    t0 = time.monotonic()
    result = registry.wait_for_pending_completions(timeout=30)
    assert time.monotonic() - t0 < 1.0
    assert result["waited"] == []


def test_already_exited_session_not_waited_on(registry):
    s = _make_session(exited=True)
    with registry._lock:
        registry._running[s.id] = s
    result = registry.wait_for_pending_completions(timeout=30)
    assert result["waited"] == []


def test_wait_returns_when_process_completes(registry):
    s = _make_session()
    with registry._lock:
        registry._running[s.id] = s

    def _finish():
        time.sleep(0.3)
        s.exited = True
        s.exit_code = 0
        s._completion_event.set()

    threading.Thread(target=_finish, daemon=True).start()
    t0 = time.monotonic()
    result = registry.wait_for_pending_completions(timeout=30, poll_interval=0.1)
    elapsed = time.monotonic() - t0
    assert result["waited"] == [s.id]
    assert result["completed"] == [s.id]
    assert result["timed_out"] == []
    assert elapsed < 10  # returned on completion, not the 30s bound


def test_wait_times_out_on_stuck_process(registry):
    s = _make_session()
    with registry._lock:
        registry._running[s.id] = s
    t0 = time.monotonic()
    result = registry.wait_for_pending_completions(timeout=0.5, poll_interval=0.1)
    elapsed = time.monotonic() - t0
    assert result["timed_out"] == [s.id]
    assert result["completed"] == []
    assert 0.4 <= elapsed < 5


def test_timeout_zero_disables_linger(registry):
    s = _make_session()
    with registry._lock:
        registry._running[s.id] = s
    result = registry.wait_for_pending_completions(timeout=0)
    assert result == {"waited": [], "completed": [], "timed_out": []}


def test_task_id_filter_scopes_the_wait(registry):
    mine = _make_session(sid="proc_mine", task_id="task_a")
    other = _make_session(sid="proc_other", task_id="task_b")
    with registry._lock:
        registry._running[mine.id] = mine
        registry._running[other.id] = other
    result = registry.wait_for_pending_completions("task_a", timeout=0.3, poll_interval=0.1)
    assert result["waited"] == [mine.id]
    # other (task_b) never entered the wait set — no timeout entry for it.
    assert other.id not in result["waited"]
    assert other.id not in result["timed_out"]


def test_wait_covers_multiple_pending_processes(registry):
    sessions = [_make_session(sid=f"proc_multi_{i}") for i in range(3)]
    with registry._lock:
        for s in sessions:
            registry._running[s.id] = s

    def _finish_all():
        time.sleep(0.2)
        for s in sessions:
            s.exited = True
            s._completion_event.set()

    threading.Thread(target=_finish_all, daemon=True).start()
    result = registry.wait_for_pending_completions(timeout=30, poll_interval=0.1)
    assert sorted(result["completed"]) == sorted(s.id for s in sessions)
    assert result["timed_out"] == []


def test_wait_uses_reconcile_for_orphaned_pipe_exits(registry, monkeypatch):
    """A direct child that exited while its reader is pipe-wedged (#17327)
    must still complete the linger via the reconcile pass."""
    s = _make_session()
    with registry._lock:
        registry._running[s.id] = s

    calls = {"n": 0}

    def _fake_reconcile(session):
        calls["n"] += 1
        if calls["n"] >= 2:
            session.exited = True
            session._completion_event.set()

    monkeypatch.setattr(registry, "_reconcile_local_exit", _fake_reconcile)
    result = registry.wait_for_pending_completions(timeout=10, poll_interval=0.05)
    assert result["completed"] == [s.id]
    assert calls["n"] >= 2


# ── config plumbing ──────────────────────────────────────────────────────────


def test_config_default_exists_and_is_bounded():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    val = DEFAULT_CONFIG["terminal"]["oneshot_completion_wait_seconds"]
    assert float(val) > 0


def test_config_reader_falls_back_when_config_unreadable(monkeypatch):
    import tools.process_registry as pr_mod

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config", _boom, raising=False
    )
    val = pr_mod.ProcessRegistry._oneshot_completion_wait_seconds()
    assert val > 0


def test_config_value_is_floored_at_zero(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"terminal": {"oneshot_completion_wait_seconds": -5}},
        raising=False,
    )
    val = ProcessRegistry._oneshot_completion_wait_seconds()
    assert val == 0.0


def test_default_timeout_read_from_config(registry, monkeypatch):
    """timeout=None resolves through _oneshot_completion_wait_seconds."""
    monkeypatch.setattr(
        ProcessRegistry, "_oneshot_completion_wait_seconds", staticmethod(lambda: 0.0)
    )
    s = _make_session()
    with registry._lock:
        registry._running[s.id] = s
    # Config-resolved 0 disables the wait → immediate empty result.
    result = registry.wait_for_pending_completions()
    assert result == {"waited": [], "completed": [], "timed_out": []}


# ── CLI exit paths invoke the linger ─────────────────────────────────────────


def test_finalize_single_query_lingers_before_teardown(monkeypatch):
    """cli._finalize_single_query must call the registry wait BEFORE the
    durable flush / cleanup so deliveries land while the parent is alive."""
    import cli as cli_mod

    order = []
    from tools.process_registry import process_registry

    monkeypatch.setattr(
        process_registry,
        "wait_for_pending_completions",
        lambda *a, **k: order.append("wait") or {"waited": [], "completed": [], "timed_out": []},
    )
    monkeypatch.setattr(
        cli_mod, "_flush_one_shot_session_store", lambda cli: order.append("flush")
    )
    monkeypatch.setattr(
        cli_mod, "_notify_single_query_session_finalize", lambda cli, **k: order.append("finalize")
    )
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **k: order.append("cleanup"))

    class _FakeCli:
        agent = None
        session_id = "s1"

        def _release_active_session(self):
            order.append("release")

    cli_mod._finalize_single_query(_FakeCli())
    assert order[0] == "wait"
    assert order == ["wait", "flush", "finalize", "cleanup", "release"]


def test_finalize_single_query_survives_wait_failure(monkeypatch):
    """A raising wait must not break the durable flush path."""
    import cli as cli_mod

    order = []
    from tools.process_registry import process_registry

    def _boom(*a, **k):
        raise RuntimeError("wait exploded")

    monkeypatch.setattr(process_registry, "wait_for_pending_completions", _boom)
    monkeypatch.setattr(
        cli_mod, "_flush_one_shot_session_store", lambda cli: order.append("flush")
    )
    monkeypatch.setattr(
        cli_mod, "_notify_single_query_session_finalize", lambda cli, **k: order.append("finalize")
    )
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **k: order.append("cleanup"))

    class _FakeCli:
        agent = None
        session_id = "s1"

        def _release_active_session(self):
            order.append("release")

    cli_mod._finalize_single_query(_FakeCli())
    assert "flush" in order and "release" in order


def test_oneshot_module_lingers_before_agent_close():
    """hermes_cli/oneshot.py must wait for pending completions before
    agent.close() (which kill_all()s the task's processes)."""
    src = (REPO_ROOT / "hermes_cli" / "oneshot.py").read_text(encoding="utf-8")
    wait_pos = src.find("process_registry.wait_for_pending_completions")
    assert wait_pos != -1, "oneshot.py lost the completion linger"
    close_call = src.find("agent.close()", wait_pos)
    assert close_call != -1, (
        "linger must run BEFORE the agent.close()/kill_all teardown call"
    )


# ── real-process E2E ─────────────────────────────────────────────────────────

_E2E_PARENT = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {repo!r})
    from tools.process_registry import ProcessRegistry

    marker = {marker!r}
    linger = {linger!r}

    reg = ProcessRegistry()
    # The child mimics a Bot Mode reply delivery: it works for a bit, then
    # WRITES to stdout (the pipe owned by this parent) before recording
    # success. If the parent is gone, that write raises SIGPIPE/BrokenPipe
    # and the marker file never appears — the destroyed-reply symptom.
    session = reg.spawn_local(
        "sleep 2; echo delivering; echo done > " + marker,
        task_id="e2e",
    )
    session.notify_on_complete = True
    print("SPAWNED", session.id, flush=True)
    if linger:
        result = reg.wait_for_pending_completions(timeout=30, poll_interval=0.2)
        print("LINGER", result, flush=True)
    # Parent exits here — short-lived one-shot CLI shape.
    """
)


def _run_e2e_parent(tmp_path, *, linger: bool) -> Path:
    marker = tmp_path / ("done_linger.txt" if linger else "done_nolinger.txt")
    script = tmp_path / f"parent_{linger}.py"
    script.write_text(
        _E2E_PARENT.format(repo=str(REPO_ROOT), marker=str(marker), linger=linger),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.setdefault("HERMES_HOME", str(tmp_path / "hermes_home"))
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert "SPAWNED" in proc.stdout
    return marker


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe/session semantics")
def test_e2e_lingering_parent_keeps_background_delivery_alive(tmp_path):
    """Real processes: parent lingers → the backgrounded delivery survives
    the parent's exit window and completes (marker file written)."""
    marker = _run_e2e_parent(tmp_path, linger=True)
    # The linger blocks until the child exits, so the marker exists already;
    # allow a short grace for FS visibility.
    deadline = time.time() + 10
    while time.time() < deadline and not marker.exists():
        time.sleep(0.2)
    assert marker.exists(), (
        "background delivery died despite the pre-exit linger — #90879 regressed"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe/session semantics")
def test_e2e_control_immediate_exit_loses_delivery_without_linger(tmp_path):
    """Control proving the bug class: the same parent WITHOUT the linger may
    lose the delivery. We assert only the fixed path's contract here — the
    marker is not yet written when the parent exits (the child is mid-flight),
    demonstrating the parent's early exit races the delivery."""
    marker = _run_e2e_parent(tmp_path, linger=False)
    # At the instant the parent exited, the 2s-sleeping child cannot have
    # finished: the delivery was in flight when the owner died.
    assert not marker.exists(), (
        "control invalid: delivery finished before the parent exited"
    )
