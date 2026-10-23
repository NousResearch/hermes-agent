"""Private sends stay bounded and tied to their original child even across replacement."""
from types import SimpleNamespace
import threading

import pytest

from tui_gateway.host_supervisor import HostSupervisor


def supervisor(tmp_path, stdin):
    host = HostSupervisor(autostart=False, registry_path=tmp_path / "registry.json",
                          expected_build_sha="test", expected_hermes_home=str(tmp_path))
    host._proc = SimpleNamespace(poll=lambda: None, stdin=stdin)
    host._hello = {"boot_id": "original"}
    return host


def test_blocked_pipe_has_bounded_admission_and_never_reroutes_old_frames(tmp_path):
    entered, release, drained = threading.Event(), threading.Event(), threading.Event()
    old_writes, new_writes = [], []
    def write(line):
        entered.set()
        assert release.wait(10)
        old_writes.append(line)
        if len(old_writes) == 9:
            drained.set()
    host = supervisor(tmp_path, SimpleNamespace(write=write, flush=lambda: None))
    try:
        host.conditional_send("original", {"action": "abort"})
        assert entered.wait(5)
        # The pipe is blocked while the gateway's supervisor lock remains usable.
        with host._lock:
            for _ in range(8):
                host.conditional_send("original", {"action": "abort"})
            with pytest.raises(RuntimeError, match="full"):
                host.conditional_send("original", {"action": "abort"})
            assert len(host._pending_controls) == 9  # rejected admission leaves no waiter
            host._proc = SimpleNamespace(poll=lambda: None,
                stdin=SimpleNamespace(write=new_writes.append, flush=lambda: None))
            host._hello = {"boot_id": "replacement"}
        with pytest.raises(RuntimeError, match="lifetime changed"):
            host.conditional_send("original", {"action": "commit"})
        release.set()
        assert drained.wait(5)
        assert len(old_writes) == 9 and not new_writes
    finally:
        release.set()
        host._close_conditional_writer()


def test_partial_pipe_failure_is_unconfirmed_and_cannot_start_a_child(tmp_path, monkeypatch):
    writes = []
    def partial_write(line):
        writes.append(line)
        raise OSError("pipe failed after accepting bytes")
    host = supervisor(tmp_path, SimpleNamespace(write=partial_write, flush=lambda: None))
    monkeypatch.setattr(host, "start", lambda: pytest.fail("conditional query started a child"))
    try:
        ticket = host.conditional_send("original", {"action": "commit"})
        assert host.conditional_receive(ticket)["error"] == 5019
        assert len(writes) == 1 and not host._pending_controls
    finally:
        host._close_conditional_writer()
