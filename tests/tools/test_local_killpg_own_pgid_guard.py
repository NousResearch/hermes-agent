"""Regression tests for #107029 — skip killpg when the tool child shares
the gateway's process group (Darwin start_new_session mitigation, #97296).

On that path, ``os.killpg(pgid, SIGTERM/SIGKILL)`` would take down the
gateway itself.  The guard must fall back to per-process kill of the
snapshotted descendants plus the wrapper, and must not fire when the
child has a distinct pgid (fail-open to the existing TERM→wait→KILL path).
"""

import os
import signal
from types import SimpleNamespace

import pytest

from tools.environments.local import LocalEnvironment, _kill_process_group_posix


pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="#107029 own-pgid guard is POSIX-only"
)


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "logs").mkdir(exist_ok=True)


class _FakeChild:
    def __init__(self, pid):
        self.pid = pid
        self.kill_calls = 0

    def kill(self):
        self.kill_calls += 1


class _FakeProc:
    def __init__(self, pid, *, hermes_pgid=None):
        self.pid = pid
        self.kill_calls = 0
        self.wait_calls = []
        if hermes_pgid is not None:
            self._hermes_pgid = hermes_pgid

    def poll(self):
        return None

    def kill(self):
        self.kill_calls += 1

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return 0


def _install_psutil_children(monkeypatch, children):
    psutil = pytest.importorskip("psutil")

    class _FakePsutilProc:
        def __init__(self, _pid):
            pass

        def children(self, recursive=True):
            return list(children)

    monkeypatch.setattr(psutil, "Process", _FakePsutilProc)


def test_kill_process_group_skips_killpg_when_pgid_is_own_pgrp(monkeypatch):
    """pgid == getpgrp → must NOT killpg; must kill proc (+ descendants)."""
    gateway_pgid = 1111
    child = _FakeChild(3333)
    proc = _FakeProc(2222)
    killpg_calls = []

    monkeypatch.setattr(os, "getpgid", lambda pid: gateway_pgid)
    monkeypatch.setattr(os, "getpgrp", lambda: gateway_pgid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))
    _install_psutil_children(monkeypatch, [child])

    _kill_process_group_posix(proc)

    assert killpg_calls == []
    assert child.kill_calls == 1
    assert proc.kill_calls == 1
    assert proc.wait_calls == [2.0]


def test_kill_process_group_still_killpg_when_pgid_distinct(monkeypatch):
    """CONTROL: distinct pgid still uses killpg — guard must not be too broad."""
    child_pgid = 2222
    gateway_pgid = 1111
    child = _FakeChild(3333)
    proc = _FakeProc(4444)
    killpg_calls = []

    monkeypatch.setattr(os, "getpgid", lambda pid: child_pgid)
    monkeypatch.setattr(os, "getpgrp", lambda: gateway_pgid)

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", fake_killpg)
    _install_psutil_children(monkeypatch, [child])

    _kill_process_group_posix(proc)

    assert killpg_calls[0] == (child_pgid, signal.SIGTERM)
    assert (child_pgid, 0) in killpg_calls
    assert proc.kill_calls == 0


def test_kill_process_group_fail_open_when_getpgrp_raises(monkeypatch):
    """getpgrp unavailable / raising must keep the old killpg path."""
    child_pgid = 2222
    proc = _FakeProc(4444)
    killpg_calls = []

    monkeypatch.setattr(os, "getpgid", lambda pid: child_pgid)

    def boom_getpgrp():
        raise OSError("getpgrp unavailable")

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "getpgrp", boom_getpgrp)
    monkeypatch.setattr(os, "killpg", fake_killpg)
    _install_psutil_children(monkeypatch, [])

    _kill_process_group_posix(proc)

    assert killpg_calls[0] == (child_pgid, signal.SIGTERM)
    assert proc.kill_calls == 0


def test_own_pgid_path_kills_proc_when_psutil_snapshot_fails(monkeypatch):
    """psutil snapshot failure → descendants=[]; own-pgid path still proc.kill()."""
    psutil = pytest.importorskip("psutil")
    gateway_pgid = 1111
    proc = _FakeProc(2222)
    killpg_calls = []

    monkeypatch.setattr(os, "getpgid", lambda pid: gateway_pgid)
    monkeypatch.setattr(os, "getpgrp", lambda: gateway_pgid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    def boom(*_a, **_k):
        raise RuntimeError("psutil exploded")

    monkeypatch.setattr(psutil, "Process", boom)

    _kill_process_group_posix(proc)  # must not raise

    assert killpg_calls == []
    assert proc.kill_calls == 1
    assert proc.wait_calls == [2.0]


def test_own_pgid_guard_uses_cached_hermes_pgid_fallback(monkeypatch):
    """Dead wrapper: _hermes_pgid == getpgrp still skips killpg."""
    gateway_pgid = 1111
    proc = _FakeProc(2222, hermes_pgid=gateway_pgid)
    killpg_calls = []

    def fake_getpgid(_pid):
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", fake_getpgid)
    monkeypatch.setattr(os, "getpgrp", lambda: gateway_pgid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))
    _install_psutil_children(monkeypatch, [])

    _kill_process_group_posix(proc)

    assert killpg_calls == []
    assert proc.kill_calls == 1


def test_own_pgid_guard_not_confused_with_getpid(monkeypatch):
    """pgid == getpid is not a substitute for getpgrp — still killpg."""
    child_pgid = 2222
    gateway_pgid = 1111
    proc = _FakeProc(child_pgid)  # pid happens to equal pgid (session leader)
    killpg_calls = []

    monkeypatch.setattr(os, "getpgid", lambda pid: child_pgid)
    monkeypatch.setattr(os, "getpgrp", lambda: gateway_pgid)
    monkeypatch.setattr(os, "getpid", lambda: child_pgid)

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", fake_killpg)
    _install_psutil_children(monkeypatch, [])

    _kill_process_group_posix(proc)

    assert killpg_calls[0] == (child_pgid, signal.SIGTERM)
    assert proc.kill_calls == 0


def test_local_environment_kill_process_routes_own_pgid_guard(monkeypatch):
    """LocalEnvironment._kill_process must honor the same own-pgid skip."""
    gateway_pgid = 1111
    proc = _FakeProc(2222)
    killpg_calls = []

    monkeypatch.setattr(os, "getpgid", lambda pid: gateway_pgid)
    monkeypatch.setattr(os, "getpgrp", lambda: gateway_pgid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))
    _install_psutil_children(monkeypatch, [])

    env = object.__new__(LocalEnvironment)
    env._kill_process(proc)

    assert killpg_calls == []
    assert proc.kill_calls == 1
