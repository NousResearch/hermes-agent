"""Orphan reaping for embedded cua-driver daemons (#126364).

On macOS the embedded daemon is launched via ``open -n -g -a CuaDriver.app`` so
LaunchServices reparents it to launchd: killing Hermes (Desktop updates/restarts)
skips atexit and leaves the daemon + socket behind. The daemon records
``<socket>.owner`` (owner PID + start-time fingerprint); the next startup reaps
stale ``hc-*.sock`` files whose owner is proven dead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest

from tools.computer_use import cua_backend_daemon as daemon_mod
from gateway import status as gateway_status


DEAD_PID = 2**30 + 12345  # PID table never reaches here -> ProcessLookupError


def _write_sock_and_marker(tmp_path, name, pid, start_time):
    sock = tmp_path / name
    sock.write_text("sock", encoding="utf-8")
    marker = tmp_path / f"{name}.owner"
    marker.write_text(json.dumps({"pid": pid, "start_time": start_time}), encoding="utf-8")
    return str(sock), str(marker)


def _patch_tempdir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        daemon_mod.tempfile, "gettempdir", lambda: str(tmp_path)
    )


def _patch_run_quiet(monkeypatch, calls, raising=None):
    def fake(argv, *, timeout=0, **kw):
        calls.append(list(argv))
        if raising is not None:
            raise raising
        m = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return m

    fake_cb = types.SimpleNamespace(_run_quiet=fake)
    monkeypatch.setattr(daemon_mod, "_cb", lambda: fake_cb)


def _make_daemon(monkeypatch, tmp_path, driver_cmd="/fake/cua-driver"):
    _patch_tempdir(monkeypatch, tmp_path)
    d = daemon_mod._EmbeddedCuaDaemon(driver_cmd, "unrestricted")
    # Keep the socket inside tmp_path even if gettempdir is restored.
    d.socket_path = str(tmp_path / f"hc-{os.getpid()}-current.sock")
    return d


# --- owner marker path -------------------------------------------------------


def test_owner_marker_path_is_socket_plus_suffix(tmp_path):
    sock = str(tmp_path / "hc-abc.sock")
    assert daemon_mod._owner_marker_path(sock) == sock + ".owner"


def test_daemon_owner_marker_property_matches_helper(monkeypatch, tmp_path):
    d = _make_daemon(monkeypatch, tmp_path)
    assert d.owner_marker_path == daemon_mod._owner_marker_path(d.socket_path)


# --- writing the marker ------------------------------------------------------


def test_write_owner_marker_records_pid_and_start_time(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    sock = str(tmp_path / "hc-write.sock")
    open(sock, "w").close()
    daemon_mod._write_owner_marker(sock)
    record = json.loads(open(sock + ".owner", encoding="utf-8").read())
    assert record["pid"] == os.getpid()
    assert record["start_time"] == gateway_status.get_process_start_time(os.getpid())


def test_write_owner_marker_never_raises(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    # Unwritable location: must not raise (startup stays up).
    daemon_mod._write_owner_marker("/definitely/not/here/hc-x.sock")


def test_write_owner_marker_skipped_on_win32(monkeypatch, tmp_path):
    stub = types.SimpleNamespace(platform="win32")
    monkeypatch.setattr(daemon_mod, "sys", stub)
    sock = str(tmp_path / "hc-win.sock")
    open(sock, "w").close()
    daemon_mod._write_owner_marker(sock)
    assert not os.path.exists(sock + ".owner")


# --- owner liveness ----------------------------------------------------------


def test_is_owner_dead_dead_pid():
    assert daemon_mod._is_owner_dead(DEAD_PID, 12345) is True


def test_is_owner_dead_live_owner_matches_fingerprint():
    pid = os.getpid()
    start = gateway_status.get_process_start_time(pid)
    assert start is not None
    assert daemon_mod._is_owner_dead(pid, start) is False


def test_is_owner_dead_recycled_pid_mismatched_fingerprint(monkeypatch):
    pid = os.getpid()
    recorded = gateway_status.get_process_start_time(pid)
    assert recorded is not None
    monkeypatch.setattr(
        gateway_status, "get_process_start_time", lambda p: int(recorded) + 100000
    )
    assert daemon_mod._is_owner_dead(pid, recorded) is True


def test_is_owner_dead_fail_closed_on_unreadable_start_time(monkeypatch):
    pid = os.getpid()
    recorded = gateway_status.get_process_start_time(pid)
    monkeypatch.setattr(gateway_status, "get_process_start_time", lambda p: None)
    assert daemon_mod._is_owner_dead(pid, recorded) is False
    # No recorded fingerprint: cannot prove recycling.
    monkeypatch.setattr(
        gateway_status, "get_process_start_time",
        lambda p: 99999999,
    )
    assert daemon_mod._is_owner_dead(pid, None) is False


def test_is_owner_dead_fail_closed_on_bad_pid():
    assert daemon_mod._is_owner_dead(None, 1) is False
    assert daemon_mod._is_owner_dead("not-a-pid", 1) is False
    # Junk fingerprint that breaks the comparator must not reap.
    assert daemon_mod._is_owner_dead(os.getpid(), "junk-{{{") is False


def test_is_owner_dead_permission_error_means_alive(monkeypatch):
    def raising(pid, sig):
        raise PermissionError("no permission")
    monkeypatch.setattr(os, "kill", raising)
    pid = os.getpid()
    start = gateway_status.get_process_start_time(pid)
    assert daemon_mod._is_owner_dead(pid, start) is False


def test_is_owner_dead_other_oserror_means_alive(monkeypatch):
    def raising(pid, sig):
        raise OSError("weird")
    monkeypatch.setattr(os, "kill", raising)
    assert daemon_mod._is_owner_dead(1234, 5678) is False


def test_is_owner_dead_uses_gateway_status_fingerprint_functions(monkeypatch):
    """The PID-reuse guard must go through gateway.status helpers."""
    seen = {}

    real_match = gateway_status.start_time_fingerprints_match

    def fake_get(pid):
        seen["get_called"] = True
        return 424242

    def fake_match(recorded, current):
        seen["match_called"] = True
        return real_match(recorded, current)

    monkeypatch.setattr(gateway_status, "get_process_start_time", fake_get)
    monkeypatch.setattr(gateway_status, "start_time_fingerprints_match", fake_match)
    # Live PID (os.kill succeeds) with a mismatched fingerprint -> dead.
    assert daemon_mod._is_owner_dead(os.getpid(), 1) is True
    assert seen.get("get_called") is True
    assert seen.get("match_called") is True


# --- reaping -----------------------------------------------------------------


def test_reap_dead_owner_stops_and_cleans(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    orphan_sock, orphan_marker = _write_sock_and_marker(
        tmp_path, "hc-orphan.sock", DEAD_PID, 111
    )
    live_start = gateway_status.get_process_start_time(os.getpid())
    live_sock, live_marker = _write_sock_and_marker(
        tmp_path, "hc-live.sock", os.getpid(), live_start
    )
    current = str(tmp_path / "hc-current.sock")
    open(current, "w").close()

    calls = []
    _patch_run_quiet(monkeypatch, calls)

    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", current, env={})

    stop_targets = [c for c in calls if c[:2] == ["/fake/cua-driver", "stop"]]
    assert any(orphan_sock in c for c in stop_targets)
    assert not any(live_sock in c for c in stop_targets)
    assert not any(current in c for c in stop_targets)
    assert not os.path.exists(orphan_sock)
    assert not os.path.exists(orphan_marker)
    assert os.path.exists(live_sock)
    assert os.path.exists(live_marker)
    assert os.path.exists(current)


def test_reap_recycled_pid_is_reaped(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    recorded = gateway_status.get_process_start_time(os.getpid())
    sock, marker = _write_sock_and_marker(
        tmp_path, "hc-recycled.sock", os.getpid(), recorded
    )
    # Same PID number, different incarnation -> owner is dead.
    monkeypatch.setattr(
        gateway_status, "get_process_start_time", lambda p: int(recorded) + 50000
    )
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", "/none/current.sock", env={})
    assert any(sock in c for c in calls)
    assert not os.path.exists(sock)
    assert not os.path.exists(marker)


def test_reap_skips_socket_without_marker(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    bare = tmp_path / "hc-bare.sock"
    bare.write_text("sock", encoding="utf-8")
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", "/none/current.sock", env={})
    assert calls == []
    assert bare.exists()


def test_reap_skips_malformed_marker(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    sock = tmp_path / "hc-broken.sock"
    sock.write_text("sock", encoding="utf-8")
    (tmp_path / "hc-broken.sock.owner").write_text("not-json{{{", encoding="utf-8")
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", "/none/current.sock", env={})
    assert calls == []
    assert sock.exists()


def test_reap_skips_current_socket_even_when_owner_dead(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    current = str(tmp_path / "hc-current.sock")
    open(current, "w").close()
    with open(current + ".owner", "w", encoding="utf-8") as fh:
        json.dump({"pid": DEAD_PID, "start_time": 1}, fh)
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", current, env={})
    assert calls == []
    assert os.path.exists(current)


def test_reap_retains_candidate_when_stop_fails(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    sock, marker = _write_sock_and_marker(tmp_path, "hc-orphan.sock", DEAD_PID, 1)
    calls = []
    _patch_run_quiet(monkeypatch, calls, raising=RuntimeError("boom"))
    # Must not raise, and must NOT unlink: a failed stop destroys the only retry
    # handle while the daemon may still be alive, so the candidate is retained
    # (no independent proof of death) for a future startup to retry.
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", "/none/current.sock", env={})
    assert os.path.exists(sock)
    assert os.path.exists(marker)


def _patch_run_quiet_per_verb(monkeypatch, calls, stop_result="ok", status_result="dead"):
    """Fake _run_quiet with independent stop/status outcomes.

    stop_result/status_result: "ok" (returncode 0), "fail" (returncode 1),
    "unknown" (return None, swallowed probe error), or an Exception to raise.
    """
    def _result(kind):
        outcome = stop_result if kind == "stop" else status_result
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "ok":
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if outcome == "fail":
            return types.SimpleNamespace(returncode=1, stdout="not running", stderr="")
        if outcome == "unknown":
            return None
        raise AssertionError(f"unknown outcome {outcome!r}")

    def fake(argv, *, timeout=0, **kw):
        calls.append(list(argv))
        kind = "stop" if len(argv) > 1 and argv[1] == "stop" else "status"
        return _result(kind)

    fake_cb = types.SimpleNamespace(_run_quiet=fake)
    monkeypatch.setattr(daemon_mod, "_cb", lambda: fake_cb)


def test_reap_cleans_when_stop_fails_but_status_confirms_dead(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    sock, marker = _write_sock_and_marker(tmp_path, "hc-orphan.sock", DEAD_PID, 1)
    calls = []
    _patch_run_quiet_per_verb(monkeypatch, calls, stop_result=RuntimeError("boom"), status_result="fail")
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", "/none/current.sock", env={})
    assert not os.path.exists(sock)
    assert not os.path.exists(marker)


def test_reap_retains_when_stop_fails_and_status_live(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    sock, marker = _write_sock_and_marker(tmp_path, "hc-orphan.sock", DEAD_PID, 1)
    calls = []
    _patch_run_quiet_per_verb(monkeypatch, calls, stop_result="fail", status_result="ok")
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", "/none/current.sock", env={})
    assert os.path.exists(sock)
    assert os.path.exists(marker)


def test_reap_retains_when_status_probe_inconclusive(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    sock, marker = _write_sock_and_marker(tmp_path, "hc-orphan.sock", DEAD_PID, 1)
    calls = []
    _patch_run_quiet_per_verb(monkeypatch, calls, stop_result="fail", status_result="unknown")
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", "/none/current.sock", env={})
    assert os.path.exists(sock)
    assert os.path.exists(marker)


def test_reap_scan_failure_never_raises(monkeypatch):
    monkeypatch.setattr(
        daemon_mod.glob, "glob", lambda *a, **k: (_ for _ in ()).throw(OSError("io"))
    )
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", "/none.sock", env={})


def test_reap_no_driver_cmd_is_noop(monkeypatch, tmp_path):
    _patch_tempdir(monkeypatch, tmp_path)
    sock, _ = _write_sock_and_marker(tmp_path, "hc-orphan.sock", DEAD_PID, 1)
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    daemon_mod._reap_orphaned_embedded_daemons("", "/none.sock", env={})
    assert calls == []
    assert os.path.exists(sock)


def test_reap_skipped_on_win32(monkeypatch, tmp_path):
    stub = types.SimpleNamespace(platform="win32")
    monkeypatch.setattr(daemon_mod, "sys", stub)
    sock, _ = _write_sock_and_marker(tmp_path, "hc-orphan.sock", DEAD_PID, 1)
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    daemon_mod._reap_orphaned_embedded_daemons("/fake/cua-driver", "/none.sock", env={})
    assert calls == []
    assert os.path.exists(sock)


# --- start()/stop() integration ----------------------------------------------


class _FakePopen:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.stderr = []
        self.terminated = False

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def communicate(self, *args, **kwargs):
        return ("", "")


def _patch_startup(monkeypatch, daemon):
    from tools.computer_use import cua_backend_driver as driver_mod

    monkeypatch.setattr(
        driver_mod, "_resolve_mcp_invocation", lambda cmd: (cmd, ["mcp"])
    )
    monkeypatch.setattr(daemon, "_sanitized_env", lambda: {})
    monkeypatch.setattr(daemon, "_socket_ready", lambda env: True)
    monkeypatch.setattr(daemon_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(daemon_mod, "_wait_or_kill", lambda proc: None)


def test_start_writes_owner_marker(monkeypatch, tmp_path):
    d = _make_daemon(monkeypatch, tmp_path)
    _patch_startup(monkeypatch, d)
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    d.start()
    try:
        assert d._running is True
        record = json.loads(open(d.owner_marker_path, encoding="utf-8").read())
        assert record["pid"] == os.getpid()
        assert record["start_time"] == gateway_status.get_process_start_time(os.getpid())
    finally:
        monkeypatch.setattr(d, "_sanitized_env", lambda: {})
        with open(d.socket_path, "w", encoding="utf-8"):
            pass
        d.stop()


def test_start_reaps_orphan_before_own_startup(monkeypatch, tmp_path):
    d = _make_daemon(monkeypatch, tmp_path)
    _patch_startup(monkeypatch, d)
    orphan_sock, orphan_marker = _write_sock_and_marker(
        tmp_path, "hc-orphan.sock", DEAD_PID, 1
    )
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    d.start()
    try:
        assert not os.path.exists(orphan_sock)
        assert not os.path.exists(orphan_marker)
        assert os.path.exists(d.owner_marker_path)
    finally:
        with open(d.socket_path, "w", encoding="utf-8"):
            pass
        d.stop()


def test_start_reap_failure_does_not_block_startup(monkeypatch, tmp_path):
    d = _make_daemon(monkeypatch, tmp_path)
    _patch_startup(monkeypatch, d)
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    monkeypatch.setattr(
        daemon_mod, "_reap_orphaned_embedded_daemons",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scan boom")),
    )
    # start() suppresses reap failures internally, so this must still come up.
    d.start()
    try:
        assert d._running is True
    finally:
        with open(d.socket_path, "w", encoding="utf-8"):
            pass
        d.stop()


def test_stop_cleans_socket_and_owner_marker(monkeypatch, tmp_path):
    d = _make_daemon(monkeypatch, tmp_path)
    _patch_startup(monkeypatch, d)
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    d.start()
    # Pretend the daemon created a real socket file.
    open(d.socket_path, "w", encoding="utf-8").close()
    assert os.path.exists(d.socket_path)
    assert os.path.exists(d.owner_marker_path)
    d.stop()
    assert not os.path.exists(d.socket_path)
    assert not os.path.exists(d.owner_marker_path)
    # Own stop verb was issued.
    assert any(d.socket_path in c for c in calls if len(c) > 2 and c[1] == "stop")


def test_stop_without_runtime_still_cleans_marker(monkeypatch, tmp_path):
    d = _make_daemon(monkeypatch, tmp_path)
    monkeypatch.setattr(d, "_sanitized_env", lambda: {})
    open(d.socket_path, "w", encoding="utf-8").close()
    with open(d.owner_marker_path, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "start_time": 1}, fh)
    d._owns_runtime = False
    d._process = None
    d.stop()
    assert not os.path.exists(d.owner_marker_path)


# --- start() prepublishes provenance -------------------------------------------


def test_start_prepublishes_owner_marker_before_launch(monkeypatch, tmp_path):
    d = _make_daemon(monkeypatch, tmp_path)
    _patch_startup(monkeypatch, d)
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    assert not os.path.exists(d.owner_marker_path)
    seen = {}

    def _RecordingPopen(*args, **kwargs):
        seen["marker_at_launch"] = os.path.exists(d.owner_marker_path)
        return _FakePopen(*args, **kwargs)

    monkeypatch.setattr(daemon_mod.subprocess, "Popen", _RecordingPopen)
    d.start()
    try:
        assert seen.get("marker_at_launch") is True
        assert os.path.exists(d.owner_marker_path)
    finally:
        with open(d.socket_path, "w", encoding="utf-8"):
            pass
        d.stop()


def test_start_cleans_prepublished_marker_when_launch_fails(monkeypatch, tmp_path):
    from tools.computer_use import cua_backend_driver as driver_mod

    d = _make_daemon(monkeypatch, tmp_path)
    monkeypatch.setattr(driver_mod, "_resolve_mcp_invocation", lambda cmd: (cmd, ["mcp"]))
    monkeypatch.setattr(d, "_sanitized_env", lambda: {})
    monkeypatch.setattr(d, "_socket_ready", lambda env: True)
    monkeypatch.setattr(daemon_mod, "_wait_or_kill", lambda proc: None)
    calls = []
    _patch_run_quiet(monkeypatch, calls)

    def _BoomPopen(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(daemon_mod.subprocess, "Popen", _BoomPopen)
    with pytest.raises(OSError, match="spawn failed"):
        d.start()
    assert not os.path.exists(d.owner_marker_path)
    assert d._process is None


def test_start_cleans_prepublished_marker_on_startup_timeout(monkeypatch, tmp_path):
    from tools.computer_use import cua_backend_driver as driver_mod

    d = _make_daemon(monkeypatch, tmp_path)
    monkeypatch.setattr(driver_mod, "_resolve_mcp_invocation", lambda cmd: (cmd, ["mcp"]))
    monkeypatch.setattr(d, "_sanitized_env", lambda: {})
    monkeypatch.setattr(d, "_socket_ready", lambda env: False)
    monkeypatch.setattr(daemon_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(daemon_mod, "_wait_or_kill", lambda proc: None)
    monkeypatch.setattr(d, "_START_TIMEOUT_SECONDS", 0.2)
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="timed out"):
        d.start()
    assert not os.path.exists(d.owner_marker_path)


class _FakeDeadPopen(_FakePopen):
    def poll(self):
        return 1


def test_start_cleans_prepublished_marker_on_early_exit(monkeypatch, tmp_path):
    from tools.computer_use import cua_backend_driver as driver_mod

    d = _make_daemon(monkeypatch, tmp_path)
    monkeypatch.setattr(driver_mod, "_resolve_mcp_invocation", lambda cmd: (cmd, ["mcp"]))
    monkeypatch.setattr(d, "_sanitized_env", lambda: {})
    monkeypatch.setattr(d, "_socket_ready", lambda env: False)
    monkeypatch.setattr(daemon_mod.subprocess, "Popen", _FakeDeadPopen)
    monkeypatch.setattr(daemon_mod, "_wait_or_kill", lambda proc: None)
    calls = []
    _patch_run_quiet(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="exited during startup"):
        d.start()
    assert not os.path.exists(d.owner_marker_path)
