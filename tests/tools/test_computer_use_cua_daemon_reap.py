"""Startup reap of embedded cua-driver daemons orphaned by a dead Hermes owner (#126364)."""

from __future__ import annotations

import os

import pytest

from tools.computer_use import cua_backend_daemon


class _RecordingBackend:
    """Stands in for the cua_backend facade; records every `_run_quiet` argv."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _run_quiet(self, argv, **kw):
        self.calls.append(list(argv))


def _stage_daemon(tmp_path, name, *, owner="4194304", start=None):
    """Create a socket file plus its owner marker; returns both paths. ``start=None`` writes the
    bare-pid marker format, ``start=<int>`` the pid+fingerprint format the reaper compares."""
    sock = tmp_path / name
    sock.write_bytes(b"")
    marker = tmp_path / (name + ".owner")
    payload = owner if start is None else f"{owner} {start}"
    marker.write_text(payload, encoding="utf-8")
    return sock, marker


def _reap(monkeypatch, tmp_path, sockets, *, owner_alive=None):
    monkeypatch.setattr(cua_backend_daemon.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cua_backend_daemon.glob, "glob", lambda pattern: list(sockets))
    # owner_alive=None runs the real `_owner_alive` (rely on the staged pid/fingerprint instead).
    if owner_alive is not None:
        monkeypatch.setattr(cua_backend_daemon, "_owner_alive", lambda pid, start: owner_alive)
    recorder = _RecordingBackend()
    monkeypatch.setattr(cua_backend_daemon, "_cb", lambda: recorder)
    cua_backend_daemon._reap_orphaned_cua_daemons("cua-driver", {})
    return recorder


@pytest.mark.platforms("posix")
def test_reap_stops_daemon_whose_owner_is_gone(monkeypatch, tmp_path):
    sock, marker = _stage_daemon(tmp_path, "hc-deadowner123.sock")

    recorder = _reap(monkeypatch, tmp_path, [str(sock)], owner_alive=False)

    assert recorder.calls == [["cua-driver", "stop", "--socket", str(sock)]]
    assert not sock.exists()
    assert not marker.exists()


@pytest.mark.platforms("posix")
def test_reap_leaves_daemon_of_live_owner_alone(monkeypatch, tmp_path):
    sock, marker = _stage_daemon(tmp_path, "hc-liveowner123.sock", owner=str(os.getpid()))

    recorder = _reap(monkeypatch, tmp_path, [str(sock)], owner_alive=True)

    assert recorder.calls == []
    assert sock.exists()
    assert marker.exists()


@pytest.mark.platforms("posix")
def test_reap_never_touches_socket_without_marker(monkeypatch, tmp_path):
    # Pre-fix leftovers and daemons another instance is still starting carry no marker: fail closed.
    sock = tmp_path / "hc-unmarked123.sock"
    sock.write_bytes(b"")

    recorder = _reap(monkeypatch, tmp_path, [str(sock)], owner_alive=False)

    assert recorder.calls == []
    assert sock.exists()


@pytest.mark.platforms("posix")
def test_reap_skips_corrupt_marker(monkeypatch, tmp_path):
    sock, marker = _stage_daemon(tmp_path, "hc-corrupt123.sock")
    marker.write_text("not-a-pid", encoding="utf-8")

    recorder = _reap(monkeypatch, tmp_path, [str(sock)], owner_alive=False)

    assert recorder.calls == []
    assert sock.exists()
    assert marker.exists()


@pytest.mark.platforms("posix")
def test_reap_continues_past_one_bad_entry(monkeypatch, tmp_path):
    unmarked = tmp_path / "hc-unmarked456.sock"
    unmarked.write_bytes(b"")
    sock, marker = _stage_daemon(tmp_path, "hc-deadowner456.sock")

    recorder = _reap(monkeypatch, tmp_path, [str(unmarked), str(sock)], owner_alive=False)

    assert recorder.calls == [["cua-driver", "stop", "--socket", str(sock)]]
    assert unmarked.exists()
    assert not sock.exists() and not marker.exists()


@pytest.mark.platforms("posix")
def test_owner_pid_alive_without_fingerprint_ignores_kill_errors(monkeypatch):
    assert cua_backend_daemon._owner_alive(os.getpid(), None) is True

    def _denied(pid, sig):
        raise PermissionError(13, "denied")

    # A bare-pid marker cannot prove pid reuse, so even a denied probe stays alive.
    monkeypatch.setattr(cua_backend_daemon.os, "kill", _denied)
    assert cua_backend_daemon._owner_alive(os.getpid(), None) is True

    def _gone(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(cua_backend_daemon.os, "kill", _gone)
    assert cua_backend_daemon._owner_alive(os.getpid(), None) is False


@pytest.mark.platforms("posix")
def test_owner_alive_treats_recycled_pid_as_dead(monkeypatch):
    # The pid answers os.kill but its live fingerprint no longer matches the recorded one.
    monkeypatch.setattr(cua_backend_daemon, "_owner_start_fingerprint", lambda pid: 424242)
    assert cua_backend_daemon._owner_alive(os.getpid(), 111111) is False
    # A reading within the drift tolerance (gateway.status uses 200) is still the same incarnation.
    assert cua_backend_daemon._owner_alive(os.getpid(), 424199) is True


@pytest.mark.platforms("posix")
def test_owner_alive_degrades_to_bare_pid_when_fingerprint_unreadable(monkeypatch):
    monkeypatch.setattr(cua_backend_daemon, "_owner_start_fingerprint", lambda pid: None)
    assert cua_backend_daemon._owner_alive(os.getpid(), 111111) is True


@pytest.mark.platforms("posix")
def test_owner_alive_checks_fingerprint_when_pid_belongs_to_another_user(monkeypatch):
    # EPERM means the pid exists under a different uid; psutil still reads its start time on
    # macOS, so a mismatched fingerprint unmasks a recycled pid that landed on a root process,
    # while an unreadable fingerprint and a matching one stay fail-closed.
    def _denied(pid, sig):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(cua_backend_daemon.os, "kill", _denied)
    monkeypatch.setattr(cua_backend_daemon, "_owner_start_fingerprint", lambda pid: 424242)
    assert cua_backend_daemon._owner_alive(os.getpid(), 111111) is False
    assert cua_backend_daemon._owner_alive(os.getpid(), 424199) is True
    monkeypatch.setattr(cua_backend_daemon, "_owner_start_fingerprint", lambda pid: None)
    assert cua_backend_daemon._owner_alive(os.getpid(), 111111) is True


@pytest.mark.platforms("posix")
def test_reap_stops_daemon_whose_owner_pid_was_recycled(monkeypatch, tmp_path):
    # Marker carries a start fingerprint; the pid is live (ours) but its fingerprint differs.
    sock, marker = _stage_daemon(tmp_path, "hc-recycled123.sock", owner=str(os.getpid()), start=111111)
    monkeypatch.setattr(cua_backend_daemon, "_owner_start_fingerprint", lambda pid: 424242)

    recorder = _reap(monkeypatch, tmp_path, [str(sock)])

    assert recorder.calls == [["cua-driver", "stop", "--socket", str(sock)]]
    assert not sock.exists()
    assert not marker.exists()


@pytest.mark.platforms("posix")
def test_reap_leaves_recycled_looking_daemon_when_fingerprint_unreadable(monkeypatch, tmp_path):
    sock, marker = _stage_daemon(tmp_path, "hc-nofingerprint123.sock", owner=str(os.getpid()), start=111111)
    monkeypatch.setattr(cua_backend_daemon, "_owner_start_fingerprint", lambda pid: None)

    recorder = _reap(monkeypatch, tmp_path, [str(sock)])

    assert recorder.calls == []
    assert sock.exists()
    assert marker.exists()


@pytest.mark.platforms("posix")
def test_reap_stops_daemon_whose_owner_pid_was_recycled_by_another_user(monkeypatch, tmp_path):
    # The recycled pid landed on a root-owned process: os.kill raises EPERM, but the start
    # fingerprint is still readable and no longer matches the recorded owner.
    sock, marker = _stage_daemon(tmp_path, "hc-rootrecycled123.sock", owner="1", start=111111)

    def _denied(pid, sig):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(cua_backend_daemon.os, "kill", _denied)
    monkeypatch.setattr(cua_backend_daemon, "_owner_start_fingerprint", lambda pid: 424242)

    recorder = _reap(monkeypatch, tmp_path, [str(sock)])

    assert recorder.calls == [["cua-driver", "stop", "--socket", str(sock)]]
    assert not sock.exists()
    assert not marker.exists()


@pytest.mark.platforms("posix")
def test_write_owner_marker_records_own_pid(monkeypatch, tmp_path):
    # Fingerprint unavailable: the marker degrades to the bare-pid format (pre-upgrade layout).
    monkeypatch.setattr(cua_backend_daemon.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cua_backend_daemon._EmbeddedCuaDaemon, "_sanitized_env", lambda self: {})
    monkeypatch.setattr(cua_backend_daemon, "_owner_start_fingerprint", lambda pid: None)
    recorder = _RecordingBackend()
    monkeypatch.setattr(cua_backend_daemon, "_cb", lambda: recorder)

    daemon = cua_backend_daemon._EmbeddedCuaDaemon("cua-driver", "unrestricted")
    daemon._write_owner_marker()

    marker = tmp_path / (os.path.basename(daemon.socket_path) + ".owner")
    assert marker.read_text(encoding="utf-8").strip() == str(os.getpid())


@pytest.mark.platforms("posix")
def test_write_owner_marker_records_pid_and_start_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(cua_backend_daemon.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cua_backend_daemon._EmbeddedCuaDaemon, "_sanitized_env", lambda self: {})
    monkeypatch.setattr(cua_backend_daemon, "_owner_start_fingerprint", lambda pid: 424242)
    recorder = _RecordingBackend()
    monkeypatch.setattr(cua_backend_daemon, "_cb", lambda: recorder)

    daemon = cua_backend_daemon._EmbeddedCuaDaemon("cua-driver", "unrestricted")
    daemon._write_owner_marker()

    marker = tmp_path / (os.path.basename(daemon.socket_path) + ".owner")
    assert marker.read_text(encoding="utf-8").strip() == f"{os.getpid()} 424242"


@pytest.mark.platforms("posix")
def test_stop_removes_socket_and_owner_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(cua_backend_daemon.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cua_backend_daemon._EmbeddedCuaDaemon, "_sanitized_env", lambda self: {})
    recorder = _RecordingBackend()
    monkeypatch.setattr(cua_backend_daemon, "_cb", lambda: recorder)

    daemon = cua_backend_daemon._EmbeddedCuaDaemon("cua-driver", "unrestricted")
    socket_path = daemon.socket_path
    marker_path = cua_backend_daemon._owner_marker_path(socket_path)
    for path in (socket_path, marker_path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
    daemon._owns_runtime, daemon._process = True, None

    daemon.stop()

    assert recorder.calls == [["cua-driver", "stop", "--socket", socket_path]]
    assert not os.path.exists(socket_path)
    assert not os.path.exists(marker_path)
