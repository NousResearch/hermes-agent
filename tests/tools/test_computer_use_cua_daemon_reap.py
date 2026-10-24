"""Startup reap of embedded cua-driver daemons orphaned by a dead Hermes owner (#126364).

Mirrors upstream PR NousResearch/hermes-agent#126378 so the local fork patch stays
rebase-friendly against the upstream fix.

On macOS the private embedded daemon is launched through LaunchServices
(``/usr/bin/open -n -g -a CuaDriver.app``), so it is parented to launchd, not to
Hermes. When Hermes dies to a signal the ``atexit`` teardown never runs and an
approval-bypassed daemon keeps serving on its socket indefinitely, keeping the
CuaDriver Accessibility / Screen Recording grants the whole time. The reap sweep
stops the daemons whose recorded owner is gone.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from tools.computer_use import cua_backend_daemon


class _RecordingBackend:
    """Stands in for the cua_backend facade; records every `_run_quiet` argv. Verbs report success
    unless the test stages `returncodes`, and `cua_daemon_listening` reports a still-answering
    daemon by default (`listening=False` makes the CLI say "not running")."""

    def __init__(self, *, returncodes=None, listening=True) -> None:
        self.calls: list[list[str]] = []
        self._returncodes = list(returncodes or [])
        # listening=None means the liveness probe itself failed to answer (inconclusive), which
        # must be distinguishable from a definite "not running" (listening=False).
        self._listening: bool | None = listening

    def _run_quiet(self, argv, **kw):
        self.calls.append(list(argv))
        returncode = self._returncodes.pop(0) if self._returncodes else 0
        return subprocess.CompletedProcess(argv, returncode)

    def cua_daemon_listening(self, driver_cmd, socket_path=None, *, timeout=3.0):
        return self._listening


def _stage_daemon(tmp_path, name, *, owner="4194304", start=None):
    """Create a socket file plus its owner marker; returns both paths. ``start=None`` writes the
    bare-pid marker format, ``start=<int>`` the pid+fingerprint format the reaper compares."""
    sock = tmp_path / name
    sock.write_bytes(b"")
    marker = tmp_path / (name + ".owner")
    payload = owner if start is None else f"{owner} {start}"
    marker.write_text(payload, encoding="utf-8")
    return sock, marker


def _reap(monkeypatch, tmp_path, sockets, *, owner_alive=None, recorder=None):
    monkeypatch.setattr(cua_backend_daemon.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cua_backend_daemon.glob, "glob", lambda pattern: list(sockets))
    # owner_alive=None runs the real `_owner_alive` (rely on the staged pid/fingerprint instead).
    if owner_alive is not None:
        monkeypatch.setattr(cua_backend_daemon, "_owner_alive", lambda pid, start: owner_alive)
    recorder = recorder or _RecordingBackend()
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


@pytest.mark.platforms("posix")
def test_reap_retains_candidate_when_stop_fails(monkeypatch, tmp_path):
    """An unconfirmed stop keeps the socket and marker so the next sweep retries.

    Custody over a still-serving daemon matters: deleting the files would strand it with no
    handle anywhere, and a later reap could never attribute it again.
    """
    sock, marker = _stage_daemon(tmp_path, "hc-stopfailed123.sock")
    recorder = _RecordingBackend(returncodes=[1], listening=True)

    _reap(monkeypatch, tmp_path, [str(sock)], owner_alive=False, recorder=recorder)

    assert recorder.calls == [["cua-driver", "stop", "--socket", str(sock)]]
    assert sock.exists()
    assert marker.exists()


@pytest.mark.platforms("posix")
def test_reap_removes_candidate_when_status_probe_confirms_death(monkeypatch, tmp_path):
    """A non-zero stop rc is not proof the daemon survived: an independent liveness probe that
    reports "not running" is."""
    sock, marker = _stage_daemon(tmp_path, "hc-stoprc1dead.sock")
    recorder = _RecordingBackend(returncodes=[1], listening=False)

    _reap(monkeypatch, tmp_path, [str(sock)], owner_alive=False, recorder=recorder)

    assert recorder.calls == [["cua-driver", "stop", "--socket", str(sock)]]
    assert not sock.exists()
    assert not marker.exists()


@pytest.mark.platforms("posix")
def test_reap_retains_candidate_when_status_probe_is_inconclusive(monkeypatch, tmp_path):
    """An unknown probe outcome (None) keeps the unconfirmed verdict, so custody errs toward retention."""
    sock, marker = _stage_daemon(tmp_path, "hc-probeunknown.sock")
    recorder = _RecordingBackend(returncodes=[1], listening=None)

    _reap(monkeypatch, tmp_path, [str(sock)], owner_alive=False, recorder=recorder)

    assert sock.exists()
    assert marker.exists()
