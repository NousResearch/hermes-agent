"""Tests: per-profile bot turn lock (#93091 — tools/bot_relay.py).

Two deliveries into the same target profile must serialize on a
cross-process flock; the queued one waits a bounded budget and then fails
with a structured 'target_busy' refusal. Real flock on real (short)
tmp_path lockfiles — flock contends between separate fds even within one
process, so threads exercise the true kernel-lock semantics.
"""

from __future__ import annotations

try:
    import fcntl  # POSIX-only; on Windows the module is skipped wholesale
except ImportError:  # pragma: no cover - Windows
    fcntl = None
import json
import os
import sys
import threading
import time
import types

import pytest

pytestmark = pytest.mark.platforms("linux")

from tools import bot_mode_dm, bot_relay
from tools.bot_relay import TurnBusyError, acquire_turn_lock, turn_lock_path


@pytest.fixture
def root(tmp_path):
    # Keep the lockfile path SHORT (macOS-safe).
    r = tmp_path / "r"
    r.mkdir()
    return r


def _hold_flock(path, hold_event, release_event):
    """Grab the profile lock on a separate fd, signal, hold until told."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    hold_event.set()
    release_event.wait(timeout=10)
    os.close(fd)  # close releases the flock — process-death semantics


def test_second_delivery_waits_then_succeeds(root):
    held = threading.Event()
    release = threading.Event()
    t = threading.Thread(
        target=_hold_flock, args=(turn_lock_path(root, "ops"), held, release)
    )
    t.start()
    assert held.wait(timeout=5)

    # Release shortly after the waiter starts probing.
    threading.Timer(0.3, release.set).start()
    start = time.monotonic()
    with acquire_turn_lock(root, "ops", timeout_seconds=5):
        waited = time.monotonic() - start
    t.join(timeout=5)
    assert waited >= 0.2, "second delivery should have queued behind the holder"


def test_timeout_is_structured_target_busy(root):
    held = threading.Event()
    release = threading.Event()
    t = threading.Thread(
        target=_hold_flock, args=(turn_lock_path(root, "ops"), held, release)
    )
    t.start()
    assert held.wait(timeout=5)
    try:
        with pytest.raises(TurnBusyError) as excinfo:
            with acquire_turn_lock(root, "ops", timeout_seconds=0.3):
                pass  # pragma: no cover — must not acquire
        err = excinfo.value
        assert err.reason == "target_busy"
        assert err.profile == "ops"
        assert err.waited_seconds >= 0.3
    finally:
        release.set()
        t.join(timeout=5)


def test_different_profiles_do_not_contend(root):
    held = threading.Event()
    release = threading.Event()
    t = threading.Thread(
        target=_hold_flock, args=(turn_lock_path(root, "ops"), held, release)
    )
    t.start()
    assert held.wait(timeout=5)
    try:
        start = time.monotonic()
        with acquire_turn_lock(root, "scout", timeout_seconds=5):
            pass
        # Upper bound generous for loaded CI runners — the point is only
        # that 'scout' never waited the busy 'ops' budget out.
        assert time.monotonic() - start < 2.5
    finally:
        release.set()
        t.join(timeout=5)


def test_lock_released_when_holder_fd_closes(root):
    """flock dies with the holder's fd — a crashed turn can't wedge the profile."""
    path = turn_lock_path(root, "ops")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.close(fd)  # simulate holder process death (kernel releases the lock)
    with acquire_turn_lock(root, "ops", timeout_seconds=0.5):
        pass  # acquires immediately — no TurnBusyError




def test_lock_path_is_short_and_sanitized(root):
    p = turn_lock_path(root, "we/ird nam√©" + "x" * 200)
    assert p.parent == bot_relay.relay_root(root) / bot_relay.LOCKS_DIR
    assert len(p.name) <= 70
    assert "/" not in p.name.replace(".lock", "")


def test_turn_wait_seconds_falls_back_to_module_constant(monkeypatch):
    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    assert bot_relay.turn_wait_seconds() == float(bot_relay.TURN_WAIT_SECONDS_FALLBACK)


def test_turn_wait_seconds_reads_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"bot_mode": {"turn_wait_seconds": 7}},
    )
    assert bot_relay.turn_wait_seconds() == 7.0


# ── Windows backend: msvcrt instead of the old no-op (#139507) ────────────────


class _FakeMsvcrt(types.ModuleType):
    """Records locking() calls; optionally refuses probes to simulate a held lock."""

    LK_LOCK, LK_UNLCK, LK_NBLCK, LK_RLCK = 0, 1, 2, 3

    def __init__(self, refuse_probes=False):
        super().__init__("msvcrt")
        self.refuse_probes = refuse_probes
        self.calls = []

    def locking(self, fd, mode, nbytes):
        self.calls.append((fd, mode, nbytes))
        if mode == self.LK_NBLCK and self.refuse_probes:
            raise OSError(13, "Permission denied")  # what msvcrt raises while held


def _as_windows(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(bot_relay.os, "name", "nt")


def test_windows_backend_takes_a_real_msvcrt_lock(root, monkeypatch):
    """#139507: with ``fcntl`` unavailable the turn lock used to be a no-op on Windows,
    so an overlapping delivery skipped the queue entirely and died at the fail-fast
    session lease. The probe must now be a real non-blocking 1-byte ``msvcrt.locking``
    — the same backends ``hermes_cli.active_sessions._flock`` uses — unlocked on exit."""
    fake = _FakeMsvcrt()
    _as_windows(monkeypatch, fake)

    with acquire_turn_lock(root, "ops", timeout_seconds=1):
        probe = fake.calls[0] if fake.calls else None
        assert probe and probe[1:] == (fake.LK_NBLCK, 1), "1-byte non-blocking probe"
    assert fake.calls[-1][1:] == (fake.LK_UNLCK, 1), "held lock is released on exit"


def test_windows_backend_queues_then_refuses_target_busy(root, monkeypatch):
    """A held Windows lock must drive the same bounded wait → TurnBusyError path POSIX
    takes, not the old silent no-op that reported the delivery lane free."""
    fake = _FakeMsvcrt(refuse_probes=True)
    _as_windows(monkeypatch, fake)
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 0.2)

    with pytest.raises(TurnBusyError) as excinfo:
        with acquire_turn_lock(root, "ops"):
            pass  # pragma: no cover — must not acquire
    assert excinfo.value.reason == "target_busy"
    assert len(fake.calls) >= 2, "probe retried within the budget before refusing"


# ── wiring: peer transport (tools/bot_mode_dm.py) never contends locally ──────
# Retired: the local query-file CLI turn under the profile lock. Local Bot Chat
# delivery is admitted by the profile authority, which owns its own FIFO.


def test_peer_stdin_delivery_skips_local_lock(root, tmp_path, monkeypatch):
    """Peer transports run their turn on the remote gateway — no local lock."""
    home = root / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    dm = tmp_path / "dm.txt"
    dm.write_text("hi", encoding="utf-8")

    held = threading.Event()
    release = threading.Event()
    t = threading.Thread(
        target=_hold_flock, args=(turn_lock_path(home, "ops"), held, release)
    )
    t.start()
    assert held.wait(timeout=5)
    try:

        def _fake_run(argv, **kwargs):
            class _P:
                returncode = 0

            return _P()

        monkeypatch.setattr(bot_mode_dm.subprocess, "run", _fake_run)
        rc = bot_mode_dm._run_delivery(
            ["hermes", "peer", "dm", "spark/ops"], str(dm), stdin_file=True
        )
        assert rc == 0  # did not contend with the held 'ops' lock
    finally:
        release.set()
        t.join(timeout=5)


# ── wiring: relay deliver RPC (tui_gateway/methods_bot_relay.py) ─────────────
