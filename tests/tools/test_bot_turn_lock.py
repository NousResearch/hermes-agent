"""Tests: per-profile bot turn lock (#93091 — tools/bot_relay.py).

Two deliveries into the same target profile must serialize on a
cross-process flock; the queued one waits a bounded budget and then fails
with a structured 'target_busy' refusal. Real flock on real (short)
tmp_path lockfiles — flock contends between separate fds even within one
process, so threads exercise the true kernel-lock semantics.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time

import pytest

# `fcntl` does not exist on Windows, and an unguarded module-level import here
# aborts collection for the whole `tests/tools/` directory rather than skipping
# this one file. Skip before importing it, matching
# `tests/cli/test_termios_drift_heal.py`. There is nothing to run here anyway:
# `acquire_turn_lock()` degrades to a no-op contextmanager on Windows (no
# `fcntl`), so the contention this module asserts on cannot occur.
if sys.platform == "win32":  # pragma: no cover
    pytest.skip("bot turn lock contention is POSIX flock-only", allow_module_level=True)

import fcntl

from tools import bot_delivery_queue as delivery_queue
from tools import bot_mode_dm, bot_relay
from tools.bot_relay import (
    TurnBusyError,
    acquire_delivery_turn_lock,
    acquire_turn_lock,
    turn_lock_path,
)


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
        assert "target_busy" in str(err)
        assert re.search(r"~\d+s", str(err))  # rough wait duration surfaced
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


def test_reentry_after_clean_release(root):
    with acquire_turn_lock(root, "ops", timeout_seconds=1):
        pass
    with acquire_turn_lock(root, "ops", timeout_seconds=1):
        pass


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


# ── wiring: local teammate delivery (tools/bot_mode_dm.py) ──────────────────


def test_run_delivery_holds_profile_lock_during_turn(root, tmp_path, monkeypatch):
    """The local `hermes -p <profile>` turn runs UNDER the profile lock."""
    home = root / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    dm = tmp_path / "dm.txt"
    dm.write_text("hi", encoding="utf-8")
    observed = {}

    def _fake_run(argv, **kwargs):
        # While the turn runs, a second acquire on the same profile must fail.
        with pytest.raises(TurnBusyError):
            with acquire_turn_lock(home, "ops", timeout_seconds=0.15):
                pass  # pragma: no cover
        observed["argv"] = argv

        class _P:
            returncode = 0
            stdout = ""
            stderr = ""

        return _P()

    monkeypatch.setattr(bot_mode_dm.subprocess, "run", _fake_run)
    rc = bot_mode_dm._run_delivery(
        ["hermes", "-p", "ops", "chat"], str(dm), stdin_file=False
    )
    assert rc == 0
    assert observed["argv"][:3] == ["hermes", "-p", "ops"]
    # …and after the turn, the lock is free again.
    with acquire_turn_lock(home, "ops", timeout_seconds=0.5):
        pass


def test_turn_busy_message_says_queued_not_failed():
    """§3.1/P10(a): a held turn slot queues the delivery — the old 'NOT delivered' text is gone."""
    error = TurnBusyError("ops", 0.05)
    assert error.reason == "target_busy"
    assert error.status_detail == "target_busy: turn slot held by another turn; delivery queued"
    assert "durably" in str(error) and "Do not resend." in str(error)
    assert "NOT delivered" not in str(error) and "retry shortly" not in str(error)


def test_delivery_turn_lock_probe_is_non_blocking(root, tmp_path, monkeypatch):
    """P10(b)/§2.5 step 1: the delivery path probes with timeout 0 — a held lock raises now."""
    home = root / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 30)  # must be irrelevant
    held = threading.Event()
    release = threading.Event()
    t = threading.Thread(target=_hold_flock, args=(turn_lock_path(home, "ops"), held, release))
    t.start()
    assert held.wait(timeout=5)
    try:
        started = time.monotonic()
        with pytest.raises(TurnBusyError):
            with acquire_delivery_turn_lock(home, "ops"):
                pass
        assert time.monotonic() - started < 1.0, "the delivery probe must never wait"
    finally:
        release.set()
        t.join(timeout=5)


def test_delivery_main_reports_target_busy_json(root, tmp_path, monkeypatch, capsys):
    """A delivery that cannot take the turn slot is RETAINED: the sender gets a receipt.

    P9(b)/§3.1 — stdout carries the 23-field envelope, the exit code stays 0, and
    ``target_busy`` appears only as the receipt's ``status_detail`` (never as a reason).
    """
    home = root / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 0.2)
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
        rc = bot_mode_dm._delivery_main(
            ["--run-delivery", "query-file", str(dm), "hermes", "-p", "ops", "chat"]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["result"] == "receipt"
        assert payload["status"] == "queued"
        assert payload["status_detail"] == "target_busy: turn slot held by another turn; delivery queued"
        assert payload["busy"] is True and payload["attempts"] == 0
        assert payload["reason"] is None
        assert payload["profile"] == "ops"
        assert list(payload) == list(delivery_queue.ENVELOPE_FIELDS)  # §1.3 order, no extras
        assert "reoffer_count" not in payload  # internal bookkeeping only
    finally:
        release.set()
        t.join(timeout=5)
    assert not dm.exists(), "DM plaintext must be reclaimed even when the delivery is retained"


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


def test_local_delivery_command_never_reenters_the_lock():
    """The gateway deliver handler runs local_delivery_command ALREADY holding
    the profile lock. That argv must stay a raw hermes CLI invocation:
    routing it through the --run-delivery wrapper would make the child hit
    _delivery_lock (hermes CLI + '-p'), burn the full wait
    budget against its parent's flock, and fail every relay delivery with
    target_busy. argv[0] may be a resolved venv path (#93590) — the lock
    matcher and this assertion both go by basename."""
    from pathlib import Path

    argv = bot_relay.local_delivery_command("ops", "/tmp/q.txt")
    assert argv[1:3] == ["-p", "ops"]
    assert Path(argv[0]).name in ("hermes", "hermes.exe")
    assert "--run-delivery" not in argv
    assert not any("bot_mode_dm" in part for part in argv)


def test_relay_deliver_returns_target_busy_error(tmp_path, monkeypatch):
    import tui_gateway.server as srv

    h = tmp_path / "h"
    (h / "profiles" / "ops").mkdir(parents=True)
    (h / "profiles" / "ops" / "config.yaml").touch()  # identity marker: bare dirs are not profiles
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 0.2)

    spawned = {}

    # Deterministic spawn detection: sentinel argv from the exact factory the
    # deliver handler uses. A global subprocess.run patch also intercepts
    # unrelated gateway-init calls (git rev-parse / ls-remote in CI), so
    # never fuzzy-match argv — mark the delivery command itself.
    monkeypatch.setattr(
        bot_relay, "local_delivery_command", lambda prof, tmp: ["__delivery__", prof]
    )

    def _fake_run(argv, **kwargs):
        argv = list(argv or [])
        if argv and argv[0] == "__delivery__":
            spawned["argv"] = argv

        class _Done:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Done()

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run)

    held = threading.Event()
    release = threading.Event()
    t = threading.Thread(
        target=_hold_flock, args=(turn_lock_path(h, "ops"), held, release)
    )
    t.start()
    assert held.wait(timeout=5)
    try:
        out = srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "x"})
        assert "error" in out
        assert out["error"]["code"] == 5096
        assert "target_busy" in out["error"]["message"]
        assert out["error"]["data"]["reason"] == "target_busy"
        assert not spawned, "turn must not spawn while the profile is busy"
    finally:
        release.set()
        t.join(timeout=5)


def test_relay_deliver_serializes_then_succeeds(tmp_path, monkeypatch):
    import tui_gateway.server as srv

    h = tmp_path / "h"
    (h / "profiles" / "ops").mkdir(parents=True)
    (h / "profiles" / "ops" / "config.yaml").touch()  # identity marker: bare dirs are not profiles
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 5.0)

    class _Proc:
        returncode = 0
        stdout = "pong"
        stderr = ""

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", lambda *a, **k: _Proc())

    held = threading.Event()
    release = threading.Event()
    t = threading.Thread(
        target=_hold_flock, args=(turn_lock_path(h, "ops"), held, release)
    )
    t.start()
    assert held.wait(timeout=5)
    threading.Timer(0.3, release.set).start()
    start = time.monotonic()
    out = srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "x"})
    t.join(timeout=5)
    assert "error" not in out, out
    assert out["result"]["reply"] == "pong"
    assert time.monotonic() - start >= 0.2, "deliver should have queued"


class _WithReason(RuntimeError):
    """An exception carrying its own ``reason``. ``reason`` is a stdlib attribute on
    ``ssl.SSLError`` and ``urllib.error.URLError`` too, so a refusal must not forward whatever
    it finds there into a channel whose consumers expect a closed vocabulary."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@pytest.mark.parametrize(
    ("failure", "code", "reason"),
    [
        (TurnBusyError("ops", 0.2), 5096, "target_busy"),
        (subprocess.TimeoutExpired(["hermes"], 600), 5093, "delivery_timeout"),
        (RuntimeError("Error code: 401 - invalid api key"), 5094, "provider_auth_or_access"),
        (RuntimeError("something nobody has a rule for"), 5094, "unknown"),
        (_WithReason("CERTIFICATE_VERIFY_FAILED", "ssl handshake failed"), 5094, "unknown"),
        (_WithReason("provider_quota_limit", "quota exhausted"), 5094, "provider_quota_limit"),
    ],
    ids=["busy", "turn-timed-out", "classifiable-failure", "unclassifiable-failure",
         "reason-outside-the-vocabulary", "reason-inside-the-vocabulary"],
)
def test_every_relay_refusal_carries_its_typed_reason(tmp_path, monkeypatch, failure, code, reason):
    """`data.reason` is the only channel the Desktop forwards: it reads `error.data.reason` and puts
    it in the sender's reply file, and the sender re-classifies from free text otherwise — which can
    never produce these codes. A refusal that ships only a JSON-RPC code reaches the sending agent as
    `[reason: unknown]`, so it cannot tell "retry shortly" from an auth failure. The turn-failure
    branch already did this; these three did not."""
    import tui_gateway.server as srv

    h = tmp_path / "h"
    (h / "profiles" / "ops").mkdir(parents=True)
    (h / "profiles" / "ops" / "config.yaml").touch()  # identity marker: bare dirs are not profiles
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(bot_relay, "local_delivery_command", lambda prof, tmp: ["__delivery__", prof])

    def _raise(argv, **kwargs):
        raise failure

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _raise)

    out = srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "x"})

    assert out["error"]["code"] == code
    assert out["error"]["data"]["reason"] == reason


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (RuntimeError("Error code: 401 - invalid api key"), "provider_auth_or_access"),
        (RuntimeError("something nobody has a rule for"), "unknown"),
        (_WithReason("CERTIFICATE_VERIFY_FAILED", "ssl handshake failed"), "unknown"),
    ],
    ids=["classifiable-failure", "unclassifiable-failure", "reason-outside-the-vocabulary"],
)
def test_delivery_main_reports_every_failure_as_typed_json(tmp_path, monkeypatch, capsys, failure, reason):
    """The local lane's runner stdout IS the sender's completion notification. A failure other
    than target_busy used to reach the sender as stderr prose with no reason, so it could not
    tell an auth failure from a transient one; it now rides the same vocabulary as the relay."""
    dm = tmp_path / "dm.txt"
    dm.write_text("hi", encoding="utf-8")

    def _raise(*args, **kwargs):
        raise failure

    monkeypatch.setattr(bot_mode_dm, "_run_delivery", _raise)

    rc = bot_mode_dm._delivery_main(["--run-delivery", "query-file", str(dm), "hermes", "-p", "ops", "chat"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {"error": str(failure), "reason": reason}
