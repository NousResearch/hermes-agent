"""Tests: the bot-chat relay WEDGE — a delivery must not hold a profile's turn lock across its
child's post-turn exit linger.

Reproduces the recurring 2026-09-20 fleet failure, where two profiles wedged each other's
delivery lanes and one instance sat for 1h36m.

The cycle, as observed: a ``bot:default`` -> ``firstmate`` delivery takes ``firstmate.lock`` and
spawns a firstmate agent session; that session's turn sends a message back to ``default``, whose
delivery needs the ``default`` session idle to inject; that session is mid-turn, so the return
never lands; the firstmate session therefore never exits; and because the delivery held
``firstmate.lock`` for the spawned session's WHOLE LIFETIME, every later delivery to firstmate
failed ``target_busy``. ``lsof`` showed both lockfiles held by processes asleep in
``poll_schedule_timeout`` and a live parent/child chain naming the cycle.

Every test drives the REAL lane (``tools.bot_mode_dm._run_delivery``) with a real child process;
only the ``hermes`` entrypoint is a stand-in that behaves like a quiet one-shot delivery turn —
it works, then reports its turn the moment the turn is over, then keeps running for the one-shot
exit linger that protects its own nested deliver replies (#113608). That linger, not the turn, is
what must not hold the lock. Nothing here reads source and no helper of the fix is imported, so
the wedge stays provable red on the un-fixed tree.

``tests/tools/test_bot_turn_lock.py`` owns the single-turn / no-double-injection safety;
``tests/tools/test_bot_relay_turn_lock_hold.py`` owns the new helper contracts.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tools import bot_mode_dm, bot_relay
from tools.bot_relay import TurnBusyError, acquire_turn_lock, turn_lock_path

# Each lane's stand-in child gives the PEER's lock this long before giving up. Reached only when
# the lanes are wedged ("unwound only when the busy session ended its turns").
_PEER_WAIT_SECONDS = 8.0
# A wedged pair spends 2 x _PEER_WAIT_SECONDS; a resolved one finishes well under this.
_ASSERT_FAST_SECONDS = 5.0


@pytest.fixture
def managed_home(tmp_path, monkeypatch):
    """Two-profile managed install (default + firstmate) under a temp HERMES_HOME."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").touch()
    d = home / "profiles" / "firstmate"
    d.mkdir(parents=True)
    (d / "profile.yaml").write_text(
        "description: teammate for tests\nui_meta:\n  hermes-bots:\n    shape: cloud\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


# A stand-in delivery turn. The lane runs `hermes -p <profile> … --query-file <dm>`, so this
# script's argv is `[shim, "-p", <profile>, <tag> <log> <turn> <linger> <peer-lock> <peer-bound>
# <counter> <mode>]`.
#   mode "ok"        : report a COMPLETED turn.
#   mode "fail-first": attempt 1 reports a FAILED turn (429 — the retry policy re-runs it) and
#                      exits 1; attempt 2 behaves like "ok".
#   mode "fail-auth" : the same, with a 401 body (never retried, so exactly one attempt).
# The turn's own work happens BEFORE the report (that is what `hermes_cli.quiet_single_query` does:
# `write_turn_report` runs after the turn). The linger AFTER the report is the exit linger whose
# nested deliver reply needs the PEER profile's lock.
_CHILD_SOURCE = """\
#!/usr/bin/env python3
import fcntl, json, os, sys, time

tag, log_path, turn, linger, peer_lock, peer_bound, counter_path, mode = sys.argv[3:11]
turn, linger, peer_bound = float(turn), float(linger), float(peer_bound)

attempt = 1
if counter_path != "-":
    try:
        attempt = int(open(counter_path, encoding="utf-8").read().strip()) + 1
    except (OSError, ValueError):
        attempt = 1
    with open(counter_path, "w", encoding="utf-8") as fh:
        fh.write(str(attempt))

def note(text):
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(text + "\\n")

BODIES = {"fail-first": "Error code: 429 - rate limit exceeded",
          "fail-auth": "Error code: 401 - invalid api key"}
failing = (mode in BODIES) and attempt == 1
body = BODIES.get(mode, "")

note("%s:turn-start" % tag)
if turn:
    time.sleep(turn)                      # the turn's own work
report = os.environ.get("HERMES_QUIET_TURN_REPORT_FILE")
if report:
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "exit_code": 1 if failing else 0, "error": body}, fh)
note("%s:turn-end" % tag)
if failing:
    sys.stderr.write(body + "\\n")
    sys.exit(1)

# The post-turn exit linger (#113608): the child protects its own nested deliver replies by
# taking the PEER profile's turn lock — its own lane still holds its own profile's lock.
if peer_lock != "-":
    fd = os.open(peer_lock, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.time() + peer_bound
    got = False
    try:
        while time.time() < deadline:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
                break
            except OSError:
                time.sleep(0.05)
    finally:
        os.close(fd)
    note("%s:peer-%s" % (tag, "lock" if got else "busy"))
    if not got:
        sys.exit(9)
if linger:
    time.sleep(linger)
note("%s:exit" % tag)
sys.exit(0)
"""


@pytest.fixture
def hermes_shim(tmp_path):
    """A ``hermes`` entrypoint stand-in for the delivery child (the lane matches it by basename)."""
    path = tmp_path / "hermes"
    path.write_text(_CHILD_SOURCE, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _lane_argv(shim: Path, profile: str, tag: str, log: Path, *, turn: float = 0.0,
               linger: float = 0.0, peer_lock: "Path | str" = "-", peer_bound: float = 0.0,
               counter: "Path | str" = "-", mode: str = "ok") -> list[str]:
    return [str(shim), "-p", profile, tag, str(log), str(turn), str(linger),
            str(peer_lock), str(peer_bound), str(counter), mode]


def _deliver(argv: list[str], dm_file: Path) -> dict:
    """One local delivery through the REAL lane; ``rc`` plus whatever the lane printed.

    A ``TurnBusyError`` escaping the lane is the lane's own refusal — reported here as
    ``rc = 1`` with ``reason: target_busy`` so every test reads one shape.
    """
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        try:
            rc = bot_mode_dm._run_delivery(argv, str(dm_file), stdin_file=False)
        except TurnBusyError as exc:
            return {"rc": 1, "payload": {"reason": exc.reason, "error": str(exc)}, "stdout": ""}
    text = buf.getvalue().strip()
    payload: dict = {}
    if text.startswith("{"):
        with contextlib.suppress(ValueError):
            payload = json.loads(text)
    return {"rc": rc, "payload": payload, "stdout": text}


def _wait_for(log: Path, marker: str, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker in _markers(log):
            return True
        time.sleep(0.05)
    return False


def _markers(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").split() if log.exists() else []


# ── the fleet symptom: `target_busy` while no turn is running ────────────────


def test_profile_is_not_busy_once_the_first_childs_turn_is_over(
        managed_home, tmp_path, monkeypatch, hermes_shim):
    """RED before the fix: the first delivery's TURN is over — its child reported it and now only
    lingers — yet a second delivery into that profile was refused ``target_busy``. That is the
    refusal the fleet saw while both lock holders sat asleep in ``poll_schedule_timeout``."""
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 1.0)
    log = tmp_path / "lane-a.log"
    dm_a, dm_b = tmp_path / "a.txt", tmp_path / "b.txt"
    dm_a.write_text("ping", encoding="utf-8")
    dm_b.write_text("pong", encoding="utf-8")

    first: dict = {}
    first_thread = threading.Thread(target=lambda: first.setdefault(
        "r", _deliver(_lane_argv(hermes_shim, "firstmate", "a", log, turn=0.3, linger=8.0), dm_a)))
    first_thread.start()
    try:
        assert _wait_for(log, "a:turn-end"), "the first delivery never reported its turn over"
        second = _deliver(_lane_argv(hermes_shim, "firstmate", "b", log), dm_b)
        assert second["rc"] == 0, (
            "a delivery into a profile whose turn is over must NOT be refused; got "
            f"{second['payload'] or second['stdout']!r}")
        assert second["payload"].get("reason") != "target_busy"
        # And the second delivery really ran its own turn (not a silently swallowed one).
        assert "b:turn-end" in _markers(log), _markers(log)
    finally:
        first_thread.join(timeout=30)
    assert first.get("r", {}).get("rc") == 0, first


# ── the cycle itself: two profiles wedging each other ────────────────────────


def test_mutual_two_profile_cycle_does_not_wedge(managed_home, tmp_path, monkeypatch, hermes_shim):
    """RED before the fix: default->firstmate and firstmate->default in flight together.

    Each lane's child, in its post-turn linger, needs the PEER's turn lock — its nested return
    delivery needs that profile's session idle — while its own lane still holds its own profile's
    lock. Circular wait: both lanes sit out the child's bounded probe.
    """
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 30.0)
    peer_lock = {
        "firstmate": turn_lock_path(managed_home, "default"),
        "default": turn_lock_path(managed_home, "firstmate"),
    }
    log = tmp_path / "cycle.log"
    outcomes: dict = {}
    dm_files = {profile: tmp_path / f"{profile}.txt" for profile in peer_lock}
    for path in dm_files.values():
        path.write_text("ping", encoding="utf-8")

    def _lane(profile: str) -> None:
        outcomes[profile] = _deliver(
            _lane_argv(hermes_shim, profile, profile, log, turn=0.5,
                       peer_lock=peer_lock[profile], peer_bound=_PEER_WAIT_SECONDS),
            dm_files[profile])

    lanes = [threading.Thread(target=_lane, args=(profile,)) for profile in peer_lock]
    start = time.monotonic()
    for t in lanes:
        t.start()
    for t in lanes:
        t.join(timeout=90)
    elapsed = time.monotonic() - start

    assert not any(t.is_alive() for t in lanes), "the two lanes deadlocked"
    assert set(outcomes) == set(peer_lock), f"a lane never reported an outcome: {outcomes}"
    wedged = {p: o for p, o in outcomes.items() if o["rc"] != 0}
    assert not wedged, (
        "each lane must drop its hold with its own turn so the peer's return delivery can land; "
        f"wedged lanes: { {p: o['payload'] or o['stdout'] for p, o in wedged.items()} } "
        f"| lane log: {_markers(log)}")
    assert elapsed < _ASSERT_FAST_SECONDS, (
        f"lanes took {elapsed:.1f}s — each waited on the other's linger instead of its own turn")
    assert not [m for m in _markers(log) if m.endswith(":peer-busy")], _markers(log)


# ── the safety this must not break (#93091) ─────────────────────────────────


def test_two_turns_into_one_profile_still_serialize(managed_home, tmp_path, monkeypatch, hermes_shim):
    """Concurrent TURNS into one profile still never overlap — no double injection."""
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 20.0)
    log = tmp_path / "serialize.log"
    dm_a, dm_b = tmp_path / "a.txt", tmp_path / "b.txt"
    dm_a.write_text("ping", encoding="utf-8")
    dm_b.write_text("pong", encoding="utf-8")

    # Lane a's TURN is 1.5s long; an un-serialized pair would interleave its markers with b's.
    first = threading.Thread(target=lambda: _deliver(
        _lane_argv(hermes_shim, "firstmate", "a", log, turn=1.5), dm_a))
    first.start()
    assert _wait_for(log, "a:turn-start"), "the first turn never started"
    second = _deliver(_lane_argv(hermes_shim, "firstmate", "b", log), dm_b)
    first.join(timeout=30)

    assert second["rc"] == 0, second
    markers = _markers(log)
    assert markers == ["a:turn-start", "a:turn-end", "a:exit",
                       "b:turn-start", "b:turn-end", "b:exit"], markers


def test_failed_turn_keeps_the_lock_for_its_retry(managed_home, tmp_path, monkeypatch, hermes_shim):
    """A FAILED attempt's turn report must not end the hold: the retry re-runs under it.

    The stand-in child fails attempt 1 (429 — the retry policy re-runs it) and, on attempt 2,
    probes its OWN profile's lock, which its lane must still hold.
    """
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 5.0)
    log = tmp_path / "retry.log"
    counter = tmp_path / "attempts"
    dm_file = tmp_path / "dm.txt"
    dm_file.write_text("ping", encoding="utf-8")

    result = _deliver(
        _lane_argv(hermes_shim, "firstmate", "retry", log, turn=0.3, peer_lock=turn_lock_path(
            managed_home, "firstmate"), peer_bound=1.0, counter=counter, mode="fail-first"),
        dm_file)

    assert result["rc"] == 0, (
        "attempt 1 fails, attempt 2 succeeds and must run under the same lock; got "
        f"{result['payload'] or result['stdout']!r}")
    markers = _markers(log)
    assert markers.count("retry:turn-start") == 2, markers
    assert "retry:peer-lock" in markers, (
        "the retry did NOT run under the lock — a concurrent turn could have injected there: "
        f"{markers}")
    assert counter.read_text(encoding="utf-8") == "2", "exactly one policy-gated re-run"


def test_auth_failure_is_never_retried(managed_home, tmp_path, monkeypatch, hermes_shim):
    """The retry budget is untouched: only classifiable transient turns get the re-run."""
    log = tmp_path / "auth.log"
    counter = tmp_path / "attempts"
    dm_file = tmp_path / "dm.txt"
    dm_file.write_text("ping", encoding="utf-8")

    result = _deliver(_lane_argv(hermes_shim, "firstmate", "auth", log, counter=counter,
                                 mode="fail-auth"), dm_file)

    assert result["rc"] == 1
    assert _markers(log) == ["auth:turn-start", "auth:turn-end"], _markers(log)
    assert counter.read_text(encoding="utf-8") == "1", "an auth failure must never be re-run"


# ── the lock file itself ─────────────────────────────────────────────────────


def test_crashed_holder_never_wedges_the_profile(tmp_path):
    """A dead holder cannot wedge anything: the kernel drops the flock with its fd."""
    root = tmp_path / "r"
    root.mkdir()
    path = turn_lock_path(root, "firstmate")
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, os, time\n"
         f"fd = os.open({str(path)!r}, os.O_RDWR | os.O_CREAT, 0o600)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\n"
         "print('held', flush=True)\n"
         "time.sleep(60)\n"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert (proc.stdout.readline() or "").strip() == "held"
        with pytest.raises(TurnBusyError):
            with acquire_turn_lock(root, "firstmate", timeout_seconds=0.3):
                pass  # pragma: no cover — the holder is alive
        proc.kill()  # no unlock, no cleanup: process death is the point
        proc.wait(timeout=10)
        with acquire_turn_lock(root, "firstmate", timeout_seconds=1.0):
            pass
    finally:
        if proc.poll() is None:  # pragma: no cover — defensive
            proc.kill()
            proc.wait(timeout=10)
