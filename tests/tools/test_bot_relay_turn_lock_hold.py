"""Tests: the per-profile turn lock's hold semantics and holder record (#93091 follow-up).

The turn lock serializes delivery TURNS. Two shapes follow from that and are pinned here:

* The hold can end EARLY, with the turn (``TurnLockHandle.release`` / ``watch_turn_end``), so the
  quiet one-shot delivery child's post-turn exit linger (#113608) — the window in which the
  spawned session was still alive and the fleet saw ``target_busy`` with no turn running — no
  longer keeps the lock.
* The holder writes who it is and since when into the lock file, so a refusal can say the holder
  is WEDGED rather than busy past the ceiling of a legitimate turn. The fleet had to reconstruct
  that from ``lsof`` and ``/proc`` process trees by hand (2026-09-20).

The wedge itself is proven end to end in ``tests/tools/test_bot_relay_wedge.py``.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time

import pytest

from tools import bot_relay
from tools.bot_relay import (
    TURN_LOCK_MAX_HOLD_SECONDS, TURN_LOCK_STALE_MARGIN_SECONDS, TURN_LOCK_WEDGED_AFTER_SECONDS,
    TurnBusyError, acquire_turn_lock, turn_completed, turn_lock_path, turn_report_env,
    turn_report_path, watch_turn_end,
)
from hermes_cli.quiet_single_query import TURN_REPORT_FILE_ENV


@pytest.fixture
def root(tmp_path):
    # Keep the lockfile path SHORT (macOS-safe), like the other turn-lock tests.
    r = tmp_path / "r"
    r.mkdir()
    return r


def _hold_flock(path, acquired_at: float, pid: int = 424242, wedged_after: float | None = None):
    """A live holder's shape: the flock on one fd plus its advisory record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    record = {"pid": pid, "profile": "ops", "acquired_at": acquired_at}
    if wedged_after is not None:
        record["wedged_after"] = wedged_after
    payload = json.dumps(record).encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, payload)
    return fd


# ── the hold ends with the turn ──────────────────────────────────────────────


def test_release_is_idempotent_and_frees_the_lock(root):
    with acquire_turn_lock(root, "ops", timeout_seconds=1) as handle:
        handle.release()
        handle.release()  # a second release must be a no-op, not an error
        assert handle.released
        with acquire_turn_lock(root, "ops", timeout_seconds=1):
            pass  # free again while the outer block is still open


def test_released_flag_is_false_while_held(root):
    with acquire_turn_lock(root, "ops", timeout_seconds=1) as handle:
        assert not handle.released
    assert handle.released


def test_watch_turn_end_releases_on_a_completed_report(root, tmp_path):
    dm_file = str(tmp_path / "dm.txt")
    report = turn_report_path(dm_file)
    with acquire_turn_lock(root, "ops", timeout_seconds=1) as handle:
        stop = watch_turn_end(handle, report, poll_seconds=0.02)
        try:
            time.sleep(0.1)
            assert not handle.released, "no report yet — the hold must stand"
            report.write_text(json.dumps({"pid": os.getpid(), "exit_code": 0, "error": ""}),
                              encoding="utf-8")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not handle.released:
                time.sleep(0.02)
            assert handle.released
        finally:
            stop.set()


def test_watch_turn_end_ignores_a_failed_report(root, tmp_path):
    """The retry re-runs the same session, so a failed attempt must not free the lock."""
    dm_file = str(tmp_path / "dm.txt")
    report = turn_report_path(dm_file)
    with acquire_turn_lock(root, "ops", timeout_seconds=1) as handle:
        stop = watch_turn_end(handle, report, poll_seconds=0.02)
        try:
            report.write_text(json.dumps({"pid": os.getpid(), "exit_code": 1, "error": "429"}),
                              encoding="utf-8")
            time.sleep(0.3)
            assert not handle.released
        finally:
            stop.set()


def test_watch_turn_end_releases_when_the_retry_completes(root, tmp_path):
    """After the re-run's report says completed, the hold ends — the retry is not a wedge."""
    dm_file = str(tmp_path / "dm.txt")
    report = turn_report_path(dm_file)
    with acquire_turn_lock(root, "ops", timeout_seconds=1) as handle:
        stop = watch_turn_end(handle, report, poll_seconds=0.02)
        try:
            report.write_text(json.dumps({"pid": os.getpid(), "exit_code": 1, "error": "429"}),
                              encoding="utf-8")
            time.sleep(0.2)
            report.write_text(json.dumps({"pid": os.getpid(), "exit_code": 0, "error": ""}),
                              encoding="utf-8")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not handle.released:
                time.sleep(0.02)
            assert handle.released
        finally:
            stop.set()


def test_stop_event_ends_the_watcher_without_releasing(root, tmp_path):
    dm_file = str(tmp_path / "dm.txt")
    report = turn_report_path(dm_file)
    with acquire_turn_lock(root, "ops", timeout_seconds=1) as handle:
        stop = watch_turn_end(handle, report, poll_seconds=0.02)
        stop.set()
        report.write_text(json.dumps({"pid": os.getpid(), "exit_code": 0, "error": ""}),
                          encoding="utf-8")
        time.sleep(0.3)
        assert not handle.released, "a stopped watcher must not release"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"pid": 1, "exit_code": 0, "error": ""}, True),
        ({"pid": 1, "exit_code": 1, "error": "boom"}, False),
        ({"pid": 1}, True),                       # absent exit_code reads as a clean turn
        ({"exit_code": "0"}, True),               # a CLI-written string still reads
        ({"exit_code": None}, True),
        ([1, 2, 3], False),                       # not an object
        ("not json", False),
    ],
)
def test_turn_completed_reads_the_report_contract(tmp_path, payload, expected):
    report = tmp_path / "x.turn.json"
    report.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    assert turn_completed(report) is expected


def test_turn_completed_is_false_for_a_missing_or_unreadable_report(tmp_path):
    assert turn_completed(tmp_path / "absent.turn.json") is False
    directory = tmp_path / "adir"
    directory.mkdir()
    assert turn_completed(directory) is False


def test_turn_report_path_matches_the_cron_and_quiet_child_contract(tmp_path):
    """One path for both lanes — the shape ``cron/scheduler_delivery`` already uses."""
    dm_file = tmp_path / "hermes-relay-dm-abc.txt"
    assert turn_report_path(dm_file) == dm_file.with_name(dm_file.name + ".turn.json")
    assert turn_report_env(dm_file) == {TURN_REPORT_FILE_ENV: str(turn_report_path(dm_file))}


# ── the holder record ────────────────────────────────────────────────────────


def test_holder_record_is_written_after_the_flock_is_taken(root):
    path = turn_lock_path(root, "ops")
    with acquire_turn_lock(root, "ops", timeout_seconds=1):
        record = json.loads(path.read_text(encoding="utf-8"))
    assert record["pid"] == os.getpid()
    assert record["profile"] == "ops"
    assert record["acquired_at"] <= time.time()


def test_holder_record_only_names_the_CURRENT_holder(root):
    """The record is advisory; exclusivity is the flock. A later holder overwrites it."""
    with acquire_turn_lock(root, "ops", timeout_seconds=1):
        pass
    with acquire_turn_lock(root, "ops", timeout_seconds=1):
        record = json.loads(turn_lock_path(root, "ops").read_text(encoding="utf-8"))
    assert record["pid"] == os.getpid()


def test_refusal_names_a_wedged_holder_past_the_ceiling(root):
    fd = _hold_flock(turn_lock_path(root, "ops"),
                     acquired_at=time.time() - (TURN_LOCK_WEDGED_AFTER_SECONDS + 900))
    try:
        with pytest.raises(TurnBusyError) as excinfo:
            with acquire_turn_lock(root, "ops", timeout_seconds=0.2):
                pass  # pragma: no cover
    finally:
        os.close(fd)

    err = excinfo.value
    assert err.reason == "target_busy"
    assert err.holder["pid"] == 424242
    message = str(err)
    assert "wedged" in message
    assert str(int(TURN_LOCK_WEDGED_AFTER_SECONDS)) in message
    assert "424242" in message


def test_a_lane_declares_its_own_wedged_budget(root):
    """A hosted-room turn outlives a delivery turn: its hold must not read as wedged."""
    room_budget = TURN_LOCK_WEDGED_AFTER_SECONDS + 5000
    with acquire_turn_lock(root, "ops", timeout_seconds=1, wedged_after_seconds=room_budget):
        record = json.loads(turn_lock_path(root, "ops").read_text(encoding="utf-8"))
        assert record["wedged_after"] == room_budget
    # ...and a peer's refusal honours that declared budget instead of the delivery default.
    held_for = TURN_LOCK_WEDGED_AFTER_SECONDS + 100
    fd = _hold_flock(turn_lock_path(root, "ops"), acquired_at=time.time() - held_for,
                     wedged_after=room_budget)
    try:
        with pytest.raises(TurnBusyError) as excinfo:
            with acquire_turn_lock(root, "ops", timeout_seconds=0.2):
                pass  # pragma: no cover
        assert "wedged" not in str(excinfo.value)
    finally:
        os.close(fd)


def test_wedged_threshold_sits_above_the_whole_turn_budget(root):
    """Past the ceiling but within slack: slow, not wedged."""
    for age in (TURN_LOCK_MAX_HOLD_SECONDS, TURN_LOCK_MAX_HOLD_SECONDS + 1):
        fd = _hold_flock(turn_lock_path(root, "ops"), acquired_at=time.time() - age)
        try:
            with pytest.raises(TurnBusyError) as excinfo:
                with acquire_turn_lock(root, "ops", timeout_seconds=0.2):
                    pass  # pragma: no cover
            assert "wedged" not in str(excinfo.value)
        finally:
            os.close(fd)
    assert TURN_LOCK_WEDGED_AFTER_SECONDS > TURN_LOCK_MAX_HOLD_SECONDS


def test_refusal_stays_plain_within_the_ceiling(root):
    """A long-but-legitimate turn must not be reported as wedged."""
    fd = _hold_flock(turn_lock_path(root, "ops"), acquired_at=time.time())
    try:
        with pytest.raises(TurnBusyError) as excinfo:
            with acquire_turn_lock(root, "ops", timeout_seconds=0.2):
                pass  # pragma: no cover
    finally:
        os.close(fd)

    message = str(excinfo.value)
    assert "wedged" not in message
    assert message.startswith("target_busy: another delivery turn is already running")
    assert re.search(r"~\d+s", message)


def test_refusal_stays_plain_without_a_usable_record(root):
    """A holder that left no record (an older build, a raw flock) is never called wedged."""
    path = turn_lock_path(root, "ops")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(TurnBusyError) as excinfo:
            with acquire_turn_lock(root, "ops", timeout_seconds=0.2):
                pass  # pragma: no cover
        assert excinfo.value.holder == {}
        assert "wedged" not in str(excinfo.value)
    finally:
        os.close(fd)


@pytest.mark.parametrize("junk", ["", "not json", "[1, 2]", '{"acquired_at": "soon"}'])
def test_malformed_holder_records_never_raise(root, junk):
    path = turn_lock_path(root, "ops")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.write(fd, junk.encode("utf-8"))
    try:
        with pytest.raises(TurnBusyError) as excinfo:
            with acquire_turn_lock(root, "ops", timeout_seconds=0.2):
                pass  # pragma: no cover
        assert "wedged" not in str(excinfo.value)
    finally:
        os.close(fd)


def test_ceiling_is_derived_from_the_delivery_turn_budget():
    """A parallel copy would drift from the lane's real worst case."""
    assert TURN_LOCK_MAX_HOLD_SECONDS == (
        bot_relay.TURN_ATTEMPT_TIMEOUT_SECONDS * bot_relay.TURN_MAX_ATTEMPTS)


def test_turn_busy_error_keeps_its_old_constructor(root):
    """``TurnBusyError(profile, waited)`` stays valid: the holder record is keyword-only."""
    err = TurnBusyError("ops", 0.5)
    assert err.reason == "target_busy" and err.holder == {}
    assert "target_busy" in str(err)
