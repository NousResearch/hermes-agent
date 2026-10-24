"""Cross-process turn-marker OWNERSHIP: a superseded runtime must never mutate the live owner's marker.

The cross-client handoff preempts a lease only when the chat's durable turn marker proves the holder
is idle (``session_lifecycle._holder_turn_fresh`` reads ``tui_gateway.turn_marker``). A cross-process
takeover cannot interrupt the peer (``session_lifecycle._take_over_cross_runtime_lease``), so the
superseded runtime keeps running — and its normal turn end, or the user pressing Stop on the old
computer, used to clear the marker BY SESSION KEY. Two runtimes over one HERMES_HOME store the same
session key, so the superseded turn deleted the NEW owner's marker: the predicate then read a live
turn as idle and let a further client start a second writer (and if that owner crashed, its crash
auto-continue silently lost the turn).

These tests drive the REAL shared marker functions with two writers on one session key: a live
sibling process ("B", the new owner) and this process ("A", the superseded runtime). Marker mutations
must be scoped to an immutable turn/writer identity captured when the marker was created, so A's
clear, retirement and heartbeat all no-op against B's marker, while B can still retire its own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tui_gateway import server
from tui_gateway import turn_marker as tm

KEY = "20260101_000000_owner"
_A_TOKEN = "a" * 32  # the superseded runtime's own turn token — never the live owner's.

_WRITER = """
import sys, time
from pathlib import Path
from tui_gateway.turn_marker import record_turn_start

home, key = Path(sys.argv[1]), sys.argv[2]
record_turn_start(home, key, "B is working on this chat")
print("ready", flush=True)
deadline = time.time() + 120
while not (home / "writer-stop").exists() and time.time() < deadline:
    time.sleep(0.05)
"""


def _spawn_writer(home: Path, key: str) -> subprocess.Popen:
    """A real second runtime: it records the marker under its OWN pid and stays alive (mid-turn)."""
    repo_root = Path(server.__file__).resolve().parents[1]
    child = subprocess.Popen(
        [sys.executable, "-c", _WRITER, str(home), key],
        cwd=str(repo_root), env={**os.environ, "PYTHONPATH": str(repo_root)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout = child.stdout
    assert stdout is not None
    ready = stdout.readline().strip()
    if ready != "ready":
        child.kill()
        _, err = child.communicate(timeout=10)
        raise AssertionError(f"marker writer never started: {ready!r} / {err!r}")
    return child


@pytest.fixture
def writer(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    child = _spawn_writer(home, KEY)
    try:
        yield home, child
    finally:
        (home / "writer-stop").touch()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=30)


def _entry(home: Path, key: str = KEY) -> dict | None:
    return tm.read_turn_marker_entry(home, key)


def _wait_entry(home: Path, key: str = KEY, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = _entry(home, key)
        if isinstance(entry, dict):
            return entry
        time.sleep(0.05)
    raise AssertionError("sibling writer produced no marker")


def _age_progress(home: Path, seconds: float, key: str = KEY) -> float:
    """Backdate the live owner's last progress so a heartbeat touch would be due."""
    path = home / "desktop" / "interrupted_turns.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[key]["progress_at"] -= seconds
    path.write_text(json.dumps(data), encoding="utf-8")
    return float(data[key]["progress_at"])


# ── A's later turn end / Stop must not delete the newer owner's marker ────────


def test_a_superseded_clear_never_deletes_the_newer_owners_marker(writer) -> None:
    home, child = writer
    entry = _wait_entry(home)
    assert entry["writer_pid"] == child.pid

    # A ends its own (superseded) turn on the same stored session key.
    tm.clear_turn_marker(home, KEY)

    assert isinstance(_entry(home), dict), "a superseded clear deleted the live owner's marker"
    survivor = _entry(home)
    assert survivor is not None and survivor["writer_pid"] == child.pid


def test_a_superseded_turn_end_never_retires_the_newer_owners_marker(writer) -> None:
    home, _child = writer
    _wait_entry(home)
    session_a = {
        "profile_home": str(home), "session_key": KEY, "history_lock": threading.RLock(),
        "_active_turn_marker_key": KEY, "_active_turn_marker_token": _A_TOKEN,
    }

    server._retire_turn_marker(session_a, KEY)  # turn end / Stop on the superseded runtime

    assert isinstance(_entry(home), dict), "the superseded turn's retire dropped the live owner's marker"


# ── A's stale heartbeat must not refresh (or recreate) the newer owner's marker ──


def _aged_entry(home: Path, key: str = KEY) -> dict:
    entry = _entry(home, key)
    assert isinstance(entry, dict)
    return entry


def test_a_superseded_heartbeat_never_refreshes_the_newer_owners_marker(writer, monkeypatch) -> None:
    home, _child = writer
    _wait_entry(home)
    aged = _age_progress(home, 120)

    session_a = {
        "profile_home": str(home), "session_key": KEY,
        "_active_turn_marker_key": KEY, "_active_turn_marker_token": _A_TOKEN,
    }
    monkeypatch.setitem(server._sessions, "sid-owner-a", session_a)

    server._touch_turn_progress("sid-owner-a")

    assert _aged_entry(home)["progress_at"] == aged, "a superseded heartbeat refreshed the live owner's marker"


def test_a_foreign_touch_reports_false_and_leaves_progress_intact(writer) -> None:
    home, _child = writer
    _wait_entry(home)
    aged = _age_progress(home, 120)

    assert tm.touch_turn_marker(home, KEY, now=time.time(), min_interval=0) is False
    assert _aged_entry(home)["progress_at"] == aged


def test_a_heartbeat_never_creates_or_revives_an_absent_marker(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    session_a = {
        "profile_home": str(home), "session_key": KEY,
        "_active_turn_marker_key": KEY, "_active_turn_marker_token": _A_TOKEN,
    }
    monkeypatch.setitem(server._sessions, "sid-owner-a", session_a)

    server._touch_turn_progress("sid-owner-a")

    assert _entry(home) is None
    assert not (home / "desktop" / "interrupted_turns.json").exists()


# ── The consequence the marker must keep fencing: the owner is NOT idle ───────


def test_the_newer_owners_marker_still_fences_a_takeover(writer, monkeypatch) -> None:
    home, _child = writer
    _wait_entry(home)
    monkeypatch.setattr(server, "_holder_turns_may_run_isolated", lambda: False)

    # Everything a superseded runtime does to its own turn on the same stored key...
    tm.clear_turn_marker(home, KEY)
    server._retire_turn_marker(
        {"profile_home": str(home), "session_key": KEY, "history_lock": threading.RLock(),
         "_active_turn_marker_key": KEY, "_active_turn_marker_token": _A_TOKEN}, KEY)

    assert server._holder_turn_fresh({"session_id": KEY}, home) is True, (
        "the live owner's turn read as idle after a superseded runtime cleared its marker")


# ── Identity is captured at creation and legacy markers are never claimed ─────


def test_a_recorded_marker_carries_an_immutable_turn_identity(tmp_path) -> None:
    tm.record_turn_start(tmp_path, KEY, "hello")

    recorded = _entry(tmp_path)
    assert recorded is not None and recorded.get("turn_token"), (
        "a recorded marker must carry a per-turn identity token")


def test_a_legacy_identityless_marker_is_never_claimed(tmp_path) -> None:
    """An entry from a build that recorded no writer/turn identity is not proof of ownership."""
    path = tmp_path / "desktop" / "interrupted_turns.json"
    path.parent.mkdir(parents=True)
    now = time.time()
    path.write_text(json.dumps({KEY: {
        "attempts": 0, "prompt": "legacy build", "started_at": now, "progress_at": now,
        "auto_continue": True}}), encoding="utf-8")

    tm.clear_turn_marker(tmp_path, KEY)

    assert _entry(tmp_path) is not None, "missing identity was treated as ownership"


# ── The owner's own lifecycle is unchanged ───────────────────────────────────


def test_the_owner_can_still_clear_its_own_marker(tmp_path) -> None:
    tm.record_turn_start(tmp_path, KEY, "mine")

    tm.clear_turn_marker(tmp_path, KEY)

    assert _entry(tmp_path) is None


def test_the_owner_retires_its_own_marker_through_its_turn_token(tmp_path) -> None:
    tm.record_turn_start(tmp_path, KEY, "mine")
    token = (_entry(tmp_path) or {}).get("turn_token") or _A_TOKEN
    session = {
        "profile_home": str(tmp_path), "session_key": KEY, "history_lock": threading.RLock(),
        "_active_turn_marker_key": KEY, "_active_turn_marker_token": token,
    }

    server._retire_turn_marker(session, KEY)

    assert _entry(tmp_path) is None
