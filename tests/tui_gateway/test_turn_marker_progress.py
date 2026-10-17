"""Progress heartbeat for a recorded turn (the lease takeover's only "is it hung?" signal).

``started_at`` cannot tell a long turn from a wedged one, and from another process nothing else is
observable. These tests pin the added signal: a turn that keeps producing progress events is fresh,
a turn that stops producing them goes stale, and a touch can never invent or resurrect a turn.
"""

import json

from tui_gateway import turn_marker as tm

SID = "20260101_000000_heartbeat"


def _read(home):
    path = home / "desktop" / "interrupted_turns.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_starting_a_turn_stamps_progress(tmp_path):
    tm.record_turn_start(tmp_path, SID, "hello")
    entry = _read(tmp_path)[SID]
    assert entry["progress_at"] == entry["started_at"]


def test_touch_refreshes_progress_and_keeps_the_marker_intact(tmp_path):
    tm.record_turn_start(tmp_path, SID, "keep me", attempts=2)
    before = _read(tmp_path)[SID]

    assert tm.touch_turn_marker(tmp_path, SID, now=before["started_at"] + 60, min_interval=15) is True

    after = _read(tmp_path)[SID]
    assert after["progress_at"] == before["started_at"] + 60
    assert after["prompt"] == "keep me" and after["attempts"] == 2
    assert after["started_at"] == before["started_at"]
    assert after["writer_pid"] == before["writer_pid"]


def test_touch_is_throttled(tmp_path):
    tm.record_turn_start(tmp_path, SID, "hi")
    start = _read(tmp_path)[SID]["started_at"]
    assert tm.touch_turn_marker(tmp_path, SID, now=start + 5, min_interval=15) is False
    assert _read(tmp_path)[SID]["progress_at"] == start


def test_a_touch_never_creates_or_revives_a_marker(tmp_path):
    assert tm.touch_turn_marker(tmp_path, SID, now=1_000.0) is False
    assert not (tmp_path / "desktop" / "interrupted_turns.json").exists()

    tm.record_turn_start(tmp_path, SID, "x")
    tm.clear_turn_marker(tmp_path, SID)
    assert tm.touch_turn_marker(tmp_path, SID, now=2_000.0) is False


def test_staleness_separates_a_long_turn_from_a_hung_one(tmp_path):
    tm.record_turn_start(tmp_path, SID, "working")
    start = _read(tmp_path)[SID]["started_at"]

    # Fresh progress: a turn that has been running for an hour is NOT stale.
    tm.touch_turn_marker(tmp_path, SID, now=start + 3600, min_interval=0)
    entry = _read(tmp_path)[SID]
    assert tm.turn_marker_stale(entry, stale_after=900, now=start + 3700) is False
    # No progress for longer than the threshold: hung.
    assert tm.turn_marker_stale(entry, stale_after=900, now=start + 4501) is True


def test_a_marker_without_the_field_falls_back_to_started_at():
    entry = {"started_at": 1_000.0}
    assert tm.turn_marker_stale(entry, stale_after=100, now=1_050) is False
    assert tm.turn_marker_stale(entry, stale_after=100, now=1_101) is True
    assert tm.turn_marker_stale(None, stale_after=100, now=1_101) is False
