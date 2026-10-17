"""Progress heartbeat for a recorded turn (the lease takeover's only "is it hung?" signal).

``started_at`` cannot tell a long turn from a wedged one, and from another process nothing else is
observable. These tests pin the added signal: a turn that keeps producing progress events is fresh,
a turn that stops producing them goes stale, and a touch can never invent or resurrect a turn.
"""

import json
import threading

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


def test_a_real_progress_event_refreshes_the_marker(tmp_path, monkeypatch):
    """The wiring contract: a turn that emits a tool step proves it is alive.

    Without this the marker stays at its start stamp, a long turn looks hung, and the handoff would
    steal a chat from a working turn.
    """
    from tui_gateway import server

    home = tmp_path / "home"
    tm.record_turn_start(home, "chat-hb", "work")
    path = home / "desktop" / "interrupted_turns.json"

    # Age the stamp past the throttle so a touch is due, exactly like a turn running for a while.
    data = json.loads(path.read_text(encoding="utf-8"))
    data["chat-hb"]["progress_at"] -= 30
    path.write_text(json.dumps(data), encoding="utf-8")
    before = json.loads(path.read_text(encoding="utf-8"))["chat-hb"]["progress_at"]

    session = {
        "session_key": "chat-hb", "source": "desktop", "profile_home": str(home),
        "history_lock": threading.Lock(), "running": True,
        "_active_turn_marker_key": "chat-hb",
    }
    monkeypatch.setitem(server._sessions, "sid-hb", session)

    server._agent_cbs("sid-hb")["tool_start_callback"]("tc-1", "terminal", {})

    after = json.loads(path.read_text(encoding="utf-8"))["chat-hb"]["progress_at"]
    assert after > before + 25, "a tool step must refresh the progress stamp"
