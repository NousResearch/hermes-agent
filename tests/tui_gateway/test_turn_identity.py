from __future__ import annotations

import threading

from tui_gateway import server


def _session():
    return {"history_lock": threading.Lock(), "inflight_turn": None}


def test_fresh_inflight_turns_get_distinct_monotonic_ids():
    session = _session()
    server._start_inflight_turn(session, "first")
    first = session["inflight_turn"]["turn_id"]
    server._start_inflight_turn(session, "second")
    second = session["inflight_turn"]["turn_id"]

    assert second > first


def test_in_place_steer_keeps_current_turn_id():
    session = _session()
    server._start_inflight_turn(session, "first")
    turn = session["inflight_turn"]
    turn_id = turn["turn_id"]
    server._record_inflight_correction(session, "steer this")

    assert session["inflight_turn"]["turn_id"] == turn_id
