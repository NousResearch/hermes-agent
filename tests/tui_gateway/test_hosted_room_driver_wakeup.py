"""Wakeup reschedule behavior of the hosted-room driver (regression for #136619)."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime


BINDING_A = HostedRoomBinding(
    room_id="room-a", gateway_id="gateway-a", authority_epoch=1
)
BINDING_B = HostedRoomBinding(
    room_id="room-b", gateway_id="gateway-a", authority_epoch=1
)


def _runtime(tmp_path: Path) -> HostedRoomRuntime:
    runtime = HostedRoomRuntime(
        db_path=tmp_path / "state.db",
        rooms=[BINDING_A, BINDING_B],
        rpc=SimpleNamespace(),
        turn_lock=lambda _profile: nullcontext(),
    )
    runtime._process_room = lambda _binding: None
    return runtime


def test_external_wakeup_marks_live_rooms_for_one_follow_up(tmp_path: Path):
    runtime = _runtime(tmp_path)
    # room-b owns a live worker slot at the moment of the room-state change, so it
    # must be revisited once its pass ends.
    runtime._room_threads[BINDING_B.room_id] = threading.current_thread()

    runtime.wakeup()

    assert runtime._wake.is_set()
    assert BINDING_B.room_id in runtime._rooms_needing_reschedule


def test_rescheduled_exit_does_not_remark_sibling_rooms(tmp_path: Path):
    runtime = _runtime(tmp_path)
    # room-a exits with a pending reschedule mark while room-b still owns its slot.
    # Waking the worker to revisit room-a must not re-mark room-b: chained re-marks
    # keep overlapping passes rescheduling each other forever (#136619 busy-loop).
    runtime._room_threads[BINDING_B.room_id] = threading.current_thread()
    runtime._rooms_needing_reschedule.add(BINDING_A.room_id)

    runtime._run_room_once(BINDING_A)

    assert runtime._wake.is_set()
    assert BINDING_A.room_id not in runtime._rooms_needing_reschedule
    assert BINDING_B.room_id not in runtime._rooms_needing_reschedule
