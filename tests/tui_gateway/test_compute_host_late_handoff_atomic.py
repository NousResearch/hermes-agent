"""Regression tests for #101824: the timeout→late-handler handoff inside
``HostSupervisor.control`` must be atomic.

Before the fix, a terminal frame that arrived in the window between the
caller's ``q.get(timeout=...)`` timing out and the pending route being
retired was routed into the (now unobserved) synchronous queue and dropped —
the late handler never fired and the caller never saw the ack. The fix
performs retire + drain + late-handler-arm under ``self._lock`` and returns
an in-flight frame synchronously (same contract as an ack that arrives
before the timeout).

The race-window case is covered deterministically by unit-testing the atomic
helper (a queued frame is drained and returned, never lost), plus a
multi-phase integration sweep asserting the frame is consumed exactly once
at every delivery phase.
"""

import queue
import sys
import threading

import pytest

from tui_gateway.host_supervisor import HostSupervisor


def _supervisor() -> tuple[HostSupervisor, list]:
    sup = HostSupervisor(argv=[sys.executable, "-c", ""], autostart=False)
    sent: list = []
    sup._send_frame = lambda frame: sent.append(frame)
    sup.start = lambda: None  # never spawn a child
    return sup, sent


def _ack(request_id: str) -> dict:
    return {"type": "control.ack", "request_id": request_id, "result": {"status": "ok"}}


class TestRetirePendingAndArmLate:
    """Direct tests of the atomic helper — the race window, deterministically."""

    def test_queued_frame_is_drained_and_returned(self):
        sup, _sent = _supervisor()
        fired: list = []
        q: queue.Queue = queue.Queue(maxsize=1)
        ack = _ack("r1")
        q.put(ack)
        with sup._lock:
            sup._pending_controls["r1"] = q
            settled = sup._retire_pending_and_arm_late("r1", fired.append)

        assert settled is ack, "a frame already in the queue must settle synchronously"
        assert fired == [], "no late handler when the frame settled inline"
        assert "r1" not in sup._pending_controls
        assert "r1" not in sup._late_control_handlers

    def test_empty_queue_arms_late_handler(self):
        sup, _sent = _supervisor()
        fired: list = []
        with sup._lock:
            sup._pending_controls["r1"] = queue.Queue(maxsize=1)
            settled = sup._retire_pending_and_arm_late("r1", fired.append)

        assert settled is None
        assert "r1" in sup._late_control_handlers
        assert "r1" not in sup._pending_controls

        sup._deliver_control_frame("r1", _ack("r1"))
        assert len(fired) == 1, "armed late handler must fire exactly once"


class TestExactlyOnceAcrossPhases:
    """At every delivery phase the frame is consumed exactly once."""

    @pytest.mark.parametrize("deliver_after_secs", [0.01, 0.25])
    def test_ack_delivered_at_any_phase_is_consumed_once(self, deliver_after_secs):
        sup, sent = _supervisor()
        fired: list = []
        result: list = []

        def _run_control():
            try:
                result.append(
                    sup.control(
                        "sid",
                        route_name="session.compress",
                        payload={"command": "/compress"},
                        wait=True,
                        timeout=0.1,
                        on_late_ack=fired.append,
                    )
                )
            except queue.Empty:
                result.append("timeout")

        worker = threading.Thread(target=_run_control)
        worker.start()
        # Give the control call a moment to register the pending route, then
        # deliver at the chosen phase (before / after the 0.1s timeout).
        import time as _time

        _time.sleep(deliver_after_secs)
        request_id = sent[0]["request_id"]
        sup._handle_host_frame(_ack(request_id))
        worker.join(timeout=5)
        assert not worker.is_alive()

        consumed = (result == [_ack(request_id)]) + len(fired)
        assert consumed == 1, (
            "the terminal frame must be consumed exactly once across the "
            "timeout→late-handler handoff (#101824)"
        )
        with sup._lock:
            assert request_id not in sup._pending_controls
        assert request_id not in sup._late_control_handlers

    def test_error_frame_handoff_matches_ack_semantics(self):
        sup, sent = _supervisor()
        fired: list = []
        with pytest.raises(queue.Empty):
            sup.control(
                "sid",
                route_name="session.compress",
                wait=True,
                timeout=0.05,
                on_late_ack=fired.append,
            )
        request_id = sent[0]["request_id"]
        sup._handle_host_frame(
            {"type": "control.error", "request_id": request_id, "message": "boom"}
        )
        assert len(fired) == 1
        assert fired[0]["type"] == "control.error"
