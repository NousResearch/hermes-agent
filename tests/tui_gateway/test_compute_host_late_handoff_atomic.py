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


class TestDeliveryUnderLock:
    """#101824 review follow-up: routing AND delivery must share one critical
    section. If ``put_nowait`` ran after the lock was released, the waiter's
    atomic handoff could retire the route in between and the frame would land
    in a detached queue. Probing the lock from inside the (patched) put proves
    delivery executes while the lock is held."""

    def test_delivery_put_executes_under_lock(self):
        sup, _sent = _supervisor()
        q: queue.Queue = queue.Queue(maxsize=1)
        with sup._lock:
            sup._pending_controls["r1"] = q

        original_put_nowait = q.put_nowait
        lock_held_during_put: list = []

        def _spying_put(frame):
            # RLock is re-entrant for the holding thread, so the lock state
            # must be probed from a DIFFERENT thread: while delivery holds
            # the RLock, a foreign-thread acquire(blocking=False) fails.
            probe = []

            def _probe():
                got = sup._lock.acquire(blocking=False)
                probe.append(got)
                if got:
                    sup._lock.release()

            probe_thread = threading.Thread(target=_probe)
            probe_thread.start()
            probe_thread.join(timeout=5)
            lock_held_during_put.append(not probe[0])
            original_put_nowait(frame)

        q.put_nowait = _spying_put
        sup._deliver_control_frame("r1", _ack("r1"))

        assert q.get_nowait()["request_id"] == "r1"
        assert lock_held_during_put == [True], (
            "put_nowait must execute while self._lock is held — otherwise the "
            "waiter's atomic retire+drain+arm handoff can interleave after "
            "route selection and drop the frame into a detached queue"
        )

    def test_delivery_after_retire_routes_to_late_handler(self):
        sup, _sent = _supervisor()
        q: queue.Queue = queue.Queue(maxsize=1)
        fired: list = []
        # The waiter's handoff already ran: pending retired, late armed.
        with sup._lock:
            sup._late_control_handlers["r1"] = (0.0, fired.append)
        sup._deliver_control_frame("r1", _ack("r1"))

        assert q.empty(), "a retired pending route must never receive frames"
        assert len(fired) == 1
        assert sup._late_control_handlers == {}
