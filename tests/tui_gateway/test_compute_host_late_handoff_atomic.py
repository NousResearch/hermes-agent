"""Regression tests for #101824: the timeout→late-handler handoff inside
``HostSupervisor.control`` must be atomic.

Before the fix, a terminal frame that arrived in the window between the
caller's ``q.get(timeout=...)`` timing out and the pending route being
retired was routed into the (now unobserved) synchronous queue and dropped —
the late handler never fired and the caller never saw the ack. The fix
performs retire + drain + late-handler-arm under ``self._lock`` and returns
an in-flight frame synchronously (same contract as an ack that arrives
before the timeout).

The race-window case is covered deterministically: unit tests exercise the
atomic helper directly, and the interleaving tests pin terminal delivery to
each phase of the timeout→handoff sequence via hooks (send time / top of
retire / after the arm) — no wall-clock phase selection.
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


class TestDeterministicHandoffInterleaving:
    """#101824 acceptance proof, ordered by construction instead of wall clock.

    A hook on the handoff (and one on ``_send_frame``) pins terminal delivery
    to each phase of the timeout→retire→drain→arm sequence, for both
    ``control.ack`` and the generic ``error`` terminal frame:

    * before the waiter blocks (delivery at send time) → settles inline;
    * at the top of the retire, same thread (delivery wins the window) → the
      handoff drains the queued frame and settles synchronously;
    * after the arm (delivery blocked on the lock until the handoff finished)
      → the late handler fires exactly once.
    """

    @staticmethod
    def _terminal(frame_kind: str, request_id: str) -> dict:
        if frame_kind == "control.ack":
            return _ack(request_id)
        assert frame_kind in ("control.error", "error"), frame_kind
        return {"type": frame_kind, "request_id": request_id, "message": "boom"}

    @staticmethod
    def _run_control(sup: HostSupervisor, request_id: str, outcome: list, fired: list,
                     timeout: float) -> threading.Thread:
        def _run():
            try:
                outcome.append(
                    sup.control(
                        "sid",
                        route_name="session.compress",
                        payload={"command": "/compress", "request_id": request_id},
                        wait=True,
                        timeout=timeout,
                        on_late_ack=fired.append,
                    )
                )
            except queue.Empty:
                outcome.append("timeout")

        worker = threading.Thread(target=_run)
        worker.start()
        return worker

    @staticmethod
    def _assert_route_maps_clean(sup: HostSupervisor, request_id: str) -> None:
        with sup._lock:
            assert request_id not in sup._pending_controls
        assert request_id not in sup._late_control_handlers

    @pytest.mark.parametrize("frame_kind", ["control.ack", "error"])
    def test_frame_before_the_wait_settles_inline(self, frame_kind):
        sup, sent = _supervisor()
        request_id = f"r-early-{frame_kind}"
        frame = self._terminal(frame_kind, request_id)
        fired: list = []

        def _send_then_deliver(outgoing):
            sent.append(outgoing)
            sup._deliver_control_frame(request_id, frame)

        sup._send_frame = _send_then_deliver
        outcome: list = []
        self._run_control(sup, request_id, outcome, fired, timeout=5.0).join(timeout=5)

        assert outcome == [frame], "a frame delivered at send time settles inline, no late handler"
        assert fired == []
        self._assert_route_maps_clean(sup, request_id)

    @pytest.mark.parametrize("frame_kind", ["control.ack", "error"])
    def test_delivery_wins_the_window_settles_via_drain(self, frame_kind):
        sup, _sent = _supervisor()
        request_id = f"r-window-{frame_kind}"
        frame = self._terminal(frame_kind, request_id)
        fired: list = []
        real_retire = sup._retire_pending_and_arm_late

        def hooked_retire(rid, handler):
            # Same thread as the waiter and still inside control()'s
            # ``with self._lock`` (the RLock re-enters): inject the frame at
            # the top of the window, before the route is retired. Delivery
            # must land in the still-pending queue and the handoff below
            # drains it for synchronous settlement.
            sup._deliver_control_frame(rid, frame)
            return real_retire(rid, handler)

        sup._retire_pending_and_arm_late = hooked_retire
        outcome: list = []
        self._run_control(sup, request_id, outcome, fired, timeout=0.05).join(timeout=5)

        assert outcome == [frame], "delivery winning the window drains through the handoff"
        assert fired == [], "no late handler when the frame settled inline"
        self._assert_route_maps_clean(sup, request_id)

    @pytest.mark.parametrize("frame_kind", ["control.ack", "error"])
    def test_delivery_after_the_arm_routes_to_late_handler(self, frame_kind):
        sup, _sent = _supervisor()
        request_id = f"r-late-{frame_kind}"
        frame = self._terminal(frame_kind, request_id)
        fired: list = []
        window_reached = threading.Event()
        real_retire = sup._retire_pending_and_arm_late

        def hooked_retire(rid, handler):
            # The waiter holds self._lock from before this point until the
            # handoff completes, so a delivery issued after this event can
            # only acquire the lock once the route is retired — the arm wins
            # the window.
            window_reached.set()
            return real_retire(rid, handler)

        sup._retire_pending_and_arm_late = hooked_retire
        outcome: list = []
        worker = self._run_control(sup, request_id, outcome, fired, timeout=0.05)
        assert window_reached.wait(timeout=5), "waiter never reached the timeout window"
        sup._deliver_control_frame(request_id, frame)
        worker.join(timeout=5)

        assert outcome == ["timeout"], "the waiter gave up; settlement is the late handler's job"
        assert fired == [frame], "armed late handler must fire exactly once with the terminal frame"
        self._assert_route_maps_clean(sup, request_id)

    @pytest.mark.parametrize("frame_kind", ["control.ack", "control.error", "error"])
    def test_host_frame_entry_routes_terminal_frames_once(self, frame_kind):
        sup, _sent = _supervisor()
        request_id = f"r-entry-{frame_kind}"
        fired: list = []
        with pytest.raises(queue.Empty):
            sup.control(
                "sid",
                route_name="session.compress",
                payload={"command": "/compress", "request_id": request_id},
                wait=True,
                timeout=0.05,
                on_late_ack=fired.append,
            )
        sup._handle_host_frame(self._terminal(frame_kind, request_id))

        assert len(fired) == 1
        assert fired[0]["type"] == frame_kind
        assert fired[0]["request_id"] == request_id
        self._assert_route_maps_clean(sup, request_id)


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
