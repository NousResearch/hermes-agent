"""Generation-pinned private controls: no start, respawn, retry or late adoption."""
from __future__ import annotations

import queue
import threading
import uuid


class ConditionalWriter:
    """Bounded nonblocking admission to a pipe writer; never hold gateway locks on I/O."""
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.queue = queue.Queue(maxsize=8)
        self.closed = threading.Event()
        from agent.memory_provider import spawn_context_thread
        spawn_context_thread(self.run, name="conditional-host-writer").start()

    def send(self, proc, frame, waiter=None):
        if self.closed.is_set():
            raise RuntimeError("conditional writer stopped")
        try:
            self.queue.put_nowait((proc, frame, waiter))
        except queue.Full as exc:
            raise RuntimeError("conditional writer full") from exc

    def run(self):
        while not self.closed.is_set():
            item = self.queue.get()
            if item is None:
                break
            proc, frame, waiter = item
            try:
                self.supervisor._write_frame_to(proc, frame)
            except (OSError, RuntimeError):
                if waiter is not None:
                    try:
                        waiter.put_nowait({"boot_id": frame["boot_id"], "error": 5019})
                    except queue.Full:
                        pass  # real reply already won; never replace it

    def close(self):
        self.closed.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass  # the departed process unblocks the writer, which observes closed


class HostConditionalSupervisorMixin:
    def conditional_dispatch_boot(self):
        # Normal deliberate dispatch may wait for startup; conditional probes may not.
        with self._lock:
            return self._hello.get("boot_id") if self.is_running() and not self._closing else None

    def conditional_boot(self):
        if not self._lock.acquire(blocking=False):
            return None  # startup/settling cannot supply conditional authority
        try:
            return self._hello.get("boot_id") if self.is_running() and not self._closing else None
        finally:
            self._lock.release()

    def _conditional_send_to_current(self, boot, payload, waiter=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Conditional host settling")
        try:
            if not boot or self.conditional_boot() != boot:
                raise RuntimeError("Conditional host lifetime changed")
            writer = getattr(self, "_conditional_writer", None)
            if writer is None or writer.closed.is_set():
                writer = self._conditional_writer = ConditionalWriter(self)
            writer.send(self._proc, {**payload, "type": "conditional", "boot_id": boot}, waiter)
        finally:
            self._lock.release()

    def conditional_send(self, boot, payload):
        request_id = uuid.uuid4().hex
        waiter = queue.Queue(maxsize=1)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Conditional host settling")
        try:
            self._pending_controls[request_id] = waiter
            try:
                self._conditional_send_to_current(boot, {**payload, "request_id": request_id}, waiter)
            except BaseException:
                self._pending_controls.pop(request_id, None)
                raise
        finally:
            self._lock.release()
        return request_id, waiter, boot

    def conditional_receive(self, ticket):
        request_id, waiter, boot = ticket
        try:
            reply = waiter.get(timeout=6.0)
            if reply.get("boot_id") != boot:
                raise RuntimeError("Conditional reply came from another child lifetime")
            return reply
        finally:
            with self._lock:
                self._pending_controls.pop(request_id, None)

    def conditional_exchange(self, boot, payload):
        return self.conditional_receive(self.conditional_send(boot, payload))

    def conditional_release(self, boot, subscription):
        self._conditional_best_effort(boot, {"action": "release", "subscription": subscription})

    def conditional_abort(self, boot, reservation):
        self._conditional_best_effort(boot, {"action": "abort", "reservation": reservation})

    def _conditional_best_effort(self, boot, payload):
        try:
            self._conditional_send_to_current(boot, payload)
        except RuntimeError:
            return  # unavailable child/queue: reservations and child membership leases expire

    def conditional_admit(self, frame):
        import logging

        try:
            allowed = self.conditional_admission_sink is not None and self.conditional_admission_sink(frame)
        except Exception:
            logging.getLogger(__name__).exception("Conditional parent admission failed")
            allowed = False
        self._conditional_best_effort(frame.get("boot_id"), {"action": "admitted",
            "turn_id": frame.get("turn_id"), "allowed": bool(allowed)})

    def conditional_membership(self, frame):
        import logging

        try:
            members = self.conditional_membership_sink(frame) if self.conditional_membership_sink else []
        except Exception:
            logging.getLogger(__name__).exception("Conditional parent membership check failed")
            members = []
        self._conditional_best_effort(frame.get("boot_id"), {"action": "membership",
            "challenge": frame.get("challenge"), "members": members})

    def conditional_track_turn(self, turn_id, sid, callback):
        with self._lock:
            self._pending_turns[turn_id] = (sid, callback)

    def conditional_untrack_turn(self, turn_id):
        with self._lock:
            self._pending_turns.pop(turn_id, None)

    def _close_conditional_writer(self):
        writer = getattr(self, "_conditional_writer", None)
        if writer is not None:
            writer.close()
