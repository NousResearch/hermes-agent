"""The process-local table of open operations."""

from __future__ import annotations

import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

from tools.operations.protocol import Operation, Owner


class OperationAlreadyOpen(RuntimeError):
    def __init__(self, existing: Operation):
        super().__init__(f"{existing.owner.key!r} already has operation {existing.op_id} open")
        self.existing = existing


class Operations:
    def __init__(self) -> None:
        self._open: Dict[Tuple[Owner, str], Operation] = {}
        self._lock = threading.Lock()

    def open(self, operation: Operation, *, exclusive: bool = False) -> None:
        """Register an operation under its owner. An operation without one is stamped with the
        opening thread's profile (the turn's override) and its own key. ``exclusive`` refuses while
        the owner already has an unsettled operation (a session shows one connection card at a time)."""
        if operation.owner is None:
            operation.owner = Owner.current(operation.key)
        with self._lock:
            if exclusive:
                held = next((o for o in self._for_owner_locked(operation.owner) if not o.settled), None)
                if held is not None and held is not operation:
                    raise OperationAlreadyOpen(held)
            self._open[(operation.owner, operation.op_id)] = operation

    def get(self, owner: Owner, op_id: str) -> Optional[Operation]:
        """The operation only when both the owner and the id match: an RPC answers the operation
        the session it authorized for opened, never one it merely knows the id of."""
        with self._lock:
            return self._open.get((owner, op_id))

    def current(self, owner: Owner) -> List[Operation]:
        """Unsettled operations of an owner, oldest first: the resume snapshot reads this."""
        with self._lock:
            live = [o for o in self._for_owner_locked(owner) if not o.settled]
        return sorted(live, key=lambda o: o.deadline_at)

    def find(self, profile: str, op_id: str) -> Optional[Operation]:
        """An unsettled operation by id within one profile, whatever key opened it."""
        with self._lock:
            return next((o for (owner, oid), o in self._open.items()
                         if owner.profile == profile and oid == op_id and not o.settled), None)

    def wait(self, operation: Operation, *, timeout: Optional[float] = None, slice_seconds: float = 0.25) -> str:
        """Block until the operation settles, its deadline passes, or ``timeout`` seconds elapse.
        Only the deadline settles the operation; a timeout returns so the caller can observe and
        wait again. The wait is sliced so the caller's interrupt flag is read between slices.
        Returns ``settled_by`` after a settle, ``"timeout"`` otherwise."""
        from tools.interrupt import is_interrupted

        until = None if timeout is None else time.time() + timeout
        while not operation.settled:
            if is_interrupted():
                operation.settle("interrupt")
                break
            to_deadline = operation.deadline_at - time.time()
            if to_deadline <= 0:
                operation.settle("deadline")
                break
            remaining = to_deadline if until is None else min(to_deadline, until - time.time())
            if remaining <= 0:
                return "timeout"
            if operation.wake.wait(min(slice_seconds, remaining)):
                operation.wake.clear()
                if not operation.settled:
                    return "woke"
        return operation.settled_by or "deadline"

    def close(self, operation: Operation) -> None:
        """Remove the operation only while the table still holds this exact object: a finished
        operation must not delete the one that replaced it under the same owner."""
        if operation.owner is None:
            return
        key = (operation.owner, operation.op_id)
        with self._lock:
            if self._open.get(key) is operation:
                del self._open[key]

    def cancel(self, owner: Optional[Owner], reason: str) -> int:
        """Settle every unsettled operation of ``owner`` (all owners when None) and wake its
        waiter. Interrupt, session teardown and shutdown all come through here."""
        with self._lock:
            targets = [o for (own, _), o in self._open.items() if (owner is None or own == owner) and not o.settled]
        for operation in targets:
            operation.settle(reason)
            operation.wake.set()
        return len(targets)

    def _for_owner_locked(self, owner: Owner) -> Iterable[Operation]:
        return (o for (own, _), o in self._open.items() if own == owner)

    def reset_for_tests(self) -> None:
        with self._lock:
            self._open.clear()


operations = Operations()
