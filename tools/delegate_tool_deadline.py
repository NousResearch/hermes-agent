"""Opt-in child deadlines; only an owning parent's approved checkpoint renews one."""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import TimeoutError


class ReviewedDeadline:
    def __init__(self, seconds: float | None, *, clock=None):
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("Reviewed timeouts require a positive finite child_timeout_seconds.")
        self.seconds = float(seconds)
        self.clock = clock or time.monotonic
        self.lock = threading.RLock()
        self.deadline = self.clock() + self.seconds
        self.pending = None
        self.latest = 0
        self.closed = False
        self.renewals = 0

    def remaining(self):
        with self.lock:
            left = self.deadline - self.clock()
            if left <= 0:
                self.closed = True
            return 0.0 if self.closed else left

    def report(self, checkpoint: int):
        with self.lock:
            if self.remaining() <= 0 or type(checkpoint) is not int or checkpoint <= self.latest:
                raise ValueError("Expired, stopped or stale checkpoint; timeout was not extended.")
            self.latest = self.pending = checkpoint

    def review(self, checkpoint: int, *, approve: bool):
        with self.lock:
            if self.remaining() <= 0 or type(checkpoint) is not int or checkpoint != self.pending:
                raise ValueError("Only the latest unreviewed checkpoint of a live child can be reviewed.")
            self.pending = None
            if approve:
                # A fresh window, not accumulating unused time; report spam cannot extend it.
                self.deadline = self.clock() + self.seconds
                self.renewals += 1
            return {"approved": approve, "checkpoint_id": checkpoint, "timeout_seconds": self.seconds,
                    "remaining_seconds": self.remaining()}

    def snapshot(self):
        with self.lock:
            return {"remaining_seconds": self.remaining(), "timeout_seconds": self.seconds,
                    "pending_checkpoint": self.pending, "closed": self.closed, "renewals": self.renewals}

    def close(self):
        with self.lock:
            self.closed = True


def wait_with_reviewed_deadline(future, lease, *, settled=None):
    """Re-check a mutable deadline, without letting keepalive activity renew it."""
    while True:
        if settled is not None and settled.is_set() and not future.done():
            raise TimeoutError("Child heartbeat declared the worker stale.")
        left = lease.remaining()
        if left <= 0:
            raise TimeoutError("Parent-reviewed child deadline expired.")
        try:
            return future.result(timeout=min(1.0, left))
        except TimeoutError:
            if future.done():  # A worker exception, not this bounded wait expiring.
                raise
