"""Keep one room's driver lease alive while a turn waits for its profile lock and runs.

The room driver otherwise renews only at checkpoints. ``turn_lock(profile)`` can block for up to
``bot_mode.turn_wait_seconds`` (120 s) behind another turn on the same profile, far past the 30 s
lease, and ``renew_lease`` refuses an expired lease, so a renewal after the lock is acquired is too
late: the turn was submitted and then fenced ``indeterminate``. A renewal that finds the lease
genuinely lost (reclaimed, released, authority moved) stops the keeper, and ``renew_now`` re-checks on
the turn's thread once the lock is held, so the attempt still fences before submit.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from agent.memory_provider import spawn_context_thread
from persistence.gateway.hosted_rooms import DriverLease, DriverStateError, RoomUnavailableError, StaleLeaseError

logger = logging.getLogger(__name__)


class LeaseKeeper:
    """Renew ``lease`` every ``interval`` seconds on a daemon thread until the context exits."""

    def __init__(
        self, lease: DriverLease, renew: Callable[[DriverLease], DriverLease], *,
        interval: float,
    ) -> None:
        self.lease, self._renew, self._interval = lease, renew, interval
        self._stop = threading.Event()
        self._lost: DriverStateError | None = None
        self._thread = spawn_context_thread(self._run, name=f"hosted-room-lease-{lease.room_id[:24]}")

    def __enter__(self) -> LeaseKeeper:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def renew_now(self) -> None:
        """Renew on the caller's thread: fences it if the lease was lost during the wait."""
        if self._lost is not None:
            raise self._lost
        self.lease = self._renew(self.lease)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.lease = self._renew(self.lease)
            except (StaleLeaseError, RoomUnavailableError) as exc:
                self._lost = exc
                return
            except Exception as exc:  # health: allow BLE001 -- transient (sqlite busy); next tick retries
                logger.warning("room %s lease renewal failed: %s", self.lease.room_id, exc)
