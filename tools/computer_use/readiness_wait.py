"""Bounded wait-for-state on top of driver readiness checks (RFC #112639).

A single readiness check answers "does the postcondition hold right now".
This module adds the missing runtime shape: poll until a real completion
signal, so a render or transition can be waited on without a capture + model
round trip per poll and without the model asking "is it done yet".

Semantics: ``satisfied`` returns at once; ``unsatisfied`` is provably false,
so the wait ends immediately (no point polling a state the driver has
negated); ``unknown`` keeps polling until the deadline; ``error`` returns as
is; deadline exceeded yields ``timeout``. Bounds are clamped; the per-check
timeout follows the verify_state driver contract inside ``verify_readiness``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools.computer_use.readiness import ReadinessResult

logger = logging.getLogger(__name__)

# Client-side poll bounds; per-check timeout is clamped by verify_readiness
# to the driver's 0-10000ms contract.
_DEADLINE_MS_MIN, _DEADLINE_MS_MAX = 0, 30000
_POLL_INTERVAL_MS_MIN = 50


@dataclass
class WaitForStateResult:
    """Outcome of a bounded wait-for-state."""

    status: str  # satisfied | unsatisfied | timeout | error
    checks: int = 0
    elapsed_ms: float = 0.0
    last_result: Optional[ReadinessResult] = field(default=None)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def wait_for_state(
    backend: Any,
    *,
    pid: int,
    window_id: int,
    expect: List[Dict[str, Any]],
    deadline_ms: int = 5000,
    poll_interval_ms: int = 250,
    check_timeout_ms: int = 2000,
    stable_samples: int = 1,
) -> WaitForStateResult:
    """Poll ``backend.verify_readiness`` until a completion signal or the deadline.

    Routes through the backend's capability-gated ``verify_readiness`` method,
    never a raw session/daemon call. No screenshots, no model calls, no retries
    beyond the poll loop itself.
    """
    deadline_ms = _clamp(deadline_ms, _DEADLINE_MS_MIN, _DEADLINE_MS_MAX)
    poll_interval_ms = max(_POLL_INTERVAL_MS_MIN, int(poll_interval_ms))
    t0 = time.perf_counter()
    deadline_s = deadline_ms / 1000.0
    checks = 0
    last: Optional[ReadinessResult] = None
    while True:
        last = backend.verify_readiness(
            pid=pid,
            window_id=window_id,
            expect=expect,
            timeout_ms=check_timeout_ms,
            stable_samples=stable_samples,
        )
        checks += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if last.status == "satisfied":
            return WaitForStateResult("satisfied", checks, elapsed_ms, last)
        if last.status == "unsatisfied":
            return WaitForStateResult("unsatisfied", checks, elapsed_ms, last)
        if last.status == "error":
            return WaitForStateResult("error", checks, elapsed_ms, last)
        # unknown: driver cannot tell yet — keep polling until the deadline.
        elapsed_s = (time.perf_counter() - t0)
        if elapsed_s >= deadline_s:
            logger.info("wait_for_state timed out after %d checks", checks)
            return WaitForStateResult("timeout", checks, elapsed_ms, last)
        time.sleep(min(poll_interval_ms / 1000.0, deadline_s - elapsed_s))
