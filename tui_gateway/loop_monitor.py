"""Event-loop lag watchdog helpers shared by TUI/WebSocket surfaces.

The gateway can run Python handlers in worker threads, but CPU/GIL-heavy work can
still starve the owning asyncio loop. These helpers install a lightweight
call_later heartbeat that logs when the loop wakes up late, giving operators a
clear breadcrumb for false disconnect/setup reports caused by loop starvation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

DEFAULT_LOOP_LAG_INTERVAL_S = 2.0
DEFAULT_LOOP_LAG_WARN_AFTER_S = 5.0


def log_loop_lag_if_stalled(
    logger: logging.Logger,
    *,
    lag_s: float,
    warn_after_s: float = DEFAULT_LOOP_LAG_WARN_AFTER_S,
    component: str = "",
) -> bool:
    """Log and return True when measured loop lag crosses the warning threshold."""
    if lag_s <= warn_after_s:
        return False

    if component:
        logger.warning(
            "%s event loop stalled %.1fs (GIL pressure suspected)",
            component,
            lag_s,
        )
    else:
        logger.warning(
            "event loop stalled %.1fs (GIL pressure suspected)",
            lag_s,
        )
    return True


def install_event_loop_lag_watchdog(
    *,
    loop: asyncio.AbstractEventLoop | Any | None = None,
    logger: logging.Logger,
    interval_s: float = DEFAULT_LOOP_LAG_INTERVAL_S,
    warn_after_s: float = DEFAULT_LOOP_LAG_WARN_AFTER_S,
    component: str = "",
) -> Any:
    """Install a self-rearming event-loop lag watchdog on *loop*.

    The returned object is the first ``call_later`` handle. The heartbeat re-arms
    itself after each tick and naturally dies with the loop. Callers should guard
    this helper so they install at most one watchdog per long-lived loop.
    """
    if loop is None:
        loop = asyncio.get_running_loop()

    interval = max(float(interval_s), 0.001)
    warn_after = max(float(warn_after_s), 0.0)

    def _loop_heartbeat(expected: float) -> None:
        now = loop.time()
        lag = now - expected
        log_loop_lag_if_stalled(
            logger,
            lag_s=lag,
            warn_after_s=warn_after,
            component=component,
        )
        try:
            loop.call_later(interval, _loop_heartbeat, now + interval)
        except RuntimeError:
            # The loop may be closing during shutdown. Best-effort diagnostic;
            # do not make teardown noisy.
            logger.debug("event-loop lag watchdog stopped: loop is closed")

    return loop.call_later(interval, _loop_heartbeat, loop.time() + interval)
