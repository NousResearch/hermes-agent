"""One idle check for a Bot Desktop screen, run by the screen's own launcher.

The screen outlives the gateway that started it on purpose (a restarted gateway re-attaches to it), so the
gateway's display watcher cannot be the only thing that ever stops it: when no gateway comes back, Xvnc +
Xfce and any browser left open run until reboot. ``launcher.sh`` therefore runs
``python -m tools.bot_desktop.idle_watch`` in a loop of its own. Each run applies
:func:`runtime.stop_if_idle` (the same ``bot_desktop.idle_stop_minutes`` window and the same rule that a
human holding the screen is never timed out) and prints how long the launcher should sleep before the next
run, so no interpreter stays resident between checks.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

from tools.bot_desktop import runtime

logger = logging.getLogger(__name__)

_MIN_DELAY_S = 30.0  # the gateway watcher's cadence; a window shorter than that is checked at its own length


def next_check_in(limit: float, idle: Optional[float]) -> int:
    """Whole seconds until the screen could next be due: when the window runs out at the current idle age,
    or a full window later while something (a human holding it) keeps it past that. With the auto-stop off
    (``limit <= 0``) the setting is re-read every default window."""
    if limit <= 0:
        return runtime.DEFAULT_IDLE_STOP_MINUTES * 60
    idle = idle or 0.0
    return math.ceil(max(min(_MIN_DELAY_S, limit), limit - idle if idle < limit else limit))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s idle-watch %(message)s")  # stderr = launcher.log
    if runtime.stop_if_idle():
        return
    delay = next_check_in(runtime.idle_stop_seconds(), runtime.idle_seconds())
    logger.info("screen kept; next idle check in %ds", delay)
    print(delay)


if __name__ == "__main__":
    # stop() kills the launcher's process group, which this check was started in; in a session of its own it
    # survives that kill to finish the stop (SIGKILL stragglers, drop the runtime files), then exits. A process
    # that already leads its group (run by hand from a shell) cannot setsid and is not in the launcher's group.
    if os.getpgrp() != os.getpid():  # windows-footgun: ok — Linux-only (launcher.sh)
        os.setsid()  # windows-footgun: ok — Linux-only (launcher.sh)  detached-spawn: ok — short-lived, outlives the group kill it performs
    main()
