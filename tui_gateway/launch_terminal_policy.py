"""Launch-profile ``TERMINAL_*`` snapshot for multiplexed TUI-gateway turns.

Once this backend serves a secondary profile, launch-profile turns bind a terminal scope instead
of reading ambient ``os.environ`` (a secondary context must never become the launch turn's
authority; #107422). A scope rebuilt from ``<launch home>/.env`` + ``config.yaml`` alone drops
the launch process's legitimate env-only policy — ``TERMINAL_ENV=ssh TERMINAL_SSH_HOST=...``
injected by systemd / ``op run`` / a launcher bridge has no file to rebuild it from and silently
became ``backend=local``. The env is trusted exactly once: frozen at multiplex activation, before
any secondary code has run in this process, and never re-read from ambient state afterwards.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional

_lock = threading.Lock()
_snapshot: Optional[Dict[str, str]] = None


def capture_launch_terminal_env() -> Dict[str, str]:
    """Freeze the process's ``TERMINAL_*`` env; the first capture wins, later calls are no-ops.

    Called by ``server._profile_home`` immediately before the first secondary home is registered
    as served — the last moment ambient env is provably the launch profile's own.
    """
    global _snapshot
    with _lock:
        if _snapshot is None:
            _snapshot = {k: v for k, v in os.environ.items() if k.startswith("TERMINAL_")}
        return dict(_snapshot)


def launch_terminal_env() -> Dict[str, str]:
    """The frozen launch ``TERMINAL_*`` overlay for a launch-profile turn's terminal scope.

    Production always captured at activation (``_profile_home`` is the only writer of
    ``_served_profile_homes``); a first capture here only happens when a harness populated the
    served set directly.
    """
    return capture_launch_terminal_env()
