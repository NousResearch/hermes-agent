"""Launch-profile policy for a process that hosts several profile homes (``hermes serve`` /
``hermes dashboard`` pooling, ``?profile=``, hosted rooms; the multiplexed gateway's own worker).

Two facts anchor this module:

* ``agent.secret_scope.get_secret`` fails closed ONLY while ``set_multiplex_active(True)`` holds.
  A ``serve`` backend that hosts a second profile home never flipped it, so every unscoped read
  for a secondary profile silently returned the LAUNCH profile's ``os.environ`` value. The flip
  happens here, at the moment the process first learns it hosts another profile home.
* Once multiplexing is active the launch profile is a profile too: its turns/RPCs must run under
  their own scope instead of ambient ``os.environ`` (a secondary context may have poisoned it,
  #107422). A scope rebuilt from ``<launch home>/.env`` + ``config.yaml`` alone would drop the
  launch process's legitimate env-only policy — ``TERMINAL_ENV=ssh`` or a provider key injected
  by systemd / ``op run`` has no file to rebuild it from. The process env is trusted exactly
  once: frozen at activation, before any secondary code has run, never re-read afterwards.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Optional

_lock = threading.Lock()
_snapshot: Optional[Dict[str, str]] = None


def capture_launch_env() -> Dict[str, str]:
    """Freeze the process env as the launch profile's own; the first capture wins.

    Called at activation, immediately before the first secondary home is registered as
    served — the last moment ambient env is provably the launch profile's.
    """
    global _snapshot
    with _lock:
        if _snapshot is None:
            _snapshot = dict(os.environ)
        return dict(_snapshot)


def activate_multi_profile_hosting() -> None:
    """This process now hosts a profile home other than its launch home: freeze the launch env
    and make unscoped credential reads fail closed (``get_secret`` raises instead of borrowing)."""
    from agent.secret_scope import set_multiplex_active
    capture_launch_env()
    set_multiplex_active(True)


def _launch_env() -> Dict[str, str]:
    """The launch profile's env: frozen once multiplexing is active; the LIVE process env before
    (no secondary has run yet, so it is provably the launch profile's, and freezing it early would
    miss values the launch process still bridges at startup)."""
    from agent.secret_scope import is_multiplex_active
    return capture_launch_env() if is_multiplex_active() else dict(os.environ)


def launch_terminal_env() -> Dict[str, str]:
    """The frozen launch ``TERMINAL_*`` overlay for a launch-profile turn's terminal scope.

    Production always captured at activation; a first capture here only happens when the
    multiplexer flag was set by another owner (the messaging gateway) or a harness.
    """
    return {k: v for k, v in capture_launch_env().items() if k.startswith("TERMINAL_")}


def launch_secret_scope(launch_home: "str | Path") -> Dict[str, str]:
    """The launch profile's secret mapping: its ``.env`` + external sources over the launch env
    (systemd / ``op run`` injection survives the fail-closed flip; a secondary never sees it because
    its scope is built from its own files only). Bound for EVERY launch-profile body, multiplexing or
    not, so the body's credential source is decided once at entry: a request that entered while
    single-profile keeps resolving from this mapping after a concurrent first secondary flips
    ``get_secret`` to fail closed (``_MULTIPLEX_ACTIVE`` is read on every ``get_secret``, the
    scope decision was made at entry)."""
    from agent.secret_scope import _is_global_env, build_profile_secret_scope
    scope = {k: v for k, v in _launch_env().items() if not _is_global_env(k)}
    scope.update(build_profile_secret_scope(Path(launch_home)))
    return scope
