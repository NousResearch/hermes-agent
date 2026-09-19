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

import contextlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterator, Mapping, Optional

_lock = threading.Lock()


@dataclass(frozen=True)
class LaunchProfileAuthority:
    """One observation of the launch environment and the home identity it owns."""

    env: Mapping[str, str]
    home: Path


_authority: Optional[LaunchProfileAuthority] = None


def _live_launch_env() -> Dict[str, str]:
    """Snapshot process values and pin a missing shell ``PWD`` to the capture cwd."""
    snapshot = dict(os.environ)
    if "PWD" not in snapshot:
        snapshot["PWD"] = os.getcwd()
    return snapshot


def capture_launch_authority() -> LaunchProfileAuthority:
    """Freeze launch values and resolved home atomically; the first capture wins."""
    global _authority
    with _lock:
        if _authority is None:
            from hermes_constants import get_process_hermes_home

            snapshot = _live_launch_env()
            home = get_process_hermes_home(snapshot)
            _authority = LaunchProfileAuthority(MappingProxyType(snapshot), home)
        return _authority


def capture_launch_env() -> Dict[str, str]:
    """Freeze the process env as the launch profile's own; the first capture wins.

    Called at activation, immediately before the first secondary home is registered as
    served — the last moment ambient env is provably the launch profile's.
    """
    return dict(capture_launch_authority().env)


def activate_multi_profile_hosting() -> None:
    """This process now hosts a profile home other than its launch home: freeze the launch env
    and make unscoped credential reads fail closed (``get_secret`` raises instead of borrowing)."""
    from agent.secret_scope import set_multiplex_active
    capture_launch_authority()
    set_multiplex_active(True)


def launch_env() -> Dict[str, str]:
    """The launch profile's env: frozen once multiplexing is active; the LIVE process env before
    (no secondary has run yet, so it is provably the launch profile's, and freezing it early would
    miss values the launch process still bridges at startup)."""
    return dict(launch_authority().env)


def launch_authority() -> LaunchProfileAuthority:
    """Return the captured authority once published, otherwise a pre-activation live view."""
    from agent.secret_scope import is_multiplex_active
    from hermes_constants import get_process_hermes_home

    with _lock:
        if _authority is not None:
            return _authority
        if not is_multiplex_active():
            env = _live_launch_env()
            home = get_process_hermes_home(env)
            return LaunchProfileAuthority(MappingProxyType(env), home)
    # A harness may flip multiplexing directly without calling activation.
    return capture_launch_authority()


def launch_home() -> Path:
    """Resolved launch-profile home from the same observation as :func:`launch_env`."""
    return launch_authority().home


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
    scope = {k: v for k, v in launch_env().items() if not _is_global_env(k)}
    scope.update(build_profile_secret_scope(Path(launch_home)))
    return scope


@contextlib.contextmanager
def launch_profile_runtime_scope(
    launch_home: "str | Path | None" = None,
) -> Iterator[None]:
    """Bind the launch profile's own runtime scope for one body: frozen home identity,
    launch secret scope, and terminal policy over the frozen ``TERMINAL_*`` overlay. For hosts whose
    launch-profile bodies are not RPC sessions (the standalone messaging gateway after a hosted
    room activated multiplexing, #112878). ``launch_home`` is retained for
    caller compatibility but cannot override the captured authority."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope

    del launch_home
    home = launch_authority().home
    home_token = set_hermes_home_override(home)
    try:
        secret_token = set_secret_scope(launch_secret_scope(home))
        try:
            terminal_token = install_profile_terminal_scope(
                home, env_overlay=launch_terminal_env()
            )
            try:
                yield
            finally:
                reset_terminal_scope(terminal_token)
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)
