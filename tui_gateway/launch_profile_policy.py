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
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)

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


def _servable_profile_homes() -> set:
    """Resolved homes this host could be asked to serve: the launch home plus every profile dir
    carrying a real servability marker.

    ``named_profile_has_identity`` accepts an EMPTY ``.env``, which is all a crashed
    ``hermes profile create`` leaves behind — counting it would flip a single-profile host
    fail-closed at its next boot.
    """
    from hermes_constants import named_profile_has_servable_identity
    from hermes_cli.profiles import profiles_to_serve

    homes = {Path(home).resolve() for name, home in profiles_to_serve(multiplex=True, include_standalone=True)
             if name == "default" or named_profile_has_servable_identity(home)}
    homes.add(Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").resolve())
    return homes


def activate_multi_profile_hosting_eagerly() -> bool:
    """Activate at HOST startup when this machine has more than one servable profile home.

    Activation used to be lazy and one-way — it fired the first time a *request* asked for a second
    home. Everything the host had already done by then (idle-reaper ticks, cron start, adapter
    connects, MCP discovery) ran under single-profile assumptions and is never re-scoped, and the
    launch profile's env had already been mutated by then, so ``capture_launch_env`` froze a
    polluted snapshot. One ``hermes serve`` / ``hermes gateway run`` per host means the process
    knows at boot whether it can be asked for a second home: decide there, once. Call it as the LAST
    boot step: the frozen snapshot is the only source for launch keys with no ``.env`` to rebuild
    from, so every credential the boot still injects must already be in ``os.environ``.

    ``gateway.multiplex_profiles: false`` is deliberately NOT consulted: it is retired as a
    topology opt-out, and a multi-home host that skipped activation because of a stale ``false``
    would serve a second profile with the LAUNCH profile's credentials — the exact fail-open this
    guard exists to prevent.

    A genuinely single-profile host still never activates (byte-identical behaviour, ``os.environ``
    precedence preserved). An unreadable profiles directory fails CLOSED: we cannot prove the host
    is single-profile, and the lazy backstop only fires once a request has already been answered.
    Returns True when this call activated hosting.
    """
    from agent.secret_scope import is_multiplex_active
    if is_multiplex_active():
        return False
    try:
        homes = _servable_profile_homes()
    except Exception:
        logger.warning(
            "Could not enumerate this host's profile homes; activating multi-profile hosting "
            "fail-closed (unscoped credential reads will raise instead of borrowing the launch "
            "profile's)", exc_info=True)
        activate_multi_profile_hosting()
        return True
    if len(homes) < 2:
        return False
    logger.info("Multi-profile hosting activated at startup (%d servable profile homes)", len(homes))
    activate_multi_profile_hosting()
    return True


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
    scope decision was made at entry).

    PRECEDENCE, and it is not the process's: ``<launch home>/.env`` (plus hydrated external sources)
    WINS over the env. For a key present in both, an unscoped read returned the ambient
    ``os.environ`` value before activation and returns the ``.env`` value after — so activation is
    not purely "stricter" for the launch tenant, it also changes which of the launch profile's own
    two values it sees. Env-only keys (systemd ``Environment=``, ``op run``, Compose) are unaffected:
    nothing in the files shadows them.
    """
    from agent.secret_scope import _is_global_env, build_profile_secret_scope
    scope = {k: v for k, v in _launch_env().items() if not _is_global_env(k)}
    scope.update(build_profile_secret_scope(Path(launch_home)))
    return scope


@contextlib.contextmanager
def launch_profile_runtime_scope(launch_home: "str | Path") -> Iterator[None]:
    """Bind the launch profile's own runtime scope for one body: HERMES_HOME override naming the
    launch home, ``launch_secret_scope``, and its terminal policy over the frozen launch
    ``TERMINAL_*`` overlay. For hosts whose launch-profile bodies are not RPC sessions (the
    standalone messaging gateway after a hosted room activated multiplexing, #112878).

    The home override is bound even though the launch home IS the process home: under multiplexing
    "override unset" is the fail-closed signal for an UNBOUND context (``serves_routed_profile``,
    plugin runtime bindings such as OMH's ``pre_tool_call`` gate, per-home slots keyed on the
    override), so a launch-profile turn without it was indistinguishable from no turn at all and
    every plugin hook it fired saw an unscoped process (#118538). Routed turns already bind theirs
    (``gateway/run.py::_profile_runtime_scope``); the launch profile is a tenant like any other."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope

    home = Path(launch_home)
    home_token = secret_token = terminal_token = None
    try:
        home_token = set_hermes_home_override(str(home))
        secret_token = set_secret_scope(launch_secret_scope(home))  # own home: no foreign stamp
        terminal_token = install_profile_terminal_scope(home, env_overlay=launch_terminal_env())
        yield
    finally:
        if terminal_token is not None:
            reset_terminal_scope(terminal_token)
        if secret_token is not None:
            reset_secret_scope(secret_token)
        if home_token is not None:
            reset_hermes_home_override(home_token)


def launch_profile_scope_if_multiplexed():
    """The launch profile's own runtime scope once this process multiplexes; ``nullcontext``
    before.

    The single seam for "no routed profile here": under the one-process-per-host ruling the launch
    profile is a tenant like any other, so a body that used to run with NO scope at all (ambient
    ``os.environ`` + the process home) must bind the launch profile explicitly — otherwise a
    secondary's context can have poisoned the ambient state, and a fail-closed ``get_secret`` raises
    on a perfectly legitimate launch-profile read. Before activation the process env IS the launch
    profile's, so binding nothing is still correct (and keeps single-profile hosts byte-identical —
    callers assert the returned object is literally a ``nullcontext``).
    """
    from agent.secret_scope import is_multiplex_active
    if not is_multiplex_active():
        return contextlib.nullcontext()
    from hermes_constants import get_process_hermes_home
    return launch_profile_runtime_scope(get_process_hermes_home())


@contextlib.asynccontextmanager
async def async_launch_profile_scope_if_multiplexed():
    """``async with`` twin of :func:`launch_profile_scope_if_multiplexed` (no I/O of its own: the
    launch scope is rebuilt from already-hydrated sources, so there is nothing to move off-loop)."""
    with launch_profile_scope_if_multiplexed():
        yield
