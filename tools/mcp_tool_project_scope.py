"""Per-project stdio transports for tools.mcp_tool (#151827, ported from #63759).

A stdio MCP connection is discovered once per process, and its child runs wherever the MCP
loop resolved a cwd at connect time: the gateway's launch dir or ``TERMINAL_CWD``, never a
session's project (the loop thread cannot see the session ``_SESSION_CWD`` ContextVar). Every
session then shared that one child, so a project-sensitive server (Hindsight picks its memory
bank from ``${PWD}``) answered every project with the launch dir's.

The connection in ``_servers`` stays the *baseline*: it owns the tool schemas, the trust
metadata and every keyed ledger, unchanged. A caller whose resolved cwd differs from the
directory the baseline child was spawned in gets a *clone* here instead: same config with
``cwd`` pinned to the caller's project, never registering tools (the agent's tool list and the
prompt cache never move), keyed ``(baseline connection key, canonical cwd)``. Single-project
processes (CLI, cron, a gateway whose sessions all sit in its launch dir) never clone.

Clones are bounded: past ``_MAX_SCOPED_TRANSPORTS_PER_SERVER`` per baseline, the least
recently used idle clones are shut down. A clone in use (a handler holding its lease, an RPC
holding ``_rpc_lock``, or a connect in flight) is never evicted; an evicted project simply
reconnects on its next call. Eviction design after kvnloo/hermes-agent@33847118.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
import os
import threading
import time
from functools import wraps
from typing import Any, Optional

from tools.mcp_tool_common import _core
from tools import mcp_tool_loop as _loop

logger = logging.getLogger("tools.mcp_tool")

# ponytail: one fixed cap per baseline connection; make it a per-server config key only if a
# real multi-project gateway needs more than 8 live project children per server.
_MAX_SCOPED_TRANSPORTS_PER_SERVER = 8

# Every mutation under ``_core._lock``.
_scoped_servers: dict[tuple[Any, str], Any] = {}
# Single-flight per clone key, held across the blocking connect (never ``_core._lock``).
_scoped_connect_locks: dict[tuple[Any, str], threading.Lock] = {}
# Waiter seats per connect lock: every thread between fetching a lock and releasing it counts
# here, so a lock whose generation never installed a clone is reaped when the last seat leaves
# instead of accumulating for the process lifetime (independent review, suggestion 3).
_scoped_connect_waiters: dict[tuple[Any, str], int] = {}

# Baseline keys a ``shutdown_mcp_servers`` pass is tearing down, stamped at the synchronous
# ``pop_clones`` and cleared once the deferred async ``_servers.pop`` has landed: in between,
# the registry still answers ``is baseline`` at the insert gate, and a clone installed there
# would strand its live child with no owner (review finding 1). Read/written only under
# ``_core._lock``.
_torn_down_baselines: set = set()

# The handler's lease list: every clone a lookup hands this handler is counted on the clone
# itself (``_scoped_leases``) until the handler returns — counted on the object, never by key,
# so a rekey or a reload that moves the key mid-call cannot strand the count (review finding 3).
# None outside a leased handler.
_call_leases: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("mcp_scoped_call_leases", default=None)


# Per-clone connect cooldown (geometric, capped), mirroring the baseline's retry backoff in
# ``tools.mcp_tool_discovery._record_connect_failure``: without it a broken project pays a full
# spawn (up to connect_timeout+5) on every call while its waiters queue behind the connect lock
# (port review finding 5). Every mutation under ``_core._lock``.
_scoped_connect_failures: dict[tuple, int] = {}
_scoped_connect_retry_after: dict[tuple, float] = {}


def _record_clone_connect_failure(clone_key: tuple) -> None:
    """Stamp a geometric, capped retry cooldown on the clone key after a failed connect.
    Caller holds ``_core._lock``."""
    n = _scoped_connect_failures.get(clone_key, 0) + 1
    _scoped_connect_failures[clone_key] = n
    backoff = min(_core._CONNECT_RETRY_BASE_BACKOFF_SEC * (2 ** (n - 1)), _core._CONNECT_RETRY_MAX_BACKOFF_SEC)
    _scoped_connect_retry_after[clone_key] = time.monotonic() + backoff


def _clear_clone_connect_failure(clone_key: tuple) -> None:
    """Clear the clone's connect-cooldown state after a successful connect. Caller holds ``_core._lock``."""
    _scoped_connect_failures.pop(clone_key, None)
    _scoped_connect_retry_after.pop(clone_key, None)


def _caller_cwd() -> str:
    """The caller's canonical project dir; must run on the CALLER's thread (the session cwd is a
    ContextVar the MCP loop cannot see). A non-local terminal backend's cwd names a path in the
    target environment, never on this host — the worker side gates on the backend too — so it
    clones nothing rather than resolving a coincidental host path (review finding 8c).
    A refusal scope (the profile's terminal policy could not be resolved) fails closed the same
    way: no clone, the baseline child serves the call — raising out of here would break every
    scoped stdio call from that session (review finding 8c residual).
    realpath: ``/p``, ``/p/`` and a symlink to it are one project."""
    from tools.terminal_scope import TerminalPolicyUnavailable, get_terminal_scope, terminal_env
    try:
        backend = terminal_env("TERMINAL_ENV", "local")
    except TerminalPolicyUnavailable:  # refusal scope: no-clone, never a mid-call raise
        return ""
    if (backend or "local").strip().lower() == "local" and get_terminal_scope() is None:
        # An in-process gateway bridges no TERMINAL_ENV for ``terminal.backend: docker/ssh`` (the
        # launcher bridge exists only for spawned children) — chain the config key like the worker
        # side's _effective_terminal_backend, or the gate clones for a coincidental host path of a
        # remote workspace (review finding 8c residual). A bound policy already merged its own
        # profile's terminal config, so the launch config is consulted only when unbound.
        from hermes_cli.config import load_config_readonly
        try:
            terminal_cfg = (load_config_readonly() or {}).get("terminal") or {}
        except OSError:  # a stat failure behind the config cache must not error the MCP call
            return ""
        backend = str(terminal_cfg.get("backend") or "").strip().lower()
        if backend and backend != "local":
            return ""
    if backend != "local":
        return ""
    from agent.runtime_cwd import resolve_agent_cwd
    try:
        return os.path.realpath(str(resolve_agent_cwd()))
    except OSError:  # the launch dir itself was deleted: nothing better to scope by
        return ""


def _clone_cwd(baseline: Any) -> str:
    """The project dir this call needs a clone for, or "" when the baseline serves it: HTTP has no
    cwd, a configured ``cwd`` opts the server out, a never-spawned baseline has nothing to compare,
    and a caller already in the baseline child's dir uses it. ``isinstance``: tests inject Mock
    stand-ins into ``_servers``, and a Mock would answer every attribute truthily."""
    if (not isinstance(baseline, _core.MCPServerTask) or baseline._is_http()
            or (baseline._config or {}).get("cwd") is not None or not baseline._spawn_cwd):
        return ""
    want = _caller_cwd()
    return "" if not want or want == os.path.realpath(baseline._spawn_cwd) else want


def _acquire_connect_lock(clone_key: tuple[Any, str]) -> threading.Lock:
    """Fetch or create the per-key single-flight lock and count this thread's waiter seat.
    Caller holds ``_core._lock``."""
    lock = _scoped_connect_locks.setdefault(clone_key, threading.Lock())
    _scoped_connect_waiters[clone_key] = _scoped_connect_waiters.get(clone_key, 0) + 1
    return lock


def _release_connect_lock(clone_key: tuple[Any, str], *, keep: bool) -> None:
    """Drop this thread's waiter seat. When the last seat leaves a generation that did not
    install a clone (refused, failed or discarded), the lock goes with it — a fresh generation
    single-flights on its own lock. Counting makes this race-free: a peer blocked on the object
    still holds a seat, so the entry survives until it is done. Caller holds ``_core._lock``."""
    remaining = _scoped_connect_waiters.get(clone_key, 1) - 1
    if remaining > 0:
        _scoped_connect_waiters[clone_key] = remaining
        return
    _scoped_connect_waiters.pop(clone_key, None)
    if not keep:
        _scoped_connect_locks.pop(clone_key, None)


def _hand_clone_pids_to_reaper(clone: Any) -> None:
    """Hand a dead transport's stdio child PIDs to the orphan reaper because its shutdown will
    never run (a failed or unschedulable close)."""
    pids = set(getattr(clone, "_stdio_child_pids", ()) or ())
    if not pids:
        return
    try:
        clone._release_spawned_children(pids)
    # health: allow BLE001 -- best-effort ledger handover; the final sweep still owns these PIDs
    except Exception:
        logger.warning("MCP server '%s': could not hand a discarded transport's PIDs to the reaper",
                       clone.name, exc_info=True)


def _discard_orphan_clone(orphan: Any, server_name: str, cwd: str) -> None:
    """Shut down a clone that lost its baseline mid-connect off the registry lock, like every
    other shutdown. When the shutdown cannot run (the prune's drain stopped the MCP loop), hand
    the child's PIDs to the orphan reaper instead of leaking them until the final include_active
    sweep (independent review, suggestion 1)."""
    try:
        _loop._run_on_mcp_loop(orphan.shutdown, timeout=15)
        return
    # health: allow BLE001 -- one failed close must not fail the call; the reaper still owns the PIDs
    except Exception:
        logger.warning("MCP server '%s': discarding an orphaned project transport for %s failed",
                       server_name, cwd, exc_info=True)
    _hand_clone_pids_to_reaper(orphan)


def _server_for_call(server_name: str, key: Any, baseline: Any, *, connect: bool = True) -> Any:
    """The transport this call must use: *baseline*, or this project's clone (connected lazily
    when *connect*). None when the clone cannot be connected (the handlers already turn a missing
    server into a clean "not connected" reply)."""
    cwd = _clone_cwd(baseline)
    if not cwd:
        return baseline
    clone_key = (key, cwd)
    with _core._lock:
        clone = _scoped_servers.get(clone_key)
        if clone is not None:
            _lease(clone)
            return clone
        if not connect:
            return None
        connect_lock = _acquire_connect_lock(clone_key)
    orphan = None
    with connect_lock:
        with _core._lock:
            clone = _scoped_servers.get(clone_key)
            # Refuse BEFORE spawning when the baseline died while this thread queued on the
            # single-flight lock: a peer never pays a child spawn for a dead generation
            # (independent review, suggestion 2).
            if clone is None and (key in _torn_down_baselines
                                  or _core._servers.get(key) is not baseline):
                # Tombstone or baseline-death while this thread queued on the single-flight
                # lock: a peer never pays a child spawn for a dead or dying generation
                # (independent review, suggestion 2; tombstone = review finding 1).
                _release_connect_lock(clone_key, keep=False)
                return None
        if clone is None:
            with _core._lock:
                if time.monotonic() < _scoped_connect_retry_after.get(clone_key, 0.0):
                    # The clone's connect cooldown: a broken project must not pay a full spawn on
                    # every call while its waiters queue behind the connect lock (review finding 5).
                    _release_connect_lock(clone_key, keep=False)
                    return None
            clone = _connect_clone(server_name, baseline, cwd)
            if clone is None:
                with _core._lock:  # transient failure: arm the backoff; peers keep single-flighting
                    _record_clone_connect_failure(clone_key)
                    _release_connect_lock(clone_key, keep=False)
                return None
        with _core._lock:
            occupied = _scoped_servers.get(clone_key)
            # A teardown (reload, config prune, shutdown) can remove the baseline while a clone
            # connect is in flight; inserting afterwards would strand a live child no owner will
            # ever reap or re-register. And two generations can hold different single-flight
            # locks for one key once a teardown pops the lock entry under a still-outstanding
            # seat — so the insert is first-wins, EXCEPT when the seat already holds this very
            # clone (a peer that queued on the same lock object and read it at the top of this
            # block): it leases and returns it, it must never discard it (#151827 port, review
            # finding 4).
            if (key not in _torn_down_baselines and _core._servers.get(key) is baseline
                    and (occupied is None or occupied is clone)):
                _scoped_servers[clone_key] = clone
                _clear_clone_connect_failure(clone_key)
                _lease(clone)
                _release_connect_lock(clone_key, keep=True)
            else:
                orphan = clone
                # keep=False may delete the lock entry of an already-installed clone when this
                # seat is the last: harmless — the fast path never consults the lock, and
                # _acquire_connect_lock recreates the entry on demand.
                _release_connect_lock(clone_key, keep=False)
    if orphan is not None:
        _discard_orphan_clone(orphan, server_name, cwd)
        return None
    _evict_idle_overflow(key, keep=clone_key)
    return clone


def _rerender_project_sensitive(config: dict, cwd: str) -> dict:
    """Re-render the clone config's project-sensitive values from their carried templates
    (#151827 port, #140205): ``${workspaceFolder}`` and ``${PWD}`` resolved at config-load time
    on the MCP loop thread, in the launch dir — a project-rooted dev server (Laravel Boost)
    would run against the launch dir's project even with the clone's correct cwd. The project
    refs bind to the clone's *cwd* directly (the workspace-folder resolver reads the
    terminal-tool session record, not the runtime-cwd ContextVar, and a ``PWD`` env ref answers
    with the parent's PWD — both the launch dir again); any remaining refs resolve through the
    ordinary interpolator. Only project-sensitive templates re-render: a value carrying no such
    template keeps the baseline's resolved form — re-rendering a credential ref would silently
    rotate what the baseline child runs with. The clone's env also gets ``PWD`` = the clone cwd
    when the config does not define it, so a server reading env ``PWD`` (instead of
    ``process.cwd()``) still lands in the right project. Returns the config."""
    import re

    from tools.mcp_tool_config import _interpolate_env_vars

    pwd_template = re.compile(r"\$\{(?:env:)?PWD\}")
    project_template = re.compile(r"\$\{(?:workspaceFolder|workspaceFolderBasename|(?:env:)?PWD)\}")

    def _value(value):
        template = getattr(value, "template", None)
        if not (isinstance(template, str) and project_template.search(template)):
            return value  # plain or session-invariant: keep the baseline's resolved form
        # The project refs bind to the clone's cwd directly: the workspace-folder resolver reads
        # the terminal-tool session record, not the runtime-cwd ContextVar this thread carries,
        # and an env ref would answer with the parent's PWD — both the launch dir again.
        template = pwd_template.sub(lambda _m: cwd, template)
        template = re.sub(r"\$\{workspaceFolderBasename\}", lambda _m: os.path.basename(cwd), template)
        template = re.sub(r"\$\{workspaceFolder\}", lambda _m: cwd, template)
        try:
            return _interpolate_env_vars(str(template))
        # health: allow BLE001 -- the baseline's resolved form is always usable; one render
        # failure (an unscoped secret ref under multiplex) must not fail the call
        except Exception:
            logger.warning("MCP server '%s': could not re-render a project template for the "
                           "per-project transport; keeping the baseline's resolved value",
                           config.get("cwd"), exc_info=True)
            return value

    env = config.get("env")
    if isinstance(env, dict):
        rendered = {key: _value(value) for key, value in env.items()}
        rendered.setdefault("PWD", cwd)
        config["env"] = rendered
    if isinstance(config.get("args"), list):
        config["args"] = [_value(arg) for arg in config["args"]]
    if isinstance(config.get("command"), str):
        config["command"] = _value(config["command"])
    return config


def _connect_clone(server_name: str, baseline: Any, cwd: str) -> Any:
    from tools import mcp_tool_discovery as _discovery  # lazy: discovery imports this module
    # The baseline's rendered config, not a config.yaml re-read: an edit mid-conversation must not
    # give one project a different server than the tools were discovered from.
    config = _rerender_project_sensitive(dict(baseline._config, cwd=cwd), cwd)
    timeout = float(config.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT))
    try:
        clone = _loop._run_on_mcp_loop(
            lambda: asyncio.wait_for(_discovery._connect_server(server_name, config, scoped_cwd=cwd), timeout=timeout),
            # Outer bound above the inner so the inner wait_for reports the real connect failure.
            timeout=timeout + 5)
    # health: allow BLE001 -- connect boundary: any spawn/handshake/timeout/interrupt failure becomes "not connected"
    except Exception:
        logger.warning("MCP server '%s': could not start a transport for project %s", server_name, cwd, exc_info=True)
        return None
    logger.info("MCP server '%s': started a dedicated stdio transport for project %s", server_name, cwd)
    return clone


def _lease(clone: Any) -> None:
    """Count *clone* as in use by the running handler. Caller holds ``_core._lock``, so a clone
    is leased in the same critical section that hands it out: eviction never sees a gap."""
    leases = _call_leases.get()
    if leases is not None:
        clone._scoped_leases += 1
        leases.append(clone)


def leased_handler(handler):
    """Wrap a tool/utility handler so every clone it touches stays un-evictable until it returns."""
    @wraps(handler)
    def _leased(args: dict, **kwargs):
        leases: list = []
        token = _call_leases.set(leases)
        try:
            return handler(args, **kwargs)
        finally:
            with _core._lock:
                for clone in leases:
                    clone._scoped_leases -= 1
            _call_leases.reset(token)
    return _leased


def _evict_idle_overflow(key: Any, *, keep: tuple) -> None:
    """Shut down the least recently used idle clones of baseline *key* beyond the cap."""
    with _core._lock:
        clones = [(k, s) for k, s in _scoped_servers.items() if k[0] == key]
        excess = len(clones) - _MAX_SCOPED_TRANSPORTS_PER_SERVER
        if excess <= 0:
            return
        victims = []
        for clone_key, clone in sorted(clones, key=lambda entry: entry[1]._last_tool_call_at):
            if len(victims) == excess:
                break
            connect_lock = _scoped_connect_locks.get(clone_key)
            if (clone_key == keep or clone._scoped_leases or clone._rpc_lock.locked()
                    or (connect_lock is not None and connect_lock.locked())):
                continue
            victims.append(clone)
            _scoped_servers.pop(clone_key)
            _scoped_connect_locks.pop(clone_key, None)
            # The victim's keyed ledgers go with it, like pop_clones: a stale breaker strike or
            # connect cooldown would resurrect when the project reconnects — opening the fresh
            # child's breaker early or pausing its first connect (design §5A "pop the clone from
            # every keyed ledger"; T00020 item 1).
            _scoped_connect_failures.pop(clone_key, None)
            _scoped_connect_retry_after.pop(clone_key, None)
            _core._server_error_counts.pop(clone_key, None)
            _core._server_breaker_opened_at.pop(clone_key, None)
            _core._server_errors_all_application.pop(clone_key, None)
    from agent.async_utils import safe_schedule_threadsafe
    for clone in victims:  # never await under _core._lock: shutdown closes the anyio transport
        # Fire-and-forget ON THE MCP LOOP, never the caller's thread: the first call from a 9th
        # project must not block 15s on another project's teardown, and an interrupt landing
        # before the shutdown starts would leave the victim popped with its child alive
        # (#151827 port, review finding 6). Scheduled unwrapped, like shutdown_mcp_servers
        # schedules baseline shutdowns: nothing in MCPServerRun.shutdown reads the caller's
        # home-override or dashboard-OAuth context.
        with _core._lock:
            loop = _core._mcp_loop
        future = safe_schedule_threadsafe(clone.shutdown(), loop, logger=logger,
                                          log_message=f"MCP server '{clone.name}': failed to schedule the evicted transport's shutdown")
        if future is None:
            _hand_clone_pids_to_reaper(clone)  # nothing will ever run this shutdown
            continue

        def _reap_if_failed(fut, _clone=clone):
            try:
                exc = fut.exception()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):  # no shutdown; reaper owns child
                logger.warning("MCP server '%s': the evicted transport's shutdown was cancelled",
                               _clone.name)
                _hand_clone_pids_to_reaper(_clone)
                return
            if exc is not None:
                # health: allow BLE001 -- one failed close must not stop the others
                logger.warning("MCP server '%s': idle project transport shutdown failed",
                               _clone.name, exc_info=exc)
                _hand_clone_pids_to_reaper(_clone)

        future.add_done_callback(_reap_if_failed)


def pop_clones(keys: Optional[set]) -> list:
    """Remove and return the clones of the baseline *keys* (every clone when None) for the caller to
    shut down with their baselines. Caller holds ``_core._lock``."""
    popped = [k for k in _scoped_servers if keys is None or k[0] in keys]
    # Tombstone the dying baselines NOW, while their registry entries are still live: the async
    # ``_shutdown`` pops ``_core._servers`` only later, on the MCP loop, and a clone connect
    # completing in that window would pass the insert gate's ``is baseline`` re-check and strand
    # its live child with no owner (review finding 1). Cleared in ``clear_selected_status`` once
    # the deferred pop has landed. Only keys with a live baseline are tombstoned — the gate's
    # ``is baseline`` check already refuses an absent baseline, and a tombstone for a key nothing
    # registered would never be cleared (a re-registered same-named server must be able to grow
    # clones again).
    _torn_down_baselines.update(set(_core._servers) if keys is None else keys & set(_core._servers))
    for clone_key in popped:
        _scoped_connect_locks.pop(clone_key, None)
        # The clone's per-project bookkeeping goes with it (design section "Lock cleanup": pop the
        # clone from every keyed ledger) — a stale cooldown or breaker would pause the project
        # after a reload that dropped it, and a re-registered one must start clean.
        _scoped_connect_failures.pop(clone_key, None)
        _scoped_connect_retry_after.pop(clone_key, None)
        _core._server_error_counts.pop(clone_key, None)
        _core._server_breaker_opened_at.pop(clone_key, None)
        _core._server_errors_all_application.pop(clone_key, None)
    return [_scoped_servers.pop(k) for k in popped]


def rekey_unscoped(scope: str) -> list:
    """Follow ``tools.mcp_tool_scope.rekey_unscoped_connections``: when a single-profile process
    starts multiplexing, every bare-name baseline key becomes ``(scope, name)`` and its project
    clones must follow — otherwise the pre-flip clones (live children, up to the cap per server)
    turn invisible to lookup, eviction and scoped teardown, and only a wildcard shutdown reaps
    them (#151827 port, review finding 3). A connect still in flight under an old key keeps its
    own lock and waiter entries: its baseline check now fails, so it discards its clone and its
    release reaps the entries. A clone whose project a post-flip twin already serves keeps the
    seat only while in use; the redundant one is returned for the caller to shut down off the
    registry lock. Caller holds ``_core._lock``."""
    orphans = []
    for clone_key in [k for k in _scoped_servers if isinstance(k[0], str)]:
        new_key = ((scope, clone_key[0]), clone_key[1])
        clone = _scoped_servers.pop(clone_key)
        existing = _scoped_servers.get(new_key)
        if existing is None or not existing._scoped_leases:
            _scoped_servers[new_key] = clone
            if existing is not None:
                orphans.append(existing)  # an idle duplicate of a live project
        else:
            orphans.append(clone)  # the live twin keeps the seat; this one is the leftover
    for lock_key in [k for k in _scoped_connect_locks if isinstance(k[0], str)]:
        # Waiter seats are left alone: their releases reap them (a seat with a free lock is a
        # thread still acquiring it — deleting the lock under it would split the generation).
        lock = _scoped_connect_locks[lock_key]
        if not lock.locked() and lock_key not in _scoped_connect_waiters:
            del _scoped_connect_locks[lock_key]
    for sidecar in (_scoped_connect_failures, _scoped_connect_retry_after):
        for clone_key in [k for k in sidecar if isinstance(k[0], str)]:
            moved: Any = sidecar.pop(clone_key)
            sidecar[((scope, clone_key[0]), clone_key[1])] = moved
    return orphans
