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
import contextvars
import logging
import os
import threading
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
# Clone key -> handlers currently holding it (lookup through release, retries included).
_scoped_active_calls: dict[tuple[Any, str], int] = {}

# The handler's lease list: every clone a lookup hands this handler is counted in
# ``_scoped_active_calls`` until the handler returns. None outside a leased handler.
_call_leases: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("mcp_scoped_call_leases", default=None)


def _caller_cwd() -> str:
    """The caller's canonical project dir; must run on the CALLER's thread (the session cwd is a
    ContextVar the MCP loop cannot see). ``resolve_agent_cwd`` only accepts real local dirs, so a
    remote-backend path or a deleted worktree falls back to the host default instead of reaching
    ``Popen(cwd=...)``. realpath: ``/p``, ``/p/`` and a symlink to it are one project."""
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
            or "cwd" in (baseline._config or {}) or not baseline._spawn_cwd):
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
    pids = set(getattr(orphan, "_stdio_child_pids", ()) or ())
    if not pids:
        return
    try:
        orphan._release_spawned_children(pids)
    # health: allow BLE001 -- best-effort ledger handover; the final sweep still owns these PIDs
    except Exception:
        logger.warning("MCP server '%s': could not hand a discarded transport's PIDs to the reaper",
                       server_name, exc_info=True)


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
            _lease(clone_key)
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
            if clone is None and _core._servers.get(key) is not baseline:
                _release_connect_lock(clone_key, keep=False)
                return None
        if clone is None:
            clone = _connect_clone(server_name, baseline, cwd)
            if clone is None:
                with _core._lock:  # transient failure: peers keep single-flighting; the lock is reaped when the last leaves
                    _release_connect_lock(clone_key, keep=False)
                return None
        with _core._lock:
            # A teardown (reload, config prune, shutdown) can remove the baseline while a clone
            # connect is in flight; inserting afterwards would strand a live child no owner will
            # ever reap or re-register. The clone goes down with its baseline or not at all.
            if _core._servers.get(key) is baseline:
                _scoped_servers[clone_key] = clone
                _lease(clone_key)
                _release_connect_lock(clone_key, keep=True)
            else:
                orphan = clone
                _release_connect_lock(clone_key, keep=False)
    if orphan is not None:
        _discard_orphan_clone(orphan, server_name, cwd)
        return None
    _evict_idle_overflow(key, keep=clone_key)
    return clone


def _connect_clone(server_name: str, baseline: Any, cwd: str) -> Any:
    from tools import mcp_tool_discovery as _discovery  # lazy: discovery imports this module
    # The baseline's rendered config, not a config.yaml re-read: an edit mid-conversation must not
    # give one project a different server than the tools were discovered from.
    config = dict(baseline._config, cwd=cwd)
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


def _lease(clone_key: tuple) -> None:
    """Count *clone_key* as in use by the running handler. Caller holds ``_core._lock``, so a
    clone is leased in the same critical section that hands it out: eviction never sees a gap."""
    leases = _call_leases.get()
    if leases is not None:
        _scoped_active_calls[clone_key] = _scoped_active_calls.get(clone_key, 0) + 1
        leases.append(clone_key)


def leased_handler(handler):
    """Wrap a tool/utility handler so every clone it touches stays un-evictable until it returns."""
    @wraps(handler)
    def _leased(args: dict, **kwargs):
        token = _call_leases.set([])
        try:
            return handler(args, **kwargs)
        finally:
            with _core._lock:
                for clone_key in _call_leases.get():
                    remaining = _scoped_active_calls.get(clone_key, 1) - 1
                    if remaining:
                        _scoped_active_calls[clone_key] = remaining
                    else:
                        _scoped_active_calls.pop(clone_key, None)
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
            if (clone_key == keep or _scoped_active_calls.get(clone_key) or clone._rpc_lock.locked()
                    or (connect_lock is not None and connect_lock.locked())):
                continue
            victims.append(clone)
            _scoped_servers.pop(clone_key)
            _scoped_connect_locks.pop(clone_key, None)
    for clone in victims:  # never await under _core._lock: shutdown closes the anyio transport
        try:
            _loop._run_on_mcp_loop(clone.shutdown, timeout=15)
        # health: allow BLE001 -- one failed close must not stop the others; the orphan reaper still owns its PIDs
        except Exception:
            logger.warning("MCP server '%s': idle project transport shutdown failed", clone.name, exc_info=True)


def pop_clones(keys: Optional[set]) -> list:
    """Remove and return the clones of the baseline *keys* (every clone when None) for the caller to
    shut down with their baselines. Caller holds ``_core._lock``."""
    popped = [k for k in _scoped_servers if keys is None or k[0] in keys]
    for clone_key in popped:
        _scoped_connect_locks.pop(clone_key, None)
    return [_scoped_servers.pop(k) for k in popped]
