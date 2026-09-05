"""MCP process lifecycle: stdio child PID tracking and orphan cleanup, graceful
server shutdown and draining of the background MCP loop."""

import logging
import asyncio
import os
import threading
import time
from typing import Dict, Optional
from tools.mcp_tool_common import _core
from tools import mcp_tool_loop as _loop

logger = logging.getLogger("tools.mcp_tool")

# Live stdio MCP children (pid -> server_name), added after connection and removed on normal
# shutdown, so they can be force-killed if SDK teardown fails.
_stdio_pids: Dict[int, str] = {}
# PIDs that survived their session context exit (detected in _run_stdio's finally, reaped by
# _kill_orphaned_mcp_children). Separate from _stdio_pids so sweeps never race active sessions.
_orphan_stdio_pids: set = set()
_orphan_stdio_pid_servers: Dict[int, str] = {}
# pid -> pgid captured at spawn. The SDK spawns with start_new_session=True (PGID == PID);
# grandchildren keep that PGID after the direct child exits, so killpg still reaches them.
# Separate from _stdio_pids so the PGID survives the child's removal. Empty on Windows.
_stdio_pgids: Dict[int, int] = {}


def _snapshot_child_pids() -> set:
    """Current direct-child PIDs: /proc on Linux, else psutil, else empty set."""
    my_pid = os.getpid()
    # /proc/<pid>/task/<tid>/children is per-THREAD, and stdio_client() spawns from the MCP
    # loop thread, so union every task's children — reading only the main thread's file
    # returns an empty set on every Linux install.
    try:
        # ``/proc/<pid>/task/<tid>/children`` is per-THREAD — a child forked from thread T is listed only
        # under T's task dir. stdio_client() spawns from the background MCP loop thread, so reading only the
        # main thread's file (``task/<pid>/children``) returned an empty set on every Linux install and left
        # ``_stdio_child_pids`` / ``_stdio_pids`` empty: the #81995 dead-child fast-fail, the #96452 respawn
        # signal, and the killpg shutdown sweep never saw the subprocess.
        task_dir = f"/proc/{my_pid}/task"
        found: set = set()
        for tid in os.listdir(task_dir):
            try:
                with open(f"{task_dir}/{tid}/children", encoding="utf-8") as f:
                    found.update(int(p) for p in f.read().split() if p.strip())
            except (FileNotFoundError, OSError, ValueError):
                continue  # thread exited between listdir and open
        return found
    except (FileNotFoundError, OSError, ValueError):
        pass
    try:
        import psutil
        return {c.pid for c in psutil.Process(my_pid).children()}
    except Exception:
        return set()


# argv markers of non-MCP gateway children that can race into the snapshot delta during an
# MCP spawn (defense-in-depth; LSP/slash_worker already use start_new_session). Matched against
# argv[1:] because Python/Java children start with the interpreter path.
_NON_MCP_CHILD_CMDLINE_MARKERS: tuple[str, ...] = (
    "tui_gateway.slash_worker", "tui_gateway.entry",
    "-dorg.eclipse.equinox.launcher", "eclipse.jdt.ls", "org.eclipse.equinox.launcher_",  # jdtls
)


def _filter_mcp_children(pids: set) -> set:
    """Drop non-MCP children from a PID snapshot delta. Tracking a stray child in _stdio_pgids
    is catastrophic if it lacks start_new_session: its pgid can be the TUI parent's, so the
    shutdown killpg() would kill the TUI itself."""
    if not pids:
        return pids
    try:
        import psutil
    except ImportError:
        return pids  # keep all PIDs (prior behavior)
    kept = set()
    for pid in pids:
        try:
            argv = psutil.Process(pid).cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue  # raced away or zombie — cannot be our fresh server, unsafe to track
        if not any(marker in arg for arg in argv[1:] for marker in _NON_MCP_CHILD_CMDLINE_MARKERS):
            kept.add(pid)
    return kept


def _clear_connect_cooldowns() -> None:
    """Drop connect-retry cooldowns: a restart must re-attempt every server immediately, not
    honour a stale per-server backoff. Caller holds ``_core._lock``."""
    _core._server_connect_retry_after.clear()
    _core._server_connect_failures.clear()


def shutdown_mcp_servers(*, scope: Optional[str] = None):
    """Close MCP server connections (in parallel) and stop the background loop. Each server
    Task is signalled to exit its own ``async with`` so the anyio cancel-scope cleanup runs in
    the Task that opened it. ``scope`` restricts teardown to one multiplexed profile's servers
    (its ``/reload-mcp`` must not kill other profiles') and leaves the shared loop running if
    anything else is still connected."""
    with _core._lock:
        selected = [name for name in _core._servers if scope is None or _core._server_scope_keys.get(name) == scope]
        servers_snapshot = [_core._servers[name] for name in selected]

    # Fast path: nothing to shut down. The connect-cooldown maps can still be populated here — a server that
    # failed to connect is never recorded in ``_servers`` (that is the very premise of the #50394 cooldown),
    # so "no live servers" is the MOST likely state in which stale backoff entries exist. Clear them so a
    # post-shutdown restart re-attempts every configured server immediately.
    if servers_snapshot:
        async def _shutdown():
            results = await asyncio.gather(*(server.shutdown() for server in servers_snapshot), return_exceptions=True)
            for server, result in zip(servers_snapshot, results):
                if isinstance(result, Exception):
                    logger.debug("Error closing MCP server '%s': %s", server.name, result)
            with _core._lock:
                for name in selected:
                    _core._servers.pop(name, None)
                    _core._server_scope_keys.pop(name, None)
                _clear_connect_cooldowns()

        with _core._lock:
            loop = _core._mcp_loop
        if loop is not None and loop.is_running():
            from agent.async_utils import safe_schedule_threadsafe
            future = safe_schedule_threadsafe(_shutdown(), loop, logger=logger, log_message="MCP shutdown: failed to schedule")
            if future is not None:
                try:
                    future.result(timeout=15)
                except BaseException as exc:
                    logger.debug("Error during MCP shutdown: %s", exc)

    # Unconditional final sweep: whether ``_shutdown`` ran, timed out, or was never scheduled
    # (a server that failed to connect is never in ``_servers`` — the most likely state for
    # stale backoff entries), no connect-cooldown state may survive shutdown.
    with _core._lock:
        _clear_connect_cooldowns()
    _loop._stop_mcp_loop(only_if_idle=scope is not None)


def _take_reapable_pids(include_active: bool, server_name: Optional[str]) -> tuple[Dict[int, str], Dict[int, int]]:
    """Pop the PIDs to reap (and their spawn-time pgids) out of the ledgers under the lock, so
    a future spawn can't collide with stale state. Returns ``(pid -> owner, pid -> pgid)``."""
    def _owned(entries: Dict[int, str]) -> Dict[int, str]:
        return {pid: owner for pid, owner in entries.items() if server_name is None or owner == server_name}

    with _core._lock:
        pids = _owned({opid: _orphan_stdio_pid_servers.get(opid, "orphan") for opid in _orphan_stdio_pids})
        _orphan_stdio_pids.difference_update(pids)
        for opid in pids:
            _orphan_stdio_pid_servers.pop(opid, None)
        if include_active:
            active = _owned(_stdio_pids)
            pids.update(active)
            for pid in active:
                _stdio_pids.pop(pid, None)
        pgids = {pid: _stdio_pgids.pop(pid) for pid in pids if pid in _stdio_pgids}
    return pids, pgids


def _signal_mcp_process(pid: int, sig: int, server_name: str, pgid: Optional[int], my_pgid: Optional[int]) -> None:
    """SIGTERM/SIGKILL via the spawn-time pgroup on POSIX (reaches reparented grandchildren),
    falling back to a per-pid signal."""
    killpg = getattr(os, "killpg", None)
    if pgid is not None and killpg is not None:
        if my_pgid is not None and pgid == my_pgid:
            # Child shares the gateway's pgroup: killpg would kill the gateway too, so use
            # per-pid kill. Warn because per-pid kill can't reach grandchildren in this group.
            logger.warning("MCP server '%s' pgid %d matches gateway pgid; skipping "
                           # Fall through to the per-pid kill() path instead. Warn because per-pid kill
                           # cannot reach grandchildren in this shared group — if the direct child has
                           # already exited, they may leak (inherent: group-killing them would also kill the
                           # gateway). See #47134.
                           "killpg to avoid self-kill and using per-pid kill — any "
                           "grandchildren in this group may not be reaped", server_name, pgid)
        else:
            try:
                killpg(pgid, sig)
                return
            except (ProcessLookupError, PermissionError, OSError) as exc:
                # Pgroup gone or refused — still try the direct child.
                logger.debug("killpg(%d, %d) failed for MCP server '%s': %s; falling back to kill(pid)",
                             pgid, sig, server_name, exc)
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _kill_orphaned_mcp_children(include_active: bool = False, server_name: Optional[str] = None) -> None:
    """Best-effort reap of stdio MCP subprocesses: SIGTERM, wait 2s, SIGKILL survivors. By
    default only ``_orphan_stdio_pids`` are reaped so concurrent cron jobs / live sessions are
    untouched; ``include_active=True`` also kills every ``_stdio_pids`` entry and is only for
    final shutdown after the MCP loop has stopped. ``server_name`` limits the sweep to one
    server (stdio reconnects cleaning up their old transport)."""
    import signal as _signal
    pids, pgids = _take_reapable_pids(include_active, server_name)
    if not pids:  # skip the 2s sleep every MCP-free shutdown would otherwise pay
        return

    try:  # our own pgid, so we never killpg() the gateway itself
        my_pgid = os.getpgrp()
    except (AttributeError, OSError):
        my_pgid = None  # Windows or restricted environment

    for pid, owner in pids.items():
        _signal_mcp_process(pid, _signal.SIGTERM, owner, pgids.get(pid), my_pgid)
        logger.debug("Sent SIGTERM to orphaned MCP process %d (%s)", pid, owner)
    time.sleep(2)
    sigkill = getattr(_signal, "SIGKILL", _signal.SIGTERM)
    from gateway.status import _pid_exists  # ``os.kill(pid, 0)`` is NOT a no-op on Windows
    for pid, owner in pids.items():
        if _pid_exists(pid):  # survived SIGTERM
            _signal_mcp_process(pid, sigkill, owner, pgids.get(pid), my_pgid)
            logger.warning("Force-killed MCP process %d (%s) after SIGTERM timeout", pid, owner)
    # These groups are reaped. Release them last, so a crash partway through the SIGTERM/SIGKILL
    # dance still leaves the supervisor holding them.
    _core._update_death_supervisor("unregister", pgids.values())


# Background orphan janitor (#81880). The teardown finally in _run_stdio
# correctly marks live children as orphans when a session closes — but in a
# long-lived desktop/gateway process, nothing reaps _orphan_stdio_pids until
# the whole process exits. Every closed session's leaked stdio children
# therefore accumulate for the lifetime of the process (field reports: ~30
# python + ~28 node processes resident after days of desktop-app use, OOM
# on 16GB Macs, sustained fan/thermal load on Windows laptops). The existing
# reapers all target different producers:
#
#   - _stop_mcp_loop's exit sweep (process exit only)
#   - the CLI session-rotation boundary reaper (#82141, cli.py only)
#   - the parent-death watchdog (ungraceful exit of this process)
#
# This janitor closes the remaining gap: one lazily-started daemon thread
# that periodically calls _kill_orphaned_mcp_children() (the sweep that
# already exists and is race-safe against live sessions by design — it only
# touches _orphan_stdio_pids). The thread starts when the first orphan is
# recorded and exits when no orphans remain for a while, so MCP-free and
# clean-teardown processes pay nothing.
_orphan_janitor_interval = 30.0  # seconds between sweeps
_orphan_janitor_idle_stops = 3  # consecutive empty sweeps before the thread exits
_orphan_janitor_thread: Optional[threading.Thread] = None
_orphan_janitor_lock = threading.Lock()
_orphan_janitor_wakeup: Optional[threading.Event] = None


def _maybe_start_orphan_janitor() -> None:
    """Start the background orphan reaper if it is not running.

    Called whenever an orphan PID is recorded, so the janitor exists exactly
    when there is something to reap. Cheaply idempotent.
    """
    global _orphan_janitor_thread, _orphan_janitor_wakeup
    with _orphan_janitor_lock:
        if _orphan_janitor_thread is not None and _orphan_janitor_thread.is_alive():
            if _orphan_janitor_wakeup is not None:
                # Nudge an idle-waiting janitor so a freshly-recorded orphan
                # is swept promptly instead of after the next poll tick.
                _orphan_janitor_wakeup.set()
            return
        _orphan_janitor_wakeup = threading.Event()

        def _janitor() -> None:
            empty_streak = 0
            while True:
                wakeup = _orphan_janitor_wakeup
                if wakeup is not None:
                    wakeup.wait(timeout=_orphan_janitor_interval)
                    wakeup.clear()
                with _core._lock:
                    pending = len(_orphan_stdio_pids)
                if pending:
                    empty_streak = 0
                    try:
                        _kill_orphaned_mcp_children()
                    except Exception:
                        # The sweep already logs per-pid failures; never let
                        # the janitor die on an unexpected error.
                        logger.debug(
                            "MCP orphan janitor sweep raised", exc_info=True
                        )
                else:
                    empty_streak += 1
                    if empty_streak >= _orphan_janitor_idle_stops:
                        # Nothing left to reap for a while. A fresh thread
                        # starts when the next orphan appears.
                        global _orphan_janitor_thread
                        with _orphan_janitor_lock:
                            if (
                                _orphan_janitor_thread is not None
                                and threading.current_thread()
                                is _orphan_janitor_thread
                            ):
                                _orphan_janitor_thread = None
                        return

        thread = threading.Thread(
            target=_janitor, name="mcp-orphan-janitor", daemon=True
        )
        _orphan_janitor_thread = thread
        thread.start()
        logger.debug(
            "MCP orphan janitor started (interval %.0fs)", _orphan_janitor_interval
        )


def _stop_mcp_loop_if_idle() -> bool:
    """Stop the MCP loop only when no registered server still owns it. Probe paths create
    temporary MCPServerTasks not placed in ``_servers``; they may clean up an idle loop but
    must not tear down the process-global loop under live agent tools."""
    return _loop._stop_mcp_loop(only_if_idle=True)


async def _drain_mcp_loop_tasks(*, timeout: Optional[float] = None) -> None:
    """Cancel every task still pending on the MCP loop and reap it. ``Task.cancel()`` only
    schedules the throw, so tasks need a cancellation cycle before the loop goes away; wait
    for them here, on their owning loop, bounded so a task that suppresses cancellation
    cannot hang process exit."""
    if timeout is None:
        timeout = _core._MCP_LOOP_DRAIN_TIMEOUT
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    if not pending:
        return
    logger.debug("Draining %d pending task(s) from the MCP loop", len(pending))
    for task in pending:
        task.cancel()
    done, still_pending = await asyncio.wait(pending, timeout=timeout)
    for task in done:
        if not task.cancelled():
            task.exception()  # mark retrieved so asyncio doesn't warn "exception was never retrieved"
    if still_pending:
        logger.warning("%d MCP loop task(s) still pending after %.1fs drain", len(still_pending), timeout)


async def _drain_and_stop_mcp_loop() -> None:
    """Drain pending tasks, then stop the loop from its owning thread. Both must run as one
    loop-owned sequence: a ``loop.stop`` queued separately by a timed-out caller can overtake
    the scheduled drain, leaving the drain coroutine itself pending when the loop is closed."""
    loop = asyncio.get_running_loop()
    try:
        await _drain_mcp_loop_tasks(timeout=_core._MCP_LOOP_DRAIN_TIMEOUT)
    finally:
        loop.call_soon(loop.stop)
