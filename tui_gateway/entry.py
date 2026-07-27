import os
import sys

# Stop a ``utils/`` (or ``proxy/``, ``ui/``) package in the launch directory
# from shadowing Hermes's own top-level modules.  ``hermes_bootstrap`` lives at
# the repo root next to this package, so importing it is safe before the guard
# runs (its name won't collide with a user package), and it owns the canonical
# path-hardening logic shared with the other entry points.
import hermes_bootstrap

hermes_bootstrap.harden_import_path()

import json
import logging
import signal
import time
import traceback

from tui_gateway._stdin_recovery import handle_spurious_eof

from tui_gateway import server
from tui_gateway.server import _CRASH_LOG, dispatch, resolve_skin, write_json
from tui_gateway.transport import TeeTransport

logger = logging.getLogger(__name__)

# Handle for the background MCP tool-discovery thread (see
# ensure_mcp_discovery_started).  The first agent build briefly joins this so
# already-spawning fast servers land before the agent snapshots its tool list
# (see wait_for_mcp_discovery).  Stays None when discovery is delegated to the
# shared owner in hermes_cli.mcp_startup — the wait/in-flight/join helpers
# below consult both owners.
_mcp_discovery_thread = None

# True once ensure_mcp_discovery_started decided this process has MCP servers
# configured and spawned discovery through the shared owner. Lets
# wait_for_mcp_discovery re-invoke the (idempotent) spawn on later agent
# builds so the retry-after-zero-connected allowance in
# hermes_cli.mcp_startup.start_background_mcp_discovery can actually fire —
# without this, the single spawn is the only call and a first run that
# connected nothing latches the process MCP-less. Kept as a flag (rather than
# re-probing config) so non-MCP sessions never pay the tools.mcp_tool import
# on the per-agent-build wait path.
_mcp_discovery_enabled = False


def _install_sidecar_publisher() -> None:
    """Mirror every dispatcher emit to the dashboard sidebar via WS.

    Activated by `HERMES_TUI_SIDECAR_URL`, set by the dashboard's
    ``/api/pty`` endpoint when a chat tab passes a ``channel`` query param.
    Best-effort: connect failure or runtime drop falls back to stdio-only.
    """
    url = os.environ.get("HERMES_TUI_SIDECAR_URL")

    if not url:
        return

    from tui_gateway.event_publisher import WsPublisherTransport

    server._stdio_transport = TeeTransport(
        server._stdio_transport, WsPublisherTransport(url)
    )


# How long to wait for orderly shutdown (atexit + finalisers) before
# falling back to ``os._exit(0)`` so a wedged worker mid-flush can't
# strand the process.  1s covers the gateway's own shutdown work
# (thread-pool drain + session finalize) on every machine we've
# tested; override via ``HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S`` if a
# slower environment needs more headroom (e.g. encrypted disks
# flushing checkpoints) and accept that a longer grace also means a
# longer wait when shutdown actually deadlocks.
_DEFAULT_SHUTDOWN_GRACE_S = 1.0


def _shutdown_grace_seconds() -> float:
    raw = (os.environ.get("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S") or "").strip()
    if not raw:
        return _DEFAULT_SHUTDOWN_GRACE_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SHUTDOWN_GRACE_S
    return value if value > 0 else _DEFAULT_SHUTDOWN_GRACE_S


def _log_signal(signum: int, frame) -> None:
    """Capture WHICH thread and WHERE a termination signal hit us.

    SIG_DFL for SIGPIPE kills the process silently the instant any
    background thread (TTS playback, beep, voice status emitter, etc.)
    writes to a stdout the TUI has stopped reading.  Without this
    handler the gateway-exited banner in the TUI has no trace — the
    crash log never sees a Python exception because the kernel reaps
    the process before the interpreter runs anything.

    Termination semantics: ``sys.exit(0)`` here used to race the worker
    pool — a thread holding ``_stdout_lock`` mid-flush would block the
    interpreter shutdown indefinitely.  We now log the stack, give the
    process the configured shutdown grace
    (``HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S``, default
    ``_DEFAULT_SHUTDOWN_GRACE_S``) to drain naturally on a background
    thread, and fall back to ``os._exit(0)`` so a wedged write/flush
    can never strand the process.
    """
    # SIGPIPE and SIGHUP don't exist on Windows — build the lookup
    # dict from attributes that actually exist on the current platform.
    _signal_names: dict[int, str] = {}
    for _attr in ("SIGPIPE", "SIGTERM", "SIGHUP", "SIGINT", "SIGBREAK"):
        _sig = getattr(signal, _attr, None)
        if _sig is not None:
            _signal_names[int(_sig)] = _attr
    name = _signal_names.get(signum, f"signal {signum}")
    try:
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== {name} received · {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            if frame is not None:
                f.write("main-thread stack at signal delivery:\n")
                traceback.print_stack(frame, file=f)
            # All live threads — signal may have been triggered by a
            # background thread (write to broken stdout from TTS, etc.).
            import threading as _threading
            for tid, th in _threading._active.items():
                f.write(f"\n--- thread {th.name} (id={tid}) ---\n")
                f.write("".join(traceback.format_stack(sys._current_frames().get(tid))))
    except Exception:
        pass
    print(f"[gateway-signal] {name}", file=sys.stderr, flush=True)

    import threading as _threading

    def _hard_exit() -> None:
        # If a worker thread is still mid-flush on a half-closed pipe,
        # ``sys.exit(0)`` would wait forever for it to drop the GIL on
        # interpreter shutdown.  ``os._exit`` skips atexit handlers but
        # breaks the deadlock.  The crash log + stderr line above are
        # the forensic trail.
        os._exit(0)

    timer = _threading.Timer(_shutdown_grace_seconds(), _hard_exit)
    timer.daemon = True
    timer.start()

    # ── Flush sessions before exit ───────────────────────────────────
    # The atexit handler (_shutdown_sessions) is registered in
    # tui_gateway/server.py, but a worker thread holding the GIL or
    # _stdout_lock can block atexit from completing within the grace
    # window.  Explicitly finalize sessions here so that unpersisted
    # messages reach state.db before the hard-exit timer fires.
    try:
        from tui_gateway.server import _shutdown_sessions

        _shutdown_sessions()
    except Exception:
        pass

    try:
        sys.exit(0)
    except SystemExit:
        # Re-raise so the main-thread interpreter unwinds and runs
        # atexit + finalisers inside the grace window.  Python signal
        # handlers always run on the main thread, but a worker thread
        # holding ``_stdout_lock`` mid-flush can keep that unwind
        # waiting indefinitely; the daemon timer above is the safety
        # net for that exact case.
        raise


# SIGPIPE: ignore, don't exit. The old SIG_DFL killed the process
# silently whenever a *background* thread (TTS playback chain, voice
# debug stderr emitter, beep thread) wrote to a pipe the TUI had gone
# quiet on — even though the main thread was perfectly fine waiting on
# stdin.  Ignoring the signal lets Python raise BrokenPipeError on the
# offending write (write_json already handles that with a clean
# sys.exit(0) + _log_exit), which keeps the gateway alive as long as
# the main command pipe is still readable.  Terminal signals still
# route through _log_signal so kills and hangups are diagnosable.
#
# SIGPIPE and SIGHUP don't exist on Windows; guard each installation
# with hasattr so ``python -m tui_gateway.entry`` (spawned by
# ``hermes --tui``) imports cleanly there.  SIGBREAK (Windows' Ctrl+Break)
# is installed when available as a weaker equivalent of SIGHUP.
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _log_signal)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _log_signal)
elif hasattr(signal, "SIGBREAK"):
    # Windows-only: Ctrl+Break in a console window delivers SIGBREAK.
    # Route it through the same handler so kills are diagnosable.
    signal.signal(signal.SIGBREAK, _log_signal)
if hasattr(signal, "SIGINT"):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _log_exit(reason: str) -> None:
    """Record why the gateway subprocess is shutting down.

    Three exit paths (startup write fail, parse-error-response write fail,
    dispatch-response write fail, stdin EOF) all collapse into a silent
    sys.exit(0) here.  Without this trail the TUI shows "gateway exited"
    with no actionable clue about WHICH broken pipe or WHICH message
    triggered it — the main reason voice-mode turns look like phantom
    crashes when the real story is "TUI read pipe closed on this event".
    """
    try:
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== gateway exit · {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"· reason={reason} ===\n"
            )
    except Exception:
        pass
    print(f"[gateway-exit] {reason}", file=sys.stderr, flush=True)


def wait_for_mcp_discovery(timeout: "float | None" = None) -> None:
    """Block until background MCP discovery finishes, up to the resolved bound.

    MCP discovery runs in a daemon thread spawned at startup (see main()) so a
    slow/dead server can't freeze ``gateway.ready``.  But the agent snapshots
    its tool list ONCE at build time and never re-reads it, so a reachable-but-
    slow server that finishes connecting *after* the first prompt would be
    invisible for the whole session.  Joining with a bounded timeout before the
    first agent build lets already-spawning servers land without re-introducing
    the startup hang: ``thread.join(timeout)`` returns the instant discovery
    completes (so fast/no-MCP startups pay ~0s), and a dead server is simply not
    waited on beyond the bound.  No-op when no discovery thread was started.

    The bound comes from ``mcp_discovery_timeout`` in config (shared with the
    CLI path via ``hermes_cli.mcp_startup``); ``timeout`` overrides it.
    """
    thread = _mcp_discovery_thread
    if thread is not None and thread.is_alive():
        try:
            from hermes_cli.mcp_startup import _resolve_discovery_timeout

            bound = _resolve_discovery_timeout(timeout)
        except Exception:
            bound = timeout if timeout is not None else 0.75
        thread.join(timeout=bound)
        return
    # Discovery is spawned via the shared owner (ensure_mcp_discovery_started
    # → hermes_cli.mcp_startup); wait on it so the first agent build still
    # catches fast servers. Re-invoke the idempotent spawn first: if the
    # previous run finished with zero connected servers,
    # start_background_mcp_discovery's retry-after-zero-connected allowance
    # kicks off a fresh discovery run here instead of leaving the process
    # latched MCP-less for the session. In multi-profile processes this
    # retry runs under the CALLER's profile context (agent build binds the
    # session profile's HERMES_HOME first), so a launch profile with no
    # mcp_servers no longer starves selected profiles of discovery (#67605).
    # Gated on _mcp_discovery_enabled so non-MCP sessions never pay the
    # tools.mcp_tool import on the per-agent-build wait path.
    if not _mcp_discovery_enabled:
        return
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(
            logger=logger, thread_name="tui-mcp-discovery"
        )
    except Exception:
        logger.debug(
            "TUI MCP discovery retry-spawn failed", exc_info=True
        )
    try:
        from hermes_cli.mcp_startup import (
            wait_for_mcp_discovery as _startup_wait,
        )

        _startup_wait(timeout)
    except Exception:
        pass


def mcp_discovery_in_flight() -> bool:
    """Return True if ANY background MCP discovery thread is still running.

    Used by the agent-build path to decide whether to schedule a late tool
    snapshot refresh: if discovery didn't land within the bounded
    ``wait_for_mcp_discovery`` join, the agent was built without those tools
    and the banner/tool count will be stale until they arrive.

    There are two independent discovery-thread owners by surface: the stdio
    ``hermes --tui`` path spawns ITS thread here (``_mcp_discovery_thread``),
    while the desktop app + dashboard WebSocket sidecar (``tui_gateway/ws.py``)
    and ``hermes dashboard`` spawn theirs via
    ``hermes_cli.mcp_startup.start_background_mcp_discovery``. The late-refresh
    scheduler imports this function regardless of surface, so it MUST consult
    both — checking only the entry thread left the desktop/dashboard surfaces
    with no late refresh, so a slow MCP server's tools never surfaced for the
    whole session (#51587).
    """
    thread = _mcp_discovery_thread
    if thread is not None and thread.is_alive():
        return True
    try:
        from hermes_cli.mcp_startup import (
            mcp_discovery_in_flight as _startup_in_flight,
        )

        return _startup_in_flight()
    except Exception:
        return False


def join_mcp_discovery(timeout: float | None = None) -> bool:
    """Block until background MCP discovery finishes, up to ``timeout`` seconds.

    Returns True if discovery has completed (both thread owners absent or no
    longer alive), False if either is still running after the timeout. Unlike
    ``wait_for_mcp_discovery`` this accepts an unbounded/long wait and reports
    the outcome, for the off-critical-path late-refresh waiter.

    Joins both discovery-thread owners (see ``mcp_discovery_in_flight``): the
    entry thread first, then the ``hermes_cli.mcp_startup`` thread used by the
    desktop/dashboard surfaces. ``timeout`` bounds EACH join, mirroring the
    pre-#51587 single-owner behavior for the entry thread.
    """
    entry_done = True
    thread = _mcp_discovery_thread
    if thread is not None:
        thread.join(timeout=timeout)
        entry_done = not thread.is_alive()
    try:
        from hermes_cli.mcp_startup import join_mcp_discovery as _startup_join

        startup_done = _startup_join(timeout=timeout)
    except Exception:
        startup_done = True
    return entry_done and startup_done


# Spurious stdin-EOF recovery tracker (shared open-file-description O_NONBLOCK flip).
_recovery_times: list[float] = []



def _has_configured_mcp_servers() -> bool:
    """Return whether startup should attempt MCP discovery.

    Keep this cheap so non-MCP users do not pay the MCP SDK import cost.
    """
    try:
        from hermes_cli.config import read_raw_config

        mcp_servers = (read_raw_config() or {}).get("mcp_servers")
        return isinstance(mcp_servers, dict) and len(mcp_servers) > 0
    except Exception:
        # Be conservative: if we can't decide, fall back to attempting
        # discovery. The caller starts it in the background.
        return True


def ensure_mcp_discovery_started() -> None:
    """Start background MCP discovery for the current profile context, once.

    ``main()`` calls this for the stdio/TUI path. WebSocket/Desktop
    entrypoints can accept sessions without running ``main()``, so the
    agent-build path (``server._start_agent_build``) also calls it AFTER
    binding the session profile's HERMES_HOME override — the shared owner in
    ``hermes_cli.mcp_startup`` captures the caller's context-local override
    and propagates it into the discovery thread, so discovery reads the
    SELECTED profile's ``mcp_servers``, not the launch profile's (#67605).

    Delegating to the shared owner (instead of a hand-rolled thread) keeps
    the process-wide start lock, the retry-after-zero-connected allowance,
    and interactive-OAuth suppression.

    Known limitation: MCP tool registration is process-global, so in a
    multi-profile process the FIRST profile that builds an agent wins the
    discovery slot. Full per-profile MCP registries are tracked in #67605.
    """
    global _mcp_discovery_enabled

    if not _has_configured_mcp_servers():
        return
    _mcp_discovery_enabled = True
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(
            logger=logger, thread_name="tui-mcp-discovery"
        )
    except Exception:
        logger.warning(
            "Background MCP tool discovery failed to start", exc_info=True
        )


def main():
    _install_sidecar_publisher()

    # MCP tool discovery — backgrounded so a slow or unreachable MCP server
    # can't freeze TUI startup (a dead stdio/http server burns 1+2+4s of
    # connect retries → ~7s of dead air before the composer appears).  The
    # agent isn't built until the first prompt, at which point _make_agent
    # briefly joins the discovery thread (wait_for_mcp_discovery, bounded) so
    # already-spawning fast servers land in the tool snapshot.  The config
    # gate inside ensure_mcp_discovery_started keeps the ~200ms MCP SDK
    # import cost entirely off the path for users with no mcp_servers.
    ensure_mcp_discovery_started()

    if not write_json({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": "gateway.ready", "payload": {"skin": resolve_skin()}},
    }):
        _log_exit("startup write failed (broken stdout pipe before first event)")
        sys.exit(0)

    # Live-apply skins Hermes activates mid-conversation.
    server._ensure_skin_watcher()

    while True:
        raw = sys.stdin.readline()
        if not raw:
            # Stdin fell through — check if spurious (O_NONBLOCK flip by a
            # child on the shared open file description) or genuine EOF.
            if not handle_spurious_eof(_recovery_times, _log_exit):
                break
            continue

        line = raw.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            if not write_json({"jsonrpc": "2.0", "error": {"code": -32700, "message": "parse error"}, "id": None}):
                _log_exit("parse-error-response write failed (broken stdout pipe)")
                sys.exit(0)
            continue

        method = req.get("method") if isinstance(req, dict) else None
        resp = dispatch(req)
        if resp is not None:
            if not write_json(resp):
                _log_exit(f"response write failed for method={method!r} (broken stdout pipe)")
                sys.exit(0)


if __name__ == "__main__":
    main()
