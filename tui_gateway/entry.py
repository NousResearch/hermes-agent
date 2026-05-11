import os
import sys

# Guard against a local utils/ (or other package) in CWD shadowing installed
# hermes modules.  hermes_cli sets HERMES_PYTHON_SRC_ROOT before spawning this
# subprocess; inserting it first ensures the installed packages win.
_src_root = os.environ.get("HERMES_PYTHON_SRC_ROOT", "")
if _src_root and _src_root not in sys.path:
    sys.path.insert(0, _src_root)
# Strip '' and '.' — both resolve to CWD at import time and can let a local
# directory shadow installed packages.
sys.path = [p for p in sys.path if p not in {"", "."}]

import json
import signal
import time
import traceback

from tui_gateway import server
from tui_gateway.server import _CRASH_LOG, dispatch, resolve_skin, write_json
from tui_gateway.transport import TeeTransport


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


def main():
    _install_sidecar_publisher()

    # MCP tool discovery — inline is safe here: TUI entry is a plain
    # sync loop with no asyncio event loop to block.  Previously ran as
    # a model_tools.py module-level side effect; moved to explicit
    # startup calls to avoid freezing the gateway's loop on lazy import
    # (#16856).
    #
    # Cold-start guard: importing ``tools.mcp_tool`` transitively pulls the
    # full MCP SDK (mcp, pydantic, httpx, jsonschema, starlette parsers —
    # ~200ms on macOS), which runs on the TUI's critical path before
    # ``gateway.ready`` can be emitted.  The overwhelming majority of users
    # have no ``mcp_servers`` configured, in which case every byte of that
    # import is wasted.  Check the config first (cheap — it's already been
    # loaded once by ``_config_mtime`` elsewhere) and only pay the import
    # cost when there's actually MCP work to do.
    try:
        from hermes_cli.config import read_raw_config
        _mcp_servers = (read_raw_config() or {}).get("mcp_servers")
        _has_mcp_servers = isinstance(_mcp_servers, dict) and len(_mcp_servers) > 0
    except Exception:
        # Be conservative: if we can't decide, fall back to the old
        # behaviour and let the discovery path handle its own errors.
        _has_mcp_servers = True
    if _has_mcp_servers:
        try:
            from tools.mcp_tool import discover_mcp_tools
            discover_mcp_tools()
        except Exception:
            pass

    if not write_json({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": "gateway.ready", "payload": {"skin": resolve_skin()}},
    }):
        _log_exit("startup write failed (broken stdout pipe before first event)")
        sys.exit(0)

    for raw in sys.stdin:
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

    _log_exit("stdin EOF (TUI closed the command pipe)")


if __name__ == "__main__":
    main()
