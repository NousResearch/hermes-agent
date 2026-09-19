"""Regression test: importing tui_gateway.entry off the main thread must not crash.

Background
----------
``entry.py`` installs signal handlers (SIGPIPE, SIGTERM, …) at *import time*.
``signal.signal`` is only legal in the main thread, so a first import of
``entry`` from a worker thread raises
``ValueError: signal only works in main thread of the main interpreter``.

The Desktop/WebSocket agent-build path (``server._build``) runs in a daemon
thread and does ``from tui_gateway.entry import ensure_mcp_discovery_started``
(server.py:1866).  On that path ``entry.main()`` is never run, so the worker
thread performs the *first* import of ``entry`` — which crashed and aborted
MCP discovery startup (#72667, regression from the 2026-07-26 websocket-MCP
discovery fix).

Fix: guard the import-time signal installs with a main-thread check.  Handlers
are process-global, so installing them only when reached in the main thread is
sufficient; importing from a worker thread becomes a safe no-op.

This test runs the import in a *fresh* subprocess worker thread to guarantee a
clean first-import (the test process already imported entry in its main
thread, so an in-process import would not reproduce the bug).
"""

import subprocess
import sys
from pathlib import Path

# Derive from __file__ so the test is not cwd-dependent (works regardless of
# where pytest is invoked from). Sibling tests use the identical pattern.
REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _spawn_worker_import_entry():
    """Run `import tui_gateway.entry` for the first time inside a worker thread.

    Returns (returncode, stdout, stderr).
    """
    code = (
        "import threading, sys, signal, os\n"
        "sys.stdout.reconfigure(line_buffering=True)\n"
        "sys.stderr.reconfigure(line_buffering=True)\n"
        "errs = []\n"
        "def _worker():\n"
        "    try:\n"
        "        import tui_gateway.entry\n"
        "    except Exception as e:\n"
        "        errs.append(repr(e))\n"
        "t = threading.Thread(target=_worker, daemon=True)\n"
        "t.start(); t.join(timeout=15)\n"
        "if errs:\n"
        "    sys.stdout.write('IMPORT_FAILED: ' + errs[0] + '\\n')\n"
        "    sys.exit(2)\n"
        "# main thread of this process still installs SIGPIPE handler\n"
        "# SIGPIPE is POSIX-only; guard so the probe works on Windows.\n"
        "if hasattr(signal, 'SIGPIPE'):\n"
        "    h = signal.getsignal(signal.SIGPIPE)\n"
        "    sys.stdout.write('OK handler_installed=' + str(h is signal.SIG_IGN or callable(h)) + '\\n')\n"
        "else:\n"
        "    sys.stdout.write('OK no_sigpipe_on_platform\\n')\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={**__import__("os").environ},
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_entry_imports_cleanly_from_worker_thread():
    """First import of tui_gateway.entry from a worker thread must succeed."""
    rc, out, err = _spawn_worker_import_entry()
    # entry import may emit to stdout or stderr depending on the runner; check both.
    combined = out + err
    assert rc == 0, f"entry import off main thread failed (rc={rc}): {err!r}"
    assert "OK" in combined, f"unexpected output: {out!r} / {err!r}"


