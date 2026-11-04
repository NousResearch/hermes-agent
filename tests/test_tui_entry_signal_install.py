"""Regression tests: ``tui_gateway.entry`` must be importable off the main thread.

The module used to call ``signal.signal()`` at import time. ``tui_gateway/server.py``
imports ``ensure_mcp_discovery_started`` from it lazily inside the dashboard /
TUI-gateway server thread, so every TUI session logged
``MCP discovery startup failed: signal only works in main thread`` and silently
lost MCP auto-discovery. Signal handlers are now installed by
``_install_signal_handlers()`` from ``main()`` only.
"""

import signal
import subprocess
import sys
import threading

from tui_gateway import entry


def test_import_from_worker_thread_does_not_touch_signals():
    """Fresh interpreter: import the module on a non-main thread and make
    sure (a) it succeeds and (b) no handler was installed by the import."""
    code = r"""
import signal, sys, threading
before = {s: signal.getsignal(getattr(signal, s)) for s in ("SIGPIPE", "SIGTERM") if hasattr(signal, s)}
res = {}
def t():
    try:
        from tui_gateway.entry import ensure_mcp_discovery_started
        res["ok"] = True
    except Exception as exc:
        res["err"] = repr(exc)
th = threading.Thread(target=t); th.start(); th.join()
after = {s: signal.getsignal(getattr(signal, s)) for s in before}
assert res.get("ok"), res
assert before == after, (before, after)
# entry.py re-routes sys.stdout (it owns the JSON-RPC pipe) — report on stderr.
print("ok", file=sys.stderr)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.strip().endswith("ok"), proc.stderr


def test_install_signal_handlers_only_installs_on_main_thread(monkeypatch):
    """``_install_signal_handlers`` wires SIGPIPE→SIG_IGN and SIGTERM→handler
    when called from the main thread (the ``main()`` path)."""
    assert threading.current_thread() is threading.main_thread()
    calls = []
    monkeypatch.setattr(
        entry.signal, "signal", lambda sig, handler: calls.append((sig, handler))
    )
    entry._install_signal_handlers()
    sigs = {sig for sig, _ in calls}
    if hasattr(signal, "SIGPIPE"):
        assert signal.SIGPIPE in sigs
        assert dict(calls)[signal.SIGPIPE] is signal.SIG_IGN
    if hasattr(signal, "SIGTERM"):
        assert signal.SIGTERM in sigs
        assert dict(calls)[signal.SIGTERM] is entry._log_signal


def test_main_installs_signal_handlers_before_anything_else(monkeypatch):
    order = []
    monkeypatch.setattr(entry, "_install_signal_handlers", lambda: order.append("signals"))
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: order.append("sidecar"))

    class _Stop(Exception):
        pass

    def _stop():
        raise _Stop()

    # Abort main() right after the two installers so we don't start a real TUI.
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", _stop)
    try:
        entry.main()
    except _Stop:
        pass
    assert order == ["signals", "sidecar"]
