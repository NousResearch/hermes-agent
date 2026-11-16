"""A fake ``chromium`` for launcher tests that behaves like the real one where the launcher depends on it.

It takes Chromium's profile singleton (a ``SingletonLock`` symlink ``<host>-<pid>``; a second instance on a
held dir hands over to the running one and exits 0), answers ``/json/version`` on an ephemeral loopback port,
and writes ``DevToolsActivePort`` once ready (after ``FAKE_CHROMIUM_READY_DELAY`` seconds, default 0). Every
start appends its pid to ``<udd>/spawns``; the instance that took the singleton also to ``<udd>/launches``.
SIGTERM releases the singleton and exits, as a clean Chromium shutdown does.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import textwrap
from pathlib import Path

FAKE_CHROMIUM = textwrap.dedent('''\
    #!{python}
    import http.server, json, os, signal, socket, sys, threading, time
    udd = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
    os.makedirs(udd, exist_ok=True)
    def note(name):
        with open(os.path.join(udd, name), "a") as f:
            f.write(f"{{os.getpid()}}\\n")
    note("spawns")
    lock = os.path.join(udd, "SingletonLock")
    try:
        os.symlink(f"{{socket.gethostname()}}-{{os.getpid()}}", lock)
    except FileExistsError:
        holder = int(os.readlink(lock).rpartition("-")[2])
        try:
            os.kill(holder, 0)
            sys.exit(0)  # handed to the running instance, like Chromium's singleton
        except OSError:
            os.unlink(lock)
            os.symlink(f"{{socket.gethostname()}}-{{os.getpid()}}", lock)
    note("launches")
    def on_term(*_):
        with __import__("contextlib").suppress(OSError):
            os.unlink(lock)
        os._exit(0)
    signal.signal(signal.SIGTERM, on_term)
    browser_path = f"/devtools/browser/{{os.getpid()}}"
    class Version(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({{"webSocketDebuggerUrl": f"ws://127.0.0.1:{{port}}{{browser_path}}"}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Version)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(float(os.environ.get("FAKE_CHROMIUM_READY_DELAY") or 0))
    tmp = os.path.join(udd, "DevToolsActivePort.tmp")
    with open(tmp, "w") as f:
        f.write(f"{{port}}\\n{{browser_path}}\\n")
    os.replace(tmp, os.path.join(udd, "DevToolsActivePort"))
    while True:
        time.sleep(600)
''')


def write_fake_chromium(path: Path) -> Path:
    path.write_text(FAKE_CHROMIUM.format(python=sys.executable))
    path.chmod(0o755)
    return path


def pids(udd: Path, name: str) -> list[int]:
    """The pids the fake noted in ``<udd>/<name>`` (``spawns`` or ``launches``)."""
    try:
        return [int(line) for line in (udd / name).read_text().split()]
    except OSError:
        return []


def kill_fake(pid: int, script: Path) -> None:
    """SIGKILL ``pid`` only while it is still our fake (its argv names the script)."""
    with contextlib.suppress(OSError):
        if str(script) in Path(f"/proc/{pid}/cmdline").read_text(errors="replace"):
            os.kill(pid, signal.SIGKILL)
