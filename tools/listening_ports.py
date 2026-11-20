"""TCP ports a background process tree is listening on.

A dev server started with ``terminal(background=true)`` usually binds a port
the user then has to fish out of its log. Every surface that lists background
processes (the ``process`` tool, the Desktop status stack, the TUI roster and
the classic CLI dock) reads this one helper, so they all show the same ports.

Only host-scoped PIDs are inspected: a sandbox PID names a process inside a
container or remote shell, and a ``wsl.exe`` launcher's PID is not the server
inside the VM, so the local process table knows nothing useful about either.
"""

from __future__ import annotations

import threading
import time

# Surfaces poll every few seconds; a short cache keeps several of them (plus the
# agent's own ``process list``) from walking the same process tree repeatedly.
_CACHE_TTL_SECONDS = 2.0
# A runaway fork tree must not turn one listing into thousands of syscalls.
_MAX_TREE_PROCESSES = 64

_cache: dict[int, tuple[float, list[int]]] = {}
_cache_lock = threading.Lock()


def _scan(pid: int) -> list[int]:
    try:
        import psutil
    except ImportError:
        return []
    try:
        root = psutil.Process(pid)
        tree = [root, *root.children(recursive=True)][:_MAX_TREE_PROCESSES]
    except (psutil.Error, OSError):
        return []
    ports: set[int] = set()
    for proc in tree:
        try:
            conns = proc.net_connections(kind="inet")
        except (psutil.Error, OSError):
            continue  # exited mid-walk, or another user's child
        ports.update(c.laddr.port for c in conns if c.status == psutil.CONN_LISTEN and c.laddr)
    return sorted(ports)


def listening_ports(pid: int | None) -> list[int]:
    """Sorted listening TCP ports of *pid* and its descendants; ``[]`` when unknown."""
    if not pid or pid <= 0:
        return []
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(pid)
        if hit and now - hit[0] < _CACHE_TTL_SECONDS:
            return list(hit[1])
    ports = _scan(pid)
    with _cache_lock:
        for stale in [p for p, (at, _) in _cache.items() if now - at >= _CACHE_TTL_SECONDS]:
            del _cache[stale]
        _cache[pid] = (now, ports)
    return list(ports)


def session_ports(session) -> dict:
    """``{"ports": [...]}`` for a running host-scoped background session, else ``{}``."""
    if session.exited or session.pid_scope != "host" or session.wsl_chain:
        return {}
    ports = listening_ports(session.pid)
    return {"ports": ports} if ports else {}
