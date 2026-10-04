"""Per-connection admission barrier for non-disruptive scoped MCP reloads.

A request claims admission before resolving its server. A reload closes admission and
waits for every admitted request (including queued RPCs and retries) to return.
Timeout never cancels a user request and never tears down its transport.
"""
from contextlib import contextmanager
import threading
import time

from tools.mcp_tool_scope import _resolve_server_key
from tools.mcp_tool_common import _core

_lock = threading.Lock()
_changed = threading.Condition(_lock)
_active = {}
_blocked = set()


def _connection_key(server_name):
    with _core._lock:
        return _resolve_server_key(server_name)


@contextmanager
def admit_call(server_name):
    key = _connection_key(server_name)
    with _changed:
        admitted = key not in _blocked
        if admitted:
            _active[key] = _active.get(key, 0) + 1
    try:
        yield admitted
    finally:
        if admitted:
            with _changed:
                _active[key] -= 1
                if not _active[key]:
                    _active.pop(key)
                    _changed.notify_all()


@contextmanager
def drain_calls(server_name, *, timeout):
    key = _connection_key(server_name)
    with _changed:
        if key in _blocked:
            yield_pending = True
        else:
            _blocked.add(key)
            yield_pending = False
    if yield_pending:
        yield False
        return
    try:
        deadline = time.monotonic() + max(0, timeout)
        with _changed:
            while _active.get(key, 0):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                _changed.wait(remaining)
            ready = not _active.get(key, 0)
        yield ready
    finally:
        with _changed:
            _blocked.discard(key)
            _changed.notify_all()
