"""Shared helpers for the gateway suites."""

from __future__ import annotations

import threading
import time


class ReplyTransport:
    def __init__(self):
        self.frames = []
        self.event = threading.Event()

    def write(self, frame):
        self.frames.append(frame)
        self.event.set()
        return True

    def close(self):
        pass


def reply(transport, method, params, *, rid=7, timeout=2.0):
    """Dispatch one request and return its reply: a short handler answers inline, a long one through the transport."""
    from tui_gateway import server

    start = len(transport.frames)
    direct = server.dispatch({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}, transport)
    if direct is not None:
        return direct
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        replies = [frame for frame in transport.frames[start:] if frame.get("id") == rid]
        if replies:
            return replies[-1]
        transport.event.wait(0.05)
        transport.event.clear()
    raise AssertionError(f"{method} did not reply")
