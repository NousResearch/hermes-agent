"""Bound collection-time server import; never replace its RPC registration."""

import atexit
import importlib
import os
import socket
import subprocess
import sys
import threading
from unittest.mock import patch
from contextlib import ExitStack


SERVER = "tui_gateway.server"
REAPER = "_start_idle_reaper.<locals>._loop"


def import_viewer_server():
    """Suppress only the known reaper, restoring real race threads on return.

    Any other thread/process/listener/network startup is a collection blocker,
    even if an imported module catches the immediate exception. Only the exact
    server-owned session shutdown callback is unregistered: it would enter
    unrelated wake-owner cleanup in this otherwise inert per-file process.
    """
    if SERVER in sys.modules:
        raise RuntimeError("viewer fixture requires a fresh per-file server import")
    suppressed, unexpected, shutdown_callbacks = [], [], []
    real_register = atexit.register
    real_start = threading.Thread.start
    old_sys_hook, old_thread_hook = sys.excepthook, threading.excepthook

    def blocked(kind):
        def reject(*args, **kwargs):
            unexpected.append(kind)
            raise RuntimeError(f"unexpected startup during viewer import: {kind}")
        return reject

    def start(thread, *args, **kwargs):
        target = getattr(thread, "_target", None)
        identity = (getattr(target, "__module__", None),
                    getattr(target, "__qualname__", None))
        if identity == (SERVER, REAPER) and thread.daemon:
            suppressed.append(identity)
            return None
        return blocked(f"thread {identity!r}")()

    def register(callback, *args, **kwargs):
        if (getattr(callback, "__module__", None) == SERVER
                and getattr(callback, "__qualname__", None) == "_shutdown_sessions"):
            shutdown_callbacks.append(callback)
        return real_register(callback, *args, **kwargs)

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(threading.Thread, "start", start))
            stack.enter_context(patch.object(atexit, "register", register))
            for owner, name in ((subprocess, "Popen"), (os, "system"),
                                (socket.socket, "connect"), (socket.socket, "connect_ex"),
                                (socket.socket, "bind"), (socket.socket, "listen")):
                stack.enter_context(patch.object(owner, name, blocked(name)))
            if hasattr(os, "fork"):
                stack.enter_context(patch.object(os, "fork", blocked("fork")))
            server = importlib.import_module(SERVER)
    finally:
        for callback in shutdown_callbacks:
            atexit.unregister(callback)
        sys.excepthook, threading.excepthook = old_sys_hook, old_thread_hook
    if unexpected:
        raise RuntimeError(f"blocked unexpected import startup: {unexpected!r}")
    if suppressed != [(SERVER, REAPER)] or len(shutdown_callbacks) != 1:
        raise RuntimeError("viewer import lifecycle differs from the inspected fixture")
    if threading.Thread.start is not real_start:
        raise RuntimeError("viewer import failed to restore real thread startup")
    return server
