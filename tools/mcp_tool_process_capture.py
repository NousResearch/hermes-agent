"""Attribute stdio children to the SDK launch that created them, not a PID snapshot.

The pinned SDK yields streams without its process. Its platform factory is the
single spawn seam; one shim observes it with a task-local callback while leaving
the SDK's native process/job lifecycle intact.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from importlib import import_module
from threading import Lock

_on_spawn = ContextVar("mcp_stdio_on_spawn", default=None)
_install_lock = Lock()


def _install_capture():
    sdk = import_module("mcp.client.stdio")
    with _install_lock:
        factory = sdk._create_platform_compatible_process
        if getattr(factory, "_hermes_stdio_capture", None) is True:
            return

        @wraps(factory)
        async def create(*args, **kwargs):
            process = await factory(*args, **kwargs)
            callback = _on_spawn.get()
            if callback is not None:
                try:
                    callback(process)
                except BaseException:
                    import anyio

                    # The SDK has not received the process yet, so its context
                    # manager cannot clean up a failed ownership registration.
                    with anyio.CancelScope(shield=True):
                        await sdk._stop_server_process(process)
                    raise
            return process

        create._hermes_stdio_capture = True
        sdk._create_platform_compatible_process = create


@contextmanager
def capture_stdio_process(on_spawn):
    """Observe only this task's SDK-created process, including partial startup."""
    _install_capture()
    token = _on_spawn.set(on_spawn)
    try:
        yield
    finally:
        _on_spawn.reset(token)
