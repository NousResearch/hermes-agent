"""The OAuth callback listener is released when the flow fails right after binding it.

The waiter binds the loopback listener, then prints the paste-fallback prompt and starts the paste
reader. Those steps ran before the ``try`` whose ``finally`` closes the listener, so a prompt write
that raised (``BrokenPipeError`` once the reader of a piped stderr has exited, e.g.
``hermes mcp login srv 2>&1 | head``) left the port listening for the life of the process and the
retry on the same pinned or cached port died with "already in use" — the leak class of #113771.
"""
import asyncio
import io
import socket
import threading

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

import tools.mcp_oauth as mo


class _BrokenStderr(io.TextIOBase):
    def write(self, _text):
        raise BrokenPipeError(32, "Broken pipe")


def _record_bound_ports(monkeypatch) -> list[int]:
    ports: list[int] = []
    original_start = mo._start_callback_server

    def recording_start(port, handler_cls):
        server = original_start(port, handler_cls)
        ports.append(server.server_address[1])
        return server

    monkeypatch.setattr(mo, "_start_callback_server", recording_start)
    return ports


def _run_waiter() -> None:
    async def main():
        with mo.force_interactive_oauth():
            await mo._make_callback_waiter(0, timeout=10)()

    asyncio.run(main())


def _accepts_connections(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            return True
    except OSError:
        return False


def test_failed_paste_prompt_releases_the_callback_port(monkeypatch):
    monkeypatch.setattr(mo.sys, "stdin", io.StringIO())
    monkeypatch.setattr(mo.sys, "stderr", _BrokenStderr())
    ports = _record_bound_ports(monkeypatch)

    with pytest.raises(BrokenPipeError):
        _run_waiter()

    assert len(ports) == 1, "the listener was never bound, so the failure under test did not happen"
    assert not _accepts_connections(ports[0])


def test_serve_thread_that_cannot_start_releases_the_callback_port(monkeypatch):
    monkeypatch.setattr(mo.sys, "stdin", io.StringIO())
    ports = _record_bound_ports(monkeypatch)
    real_thread = threading.Thread

    def thread_factory(*args, target=None, **kwargs):
        if getattr(target, "__name__", "") == "serve_forever":
            raise RuntimeError("can't start new thread")
        return real_thread(*args, target=target, **kwargs)

    monkeypatch.setattr(mo.threading, "Thread", thread_factory)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        _run_waiter()

    assert len(ports) == 1
    assert not _accepts_connections(ports[0])
