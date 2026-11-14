"""Preparation failures must reach the MCP startup waiter (regression for #131835)."""

import asyncio
import subprocess
import sys
import textwrap

import pytest


def test_cold_sdk_import_failure_reaches_start():
    # A fresh interpreter exercises the real SDK import, without removing modules
    # that another test or an active connection may still own.
    script = textwrap.dedent("""
        import asyncio
        import gc
        import sys

        from opentelemetry.util import _importlib_metadata as metadata
        assert metadata.entry_points(group="opentelemetry_context")
        assert "opentelemetry.context" not in sys.modules
        metadata.entry_points = lambda **params: ()

        from tools.mcp_tool import MCPServerTask

        async def check():
            unhandled = []
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(lambda loop, context: unhandled.append(context))
            server = MCPServerTask("cold-sdk")
            try:
                await asyncio.wait_for(server.start({"command": "unused"}), timeout=5)
            except RuntimeError as exc:
                assert str(exc) == "coroutine raised StopIteration"
                assert isinstance(exc.__cause__, StopIteration)
                assert server._error is exc
            else:
                raise AssertionError("broken SDK import was accepted")
            assert server._ready.is_set()
            await server._task
            assert server._task.exception() is None
            await server.shutdown()
            del server
            gc.collect()
            await asyncio.sleep(0)
            assert not unhandled, unhandled

        asyncio.run(check())

        # The same failure must cross the native Desktop/CLI probe bridge
        # without becoming a connect timeout.
        from hermes_cli.mcp_config import _probe_single_server
        from tools import mcp_tool
        details = {}
        try:
            _probe_single_server("cold-sdk-probe", {"command": "unused"},
                                 connect_timeout=5, details=details)
        except RuntimeError as exc:
            assert str(exc) == "coroutine raised StopIteration"
            # The probe's existing redaction boundary suppresses exception causes.
            assert exc.__cause__ is None
        else:
            raise AssertionError("broken SDK probe was accepted")
        assert details["initialized"] is False
        assert mcp_tool._mcp_loop is None
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_preparation_errors_do_not_retry_and_cancellation_propagates(monkeypatch):
    from tools.mcp_tool import MCPServerTask

    async def unexpected_transport(self, config):
        pytest.fail("preparation must finish before a transport starts")

    monkeypatch.setattr(MCPServerTask, "_run_stdio", unexpected_transport)

    async def check():
        server = MCPServerTask("invalid-config")
        with pytest.raises(AttributeError, match="lower") as failure:
            await asyncio.wait_for(
                server.start({"command": "unused", "auth": 42}), timeout=5,
            )
        assert server._error is failure.value
        await server._task
        await server.shutdown()

        preparing = asyncio.Event()
        cancelled = asyncio.Event()

        async def preflight(self, *args, **kwargs):
            preparing.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(MCPServerTask, "_preflight_content_type", preflight)
        server = MCPServerTask("cancel-preflight")
        waiter = asyncio.create_task(server.start({"url": "https://example.com/mcp"}))
        await asyncio.wait_for(preparing.wait(), timeout=5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        with pytest.raises(asyncio.CancelledError):
            await server._task
        assert cancelled.is_set()
        assert server._error is None
        assert not server._ready.is_set()
        await server.shutdown()

    asyncio.run(check())
