"""Matrix tool dispatch must use GatewayRunner's owning event loop."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import matrix_tool as mt


@pytest.fixture
def gateway_loop_thread():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, name="fake-gateway-loop", daemon=True)
    thread.start()
    assert ready.wait(timeout=2)

    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def test_run_dispatches_coroutine_onto_gateway_loop(gateway_loop_thread):
    """Live adapter coros must execute on GatewayRunner._gateway_loop.

    Does NOT mock ``_run`` — this is the handoff regression the sweeper asked for.
    """
    gateway_loop = gateway_loop_thread
    observed = {}

    async def _probe():
        observed["thread"] = threading.current_thread().name
        observed["loop_id"] = id(asyncio.get_running_loop())
        return "ok"

    runner = SimpleNamespace(_gateway_loop=gateway_loop, adapters={})

    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        # Call the real bridge; do not patch mt._run.
        result = mt._run(_probe())

    assert result == "ok"
    assert observed["loop_id"] == id(gateway_loop)
    assert observed["thread"] == "fake-gateway-loop"


def test_run_falls_back_without_gateway_loop():
    """Unit/CLI path without a live gateway still works via _run_async."""
    observed = {}

    async def _probe():
        observed["ran"] = True
        return 42

    with patch("gateway.run._gateway_runner_ref", return_value=None):
        result = mt._run(_probe())

    assert result == 42
    assert observed["ran"] is True
