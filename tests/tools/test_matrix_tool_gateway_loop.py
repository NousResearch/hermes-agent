"""Matrix tool dispatch must use GatewayRunner's owning event loop + profile adapter."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def test_matrix_adapter_uses_default_profile_map():
    from gateway.config import Platform

    default_adapter = object()
    secondary_adapter = object()

    def _authorization_adapter(platform, profile=None):
        if profile and profile != "default":
            return secondary_adapter if platform is Platform.MATRIX else None
        return default_adapter if platform is Platform.MATRIX else None

    runner = SimpleNamespace(
        adapters={Platform.MATRIX: default_adapter},
        _profile_adapters={"secondary": {Platform.MATRIX: secondary_adapter}},
        _authorization_adapter=_authorization_adapter,
    )

    with patch("gateway.run._gateway_runner_ref", return_value=runner), patch(
        "tools.matrix_tool.get_session_env",
        side_effect=lambda key, default="": {
            "HERMES_SESSION_PLATFORM": "matrix",
            "HERMES_SESSION_PROFILE": "",
        }.get(key, default),
    ):
        adapter, err = mt._matrix_adapter()
    assert err == ""
    assert adapter is default_adapter


def test_matrix_adapter_uses_secondary_profile_map():
    from gateway.config import Platform

    default_adapter = object()
    secondary_adapter = object()

    def _authorization_adapter(platform, profile=None):
        if profile == "secondary":
            return secondary_adapter if platform is Platform.MATRIX else None
        return default_adapter if platform is Platform.MATRIX else None

    runner = SimpleNamespace(
        adapters={Platform.MATRIX: default_adapter},
        _profile_adapters={"secondary": {Platform.MATRIX: secondary_adapter}},
        _authorization_adapter=_authorization_adapter,
    )

    with patch("gateway.run._gateway_runner_ref", return_value=runner), patch(
        "tools.matrix_tool.get_session_env",
        side_effect=lambda key, default="": {
            "HERMES_SESSION_PLATFORM": "matrix",
            "HERMES_SESSION_PROFILE": "secondary",
        }.get(key, default),
    ):
        adapter, err = mt._matrix_adapter()
    assert err == ""
    assert adapter is secondary_adapter


def test_matrix_adapter_secondary_missing_does_not_fall_back_to_default():
    from gateway.config import Platform

    default_adapter = object()

    def _authorization_adapter(platform, profile=None):
        # Fail closed for stamped secondary with no entry.
        if profile == "ghost":
            return None
        return default_adapter if platform is Platform.MATRIX else None

    runner = SimpleNamespace(
        adapters={Platform.MATRIX: default_adapter},
        _profile_adapters={},
        _authorization_adapter=_authorization_adapter,
    )

    with patch("gateway.run._gateway_runner_ref", return_value=runner), patch(
        "tools.matrix_tool.get_session_env",
        side_effect=lambda key, default="": {
            "HERMES_SESSION_PLATFORM": "matrix",
            "HERMES_SESSION_PROFILE": "ghost",
        }.get(key, default),
    ):
        adapter, err = mt._matrix_adapter()
    assert adapter is None
    assert "ghost" in err
    assert "not connected" in err.lower()


def test_send_reaction_uses_gateway_loop_and_profile_adapter(gateway_loop_thread):
    """End-to-end action path without mocking ``_run``."""
    from gateway.config import Platform

    gateway_loop = gateway_loop_thread
    observed = {}

    class FakeAdapter:
        async def _send_reaction(self, room_id, event_id, emoji):
            observed["thread"] = threading.current_thread().name
            observed["loop_id"] = id(asyncio.get_running_loop())
            observed["args"] = (room_id, event_id, emoji)
            return "$rxn1"

    adapter = FakeAdapter()

    def _authorization_adapter(platform, profile=None):
        assert platform is Platform.MATRIX
        assert profile == "secondary"
        return adapter

    runner = SimpleNamespace(
        _gateway_loop=gateway_loop,
        adapters={},
        _profile_adapters={"secondary": {Platform.MATRIX: adapter}},
        _authorization_adapter=_authorization_adapter,
    )

    def _env(key, default=""):
        return {
            "HERMES_SESSION_PLATFORM": "matrix",
            "HERMES_SESSION_PROFILE": "secondary",
            "HERMES_SESSION_CHAT_ID": "!room:example.org",
        }.get(key, default)

    with patch("gateway.run._gateway_runner_ref", return_value=runner), patch(
        "tools.matrix_tool.get_session_env", side_effect=_env
    ), patch("tools.matrix_tool._matrix_tools_cfg", return_value={}):
        result = mt._run_matrix_action(
            "send_reaction",
            mt._CORE_ACTIONS,
            "matrix",
            event_id="$evt",
            emoji="✅",
        )

    assert '"success": true' in result or '"success":true' in result.replace(" ", "")
    assert observed["loop_id"] == id(gateway_loop)
    assert observed["thread"] == "fake-gateway-loop"
    assert observed["args"] == ("!room:example.org", "$evt", "✅")
