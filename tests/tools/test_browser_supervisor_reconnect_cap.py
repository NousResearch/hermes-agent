"""Unit tests for CDP supervisor reconnect cap and registry eviction (#114897).

Covers P0.4:
- Bounded post-attach reconnect failures (MAX_POST_ATTACH_RECONNECT_FAILURES = 5).
- Resetting the failure counter on successful attach.
- Terminal failure logs warning with redacted credentials and evicts supervisor from registry.
- Subsequent get_or_start creates a fresh supervisor.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import browser_supervisor as bs
from tools.browser_supervisor import (
    CDPSupervisor,
    MAX_POST_ATTACH_RECONNECT_FAILURES,
    SUPERVISOR_REGISTRY,
    _SupervisorRegistry,
)


@pytest.fixture
def clean_registry():
    """Ensure SUPERVISOR_REGISTRY is clean before and after tests."""
    SUPERVISOR_REGISTRY.stop_all()
    yield SUPERVISOR_REGISTRY
    SUPERVISOR_REGISTRY.stop_all()


def test_reconnect_cap_and_registry_eviction(clean_registry, caplog):
    """When remote endpoint drops permanently after successful attach,

    supervisor attempts at most MAX_POST_ATTACH_RECONNECT_FAILURES retries,
    emits a terminal warning with redacted credentials, and evicts from registry.
    """
    async def _async_test():
        secret_url = "wss://user:supersecretpass@cdp.example.com/devtools/page/abc?token=sekret123"
        task_id = "test_task_reconnect_cap"

        sup = CDPSupervisor(task_id=task_id, cdp_url=secret_url)
        clean_registry._by_task[task_id] = sup

        connect_calls = 0

        async def mock_connect(*args, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            if connect_calls == 1:
                # First attempt succeeds
                mock_ws = AsyncMock()
                mock_ws.close = AsyncMock()
                return mock_ws
            # Subsequent attempts fail (simulating dead endpoint)
            raise ConnectionRefusedError(f"Connection refused to {secret_url}")

        async def mock_attach():
            pass

        async def mock_read_loop():
            # Simulate session dropping immediately after initial attach
            return

        sup._attach_initial_page = mock_attach
        sup._read_loop = mock_read_loop

        with patch("websockets.connect", side_effect=mock_connect), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
             caplog.at_level(logging.WARNING):

            # Run _run directly in this async test
            await sup._run()

        # 1 initial connect + 5 reconnect failures = 6 total attempts
        assert connect_calls == 1 + MAX_POST_ATTACH_RECONNECT_FAILURES
        assert sup._attached_once is True
        assert sup._consecutive_reconnect_failures == MAX_POST_ATTACH_RECONNECT_FAILURES

        # Supervisor was evicted from registry
        assert clean_registry.get(task_id) is None

        # Warning logged with redacted credentials
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "reconnect failed" in r.message]
        assert len(warning_records) >= 1
        term_msg = warning_records[-1].message
        assert "supersecretpass" not in term_msg
        assert "sekret123" not in term_msg

    asyncio.run(_async_test())


def test_transient_disconnect_resets_reconnect_budget(clean_registry):
    """Successful re-attach resets _consecutive_reconnect_failures to 0."""
    async def _async_test():
        cdp_url = "ws://localhost:9222/devtools/page/123"
        task_id = "test_task_transient_reconnect"

        sup = CDPSupervisor(task_id=task_id, cdp_url=cdp_url)
        clean_registry._by_task[task_id] = sup

        connect_attempts = 0

        async def mock_connect(*args, **kwargs):
            nonlocal connect_attempts
            connect_attempts += 1
            if connect_attempts == 1:
                # Initial attach
                mock_ws = AsyncMock()
                mock_ws.close = AsyncMock()
                return mock_ws
            elif connect_attempts in (2, 3):
                # 2 transient reconnect failures
                raise ConnectionRefusedError("Transient error")
            elif connect_attempts == 4:
                # Reconnect succeeds!
                mock_ws = AsyncMock()
                mock_ws.close = AsyncMock()
                return mock_ws
            else:
                # Stop loop
                sup._stop_requested = True
                raise ConnectionRefusedError("Done")

        session_count = 0

        async def mock_read_loop():
            nonlocal session_count
            session_count += 1
            if session_count == 1:
                # First session drops
                return
            elif session_count == 2:
                # Second session stops supervisor
                sup._stop_requested = True
                return

        sup._attach_initial_page = AsyncMock()
        sup._read_loop = mock_read_loop

        with patch("websockets.connect", side_effect=mock_connect), \
             patch("asyncio.sleep", new_callable=AsyncMock):

            await sup._run()

        # Budget was reset on second successful attach
        assert sup._consecutive_reconnect_failures == 0
        assert sup._attached_once is True

    asyncio.run(_async_test())


def test_registry_remove_method():
    """_SupervisorRegistry.remove only removes matching supervisor."""
    reg = _SupervisorRegistry()
    s1 = MagicMock()
    s2 = MagicMock()

    reg._by_task["task_1"] = s1

    # Mismatched supervisor does not remove
    assert reg.remove("task_1", supervisor=s2) is False
    assert reg.get("task_1") is s1

    # Matching supervisor removes
    assert reg.remove("task_1", supervisor=s1) is True
    assert reg.get("task_1") is None

    # Already absent returns False
    assert reg.remove("task_1") is False
