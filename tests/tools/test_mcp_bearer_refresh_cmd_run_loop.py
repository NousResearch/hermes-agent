"""Run-loop lifecycle tests for `bearer_refresh_cmd`.

Proves the opt-in exception to the initial-auth-stop contract is safe.

The existing contract (see `test_initial_oauth_failure_does_not_retry`
in test_mcp_tool.py) is: on the FIRST connect attempt, an auth error
must stop immediately — no retry — to avoid looping a broken OAuth
config through repeated browser prompts.

This module verifies that adding `bearer_refresh_cmd` preserves that
contract in every path except the deliberate opt-in success case:

  path A (unopted):  no bearer_refresh_cmd + initial-auth-fail → STOPS
                     (contract unchanged for anyone who did not opt in)

  path B (opt-in fail):  bearer_refresh_cmd set + initial-auth-fail
                         + refresh returns False (cmd exited nonzero,
                         cooldown, bad token, headers misconfigured)
                         → STOPS (contract preserved when the operator's
                         cmd cannot recover; no infinite loop)

  path C (opt-in success):  bearer_refresh_cmd set + initial-auth-fail
                            + refresh returns True → RETRIES with the
                            new bearer (the whole point of the feature)

Together these three tests guarantee that any regression which starts
looping a broken OAuth config would surface here as a hang or an
assertion failure on `run_count == 1`.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask


def _make_server(name: str, config: dict) -> MCPServerTask:
    """Real MCPServerTask instance with the config already stashed on
    it. We avoid running the constructor's connection logic by only
    building the object shell + setting the two fields the run() loop
    reads. `run(config)` still gets the config passed explicitly."""
    server = MCPServerTask(name)
    server._config = config
    return server


def test_no_refresh_cmd_initial_auth_failure_still_stops():
    """Path A: unopted-in path preserves the original contract exactly.

    Regression guard: this is a paraphrase of
    `test_initial_oauth_failure_does_not_retry`, restated from the
    perspective of `bearer_refresh_cmd`. If someone deletes the
    `_refresh_bearer_via_command() returns False when unconfigured`
    short-circuit, this test fails."""
    run_count = 0
    target_server = None
    oauth_error = RuntimeError("Token exchange failed (401): expired")

    original_run_stdio = MCPServerTask._run_stdio

    async def patched_run_stdio(self_srv, config):
        nonlocal run_count
        run_count += 1
        if target_server is not self_srv:
            return await original_run_stdio(self_srv, config)
        raise oauth_error

    async def _test():
        nonlocal target_server
        server = _make_server("no_cmd_srv", {"command": "test"})
        target_server = server

        with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
             patch("tools.mcp_tool._is_auth_error", return_value=True), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await server.run({"command": "test"})

        assert run_count == 1, "unopted initial-auth failure must NOT retry"
        assert server._error is oauth_error
        assert server._ready.is_set()
        assert mock_sleep.await_count == 0

    asyncio.run(_test())


def test_refresh_cmd_configured_but_refresh_fails_still_stops():
    """Path B: opt-in path preserves the contract when the operator's
    refresh cmd can't recover.

    This is the failure mode that most concerned the reviewer: if the
    refresh path can EVER swallow an initial 401 without stopping, an
    operator with a broken refresh cmd would spin forever. This test
    asserts we stop even when the cmd is configured — the difference
    from path A is only that _refresh_bearer_via_command() gets called
    once (and returns False)."""
    run_count = 0
    target_server = None
    oauth_error = RuntimeError("Token exchange failed (401): expired")
    refresh_calls = 0

    original_run_stdio = MCPServerTask._run_stdio

    async def patched_run_stdio(self_srv, config):
        nonlocal run_count
        run_count += 1
        if target_server is not self_srv:
            return await original_run_stdio(self_srv, config)
        raise oauth_error

    async def fake_refresh(self_srv):
        nonlocal refresh_calls
        refresh_calls += 1
        return False  # cmd exited nonzero, or bad token, or cooldown, etc.

    async def _test():
        nonlocal target_server
        server = _make_server(
            "cmd_but_fails_srv",
            {"command": "test", "bearer_refresh_cmd": "/etc/hermes/broken.sh"},
        )
        target_server = server

        with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
             patch.object(MCPServerTask, "_refresh_bearer_via_command", fake_refresh), \
             patch("tools.mcp_tool._is_auth_error", return_value=True), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await server.run({"command": "test", "bearer_refresh_cmd": "/etc/hermes/broken.sh"})

        assert refresh_calls == 1, "refresh should have been attempted exactly once"
        assert run_count == 1, "failed refresh must NOT cause a retry"
        assert server._error is oauth_error
        assert server._ready.is_set()
        assert mock_sleep.await_count == 0

    asyncio.run(_test())


def test_refresh_cmd_success_on_initial_auth_failure_retries_and_succeeds():
    """Path C: opt-in exception success case.

    An operator configured `bearer_refresh_cmd`. The env-baked bearer
    is stale so the first connect fails with 401. The refresh cmd
    succeeds. The run-loop is expected to retry immediately (backoff
    reset) and succeed on the second attempt — this is the whole point
    of the feature and what makes hermes recover from a stale bearer
    without a service restart."""
    run_count = 0
    target_server = None
    oauth_error = RuntimeError("Token exchange failed (401): expired")

    async def fake_refresh(self_srv):
        # After refresh, the connection would succeed with the new
        # bearer. We simulate that by having the next _run_stdio call
        # signal shutdown so the loop exits cleanly.
        return True

    original_run_stdio = MCPServerTask._run_stdio

    async def patched_run_stdio(self_srv, config):
        nonlocal run_count
        run_count += 1
        if target_server is not self_srv:
            return await original_run_stdio(self_srv, config)
        if run_count == 1:
            raise oauth_error
        # Second attempt: succeed. Set _ready and signal shutdown so
        # the loop exits without further retries.
        self_srv.session = MagicMock()
        self_srv._tools = []
        self_srv._ready.set()
        self_srv._shutdown_event.set()
        await self_srv._shutdown_event.wait()

    async def _test():
        nonlocal target_server
        server = _make_server(
            "opt_in_success_srv",
            {"command": "test", "bearer_refresh_cmd": "/etc/hermes/works.sh"},
        )
        target_server = server

        with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
             patch.object(MCPServerTask, "_refresh_bearer_via_command", fake_refresh), \
             patch("tools.mcp_tool._is_auth_error", return_value=True), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await server.run({"command": "test", "bearer_refresh_cmd": "/etc/hermes/works.sh"})

        assert run_count == 2, "must retry once after successful refresh"
        assert server._error is None
        assert server._ready.is_set()

    asyncio.run(_test())
