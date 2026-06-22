"""Regression test for #50394.

A single failing stdio MCP server must not churn the whole MCP bridge.

Root cause: a server that fails to connect is never recorded in
``_servers`` (``start()`` raises before the ``_servers[name] = server``
line in ``_discover_and_register_server``). Without a post-failure
cooldown, every subsequent ``register_mcp_servers`` pass (one per agent
worker session) re-spawns the failing server from scratch -- a restart
storm that destabilises the healthy co-located servers. The fix arms a
per-server exponential backoff so a chronically failing server is retried
on a schedule, isolated from the rest of the bridge.
"""

from unittest.mock import patch

import pytest

import tools.mcp_tool as mcp_mod


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    """Snapshot and restore the module-level MCP state around each test."""
    snapshot = (
        dict(mcp_mod._servers),
        set(mcp_mod._server_connecting),
        dict(mcp_mod._server_connect_errors),
        dict(mcp_mod._server_connect_retry_after),
        dict(mcp_mod._server_connect_failures),
    )
    mcp_mod._servers.clear()
    mcp_mod._server_connecting.clear()
    mcp_mod._server_connect_errors.clear()
    mcp_mod._server_connect_retry_after.clear()
    mcp_mod._server_connect_failures.clear()
    try:
        yield
    finally:
        (servers, connecting, errs, retry_after, failures) = snapshot
        mcp_mod._servers.clear(); mcp_mod._servers.update(servers)
        mcp_mod._server_connecting.clear(); mcp_mod._server_connecting.update(connecting)
        mcp_mod._server_connect_errors.clear(); mcp_mod._server_connect_errors.update(errs)
        mcp_mod._server_connect_retry_after.clear(); mcp_mod._server_connect_retry_after.update(retry_after)
        mcp_mod._server_connect_failures.clear(); mcp_mod._server_connect_failures.update(failures)


class TestConnectCooldownHelpers:
    def test_failure_arms_exponential_backoff(self):
        now = 1000.0
        with patch("tools.mcp_tool.time.monotonic", return_value=now):
            mcp_mod._record_connect_failure("bad")
            d1 = mcp_mod._server_connect_retry_after["bad"]
            mcp_mod._record_connect_failure("bad")
            d2 = mcp_mod._server_connect_retry_after["bad"]
        assert d1 == now + mcp_mod._CONNECT_RETRY_BASE_BACKOFF_SEC
        assert d2 == now + mcp_mod._CONNECT_RETRY_BASE_BACKOFF_SEC * 2
        assert mcp_mod._server_connect_failures["bad"] == 2

    def test_backoff_is_capped(self):
        for _ in range(50):
            mcp_mod._record_connect_failure("bad")
        deadline = mcp_mod._server_connect_retry_after["bad"]
        assert deadline <= mcp_mod.time.monotonic() + mcp_mod._CONNECT_RETRY_MAX_BACKOFF_SEC + 1

    def test_cooldown_active_then_clears(self):
        now = 5000.0
        with patch("tools.mcp_tool.time.monotonic", return_value=now):
            mcp_mod._record_connect_failure("bad")
            assert mcp_mod._connect_cooldown_active("bad") is True
        later = now + mcp_mod._CONNECT_RETRY_MAX_BACKOFF_SEC + 1
        with patch("tools.mcp_tool.time.monotonic", return_value=later):
            assert mcp_mod._connect_cooldown_active("bad") is False
        mcp_mod._clear_connect_failure("bad")
        assert "bad" not in mcp_mod._server_connect_retry_after
        assert "bad" not in mcp_mod._server_connect_failures

    def test_unknown_server_not_in_cooldown(self):
        assert mcp_mod._connect_cooldown_active("never-seen") is False


@pytest.mark.skipif(not mcp_mod._MCP_AVAILABLE, reason="mcp SDK not installed")
class TestRegisterMcpServersIsolation:
    """register_mcp_servers must not re-spawn a server still in cooldown."""

    def _run_with_mocked_connect(self, attempts):
        async def fake_connect(name, config):
            attempts.append(name)
            if name == "bad":
                raise ConnectionError("exec: bad: not found")
            server = mcp_mod.MCPServerTask(name)
            server._registered_tool_names = []
            server._tools = []
            return server

        return patch("tools.mcp_tool._connect_server", side_effect=fake_connect)

    def test_failing_server_skipped_on_second_pass(self):
        attempts = []
        cfg = {
            "good": {"command": "good-cmd"},
            "bad": {"command": "bad-cmd"},
        }
        with self._run_with_mocked_connect(attempts), \
                patch("tools.mcp_tool._register_server_tools", return_value=[]), \
                patch("tools.mcp_tool._filter_suspicious_mcp_servers", side_effect=lambda x: x):
            mcp_mod.register_mcp_servers(cfg)
            assert "good" in mcp_mod._servers
            assert "bad" not in mcp_mod._servers
            assert mcp_mod._connect_cooldown_active("bad") is True
            assert "bad" in attempts

            attempts.clear()
            mcp_mod.register_mcp_servers(cfg)
            assert "bad" not in attempts, (
                "failing server was re-spawned despite active cooldown -- "
                "restart storm not isolated (#50394)"
            )

    def test_cooldown_expiry_allows_retry(self):
        attempts = []
        cfg = {"bad": {"command": "bad-cmd"}}
        with self._run_with_mocked_connect(attempts), \
                patch("tools.mcp_tool._register_server_tools", return_value=[]), \
                patch("tools.mcp_tool._filter_suspicious_mcp_servers", side_effect=lambda x: x):
            mcp_mod.register_mcp_servers(cfg)
            assert mcp_mod._connect_cooldown_active("bad") is True

            mcp_mod._server_connect_retry_after["bad"] = mcp_mod.time.monotonic() - 1
            attempts.clear()
            mcp_mod.register_mcp_servers(cfg)
            assert "bad" in attempts, "elapsed cooldown should permit a retry"
