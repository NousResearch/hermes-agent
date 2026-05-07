"""Tests for pre_connect hook in MCPServerTask.run() (tools.mcp_tool)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest  # ty: ignore[unresolved-import]


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    """Ensure clean MCP module state before/after each test."""
    import tools.mcp_tool as mcp

    old_loop = mcp._mcp_loop
    old_thread = mcp._mcp_thread
    old_servers = dict(mcp._servers)
    yield
    mcp._servers.clear()
    mcp._servers.update(old_servers)
    mcp._mcp_loop = old_loop
    mcp._mcp_thread = old_thread


async def _run_once_with_stubbed_transport(server, config):
    """Run server.run once with transports stubbed to avoid real subprocesses."""
    server._shutdown_event.set()
    with (
        patch.object(type(server), "_run_stdio", new_callable=AsyncMock) as mock_stdio,
        patch.object(type(server), "_run_http", new_callable=AsyncMock) as mock_http,
    ):
        await asyncio.wait_for(server.run(config), timeout=1)
    return mock_stdio, mock_http


class TestPreConnectHook:
    """Tests for the optional pre_connect key in MCP server config."""

    @pytest.mark.asyncio
    async def test_pre_connect_runs_before_transport(self):
        """pre_connect subprocess runs, then transport proceeds."""
        config = {
            "command": "echo server",
            "pre_connect": "echo prepped",
            "pre_connect_timeout": 1,
        }

        import tools.mcp_tool as mcp

        server = mcp.MCPServerTask("test-preconnect")
        events = []

        async def fake_subproc(*_args, **_kwargs):
            events.append("pre_connect")
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(None, b""))
            mock_proc.returncode = 0
            return mock_proc

        async def fake_stdio(_config):
            events.append("transport")
            server._shutdown_event.set()

        with (
            patch.object(
                mcp.asyncio,
                "create_subprocess_shell",
                new_callable=AsyncMock,
                side_effect=fake_subproc,
            ) as mock_subproc,
            patch.object(type(server), "_run_stdio", new_callable=AsyncMock) as mock_stdio,
            patch.object(type(server), "_run_http", new_callable=AsyncMock) as mock_http,
        ):
            mock_stdio.side_effect = fake_stdio

            await asyncio.wait_for(server.run(config), timeout=1)

            mock_subproc.assert_awaited_once_with(
                "echo prepped",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            mock_stdio.assert_awaited_once_with(config)
            mock_http.assert_not_awaited()

        assert events == ["pre_connect", "transport"]

    @pytest.mark.asyncio
    async def test_pre_connect_runs_on_reconnect_cycles(self):
        """pre_connect runs before every reconnect transport cycle."""
        config = {
            "command": "echo server",
            "pre_connect": "echo prepped",
            "pre_connect_timeout": 1,
        }

        import tools.mcp_tool as mcp

        server = mcp.MCPServerTask("test-preconnect-reconnect")
        events = []
        transport_calls = 0

        async def fake_subproc(*_args, **_kwargs):
            events.append("pre_connect")
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(None, b""))
            mock_proc.returncode = 0
            return mock_proc

        async def fake_stdio(_config):
            nonlocal transport_calls
            events.append("transport")
            transport_calls += 1
            if transport_calls == 2:
                server._shutdown_event.set()

        with (
            patch.object(
                mcp.asyncio,
                "create_subprocess_shell",
                new_callable=AsyncMock,
                side_effect=fake_subproc,
            ) as mock_subproc,
            patch.object(type(server), "_run_stdio", new_callable=AsyncMock) as mock_stdio,
            patch.object(type(server), "_run_http", new_callable=AsyncMock) as mock_http,
        ):
            mock_stdio.side_effect = fake_stdio

            await asyncio.wait_for(server.run(config), timeout=1)

            assert mock_subproc.await_count == 2
            assert mock_stdio.await_count == 2
            mock_http.assert_not_awaited()

        assert events == [
            "pre_connect",
            "transport",
            "pre_connect",
            "transport",
        ]

    @pytest.mark.asyncio
    async def test_pre_connect_failure_is_non_fatal(self):
        """Failed pre_connect should NOT prevent transport from running."""
        config = {
            "command": "echo server",
            "pre_connect": "exit 1",
            "pre_connect_timeout": 1,
        }

        import tools.mcp_tool as mcp

        server = mcp.MCPServerTask("test-preconnect-fail")

        with patch.object(
            mcp.asyncio,
            "create_subprocess_shell",
            new_callable=AsyncMock,
        ) as mock_subproc:
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(None, b"pull failed"))
            mock_proc.returncode = 1
            mock_subproc.return_value = mock_proc

            mock_stdio, mock_http = await _run_once_with_stubbed_transport(
                server, config
            )

            mock_subproc.assert_awaited_once()
            mock_stdio.assert_awaited_once_with(config)
            mock_http.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_pre_connect_when_not_configured(self):
        """Without pre_connect key, no subprocess is spawned."""
        config = {"command": "echo server", "connect_timeout": 1}

        import tools.mcp_tool as mcp

        server = mcp.MCPServerTask("test-no-preconnect")

        with patch.object(
            mcp.asyncio,
            "create_subprocess_shell",
            new_callable=AsyncMock,
        ) as mock_subproc:
            mock_stdio, mock_http = await _run_once_with_stubbed_transport(
                server, config
            )

            mock_subproc.assert_not_called()
            mock_stdio.assert_awaited_once_with(config)
            mock_http.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pre_connect_exception_is_non_fatal(self):
        """OSError in pre_connect subprocess should not crash server."""
        config = {
            "command": "echo server",
            "pre_connect": "docker pull nosuchimage",
            "pre_connect_timeout": 1,
        }

        import tools.mcp_tool as mcp

        server = mcp.MCPServerTask("test-preconnect-exc")

        with patch.object(
            mcp.asyncio,
            "create_subprocess_shell",
            new_callable=AsyncMock,
            side_effect=OSError("no such file"),
        ) as mock_subproc:
            mock_stdio, mock_http = await _run_once_with_stubbed_transport(
                server, config
            )

            mock_subproc.assert_awaited_once()
            mock_stdio.assert_awaited_once_with(config)
            mock_http.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pre_connect_timeout_is_non_fatal(self):
        """Hung pre_connect command should not block server startup."""
        import tools.mcp_tool as mcp

        config = {
            "command": "echo server",
            "pre_connect": "sleep 999",
            "pre_connect_timeout": 0.01,
        }

        server = mcp.MCPServerTask("test-preconnect-timeout")

        async def fake_communicate():
            await asyncio.sleep(999)
            return None, b""

        with patch.object(
            mcp.asyncio,
            "create_subprocess_shell",
            new_callable=AsyncMock,
        ) as mock_subproc:
            mock_proc = MagicMock()
            mock_proc.communicate = fake_communicate
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock(return_value=None)
            mock_subproc.return_value = mock_proc

            mock_stdio, mock_http = await _run_once_with_stubbed_transport(
                server, config
            )

            mock_subproc.assert_awaited_once()
            mock_proc.kill.assert_called_once()
            mock_proc.wait.assert_awaited_once()
            mock_stdio.assert_awaited_once_with(config)
            mock_http.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_string_pre_connect_skipped(self, caplog):
        """Non-string pre_connect values log a warning and are skipped."""
        import tools.mcp_tool as mcp

        config = {
            "command": "echo server",
            "pre_connect": ["not", "a", "string"],
            "connect_timeout": 1,
        }

        server = mcp.MCPServerTask("test-preconnect-nonstr")

        with patch.object(
            mcp.asyncio,
            "create_subprocess_shell",
            new_callable=AsyncMock,
        ) as mock_subproc:
            mock_stdio, mock_http = await _run_once_with_stubbed_transport(
                server, config
            )

            mock_subproc.assert_not_called()
            mock_stdio.assert_awaited_once_with(config)
            mock_http.assert_not_awaited()
            assert "pre_connect is not a string" in caplog.text
