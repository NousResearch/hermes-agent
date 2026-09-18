"""A parked MCP server's timed self-probe is unattended: it must never start a browser OAuth
flow. Left interactive, an expired refresh token opened a new authorize tab every
``_PARKED_RETRY_INTERVAL`` for the life of the gateway (92 tabs in one night)."""

import asyncio

import pytest


@pytest.mark.no_isolate
def test_self_probe_revival_runs_with_interactive_oauth_suppressed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_oauth import _is_interactive, force_interactive_oauth
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)
    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)
    attempts: list = []

    class _Task(MCPServerTask):
        def _is_http(self):
            return False

        def _deregister_tools(self):
            self._registered_tool_names = []

        async def _run_stdio(self, config):
            attempts.append(_is_interactive())
            if len(attempts) == 1:
                raise RuntimeError("down")  # parks at once (initial ladder = 0 retries)
            self.session = object()
            await self._wait_for_lifecycle_event()
            return "shutdown"

    async def _scenario():
        # The gateway/TTY parent context IS interactive; the self-probe must still not be.
        with force_interactive_oauth():
            task = _Task("srv")
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            for _ in range(400):
                await _real_sleep(0.01)
                if len(attempts) >= 2:
                    break
            await task.shutdown()
            await asyncio.wait_for(run_task, timeout=5)

    asyncio.run(_scenario())
    assert attempts[0] is True, "the attended initial connect may still prompt"
    assert len(attempts) >= 2 and attempts[-1] is False, (
        "timed self-probe revival ran with interactive OAuth enabled -> browser tab per probe")


def test_explicit_reconnect_request_keeps_the_callers_interactivity():
    """Only the TIMED wake is unattended; ``_reconnect_event`` (OAuth recovery, /mcp refresh) is an
    explicit request and is reported as such."""
    from tools.mcp_tool import MCPServerTask

    async def _scenario():
        task = MCPServerTask("srv")
        task._reconnect_event.set()
        assert await task._wait_for_reconnect_or_shutdown(timeout=1) == "reconnect"
        assert await task._wait_for_reconnect_or_shutdown(timeout=0.01) == "self-probe"
        task._shutdown_event.set()
        assert await task._wait_for_reconnect_or_shutdown(timeout=0.01) == "shutdown"

    asyncio.run(_scenario())
