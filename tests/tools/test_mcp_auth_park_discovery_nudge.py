"""An OAuth server parked on an auth failure must not be revived by discovery passes until a
re-authorization rewrites its credentials: every nudge replayed the same rejected refresh token,
once per agent build, until the authorization server rate-limited the user's IP."""

import asyncio
import json

import pytest


@pytest.mark.no_isolate
def test_auth_parked_oauth_server_ignores_discovery_until_credentials_change(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_oauth import HermesTokenStorage, OAuthNonInteractiveError
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_discovery import _select_new_servers

    cfg = {"url": "https://mcp.example.test/mcp", "auth": "oauth"}
    token_file = HermesTokenStorage("cloud")._tokens_path()
    token_file.parent.mkdir(parents=True)
    token_file.write_text(json.dumps({"access_token": "a", "token_type": "Bearer", "refresh_token": "dead"}))
    attempts = []

    class _Task(MCPServerTask):
        def _deregister_tools(self):
            self._registered_tool_names = []

        async def _run_http(self, config):
            attempts.append(config["url"])
            raise OAuthNonInteractiveError("refresh token rejected; browser authorization required")

    async def _settle():
        for _ in range(50):
            await asyncio.sleep(0)

    async def scenario():
        task = _Task("cloud")
        monkeypatch.setitem(mcp_tool._servers, "cloud", task)
        run = asyncio.ensure_future(task.run(cfg))
        await _settle()
        assert attempts and task._was_parked and not run.done()

        for _ in range(5):  # five agent builds re-running discovery
            _select_new_servers({"cloud": cfg})
            await _settle()
        assert len(attempts) == 1

        token_file.write_text(json.dumps({"access_token": "b", "token_type": "Bearer", "refresh_token": "new"}))
        _select_new_servers({"cloud": cfg})  # `hermes mcp login` landed: the next pass revives it
        await _settle()
        assert len(attempts) == 2

        task._shutdown_event.set()
        await asyncio.wait_for(run, 5)

    asyncio.run(scenario())
