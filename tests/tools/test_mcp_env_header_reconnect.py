"""Exercise static-header rotation through the real MCP reconnect/config pipeline."""

import asyncio

import pytest

from tools import mcp_tool_config, mcp_tool_transport
from tools.mcp_tool import MCPServerTask

TOKEN_NAME = "RECONNECT_PROBE_TOKEN"


@pytest.mark.no_isolate
@pytest.mark.parametrize("rotation_source", ["process", "dotenv"])
@pytest.mark.parametrize("env_prefix", ["", "env:"])
def test_static_header_rotation_on_next_reconnect(monkeypatch, tmp_path, rotation_source, env_prefix):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(TOKEN_NAME, "before-rotation")
    (tmp_path / "config.yaml").write_text(
        "mcp_servers:\n  rotation-probe:\n    url: https://example.invalid/mcp\n"
        f"    headers:\n      Authorization: 'Bearer ${{{env_prefix}{TOKEN_NAME}}}'\n"
        "      X-Static: retained\n",
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    if rotation_source == "dotenv":
        env_path.write_text(f"{TOKEN_NAME}=before-rotation\n", encoding="utf-8")
    initial = mcp_tool_config._load_mcp_config()["rotation-probe"]
    seen = []

    class ReconnectingTask(MCPServerTask):
        async def _prepare_run(self, config):
            self._config = config
            self._auth_type = ""
            return True

        async def _run_http(self, config):
            # Stop at the network boundary, using the production transport-input builder.
            inputs, _ = mcp_tool_transport._connect_inputs(self.name, config)
            seen.append(dict(inputs[1]))
            self._ready.set()
            self.session = object()
            self._session_proven = True
            if len(seen) == 1:
                if rotation_source == "dotenv":
                    env_path.write_text(f"{TOKEN_NAME}=after-rotation\n", encoding="utf-8")
                else:
                    monkeypatch.setenv(TOKEN_NAME, "after-rotation")
                return "reconnect"
            self._shutdown_event.set()
            return "shutdown"

    async def scenario():
        await asyncio.wait_for(ReconnectingTask("rotation-probe").run(initial), timeout=10)

    asyncio.run(scenario())
    assert seen == [
        {"Authorization": "Bearer before-rotation", "X-Static": "retained"},
        {"Authorization": "Bearer after-rotation", "X-Static": "retained"},
    ]
