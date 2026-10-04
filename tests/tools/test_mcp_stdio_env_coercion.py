"""Regression tests: an unquoted YAML scalar in ``mcp_servers.<name>.env`` must not break the server.

``MS365_MCP_USE_KEYTAR: 0`` (and ``SOME_FLAG: false``, ``PORT: 8080``) parse as int/bool, and the
MCP SDK's ``StdioServerParameters`` declares ``env: dict[str, str]`` — pydantic rejects the value
BEFORE the child is spawned, so the server connects as zero tools with only an opaque
``Input should be a valid string`` error, and any job whose ``enabled_toolsets`` names it is blocked
by pre-dispatch config validation. The config env IS delivered (that is what these tests pin down);
it just has to be delivered as strings.
"""

import asyncio
import json
import os
import sys


import pytest

from tools.mcp_tool import _MCP_AVAILABLE
from tools.mcp_tool_config import _build_safe_env

pytestmark = pytest.mark.skipif(not _MCP_AVAILABLE, reason="MCP SDK not installed")

# Child: dump the environment it was spawned with, then hold the pipes open briefly.
_CHILD_DUMP_ENV = (
    "import json, os, time\n"
    "json.dump(dict(os.environ), open(os.environ['MCP_TEST_ENV_DUMP'], 'w'))\n"
    "time.sleep(8)\n"
)

_CONFIG_ENV = {
    "MCP_TEST_ENV_INT": 0,          # MS365_MCP_USE_KEYTAR: 0     -> int
    "MCP_TEST_ENV_BOOL": False,     # SOME_FLAG: false            -> bool
    "MCP_TEST_ENV_PORT": 8080,      # PORT: 8080                  -> int
    "MCP_TEST_ENV_STR": "as-written",
    "MCP_TEST_ENV_EMPTY": None,     # KEY:                        -> None
}


class TestEnvValueCoercion:
    def test_build_safe_env_stringifies_the_configs_own_env(self):
        """A YAML scalar the user did not quote (``KEY: 0``) must still reach the child as a string."""
        env = _build_safe_env(dict(_CONFIG_ENV))
        assert env["MCP_TEST_ENV_INT"] == "0"
        assert env["MCP_TEST_ENV_BOOL"] == "false"  # never "False": a YAML false means false
        assert env["MCP_TEST_ENV_PORT"] == "8080"
        assert env["MCP_TEST_ENV_EMPTY"] == ""
        assert env["MCP_TEST_ENV_STR"] == "as-written"
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), (
            f"non-string entry left in the child env: "
            f"{ {k: v for k, v in env.items() if not isinstance(v, str)} }")


class TestSdkAcceptsTheSpawnParameters:
    """The exact call ``MCPServerTask._run_stdio`` makes (tools/mcp_tool_transport.py)."""

    def test_uncoerced_yaml_scalar_is_what_breaks_the_spawn(self):
        from mcp import StdioServerParameters

        with pytest.raises(Exception) as exc:
            StdioServerParameters(command="npx", args=["-y", "pkg"], env={"KEY": 0})
        assert "Input should be a valid string" in str(exc.value)

    def test_spawn_params_accept_the_env_built_from_config(self):
        from mcp import StdioServerParameters

        env = _build_safe_env(dict(_CONFIG_ENV))
        params = StdioServerParameters(command="npx", args=["-y", "pkg"], env=env)
        assert params.env["MCP_TEST_ENV_INT"] == "0"
        assert params.env["MCP_TEST_ENV_BOOL"] == "false"


class TestChildReceivesTheConfiguredEnv:
    def test_spawned_child_gets_the_stringified_config_env(self, tmp_path):
        """End-to-end: config env -> child process environment (the hypothesis under test)."""
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        dump = tmp_path / "child-env.json"
        config_env = dict(_CONFIG_ENV, MCP_TEST_ENV_DUMP=str(dump))

        async def _spawn():
            params = StdioServerParameters(command=sys.executable, args=["-c", _CHILD_DUMP_ENV],
                                           env=_build_safe_env(config_env))
            async with stdio_client(params):
                for _ in range(150):
                    if dump.exists() and dump.stat().st_size:
                        return json.loads(dump.read_text(encoding="utf-8"))
                    await asyncio.sleep(0.2)
                raise AssertionError("the spawned child never wrote its environment")

        delivered = asyncio.run(asyncio.wait_for(_spawn(), timeout=60))
        assert delivered["MCP_TEST_ENV_INT"] == "0"
        assert delivered["MCP_TEST_ENV_BOOL"] == "false"
        assert delivered["MCP_TEST_ENV_PORT"] == "8080"
        assert delivered["MCP_TEST_ENV_EMPTY"] == ""
        assert delivered["MCP_TEST_ENV_STR"] == "as-written"


def test_build_safe_env_does_not_mutate_the_configs_env():
    """The coercion is a copy: the loaded config dict keeps its own (typed) values."""
    config_env = dict(_CONFIG_ENV)
    _build_safe_env(config_env)
    assert config_env["MCP_TEST_ENV_INT"] == 0
    assert config_env["MCP_TEST_ENV_BOOL"] is False
