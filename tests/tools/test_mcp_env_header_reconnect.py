"""MCP static-token headers must be re-resolved from the environment at every
(re)connect, not frozen at config load (#97107).

A rotated static token (e.g. ``Authorization: Bearer *** was
previously expanded once when the server config was loaded, so every
automatic reconnect kept presenting the stale value and the server could
never recover on its own after a rotation. The fix keeps MCP ``headers``
RAW in the stored config and re-resolves ``${ENV}`` refs at each transport
build.
"""

from __future__ import annotations

from unittest.mock import patch

from tools import mcp_tool_config


class TestResolveHeadersWithEnv:
    """``_resolve_headers_with_env`` re-reads env refs on every call."""

    def test_resolves_env_ref_in_header(self):
        config = {
            "url": "https://mcp.example.com",
            "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
        }
        with patch.dict("os.environ", {"MCP_TOKEN": "tok-A"}, clear=False):
            headers = mcp_tool_config._resolve_headers_with_env(config)
            assert headers["Authorization"] == "Bearer tok-A"

    def test_rotated_value_is_picked_up_on_second_call(self):
        """The whole point: a rotated token is resolved fresh on each build."""
        config = {"headers": {"Authorization": "Bearer ${MCP_TOKEN}"}}
        with patch.dict("os.environ", {"MCP_TOKEN": "tok-A"}, clear=False):
            assert (
                mcp_tool_config._resolve_headers_with_env(config)["Authorization"]
                == "Bearer tok-A"
            )
        # token rotates
        with patch.dict("os.environ", {"MCP_TOKEN": "tok-B"}, clear=False):
            assert (
                mcp_tool_config._resolve_headers_with_env(config)["Authorization"]
                == "Bearer tok-B"
            )

    def test_unset_var_keeps_literal_placeholder(self):
        config = {"headers": {"Authorization": "Bearer ${MISSING_VAR}"}}
        with patch.dict("os.environ", {}, clear=False):
            headers = mcp_tool_config._resolve_headers_with_env(config)
            assert headers["Authorization"] == "Bearer ${MISSING_VAR}"

    def test_non_string_values_pass_through(self):
        config = {"headers": {"X-Counts": 3, "X-List": ["${A}", 1]}}
        result = mcp_tool_config._resolve_headers_with_env(config)
        assert result["X-Counts"] == 3
        assert result["X-List"] == ["${A}", 1]

    def test_idempotent_with_unchanged_env(self):
        """RAW storage contract: re-resolving the same config must not
        entangle state across calls. Because headers are stored with `${ENV}`
        intact (see _load_mcp_config), a second call with an unchanged env
        yields the same header — the resolved token is never fed back into a
        later expansion (no nested/second-order expansion of a value that was
        already materialized)."""
        config = {"headers": {"Authorization": "Bearer ${MCP_TOKEN}"}}
        with patch.dict("os.environ", {"MCP_TOKEN": "tok-A"}, clear=False):
            first = mcp_tool_config._resolve_headers_with_env(config)
        with patch.dict("os.environ", {"MCP_TOKEN": "tok-A"}, clear=False):
            second = mcp_tool_config._resolve_headers_with_env(config)
        assert first == second == {"Authorization": "Bearer tok-A"}

    def test_no_second_order_expansion_of_a_materialized_value(self):
        """A resolved value is not re-interpolated on a later call: if the env
        var's own value contains a `${...}`-looking substring, it is emitted
        verbatim (the RAW header only carries the outer ref)."""
        config = {"headers": {"Authorization": "Bearer ${MCP_TOKEN}"}}
        with patch.dict("os.environ", {"MCP_TOKEN": "A${B}"}, clear=False):
            out = mcp_tool_config._resolve_headers_with_env(config)
        assert out["Authorization"] == "Bearer A${B}"


def test_load_mcp_config_keeps_headers_raw():
    """``_load_mcp_config`` stores headers WITHOUT expanding ${ENV}, while
    other fields (url) still interpolate at load time."""
    servers = {
        "api": {
            "url": "https://mcp/${HOST}",
            "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
        }
    }
    with patch("hermes_cli.config.load_config", return_value={"mcp_servers": servers}), \
         patch.dict("os.environ", {"HOST": "example", "MCP_TOKEN": "tok-A"}, clear=False):
        config = mcp_tool_config._load_mcp_config()
        assert "api" in config
        # url expands at load (only headers are re-resolved later)
        assert config["api"]["url"] == "https://mcp/example"
        # headers stay RAW so each connect re-resolves the token
        assert config["api"]["headers"] == {
            "Authorization": "Bearer ${MCP_TOKEN}"
        }


def test_load_mcp_config_empty_env_still_returns_dict():
    """Smoke: _load_mcp_config works with no servers configured."""
    with patch("hermes_cli.config.load_config", return_value={}):
        assert isinstance(mcp_tool_config._load_mcp_config(), dict)