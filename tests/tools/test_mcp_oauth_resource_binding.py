"""OAuth state is bound to the MCP server URL it was granted for, not only the server name.

State is filed as ``mcp-tokens/<name>.*``. When a configured name is re-pointed at a different URL
(config edit, ``hermes mcp add`` over an existing name, dashboard replace, remove + re-add), the old
server's tokens, client registration and metadata must not be used for the new one.
"""

import asyncio

import pytest

pytest.importorskip("mcp")

from mcp.shared.auth import OAuthToken  # noqa: E402

from tools.mcp_oauth_provider import prepare_oauth_config  # noqa: E402

OLD, NEW = "https://old.example.com/mcp", "https://new.example.com/mcp"


def _storage(url):
    """The storage the MCP client opens for server ``srv`` at ``url`` (``build_oauth_auth`` path)."""
    return prepare_oauth_config("srv", url, {})[1]


def test_state_granted_for_one_url_is_not_used_for_another(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    asyncio.run(_storage(OLD).set_tokens(OAuthToken(access_token="old-access", token_type="Bearer",
                                                    refresh_token="r")))

    same = _storage(OLD + "/")
    assert asyncio.run(same.get_tokens()).access_token == "old-access"

    moved = _storage(NEW)
    assert asyncio.run(moved.get_tokens()) is None
    assert asyncio.run(moved.get_client_info()) is None and moved.load_oauth_metadata() is None
