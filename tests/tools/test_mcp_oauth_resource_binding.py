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


def test_client_registration_and_metadata_carry_their_own_binding(tmp_path, monkeypatch):
    """A partial authorization (registration + metadata saved, no token) or an unreadable token file
    must not hand the old server's client_id and endpoints to a re-pointed name."""
    from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old = _storage(OLD)
    asyncio.run(old.set_client_info(OAuthClientInformationFull(
        client_id="old-client", redirect_uris=["http://127.0.0.1:8420/callback"])))
    old.save_oauth_metadata(OAuthMetadata(
        issuer="https://auth.old.example.com", authorization_endpoint="https://auth.old.example.com/authorize",
        token_endpoint="https://auth.old.example.com/token"))

    for token_file in (None, "{not json"):
        if token_file is not None:
            old._tokens_path().write_text(token_file)
        moved = _storage(NEW)
        assert asyncio.run(moved.get_client_info()) is None and moved.load_oauth_metadata() is None
        same = _storage(OLD)
        assert asyncio.run(same.get_client_info()).client_id == "old-client"
        assert same.load_oauth_metadata() is not None


def test_token_presence_checks_follow_the_binding(tmp_path, monkeypatch):
    """"A token landed" / "cached tokens exist" must not answer yes off another server's grant."""
    from hermes_cli.mcp_config import _oauth_tokens_present

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    asyncio.run(_storage(OLD).set_tokens(OAuthToken(access_token="old-access", token_type="Bearer")))

    assert not _storage(NEW).has_cached_tokens()
    assert not _oauth_tokens_present("srv", NEW)
    assert _storage(OLD).has_cached_tokens() and _oauth_tokens_present("srv", OLD)
