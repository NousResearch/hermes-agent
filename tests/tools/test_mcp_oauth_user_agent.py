"""Tests for the per-server ``oauth.user_agent`` on MCP OAuth token requests.

Some authorization servers and WAFs reject httpx's default User-Agent on the
token endpoint (#75576). The header is opt-in, per-server, and applied ONLY to
the two token-endpoint requests (authorization-code exchange and refresh) —
never to MCP traffic or discovery.

The tests drive the REAL provider classes' request builders end to end: the
``httpx.Request`` the SDK would send is what gets inspected, not a mocked
constructor call.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK required for OAuth support",
)

from tools.mcp_oauth import (  # noqa: E402 — after the SDK availability gate
    build_oauth_auth,
    token_request_user_agent,
)


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


@pytest.fixture(autouse=True)
def clean_port_state():
    import tools.mcp_oauth as mod

    mod._assigned_cimd_ports.clear()
    yield
    mod._assigned_cimd_ports.clear()
    for port in list(mod._reserved_sockets):
        sock = mod._reserved_sockets.pop(port, None)
        if sock is not None:
            sock.close()


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_configured_user_agent_is_returned():
    assert token_request_user_agent({"user_agent": "My-MCP-Client/1.0"}) == "My-MCP-Client/1.0"


@pytest.mark.parametrize("cfg", [
    pytest.param({}, id="absent"),
    pytest.param({"user_agent": None}, id="null"),
    pytest.param({"user_agent": ""}, id="empty"),
    pytest.param({"user_agent": "   "}, id="whitespace-only"),
    pytest.param({"user_agent": 7}, id="non-string"),
])
def test_unset_user_agent_values_are_treated_as_absent(cfg):
    assert token_request_user_agent(cfg) is None


def test_user_agent_is_stripped():
    assert token_request_user_agent({"user_agent": "  UA/2 "}) == "UA/2"


# ---------------------------------------------------------------------------
# The requests the SDK actually sends
# ---------------------------------------------------------------------------


def _ready_for_token_requests(provider):
    """Give the provider the minimum context both builders require."""
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    provider.context.oauth_metadata = SimpleNamespace(
        token_endpoint="https://idp.example.com/oauth/token"
    )
    provider.context.client_info = OAuthClientInformationFull.model_validate({
        "client_id": "client-1",
        "redirect_uris": ["http://127.0.0.1:33333/callback"],
    })
    provider.context.current_tokens = OAuthToken.model_validate({
        "access_token": "at",
        "token_type": "Bearer",
        "refresh_token": "rt",
    })


def _build_provider_via(builder, monkeypatch, tmp_path, cfg):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    return builder("srv", "https://mcp.example.com/mcp", cfg)


def _manager_builder(server_name, server_url, cfg):
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    reset_manager_for_tests()
    return MCPOAuthManager().get_or_build_provider(server_name, server_url, cfg)


@pytest.mark.parametrize("builder", [
    pytest.param(build_oauth_auth, id="build_oauth_auth"),
    pytest.param(_manager_builder, id="oauth_manager"),
])
def test_token_requests_carry_the_configured_user_agent(
    builder, tmp_path, monkeypatch
):
    """Both token-endpoint requests, on both provider construction paths."""
    provider = _build_provider_via(
        builder, monkeypatch, tmp_path, {"user_agent": "My-MCP-Client/1.0"}
    )
    _ready_for_token_requests(provider)

    exchange = asyncio.run(
        provider._exchange_token_authorization_code("code", "verifier")
    )
    refresh = asyncio.run(provider._refresh_token())

    assert exchange.headers["User-Agent"] == "My-MCP-Client/1.0"
    assert refresh.headers["User-Agent"] == "My-MCP-Client/1.0"


@pytest.mark.parametrize("builder", [
    pytest.param(build_oauth_auth, id="build_oauth_auth"),
    pytest.param(_manager_builder, id="oauth_manager"),
])
def test_unconfigured_user_agent_leaves_the_default_header(
    builder, tmp_path, monkeypatch
):
    """No config → httpx's own default, exactly as before the feature."""
    import httpx

    provider = _build_provider_via(builder, monkeypatch, tmp_path, {})
    _ready_for_token_requests(provider)

    exchange = asyncio.run(
        provider._exchange_token_authorization_code("code", "verifier")
    )
    refresh = asyncio.run(provider._refresh_token())

    default_ua = httpx.Request("POST", "https://x.example/").headers.get("User-Agent")
    assert exchange.headers.get("User-Agent") == default_ua
    assert refresh.headers.get("User-Agent") == default_ua


def test_user_agent_does_not_disturb_token_auth_preparation(tmp_path, monkeypatch):
    """The stamp runs after prepare_token_auth — a confidential client's
    Authorization header must survive alongside the custom User-Agent."""
    provider = _build_provider_via(
        build_oauth_auth, monkeypatch, tmp_path,
        {"user_agent": "UA/1", "client_id": "pre", "client_secret": "shh",
         "token_endpoint_auth_method": "client_secret_basic"},
    )
    _ready_for_token_requests(provider)
    from mcp.shared.auth import OAuthClientInformationFull

    provider.context.client_info = OAuthClientInformationFull.model_validate({
        "client_id": "pre",
        "client_secret": "shh",
        "token_endpoint_auth_method": "client_secret_basic",
        "redirect_uris": ["http://127.0.0.1:33333/callback"],
    })

    exchange = asyncio.run(
        provider._exchange_token_authorization_code("code", "verifier")
    )

    assert exchange.headers["User-Agent"] == "UA/1"
    assert exchange.headers.get("Authorization", "").startswith("Basic ")
