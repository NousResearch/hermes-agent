"""oauth.scope is a hard ceiling through SDK discovery and DCR (#101467)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK OAuth required")


CONFIGURED = "user:read company:read recognition:read rewards:read recognition:write"
ADVERTISED = ["user:read", "company:read", "billing:administer", "finance:administer"]


def _set_interactive_stdin(monkeypatch) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


def test_configured_scope_survives_sdk_assignment(tmp_path, monkeypatch, caplog):
    from tools.mcp_oauth import HermesTokenStorage, _build_client_metadata, _configure_callback_port

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {"scope": CONFIGURED, "redirect_port": 0}
    storage = HermesTokenStorage("bonusly")
    _configure_callback_port(cfg, storage)
    metadata = _build_client_metadata(cfg)

    assert metadata.scope == CONFIGURED
    metadata.scope = " ".join(ADVERTISED)
    assert metadata.scope == CONFIGURED
    assert "keeping configured oauth.scope" in caplog.text


def test_unset_scope_still_accepts_sdk_assignment(tmp_path, monkeypatch):
    from tools.mcp_oauth import HermesTokenStorage, _build_client_metadata, _configure_callback_port

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {"redirect_port": 0}
    storage = HermesTokenStorage("open")
    _configure_callback_port(cfg, storage)
    metadata = _build_client_metadata(cfg)

    metadata.scope = " ".join(ADVERTISED)
    assert metadata.scope == " ".join(ADVERTISED)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_scope", "expected_scope"),
    [
        (CONFIGURED, CONFIGURED),
        (None, " ".join(ADVERTISED)),
    ],
)
async def test_registration_honors_configured_scope_ceiling(
    tmp_path, monkeypatch, configured_scope, expected_scope
):
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
    )
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    cfg = {"redirect_port": 0}
    if configured_scope is not None:
        cfg["scope"] = configured_scope
    storage = HermesTokenStorage("scoped-registration")
    _configure_callback_port(cfg, storage)
    metadata = _build_client_metadata(cfg)
    provider = _HERMES_PROVIDER_CLS(
        server_name="scoped-registration",
        server_url="https://mcp.example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    flow = provider.async_auth_flow(
        httpx.Request("POST", "https://mcp.example.com/mcp")
    )
    outbound = await flow.__anext__()
    outbound = await flow.asend(
        httpx.Response(
            401,
            request=outbound,
            headers={
                "www-authenticate": (
                    'Bearer resource_metadata="https://mcp.example.com/'
                    '.well-known/oauth-protected-resource"'
                )
            },
        )
    )

    for _ in range(8):
        url = str(outbound.url)
        if url.endswith("/oauth/register"):
            registration = json.loads(outbound.content)
            assert registration["scope"] == expected_scope
            await flow.aclose()
            return

        if url.endswith("/.well-known/oauth-protected-resource"):
            response = httpx.Response(
                200,
                request=outbound,
                json={
                    "resource": "https://mcp.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                    "scopes_supported": ADVERTISED,
                    "bearer_methods_supported": ["header"],
                },
            )
        elif "/.well-known/oauth-authorization-server" in url:
            response = httpx.Response(
                200,
                request=outbound,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "https://auth.example.com/oauth/authorize",
                    "token_endpoint": "https://auth.example.com/oauth/token",
                    "registration_endpoint": "https://auth.example.com/oauth/register",
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": ["none"],
                    "scopes_supported": ADVERTISED,
                },
            )
        else:
            response = httpx.Response(404, request=outbound)

        outbound = await flow.asend(response)

    await flow.aclose()
    raise AssertionError("OAuth flow never reached dynamic client registration")


def test_manager_rebuilds_provider_when_scope_tightens(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    reset_manager_for_tests()
    manager = MCPOAuthManager()
    url = "https://mcp.example.com/mcp"

    wide = manager.get_or_build_provider("mesh", url, {"scope": "openid read write"})
    tight = manager.get_or_build_provider("mesh", url, {"scope": "openid read"})

    assert tight is not wide
    assert tight.context.client_metadata.scope == "openid read"
    tight.context.client_metadata.scope = "openid read write"
    assert tight.context.client_metadata.scope == "openid read"


async def _noop_redirect(_url: str) -> None:
    return None


async def _noop_callback() -> tuple[str, str | None]:
    raise AssertionError("callback handler should not be invoked")
