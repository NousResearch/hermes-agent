"""Tests for noninteractive MCP OAuth client_credentials auth."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import pytest


def test_client_credentials_does_not_use_browser_callback(monkeypatch, tmp_path):
    """client_credentials must not route through the OAuth browser callback flow."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import tools.mcp_oauth as mcp_oauth

    async def _fail_callback():  # pragma: no cover - should never be called
        raise AssertionError("client_credentials called browser callback")

    monkeypatch.setattr(mcp_oauth, "_wait_for_callback", _fail_callback)

    from tools.mcp_client_credentials import build_client_credentials_auth

    auth = build_client_credentials_auth(
        "linear",
        "https://mcp.linear.app/mcp",
        {
            "client_id": "linear-client-id",
            "client_secret": "linear-client-secret",
            "scope": "read,comments:create",
        },
    )

    assert auth is not None


def test_refreshes_in_memory_token_before_expiry(monkeypatch, tmp_path):
    """A near-expiry token is reminted in memory and is not written to mcp-tokens."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.mcp_client_credentials import MCPClientCredentialsAuth

    now = {"value": 1_000.0}
    minted_tokens = iter([
        {"access_token": "first-token", "token_type": "Bearer", "expires_in": 90},
        {"access_token": "second-token", "token_type": "Bearer", "expires_in": 300},
    ])
    token_requests: list[httpx.Request] = []

    async def token_handler(request: httpx.Request) -> httpx.Response:
        token_requests.append(request)
        return httpx.Response(200, json=next(minted_tokens))

    seen_auth_headers: list[str | None] = []

    async def mcp_handler(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"ok": True})

    def token_client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(token_handler))

    auth = MCPClientCredentialsAuth(
        server_name="linear",
        token_url="https://api.linear.app/oauth/token",
        client_id="linear-client-id",
        client_secret="linear-client-secret",
        scope="read,comments:create",
        refresh_skew_seconds=60,
        now=lambda: now["value"],
        token_client_factory=token_client_factory,
    )

    async def drive():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(mcp_handler), auth=auth
        ) as client:
            await client.get("https://mcp.linear.app/mcp")
            now["value"] = 1_031.0
            await client.get("https://mcp.linear.app/mcp")

    asyncio.run(drive())

    assert seen_auth_headers == ["Bearer first-token", "Bearer second-token"]
    assert len(token_requests) == 2
    assert not (Path(tmp_path) / "mcp-tokens" / "linear.json").exists()


def test_remints_and_retries_once_after_401(monkeypatch, tmp_path):
    """A 401 from the MCP endpoint forces a fresh client_credentials token."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.mcp_client_credentials import MCPClientCredentialsAuth

    token_payloads = iter([
        {"access_token": "expired-token", "token_type": "Bearer", "expires_in": 300},
        {"access_token": "fresh-token", "token_type": "Bearer", "expires_in": 300},
    ])
    token_call_count = 0

    async def token_handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_call_count
        token_call_count += 1
        return httpx.Response(200, json=next(token_payloads))

    mcp_auth_headers: list[str | None] = []

    async def mcp_handler(request: httpx.Request) -> httpx.Response:
        mcp_auth_headers.append(request.headers.get("Authorization"))
        if len(mcp_auth_headers) == 1:
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(200, json={"ok": True})

    def token_client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(token_handler))

    auth = MCPClientCredentialsAuth(
        server_name="linear",
        token_url="https://api.linear.app/oauth/token",
        client_id="linear-client-id",
        client_secret="linear-client-secret",
        scope="read,comments:create",
        now=lambda: 2_000.0,
        token_client_factory=token_client_factory,
    )

    async def drive():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(mcp_handler), auth=auth
        ) as client:
            return await client.get("https://mcp.linear.app/mcp")

    response = asyncio.run(drive())

    assert response.status_code == 200
    assert mcp_auth_headers == ["Bearer expired-token", "Bearer fresh-token"]
    assert token_call_count == 2


def test_token_mint_errors_redact_secrets(monkeypatch, tmp_path, caplog):
    """Token endpoint failures must not log or raise client secrets or tokens."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.mcp_client_credentials import (
        ClientCredentialsTokenError,
        MCPClientCredentialsAuth,
    )

    secret = "linear-client-secret-value"
    leaked_token = "tok_1234567890"

    async def token_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "invalid_client",
                "error_description": f"bad secret {secret}; access_token={leaked_token}",
            },
        )

    def token_client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(token_handler))

    auth = MCPClientCredentialsAuth(
        server_name="linear",
        token_url="https://api.linear.app/oauth/token",
        client_id="linear-client-id",
        client_secret=secret,
        scope="read",
        token_client_factory=token_client_factory,
    )

    async def drive():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            auth=auth,
        ) as client:
            return await client.get("https://mcp.linear.app/mcp")

    caplog.set_level(logging.WARNING, logger="tools.mcp_client_credentials")
    with pytest.raises(ClientCredentialsTokenError) as exc_info:
        asyncio.run(drive())

    message = str(exc_info.value)
    assert secret not in message
    assert leaked_token not in message
    assert secret not in caplog.text
    assert leaked_token not in caplog.text
    assert "[REDACTED]" in message


def test_linear_api_key_is_not_used_as_client_credentials(monkeypatch, tmp_path):
    """LINEAR_API_KEY is not a fallback for the client_credentials lane."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LINEAR_API_KEY", "linear-api-key-that-must-not-be-used")

    from tools.mcp_client_credentials import (
        ClientCredentialsConfigError,
        build_client_credentials_auth,
    )

    with pytest.raises(ClientCredentialsConfigError) as exc_info:
        build_client_credentials_auth(
            "linear",
            "https://mcp.linear.app/mcp",
            {"scope": "read"},
        )

    assert "LINEAR_API_KEY" not in str(exc_info.value)
    assert "client_id" in str(exc_info.value)
    assert "client_secret" in str(exc_info.value)
