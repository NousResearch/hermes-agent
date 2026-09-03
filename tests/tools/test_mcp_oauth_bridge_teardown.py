"""Regression test for #101756: the ``async_auth_flow`` bridge must close the
SDK's inner generator in the same task that drove it.

``httpx`` closes the **outer** bridge generator deterministically when the
auth flow ends. Before the fix, the wrapped SDK generator (``inner``) was
simply abandoned: asyncio's asyncgen finalization hook later ran
``athrow(GeneratorExit)`` through it **in a new task**, and the SDK's
``async with self.context.lock`` teardown release is task-affine —
``anyio.Lock.release()`` raises *before* clearing the owner task, so the lock
stayed owned by a dead task forever. Every later OAuth flow for that server
then deadlocked inside ``async with self.context.lock`` before issuing any
HTTP, and the provider was cached for the process lifetime.
"""

from __future__ import annotations

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


@pytest.mark.asyncio
async def test_outer_close_releases_context_lock(tmp_path, monkeypatch):
    """Closing the outer bridge must close ``inner`` in the same task.

    Deterministic assertion (no GC/hook timing): after the outer generator is
    closed at its yield point, the bridge's lock-balancing finally re-acquires
    ``context.lock`` on behalf of the suspended SDK ``async with``; only a
    same-task ``inner.aclose()`` lets the SDK's teardown release run legally
    and leave the lock FREE. Without the fix the lock reads as still held.
    """
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="old_access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="old_refresh",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)

    # Drive to the suspension point: the SDK yielded the outbound request and
    # the bridge released context.lock around the resource call.
    outbound = await flow.__anext__()
    assert outbound is not None

    # httpx tears the auth flow down exactly like this when the transport
    # dies mid-flow (e.g. a 405 on the GET stream) — httpx2/_client.py:1845.
    await flow.aclose()

    # Without the same-task inner.aclose() the SDK's lock teardown release
    # either never runs (abandoned generator) or runs cross-task and raises
    # before clearing the owner — either way the lock reads as held and the
    # provider is poisoned for the process lifetime.
    import anyio

    lock_free = True
    try:
        provider.context.lock.acquire_nowait()
    except anyio.WouldBlock:
        lock_free = False
    else:
        provider.context.lock.release()
    assert lock_free, (
        "context.lock must be free after the auth flow is torn down; a held "
        "lock deadlocks every later OAuth flow for this server (#101756)"
    )


async def _noop_redirect(_url: str) -> None:
    """Redirect handler that does nothing (won't be invoked in these tests)."""
    return None


async def _noop_callback() -> tuple[str, str | None]:
    """Callback handler that won't be invoked in these tests."""
    raise AssertionError(
        "callback handler should not be invoked in teardown tests"
    )
