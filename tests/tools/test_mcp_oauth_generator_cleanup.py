"""Regression test: closing HermesMCPOAuthProvider.async_auth_flow wrapper
must close the inner SDK auth generator.

When httpx's auth_flow driver closes the auth generator, the Hermes wrapper
generator is closed but the inner SDK generator — which is suspended inside
``async with self.context.lock`` — is left orphaned. Python later finalizes
that orphaned inner async generator from a different asyncio task. The SDK
lock is an ``anyio.Lock``, which enforces task ownership on release, so
release raises::

    RuntimeError("The current task is not holding this lock")

and the lock stays permanently held. Subsequent reconnect attempts on the
cached provider deadlock, and the MCP server becomes unusable until the
gateway process is restarted.

This test proves that closing the Hermes wrapper currently does NOT close
the inner generator and the lock is left held. It will pass once the fix
(adding ``finally: await inner.aclose()``) is applied.
"""
from __future__ import annotations

import asyncio
import gc
import sys

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


@pytest.mark.asyncio
async def test_aclose_hermes_wrapper_releases_sdk_lock(tmp_path, monkeypatch):
    """Closing the Hermes wrapper generator must release the SDK's anyio.Lock.

    Without the fix, the inner SDK auth-flow generator remains suspended
    inside ``async with self.context.lock``, and the lock is never released
    from the owning task. Python's async generator finalizer then tries to
    release it from a different task, anyio raises RuntimeError, and the
    lock stays stuck permanently.
    """
    import httpx
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    # Seed valid-looking tokens so the SDK won't attempt full registration.
    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="test_access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="test_refresh",
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

    # Drive the wrapper so the inner SDK generator reaches its yield point
    # inside ``async with self.context.lock``.
    outbound = await flow.__anext__()
    assert outbound is not None, "wrapper must yield the outbound request"
    assert outbound.url.host == "example.com"

    # The SDK's context.lock should be held at this point — the inner
    # generator is suspended inside the locked section.
    assert provider.context.lock.locked(), (
        "SDK context lock must be held while the inner auth generator "
        "is suspended at its yield point"
    )

    # Install a temporary event-loop exception handler to capture any
    # async-generator finalizer errors (the RuntimeError about cross-task
    # lock release).
    captured_exceptions: list[dict] = []

    def _capture_exc(loop, context):
        captured_exceptions.append({
            "message": context.get("message", ""),
            "exception": context.get("exception"),
        })

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_capture_exc)

    try:
        # Close the Hermes wrapper. Without the fix, this only closes the
        # outer wrapper; the inner SDK generator stays suspended with the
        # lock held.
        await flow.aclose()

        # Let the event loop breathe and give async generator finalizers a
        # chance to run.
        gc.collect()
        await asyncio.sleep(0.1)
        await asyncio.sleep(0)  # extra yield to drain finalizer callbacks

        # After the fix: the lock should be released because the inner
        # generator was explicitly closed.
        assert not provider.context.lock.locked(), (
            "SDK context lock must be released after aclose() — "
            "the inner SDK auth generator should have been properly closed, "
            "releasing the anyio.Lock it acquired"
        )

        # No captured exception should mention cross-task lock release.
        for exc_info in captured_exceptions:
            msg = exc_info["message"]
            if msg and "not holding this lock" in msg:
                pytest.fail(
                    f"anyio cross-task lock error captured: {msg}. "
                    "The inner SDK generator was finalized from a different "
                    "task than the one that acquired the lock."
                )

    finally:
        # Restore the default exception handler.
        loop.set_exception_handler(None)


async def _noop_redirect(_url: str) -> None:
    """Redirect handler that does nothing (won't be invoked in this test)."""
    return None


async def _noop_callback() -> tuple[str, str | None]:
    """Callback handler that won't be invoked in this test."""
    raise AssertionError(
        "callback handler should not be invoked in generator-cleanup test"
    )
