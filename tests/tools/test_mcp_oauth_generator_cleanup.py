"""Regression tests for MCP OAuth auth-flow generator cleanup.

Field report (``hermes mcp login <server>`` against a real OAuth MCP server):

1. Hermes printed the authorization URL (PKCE + loopback callback).
2. The callback landed — the local server answered
   ``<h2>Authorization Successful</h2>``.
3. Login never completed. No tokens were written; the bounded wait timed out.
4. The console showed ``GeneratorExit`` followed by
   ``RuntimeError: The current task is not holding this lock`` raised from
   ``mcp/client/auth/oauth2.py`` while closing ``async_auth_flow`` — as an
   unretrieved task exception.
5. ``hermes mcp test <server>`` still reported "no cached tokens found".

Root cause: ``HermesMCPOAuthProvider.async_auth_flow`` bridges httpx's
bidirectional auth-flow protocol onto the SDK's generator, but never closed
the inner SDK generator. When httpx abandoned a flow mid-stream (cancelled
request, connect timeout, reconnect) it called ``aclose()`` on our wrapper;
the wrapper exited and left ``inner`` suspended inside
``async with self.context.lock``. The garbage collector later handed ``inner``
to asyncio's async-generator finalizer, which runs ``aclose()`` in a *different*
task — and anyio's ``Lock.release()`` is task-bound, so it raised and
``_owner_task`` stayed pointing at the dead flow.

Net effect: ``context.lock`` was held forever, so every later auth flow for
that server blocked on it indefinitely. That is the wedge — the OAuth callback
succeeded but the token exchange behind the lock could never run.

These tests drive the real bridge (no mocked provider) and assert the lock is
released and a subsequent flow can proceed. Nothing here performs network I/O:
``_prefetch_oauth_metadata`` is stubbed out and every HTTP response is
synthesised.
"""
from __future__ import annotations

import asyncio

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


async def _noop_redirect(authorization_url: str) -> None:
    return None


async def _noop_callback() -> tuple[str, str]:
    return "test-auth-code", "test-state"


async def _build_provider(tmp_path, monkeypatch, name: str = "srv"):
    """Build a real HermesMCPOAuthProvider backed by a temp HERMES_HOME."""
    from mcp.shared.auth import (
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage(name)
    # Seed tokens + client info so _initialize has state to load and the flow
    # reaches the first ``yield request`` without registration round-trips.
    await storage.set_tokens(
        OAuthToken(
            access_token="seeded_access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="seeded_refresh",
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

    provider = _HERMES_PROVIDER_CLS(
        server_name=name,
        server_url="https://example.invalid/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    # Keep the test hermetic: _initialize() would otherwise run live PRM/ASM
    # discovery over the network because tokens exist but metadata does not.
    async def _no_prefetch(self=provider) -> None:
        return None

    monkeypatch.setattr(provider, "_prefetch_oauth_metadata", _no_prefetch)
    return provider


def _fake_401(request):
    import httpx

    return httpx.Response(
        401,
        request=request,
        headers={
            "www-authenticate": (
                'Bearer resource_metadata='
                '"https://example.invalid/.well-known/oauth-protected-resource"'
            )
        },
    )


async def _drive_into_oauth_branch(provider):
    """Start a flow and park it mid-OAuth, holding ``context.lock``.

    Returns the live wrapper generator, suspended after yielding the SDK's
    metadata-discovery request — exactly the state a cancelled login leaves
    behind.
    """
    import httpx

    flow = provider.async_auth_flow(
        httpx.Request("POST", "https://example.invalid/mcp")
    )
    outbound = await flow.__anext__()
    discovery_request = await flow.asend(_fake_401(outbound))
    assert isinstance(discovery_request, httpx.Request)
    assert provider.context.lock.locked(), (
        "precondition: the SDK holds context.lock for the duration of the flow"
    )
    return flow


async def _run_flow_to_completion(provider):
    """Drive one non-401 flow start-to-finish and return the outbound request.

    Every step stays inside the *calling* task, matching httpx's contract
    (``_send_handling_auth`` drives one auth flow from a single task) — anyio's
    ``Lock`` is task-bound, so splitting ``__anext__`` and ``asend`` across
    tasks would fail for reasons unrelated to what these tests cover.
    """
    import httpx

    flow = provider.async_auth_flow(
        httpx.Request("POST", "https://example.invalid/mcp")
    )
    outbound = await flow.__anext__()
    try:
        await flow.asend(httpx.Response(200, request=outbound))
    except StopAsyncIteration:
        return outbound
    raise AssertionError("flow did not finish on a 200 response")


# ---------------------------------------------------------------------------
# The wedge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_mid_flow_releases_the_sdk_auth_lock(tmp_path, monkeypatch):
    """Abandoning a flow must not strand ``context.lock``.

    Before the fix the wrapper dropped the inner SDK generator, which kept the
    lock until GC finalized it off-task — where anyio refused the release and
    the lock stayed held forever.
    """
    provider = await _build_provider(tmp_path, monkeypatch)
    flow = await _drive_into_oauth_branch(provider)

    await flow.aclose()

    assert not provider.context.lock.locked(), (
        "aclose() on the wrapper must close the inner SDK generator so its "
        "``async with self.context.lock`` unwinds"
    )


@pytest.mark.asyncio
async def test_flow_after_abandoned_login_does_not_wedge(tmp_path, monkeypatch):
    """The user-visible symptom: the *next* auth flow must still run.

    This is the wedge itself. With a stranded lock, the follow-up flow — the
    one that would exchange the authorization code for tokens — blocks on
    ``async with self.context.lock`` forever, so a successful OAuth callback
    never produces cached tokens.
    """
    provider = await _build_provider(tmp_path, monkeypatch)
    flow = await _drive_into_oauth_branch(provider)
    await flow.aclose()

    try:
        outbound = await asyncio.wait_for(
            _run_flow_to_completion(provider), timeout=5
        )
    except asyncio.TimeoutError:  # pragma: no cover — this is the bug
        pytest.fail(
            "second auth flow deadlocked on a stranded context.lock — "
            "OAuth login would hang after a successful callback"
        )
    assert outbound.url.host == "example.invalid"


@pytest.mark.asyncio
async def test_cross_task_cleanup_is_silent_and_still_frees_the_lock(
    tmp_path, monkeypatch
):
    """Closing the wrapper from another task must not raise or leave it locked.

    This is the GC-finalizer shape: asyncio's async-generator finalizer runs
    ``aclose()`` in a task that never acquired the lock. anyio's task-bound
    ``release()`` raises ``RuntimeError: The current task is not holding this
    lock`` there. That RuntimeError must be swallowed (it used to escape as an
    unretrieved task exception) and the lock reclaimed anyway.
    """
    provider = await _build_provider(tmp_path, monkeypatch)
    flow = await _drive_into_oauth_branch(provider)

    # A separate task — like the async-generator finalizer — does the close.
    closer = asyncio.create_task(flow.aclose())
    await asyncio.wait_for(closer, timeout=5)

    assert closer.exception() is None, (
        "cross-task auth-flow cleanup must not surface the SDK's task-bound "
        "lock RuntimeError"
    )
    assert not provider.context.lock.locked(), (
        "a lock stranded by cross-task cleanup must be reclaimed"
    )


@pytest.mark.asyncio
async def test_garbage_collected_flow_emits_no_unhandled_task_exception(
    tmp_path, monkeypatch
):
    """Dropping the wrapper entirely must not log an unretrieved task error.

    Reproduces the console noise from the field report: ``GeneratorExit`` then
    ``RuntimeError: The current task is not holding this lock``, surfaced via
    the event loop's exception handler because the finalizer task's exception
    was never retrieved.
    """
    import gc

    provider = await _build_provider(tmp_path, monkeypatch)
    flow = await _drive_into_oauth_branch(provider)

    handled: list[dict] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: handled.append(context))
    try:
        del flow
        gc.collect()
        # Let the finalizer task asyncio schedules for the dropped generator run.
        for _ in range(20):
            await asyncio.sleep(0)
        await asyncio.sleep(0.1)
    finally:
        loop.set_exception_handler(previous_handler)

    lock_errors = [
        ctx for ctx in handled
        if "not holding this lock" in str(ctx.get("exception", ""))
    ]
    assert not lock_errors, (
        f"SDK generator-cleanup noise leaked to the event loop: {lock_errors}"
    )
    assert not provider.context.lock.locked(), (
        "a GC-finalized auth flow must not leave context.lock held"
    )


# ---------------------------------------------------------------------------
# The happy path must be untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_completion_still_ends_with_stop_async_iteration(
    tmp_path, monkeypatch
):
    """Adding cleanup must not change the non-401 completion contract.

    httpx relies on ``StopAsyncIteration`` to know the flow is done and to
    return the final response.
    """
    import httpx

    provider = await _build_provider(tmp_path, monkeypatch)
    flow = provider.async_auth_flow(
        httpx.Request("POST", "https://example.invalid/mcp")
    )
    outbound = await flow.__anext__()

    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200, request=outbound))

    assert not provider.context.lock.locked(), (
        "a normally-completed flow releases the lock via the SDK's own unwind"
    )
    # Closing an exhausted generator is a no-op and must stay quiet.
    await flow.aclose()


@pytest.mark.asyncio
async def test_repeated_flows_reuse_the_lock(tmp_path, monkeypatch):
    """Back-to-back flows on one provider must all acquire the lock."""
    provider = await _build_provider(tmp_path, monkeypatch)
    for _ in range(3):
        await asyncio.wait_for(_run_flow_to_completion(provider), timeout=5)
        assert not provider.context.lock.locked()


# ---------------------------------------------------------------------------
# _reclaim_stranded_auth_lock must never steal a live lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reclaim_ignores_a_lock_held_by_a_different_flow():
    """Only the exact recorded owner is reclaimable.

    A clean release either clears ``_owner_task`` or hands it to a waiter, so
    the identity check is what keeps the reclaim from stealing a lock held by
    a genuinely concurrent auth flow.
    """
    import anyio

    from tools.mcp_oauth_manager import _reclaim_stranded_auth_lock

    lock = anyio.Lock()

    async def _hold(started: asyncio.Event, release: asyncio.Event) -> None:
        async with lock:
            started.set()
            await release.wait()

    started, release = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(_hold(started, release))
    await started.wait()

    stale_owner = asyncio.current_task()  # never actually owned this lock
    assert _reclaim_stranded_auth_lock(lock, stale_owner, server_name="srv") is False
    assert lock.locked(), "a lock held by another running flow must be left alone"

    release.set()
    await holder
    assert not lock.locked()


@pytest.mark.asyncio
async def test_reclaim_is_a_noop_without_a_recorded_owner():
    """No recorded owner (e.g. a plain ``asyncio.Lock``) means nothing to reclaim."""
    import anyio

    from tools.mcp_oauth_manager import _reclaim_stranded_auth_lock

    assert _reclaim_stranded_auth_lock(None, None) is False
    assert _reclaim_stranded_auth_lock(anyio.Lock(), None) is False
    # Unlocked lock with an owner recorded from a prior flow: nothing to do.
    assert _reclaim_stranded_auth_lock(anyio.Lock(), asyncio.current_task()) is False


@pytest.mark.asyncio
async def test_reclaim_hands_the_lock_to_a_waiting_flow():
    """Reclaiming must not drop a queued waiter on the floor."""
    import anyio

    from tools.mcp_oauth_manager import _reclaim_stranded_auth_lock

    lock = anyio.Lock()
    await lock.acquire()
    owner = asyncio.current_task()

    acquired = asyncio.Event()

    async def _waiter() -> None:
        async with lock:
            acquired.set()

    waiting = asyncio.create_task(_waiter())
    await asyncio.sleep(0)  # let the waiter queue up

    assert _reclaim_stranded_auth_lock(lock, owner, server_name="srv") is True
    await asyncio.wait_for(acquired.wait(), timeout=5)
    await waiting
    assert not lock.locked()
