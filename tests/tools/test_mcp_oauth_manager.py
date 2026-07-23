"""Tests for the MCP OAuth manager (tools/mcp_oauth_manager.py).

The manager consolidates the eight scattered MCP-OAuth call sites into a
single object with disk-mtime watch, dedup'd 401 handling, and a provider
cache. See `tools/mcp_oauth_manager.py` for design rationale.
"""
import json
import os
import time
from unittest.mock import MagicMock

import pytest


def test_manager_isolates_same_named_servers_by_profile_home(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    for home, access_token in ((profile_a, "TOKEN_A"), (profile_b, "TOKEN_B")):
        token = set_hermes_home_override(home)
        try:
            storage = HermesTokenStorage("shared")
            storage._tokens_path().parent.mkdir(parents=True, exist_ok=True)
            storage._tokens_path().write_text(
                '{"access_token":"%s","token_type":"Bearer","expires_in":3600}'
                % access_token
            )
        finally:
            reset_hermes_home_override(token)

    manager = MCPOAuthManager()
    providers = []
    for home in (profile_a, profile_b):
        token = set_hermes_home_override(home)
        try:
            provider = manager.get_or_build_provider("shared", "https://mcp.example/mcp", {})
            asyncio.run(provider._initialize())
            providers.append(provider)
        finally:
            reset_hermes_home_override(token)

    assert providers[0] is not providers[1]
    assert providers[0].context.current_tokens.access_token == "TOKEN_A"
    assert providers[1].context.current_tokens.access_token == "TOKEN_B"


def test_manager_explicit_home_removes_only_that_profiles_tokens(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    paths = []
    for home in (profile_a, profile_b):
        token = set_hermes_home_override(home)
        try:
            storage = HermesTokenStorage("shared")
            storage._tokens_path().parent.mkdir(parents=True, exist_ok=True)
            storage._tokens_path().write_text('{"access_token":"x","token_type":"Bearer"}')
            paths.append(storage._tokens_path())
        finally:
            reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_a)
    try:
        MCPOAuthManager().remove("shared", hermes_home=profile_b)
    finally:
        reset_hermes_home_override(token)

    assert paths[0].exists()
    assert not paths[1].exists()


def test_manager_can_restore_removed_entry_after_failed_reauth(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    manager = MCPOAuthManager()
    provider = manager.get_or_build_provider("shared", "https://mcp.example", {})

    entry = manager.remove("shared")
    manager.restore_entry("shared", entry)

    assert manager.get_or_build_provider("shared", "https://mcp.example", {}) is provider


def test_manager_restore_entry_preserves_newer_concurrent_entry(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    manager = MCPOAuthManager()
    old_provider = manager.get_or_build_provider("shared", "https://old.example", {})
    old_entry = manager.remove("shared")
    new_provider = manager.get_or_build_provider("shared", "https://new.example", {})

    manager.restore_entry("shared", old_entry)

    assert manager.get_or_build_provider("shared", "https://new.example", {}) is new_provider
    assert new_provider is not old_provider

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


def test_manager_is_singleton():
    """get_manager() returns the same instance across calls."""
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    reset_manager_for_tests()
    m1 = get_manager()
    m2 = get_manager()
    assert m1 is m2


def test_manager_get_or_build_provider_caches(tmp_path, monkeypatch):
    """Calling get_or_build_provider twice with same name returns same provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    p2 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is p2


def test_manager_get_or_build_rebuilds_on_url_change(tmp_path, monkeypatch):
    """Changing the URL discards the cached provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://a.example.com/mcp", None)
    p2 = mgr.get_or_build_provider("srv", "https://b.example.com/mcp", None)
    assert p1 is not p2


def test_manager_remove_evicts_cache(tmp_path, monkeypatch):
    """remove(name) evicts the provider from cache AND deletes disk files."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    # Pre-seed tokens on disk
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(json.dumps({
        "access_token": "TOK",
        "token_type": "Bearer",
    }))

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is not None
    assert (token_dir / "srv.json").exists()

    mgr.remove("srv")

    assert not (token_dir / "srv.json").exists()
    p2 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is not p2


def test_hermes_provider_subclass_exists():
    """HermesMCPOAuthProvider is defined and subclasses OAuthClientProvider."""
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
    from mcp.client.auth.oauth2 import OAuthClientProvider

    assert _HERMES_PROVIDER_CLS is not None
    assert issubclass(_HERMES_PROVIDER_CLS, OAuthClientProvider)


@pytest.mark.asyncio
async def test_disk_watch_invalidates_on_mtime_change(tmp_path, monkeypatch):
    """When the tokens file mtime changes, provider._initialized flips False.

    This is the behaviour Claude Code ships as
    invalidateOAuthCacheIfDiskChanged (CC-1096 / GH#24317) and is the core
    fix for Cthulhu's external-cron refresh workflow.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    tokens_file = token_dir / "srv.json"
    tokens_file.write_text(json.dumps({
        "access_token": "OLD",
        "token_type": "Bearer",
    }))

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    # First call: records mtime (zero -> real) -> returns True
    changed1 = await mgr.invalidate_if_disk_changed("srv")
    assert changed1 is True

    # No file change -> False
    changed2 = await mgr.invalidate_if_disk_changed("srv")
    assert changed2 is False

    # Touch file with a newer mtime
    future_mtime = time.time() + 10
    os.utime(tokens_file, (future_mtime, future_mtime))

    changed3 = await mgr.invalidate_if_disk_changed("srv")
    assert changed3 is True
    # _initialized flipped — next async_auth_flow will re-read from disk
    assert provider._initialized is False


@pytest.mark.asyncio
async def test_handle_401_tracks_inflight_task_to_prevent_gc(tmp_path, monkeypatch):
    """The 401 handler task must be strongly referenced by the manager.

    ``asyncio.create_task`` returns a task the event loop only weakly
    references. If the manager discards its handle, the background coroutine
    can be garbage-collected mid-run and every concurrent waiter stuck on
    ``await pending`` hangs forever. See the design note on
    ``MCPOAuthManager._inflight_tasks``.
    """
    import asyncio

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry

    class _TrackedSet(set):
        """set subclass that records every element ever inserted."""

        def __init__(self):
            super().__init__()
            self.ever_added: list = []

        def add(self, item):  # noqa: A003
            self.ever_added.append(item)
            super().add(item)

    mgr = MCPOAuthManager()
    mgr._inflight_tasks = _TrackedSet()

    class _DummyProvider:
        context = None  # forces the can_refresh=False branch

    mgr._entries[mgr._key("srv")] = _ProviderEntry(
        server_url="https://example.com/mcp",
        oauth_config=None,
        provider=_DummyProvider(),
    )

    result = await mgr.handle_401("srv", failed_access_token="TOK")

    # The discard done-callback is scheduled via loop.call_soon, so it runs on
    # a later loop iteration than the one that resolved `pending` and let
    # handle_401 return. Yield once so the callback fires before we assert the
    # task was removed from the live set.
    await asyncio.sleep(0)

    # Exactly one handler task was created and tracked.
    assert len(mgr._inflight_tasks.ever_added) == 1
    tracked_task = mgr._inflight_tasks.ever_added[0]
    assert isinstance(tracked_task, asyncio.Task)
    # done_callback must have removed the finished task from the live set,
    # otherwise the set would grow unbounded across repeated 401s.
    assert tracked_task not in mgr._inflight_tasks
    assert len(mgr._inflight_tasks) == 0
    assert tracked_task.done()
    # With provider.context=None, there's nothing to refresh — result False.
    assert result is False


@pytest.mark.asyncio
async def test_handle_401_dedup_survives_even_if_task_reference_dropped(tmp_path, monkeypatch):
    """Concurrent 401s share one handler task and all callers resolve.

    Regression guard: if the manager ever stops holding a strong reference
    to the `_do_handle` task, this test can intermittently hang when the
    task is GC'd between the ``await`` checkpoints inside ``_do_handle``.
    Running it in CI with ``gc.collect()`` mid-flight (below) exercises
    that window.
    """
    import asyncio
    import gc

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry

    mgr = MCPOAuthManager()

    class _DummyProvider:
        context = None

    mgr._entries[mgr._key("srv")] = _ProviderEntry(
        server_url="https://example.com/mcp",
        oauth_config=None,
        provider=_DummyProvider(),
    )

    # Fan out N concurrent callers sharing the same failed token so all
    # collapse onto a single deduped handler future.
    async def _caller():
        return await mgr.handle_401("srv", failed_access_token="TOK")

    tasks = [asyncio.create_task(_caller()) for _ in range(8)]
    # Give the event loop one tick to schedule _do_handle, then force GC.
    await asyncio.sleep(0)
    gc.collect()

    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
    assert results == [False] * 8
    # Let the shared _do_handle task's discard done-callback (call_soon) run.
    await asyncio.sleep(0)
    assert len(mgr._inflight_tasks) == 0


def test_manager_builds_hermes_provider_subclass(tmp_path, monkeypatch):
    """get_or_build_provider returns HermesMCPOAuthProvider, not plain OAuthClientProvider."""
    from tools.mcp_oauth_manager import (
        MCPOAuthManager, _HERMES_PROVIDER_CLS, reset_manager_for_tests,
    )
    reset_manager_for_tests()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    assert _HERMES_PROVIDER_CLS is not None
    assert isinstance(provider, _HERMES_PROVIDER_CLS)
    assert provider._hermes_server_name == "srv"


def test_manager_fails_fast_noninteractive_without_cached_tokens(tmp_path, monkeypatch):
    """A daemon without cached MCP OAuth tokens must not enter browser auth."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch, is_tty=False)
    from tools.mcp_oauth import OAuthNonInteractiveError
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()

    with pytest.raises(OAuthNonInteractiveError, match="non-interactive"):
        mgr.get_or_build_provider("linear", "https://mcp.linear.app/mcp", None)

    assert mgr._entries[mgr._key("linear")].provider is None


# ---------------------------------------------------------------------------
# invalid_client auto-heal (GH#36767) — _maybe_flag_poisoned_client
# ---------------------------------------------------------------------------

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock


def _fake_response(status, url, body):
    """A minimal stand-in for the httpx.Response the SDK feeds our bridge."""
    resp = MagicMock()
    resp.status_code = status
    resp.request = SimpleNamespace(url=url)

    async def _aread():
        return body

    resp.aread = _aread
    return resp


def _provider_with_token_endpoint(tmp_path, oauth_config, token_endpoint, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()
    # Provider construction fails fast in a non-interactive environment with no
    # cached tokens (mcp_oauth_manager.py guard). The hermetic test env has no
    # TTY, so present an interactive stdin to reach the code under test.
    _set_interactive_stdin(monkeypatch)
    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://mcp.example.com", oauth_config)
    provider.context.oauth_metadata = SimpleNamespace(token_endpoint=token_endpoint)
    provider._initialized = True
    return provider


def test_invalid_client_at_token_endpoint_poisons(tmp_path, monkeypatch):
    """400 invalid_client on the token endpoint deletes the dead client.json."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "dead"}')
    (d / "srv.meta.json").write_text("{}")
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_client"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert not (d / "srv.client.json").exists()
    assert (d / "srv.client.json.bak").exists()
    assert provider._initialized is False
    assert provider.context.client_info is None


def test_invalid_client_at_other_endpoint_is_ignored(tmp_path, monkeypatch):
    """An invalid_client body from a non-token endpoint must not poison."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "live"}')
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    resp = _fake_response(
        400, "https://mcp.example.com/messages", b'{"error":"invalid_client"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert (d / "srv.client.json").exists()
    assert provider._initialized is True


def test_success_response_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "live"}')
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    resp = _fake_response(
        200, "https://idp.example.com/oauth/token", b'{"access_token":"x"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert (d / "srv.client.json").exists()
    assert provider._initialized is True


def test_preregistered_client_is_never_poisoned(tmp_path, monkeypatch):
    """A config-supplied client_id is never auto-deleted (re-reg can't help)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {"client_id": "from-config"}, "https://idp.example.com/oauth/token", monkeypatch
    )
    d = tmp_path / "mcp-tokens"
    # _maybe_preregister_client wrote client.json from config during build.
    assert (d / "srv.client.json").exists()
    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_client"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert (d / "srv.client.json").exists()
    assert provider._initialized is True


def test_invalid_client_metadata_does_not_trip(tmp_path, monkeypatch):
    """RFC 7591 `invalid_client_metadata` must NOT be mistaken for invalid_client."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "live"}')
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_client_metadata"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert (d / "srv.client.json").exists()
    assert provider._initialized is True


class _FakeMeta:
    """Metadata stub usable by both detection and the post-flow persist hook."""

    def __init__(self, token_endpoint):
        self.token_endpoint = token_endpoint

    def model_dump(self, **kwargs):
        return {"token_endpoint": self.token_endpoint}


def test_bridge_forwards_requests_and_poisons_on_token_endpoint_400(
    tmp_path, monkeypatch
):
    """Drive the REAL async_auth_flow bridge to prove the inserted detection
    hook does not break the bidirectional asend() forwarding contract — the
    genuinely fragile part. A patched SDK base generator stands in for the
    real OAuth flow so we control exactly which response the bridge sees.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token_ep = "https://idp.example.com/oauth/token"
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "dead"}')

    forwarded = []

    async def fake_base_flow(self, request):
        # Mimic the SDK: yield the request, receive the response, then finish.
        forwarded.append(("out", request))
        response = yield request
        forwarded.append(("in", response))

    from mcp.client.auth.oauth2 import OAuthClientProvider
    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", fake_base_flow)

    provider = _provider_with_token_endpoint(tmp_path, {}, token_ep, monkeypatch)
    provider.context.oauth_metadata = _FakeMeta(token_ep)

    sentinel_request = object()
    poison_resp = _fake_response(400, token_ep, b'{"error":"invalid_client"}')

    async def drive():
        gen = provider.async_auth_flow(sentinel_request)
        out0 = await gen.__anext__()
        assert out0 is sentinel_request  # request forwarded unchanged
        try:
            await gen.asend(poison_resp)
        except StopAsyncIteration:
            pass

    asyncio.run(drive())

    # The poison response reached the inner generator (forwarding intact)...
    assert ("in", poison_resp) in forwarded
    # ...and the detection hook fired.
    assert not (d / "srv.client.json").exists()
    assert provider._initialized is False
    assert provider.context.client_info is None
