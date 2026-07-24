"""Tests for MCP tool-level OAuth challenge handling (#69811).

Servers may accept anonymous ``initialize`` but protect individual tools,
returning ``isError: true`` with ``CallToolResult._meta["mcp/www_authenticate"]``
instead of a transport-level HTTP 401. The client must:

  1. Preserve the structured challenge while mapping the result.
  2. Route it through the existing PKCE OAuth manager (protected-resource
     discovery seeded from the challenge's ``resource_metadata`` + ``scope``).
  3. Reconnect with ``Authorization: Bearer`` transport auth.
  4. Retry the original tool at most once — a second protected failure must
     not trigger an unbounded retry loop.
  5. Leave ordinary MCP tool errors (no challenge metadata) unchanged.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


# The reproduction challenge from GH#69811 (Resin8 staging).
CHALLENGE = (
    'Bearer error="unauthorized", scope="resin8.buyer.profile.read", '
    'resource_metadata="https://api.stage.resin8.ai/.well-known/'
    'oauth-protected-resource/mcp"'
)


def _tool_result(is_error: bool, meta=None, text: str = "denied"):
    from mcp.types import CallToolResult

    payload = {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }
    if meta is not None:
        payload["_meta"] = meta
    return CallToolResult.model_validate(payload)


# ---------------------------------------------------------------------------
# _extract_tool_auth_challenges
# ---------------------------------------------------------------------------


def test_extract_challenges_from_list_meta():
    from tools.mcp_tool import _extract_tool_auth_challenges

    result = _tool_result(True, meta={"mcp/www_authenticate": [CHALLENGE]})
    assert _extract_tool_auth_challenges(result) == [CHALLENGE]


def test_extract_challenges_from_string_meta():
    from tools.mcp_tool import _extract_tool_auth_challenges

    result = _tool_result(True, meta={"mcp/www_authenticate": CHALLENGE})
    assert _extract_tool_auth_challenges(result) == [CHALLENGE]


def test_extract_challenges_absent_meta():
    from tools.mcp_tool import _extract_tool_auth_challenges

    assert _extract_tool_auth_challenges(_tool_result(True)) == []
    assert _extract_tool_auth_challenges(
        _tool_result(True, meta={"other": "value"})
    ) == []


def test_extract_challenges_filters_non_strings_and_blanks():
    from tools.mcp_tool import _extract_tool_auth_challenges

    result = _tool_result(
        True,
        meta={"mcp/www_authenticate": [42, "", "  ", None, f"  {CHALLENGE}  ", {}]},
    )
    assert _extract_tool_auth_challenges(result) == [CHALLENGE]


def test_extract_challenges_rejects_non_list_shapes_and_caps():
    from tools.mcp_tool import _extract_tool_auth_challenges

    assert _extract_tool_auth_challenges(
        _tool_result(True, meta={"mcp/www_authenticate": {"nested": "dict"}})
    ) == []
    flooded = _tool_result(
        True, meta={"mcp/www_authenticate": [f"Bearer n={i}" for i in range(50)]}
    )
    assert len(_extract_tool_auth_challenges(flooded)) == 8


def test_synthetic_response_feeds_sdk_extractors():
    """The joined challenge header must be parseable by the SDK's own
    WWW-Authenticate field extractors — that is what seeds protected-resource
    discovery (resource_metadata, path included) and scope selection."""
    import httpx
    from mcp.client.auth.utils import (
        extract_resource_metadata_from_www_auth,
        extract_scope_from_www_auth,
    )

    request = httpx.Request("POST", "https://api.stage.resin8.ai/mcp")
    response = httpx.Response(
        401, headers={"WWW-Authenticate": CHALLENGE}, request=request
    )
    assert extract_resource_metadata_from_www_auth(response) == (
        "https://api.stage.resin8.ai/.well-known/oauth-protected-resource/mcp"
    )
    assert extract_scope_from_www_auth(response) == "resin8.buyer.profile.read"


# ---------------------------------------------------------------------------
# MCPOAuthManager.handle_tool_challenge — generator driver
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Scripted OAuthClientProvider standing in for the SDK.

    Yields the anchor request; on a 401 with the challenge header it marks
    tokens valid (as the real PKCE flow would) and yields the anchor once
    more, expecting the driver to unwind with a synthetic 200.
    """

    def __init__(self, server_url: str, valid_upfront: bool = False):
        self._valid = valid_upfront
        self.flow_runs = 0
        self.responses_seen = []
        self.context = SimpleNamespace(
            server_url=server_url,
            is_token_valid=lambda: self._valid,
        )

    async def async_auth_flow(self, request):
        self.flow_runs += 1
        response = yield request
        self.responses_seen.append(response)
        if response.status_code == 401:
            self._valid = True  # simulate completed authorization
            response = yield request
            self.responses_seen.append(response)


def _manager_with_fake_provider(monkeypatch, provider):
    from tools.mcp_oauth_manager import MCPOAuthManager

    manager = MCPOAuthManager()
    monkeypatch.setattr(
        manager, "get_or_build_provider", lambda *a, **kw: provider,
    )
    return manager


def _seed_entry(manager, server_name, server_url, provider):
    from tools.mcp_oauth_manager import _ProviderEntry

    entry = _ProviderEntry(server_url=server_url, oauth_config=None)
    entry.provider = provider
    manager._entries[manager._key(server_name)] = entry
    return entry


@pytest.mark.asyncio
async def test_handle_tool_challenge_runs_flow_from_synthetic_401(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _FakeProvider("https://api.stage.resin8.ai/mcp")
    manager = _manager_with_fake_provider(monkeypatch, provider)
    _seed_entry(manager, "srv", "https://api.stage.resin8.ai/mcp", provider)

    ok = await manager.handle_tool_challenge(
        "srv", "https://api.stage.resin8.ai/mcp", None, [CHALLENGE],
    )

    assert ok is True
    assert provider.flow_runs == 1
    first, second = provider.responses_seen
    assert first.status_code == 401
    assert first.headers["WWW-Authenticate"] == CHALLENGE
    assert second.status_code == 200


@pytest.mark.asyncio
async def test_handle_tool_challenge_joins_multiple_values(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _FakeProvider("https://api.example/mcp")
    manager = _manager_with_fake_provider(monkeypatch, provider)
    _seed_entry(manager, "srv", "https://api.example/mcp", provider)

    ok = await manager.handle_tool_challenge(
        "srv", "https://api.example/mcp", None,
        ['Bearer error="unauthorized"', 'Basic realm="x"'],
    )

    assert ok is True
    assert provider.responses_seen[0].headers["WWW-Authenticate"] == (
        'Bearer error="unauthorized", Basic realm="x"'
    )


@pytest.mark.asyncio
async def test_handle_tool_challenge_skips_reauth_when_tokens_valid(
    monkeypatch, tmp_path,
):
    """Cached tokens from a previous process: no browser flow, no 401 —
    the driver unwinds immediately and the caller just reconnects."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _FakeProvider("https://api.example/mcp", valid_upfront=True)
    manager = _manager_with_fake_provider(monkeypatch, provider)
    _seed_entry(manager, "srv", "https://api.example/mcp", provider)

    ok = await manager.handle_tool_challenge(
        "srv", "https://api.example/mcp", None, [CHALLENGE],
    )

    assert ok is True
    assert [r.status_code for r in provider.responses_seen] == [200]


@pytest.mark.asyncio
async def test_handle_tool_challenge_flow_failure_returns_false(
    monkeypatch, tmp_path, caplog,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _ExplodingProvider(_FakeProvider):
        async def async_auth_flow(self, request):
            self.flow_runs += 1
            yield request
            raise RuntimeError("callback timed out")

    provider = _ExplodingProvider("https://api.example/mcp")
    manager = _manager_with_fake_provider(monkeypatch, provider)
    _seed_entry(manager, "srv", "https://api.example/mcp", provider)

    ok = await manager.handle_tool_challenge(
        "srv", "https://api.example/mcp", None, [CHALLENGE],
    )

    assert ok is False
    # No auth-flow material in diagnostics: the challenge string (scope,
    # resource_metadata URL) must not be logged.
    assert CHALLENGE not in caplog.text
    assert "resin8.buyer.profile.read" not in caplog.text


@pytest.mark.asyncio
async def test_handle_tool_challenge_provider_setup_failure_returns_false(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager

    manager = MCPOAuthManager()

    def _raise(*a, **kw):
        raise RuntimeError("non-interactive environment")

    monkeypatch.setattr(manager, "get_or_build_provider", _raise)
    ok = await manager.handle_tool_challenge(
        "srv", "https://api.example/mcp", None, [CHALLENGE],
    )
    assert ok is False


@pytest.mark.asyncio
async def test_handle_tool_challenge_single_flight(monkeypatch, tmp_path):
    """Concurrent protected calls share one flow — no browser-window storm."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _SlowProvider(_FakeProvider):
        async def async_auth_flow(self, request):
            self.flow_runs += 1
            response = yield request
            self.responses_seen.append(response)
            if response.status_code == 401:
                await asyncio.sleep(0.05)
                self._valid = True
                response = yield request
                self.responses_seen.append(response)

    provider = _SlowProvider("https://api.example/mcp")
    manager = _manager_with_fake_provider(monkeypatch, provider)
    _seed_entry(manager, "srv", "https://api.example/mcp", provider)

    results = await asyncio.gather(*[
        manager.handle_tool_challenge(
            "srv", "https://api.example/mcp", None, [CHALLENGE],
        )
        for _ in range(3)
    ])

    assert results == [True, True, True]
    assert provider.flow_runs == 1


# ---------------------------------------------------------------------------
# Tool handler integration — challenge → OAuth → reconnect → one retry
# ---------------------------------------------------------------------------


def _stub_server(name: str, results):
    """MagicMock MCPServerTask whose call_tool pops scripted results."""
    server = MagicMock()
    server.name = name
    server._config = {"url": "https://api.example/mcp"}
    server._is_http = lambda: True
    server._ready = MagicMock()
    server._ready.is_set.return_value = True
    session = MagicMock()
    calls = {"count": 0}

    async def _call_tool(*a, **kw):
        calls["count"] += 1
        return results.pop(0)

    session.call_tool = _call_tool
    server.session = session
    server._rpc_lock = asyncio.Lock()
    return server, calls


def _install_server(monkeypatch, server):
    from tools import mcp_tool

    mcp_tool._servers[server.name] = server
    mcp_tool._server_error_counts.pop(server.name, None)
    mcp_tool._ensure_mcp_loop()


def _patch_manager_challenge(monkeypatch, outcome, record=None):
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests

    reset_manager_for_tests()
    manager = get_manager()

    async def _handle(name, url, oauth_cfg, challenges):
        if record is not None:
            record.append((name, url, oauth_cfg, list(challenges)))
        return outcome

    monkeypatch.setattr(manager, "handle_tool_challenge", _handle)
    return manager


def test_challenge_triggers_oauth_reconnect_and_single_retry(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    challenge_result = _tool_result(
        True, meta={"mcp/www_authenticate": [CHALLENGE]},
    )
    success_result = _tool_result(False, text="alice@example.com")
    server, calls = _stub_server("srv-chal", [challenge_result, success_result])
    _install_server(monkeypatch, server)

    seen = []
    _patch_manager_challenge(monkeypatch, True, record=seen)
    reconnects = []
    monkeypatch.setattr(
        mcp_tool, "_signal_reconnect_and_wait",
        lambda name, srv, **kw: reconnects.append(name) or True,
    )

    try:
        handler = _make_tool_handler("srv-chal", "whoami", 10.0)
        out = json.loads(handler({}))
    finally:
        mcp_tool._servers.pop("srv-chal", None)

    assert out == {"result": "alice@example.com"}
    assert calls["count"] == 2  # original + exactly one retry
    assert seen == [(
        "srv-chal", "https://api.example/mcp", None, [CHALLENGE],
    )]
    assert reconnects == ["srv-chal"]
    # The rebuilt transport must carry OAuth auth (Authorization: Bearer).
    assert server._config["auth"] == "oauth"
    assert server._auth_type == "oauth"


def test_challenge_unrecoverable_returns_needs_reauth_without_retry(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    challenge_result = _tool_result(
        True, meta={"mcp/www_authenticate": [CHALLENGE]},
    )
    server, calls = _stub_server("srv-noauth", [challenge_result])
    _install_server(monkeypatch, server)
    _patch_manager_challenge(monkeypatch, False)

    try:
        handler = _make_tool_handler("srv-noauth", "whoami", 10.0)
        out = json.loads(handler({}))
    finally:
        mcp_tool._servers.pop("srv-noauth", None)

    assert out["needs_reauth"] is True
    assert out["server"] == "srv-noauth"
    assert calls["count"] == 1  # no retry without authorization
    assert "auth" not in server._config  # unchanged on failure


def test_second_challenge_does_not_loop(monkeypatch, tmp_path):
    """A second protected failure after OAuth + reconnect is a hard stop."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    challenge_result = _tool_result(
        True, meta={"mcp/www_authenticate": [CHALLENGE]},
    )
    challenge_again = _tool_result(
        True, meta={"mcp/www_authenticate": [CHALLENGE]},
    )
    server, calls = _stub_server(
        "srv-loop", [challenge_result, challenge_again],
    )
    _install_server(monkeypatch, server)
    _patch_manager_challenge(monkeypatch, True)
    monkeypatch.setattr(
        mcp_tool, "_signal_reconnect_and_wait", lambda *a, **kw: True,
    )

    try:
        handler = _make_tool_handler("srv-loop", "whoami", 10.0)
        out = json.loads(handler({}))
    finally:
        mcp_tool._servers.pop("srv-loop", None)

    assert out["needs_reauth"] is True
    assert calls["count"] == 2  # original + exactly one retry, then stop


def test_ordinary_tool_error_without_challenge_is_unchanged(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    plain_error = _tool_result(True, text="row not found")
    server, calls = _stub_server("srv-plain", [plain_error])
    _install_server(monkeypatch, server)

    manager = _patch_manager_challenge(monkeypatch, True)

    async def _must_not_run(*a, **kw):  # pragma: no cover — assertion guard
        raise AssertionError("OAuth flow must not run for ordinary errors")

    monkeypatch.setattr(manager, "handle_tool_challenge", _must_not_run)

    try:
        handler = _make_tool_handler("srv-plain", "lookup", 10.0)
        out = json.loads(handler({}))
    finally:
        mcp_tool._servers.pop("srv-plain", None)

    assert out == {"error": "row not found"}
    assert calls["count"] == 1


def test_challenge_on_stdio_server_stays_ordinary_error(
    monkeypatch, tmp_path,
):
    """OAuth over stdio is not a thing — challenge metadata on a non-HTTP
    server maps to the ordinary error path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    challenge_result = _tool_result(
        True, meta={"mcp/www_authenticate": [CHALLENGE]}, text="denied",
    )
    server, calls = _stub_server("srv-stdio", [challenge_result])
    server._config = {"command": "some-binary"}
    server._is_http = lambda: False
    _install_server(monkeypatch, server)

    try:
        handler = _make_tool_handler("srv-stdio", "whoami", 10.0)
        out = json.loads(handler({}))
    finally:
        mcp_tool._servers.pop("srv-stdio", None)

    assert out == {"error": "denied"}
    assert calls["count"] == 1
