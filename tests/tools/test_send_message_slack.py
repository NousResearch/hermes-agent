"""Slack-specific send_message delivery regressions.

Salvaged from #47547 and adapted to the post-#41112 plugin layout: the legacy
``_send_slack`` helper moved to ``plugins/platforms/slack/adapter.py::
_standalone_send`` and text sends now route through ``_send_via_adapter``
(live adapter first, registry standalone fallback).
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from tools.send_message_tool import _send_to_platform


def _ensure_slack_mock(monkeypatch):
    """Install lightweight Slack modules when optional Slack deps are absent."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_slack_send_to_platform_routes_through_send_via_adapter(monkeypatch):
    """Slack text sends go through _send_via_adapter (live adapter first)."""
    _ensure_slack_mock(monkeypatch)

    live_send = AsyncMock(return_value={"success": True, "message_id": "live-ts"})

    with patch("tools.send_message_tool._send_via_adapter", live_send):
        result = asyncio.run(
            _send_to_platform(
                Platform.SLACK,
                SimpleNamespace(enabled=True, token="bad-token,good-token", extra={}),
                "C123",
                "**hello** from Hermes",
                thread_id="171.1",
            )
        )

    assert result == {"success": True, "message_id": "live-ts"}
    live_send.assert_awaited_once()
    call = live_send.await_args
    assert call.args[0] == Platform.SLACK
    assert call.args[2] == "C123"
    assert call.kwargs["thread_id"] == "171.1"


class _SlackResponse:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _SlackPostContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SlackSession:
    """Fake aiohttp session whose good-token posts succeed."""

    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers, json, **kwargs):
        token = headers["Authorization"].removeprefix("Bearer ")
        self.calls.append((token, json))
        if token == "good-token":
            payload = {"ok": True, "ts": "171.123"}
        else:
            payload = {"ok": False, "error": "invalid_auth"}
        return _SlackPostContext(_SlackResponse(payload))


@pytest.fixture
def _standalone_send(monkeypatch):
    _ensure_slack_mock(monkeypatch)
    from plugins.platforms.slack import adapter as slack_adapter

    return slack_adapter._standalone_send


def test_standalone_send_stops_on_non_token_error(monkeypatch, _standalone_send):
    """Terminal errors (not token-scoped) must not burn the remaining tokens."""

    class _FatalSession(_SlackSession):
        def post(self, url, *, headers, json, **kwargs):
            token = headers["Authorization"].removeprefix("Bearer ")
            self.calls.append((token, json))
            return _SlackPostContext(
                _SlackResponse({"ok": False, "error": "msg_too_long"})
            )

    fake_session = _FatalSession()
    monkeypatch.setattr(
        "aiohttp.ClientSession", lambda *args, **kwargs: fake_session
    )

    pconfig = SimpleNamespace(enabled=True, token="tok-a,tok-b", extra={})
    result = asyncio.run(_standalone_send(pconfig, "C123", "hello"))

    assert result == {"error": "Slack API error: msg_too_long"}
    assert len(fake_session.calls) == 1


# ---------------------------------------------------------------------------
# Workspace-qualified (team_id) propagation — cron slack:<team>:<channel>
# delivery must carry the explicit workspace identity end to end.
# ---------------------------------------------------------------------------


class _RoutingSlackSession:
    """Fake aiohttp session routing by URL; records (url, token, payload)."""

    def __init__(self, *, auth_teams=None, auth_errors=None, post_payload=None):
        self.calls = []
        self._auth_teams = auth_teams or {}
        self._auth_errors = set(auth_errors or ())
        self._post_payload = (
            post_payload if post_payload is not None else {"ok": True, "ts": "171.123"}
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers, json=None, **kwargs):
        token = headers["Authorization"].removeprefix("Bearer ")
        self.calls.append((url, token, json))
        if "auth.test" in url:
            if token in self._auth_errors:
                raise RuntimeError(f"transient auth.test failure for {token}")
            team = self._auth_teams.get(token)
            payload = (
                {"ok": True, "team_id": team}
                if team is not None
                else {"ok": False, "error": "invalid_auth"}
            )
        else:
            payload = self._post_payload
        return _SlackPostContext(_SlackResponse(payload))


def test_send_via_adapter_live_path_propagates_team_metadata(monkeypatch):
    """The live in-process adapter must receive the workspace identity.

    The Slack adapter's ``_get_client`` reads team_id/scope_id/slack_team_id
    from send metadata to pick the workspace client — a workspace-qualified
    cron target must supply all three.
    """
    from tools.send_message_tool import _send_via_adapter

    adapter = AsyncMock()
    adapter.send.return_value = SimpleNamespace(success=True, message_id="ts-1")
    runner = SimpleNamespace(adapters={Platform.SLACK: adapter})
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)

    result = asyncio.run(
        _send_via_adapter(
            Platform.SLACK,
            SimpleNamespace(enabled=True, token="xoxb-a", extra={}),
            "C123",
            "hello",
            team_id="T111",
        )
    )

    assert result == {"success": True, "message_id": "ts-1"}
    metadata = adapter.send.await_args.kwargs["metadata"]
    assert metadata["team_id"] == "T111"
    assert metadata["scope_id"] == "T111"
    assert metadata["slack_team_id"] == "T111"
    assert metadata["workspace_pinned"] == "true"


def test_send_via_adapter_standalone_fallback_propagates_team_id(monkeypatch):
    """The plugin standalone sender must receive the explicit team identity
    so it can pick the matching workspace token instead of tokens[0]."""
    from tools.send_message_tool import _send_via_adapter

    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    standalone = AsyncMock(return_value={"success": True, "message_id": "ts-2"})
    entry = SimpleNamespace(standalone_sender_fn=standalone)
    monkeypatch.setattr(
        "gateway.platform_registry.platform_registry",
        SimpleNamespace(get=lambda name: entry),
    )

    result = asyncio.run(
        _send_via_adapter(
            Platform.SLACK,
            SimpleNamespace(enabled=True, token="xoxb-a", extra={}),
            "C123",
            "hello",
            team_id="T111",
        )
    )

    assert result == {"success": True, "message_id": "ts-2"}
    assert standalone.await_args.kwargs["team_id"] == "T111"


def test_slack_send_to_platform_propagates_team_id(monkeypatch):
    """_send_to_platform must thread team_id through to _send_via_adapter."""
    _ensure_slack_mock(monkeypatch)

    live_send = AsyncMock(return_value={"success": True, "message_id": "live-ts"})

    with patch("tools.send_message_tool._send_via_adapter", live_send):
        result = asyncio.run(
            _send_to_platform(
                Platform.SLACK,
                SimpleNamespace(enabled=True, token="bad-token,good-token", extra={}),
                "C123",
                "**hello** from Hermes",
                thread_id="171.1",
                team_id="T111",
            )
        )

    assert result == {"success": True, "message_id": "live-ts"}
    live_send.assert_awaited_once()
    assert live_send.await_args.kwargs["team_id"] == "T111"
    assert live_send.await_args.kwargs["thread_id"] == "171.1"


def test_slack_media_send_to_platform_propagates_team_id(monkeypatch, tmp_path):
    """Slack MEDIA sends bypass _send_via_adapter (plugin standalone sender is
    called directly) — the team identity must reach it on that path too."""
    _ensure_slack_mock(monkeypatch)
    media = tmp_path / "report.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\n")

    standalone = AsyncMock(return_value={"success": True, "message_id": "ts-3"})
    entry = SimpleNamespace(standalone_sender_fn=standalone, max_message_length=4000)
    monkeypatch.setattr(
        "gateway.platform_registry.platform_registry",
        SimpleNamespace(get=lambda name: entry),
    )

    result = asyncio.run(
        _send_to_platform(
            Platform.SLACK,
            SimpleNamespace(enabled=True, token="xoxb-a", extra={}),
            "C123",
            "hello",
            media_files=[(str(media), False)],
            team_id="T222",
        )
    )

    assert result == {"success": True, "message_id": "ts-3"}
    standalone.assert_awaited()
    assert standalone.await_args.kwargs["team_id"] == "T222"


def test_standalone_send_team_id_selects_matching_workspace_token(
    monkeypatch, _standalone_send
):
    """Explicit team identity must select that workspace's token deterministically.

    OAuth installs persist per-workspace tokens in slack_tokens.json keyed by
    team_id — with team_id=T222 the sender must use T222's token even though
    T111's token is first in the configured list.
    """
    import json as _json

    from hermes_constants import get_hermes_home

    (get_hermes_home() / "slack_tokens.json").write_text(
        _json.dumps(
            {
                "T111": {"token": "xoxb-team-111", "team_name": "One"},
                "T222": {"token": "xoxb-team-222", "team_name": "Two"},
            }
        )
    )

    session = _RoutingSlackSession()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: session)

    pconfig = SimpleNamespace(enabled=True, token="xoxb-team-111", extra={})
    result = asyncio.run(_standalone_send(pconfig, "C123", "hello", team_id="T222"))

    assert result["success"] is True
    post_calls = [c for c in session.calls if "chat.postMessage" in c[0]]
    assert post_calls, "expected a chat.postMessage call"
    assert all(token == "xoxb-team-222" for _, token, _ in post_calls)


def test_standalone_send_team_id_probes_env_tokens_via_auth_test(
    monkeypatch, _standalone_send
):
    """Env-configured tokens carry no local team mapping — the sender must
    verify each via auth.test and use the one belonging to the workspace."""
    session = _RoutingSlackSession(auth_teams={"xoxb-a": "T111", "xoxb-b": "T222"})
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: session)

    pconfig = SimpleNamespace(enabled=True, token="xoxb-a,xoxb-b", extra={})
    result = asyncio.run(_standalone_send(pconfig, "C123", "hello", team_id="T222"))

    assert result["success"] is True
    post_calls = [c for c in session.calls if "chat.postMessage" in c[0]]
    assert post_calls, "expected a chat.postMessage call"
    assert all(token == "xoxb-b" for _, token, _ in post_calls)


def test_standalone_workspace_probe_continues_after_one_token_error(
    monkeypatch, _standalone_send
):
    """A transient probe failure for token A must not hide matching token B."""
    session = _RoutingSlackSession(
        auth_teams={"xoxb-b": "T222"}, auth_errors={"xoxb-a"}
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: session)

    pconfig = SimpleNamespace(enabled=True, token="xoxb-a,xoxb-b", extra={})
    result = asyncio.run(_standalone_send(pconfig, "C123", "hello", team_id="T222"))

    assert result["success"] is True
    post_calls = [c for c in session.calls if "chat.postMessage" in c[0]]
    assert post_calls
    assert all(token == "xoxb-b" for _, token, _ in post_calls)


def test_standalone_send_unknown_team_id_fails_closed(monkeypatch, _standalone_send):
    """No token matching the explicit workspace must NOT fall back to tokens[0]
    — that would deliver to the wrong workspace. Fail closed with an error."""
    session = _RoutingSlackSession(auth_teams={"xoxb-a": "T111"})
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: session)

    pconfig = SimpleNamespace(enabled=True, token="xoxb-a", extra={})
    result = asyncio.run(_standalone_send(pconfig, "C123", "hello", team_id="T999"))

    assert "error" in result
    assert "T999" in result["error"]
    post_calls = [c for c in session.calls if "chat.postMessage" in c[0]]
    assert post_calls == []
