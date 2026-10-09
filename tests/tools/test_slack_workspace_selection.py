"""Workspace selection through real Slack send consumers, without network I/O."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter, _standalone_send
from tools.send_message_tool import _send_to_platform


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [False, True])
async def test_regression_pr4_live_explicit_workspace_never_falls_back(monkeypatch, available):
    """A colliding channel map cannot override explicit workspace identity."""
    primary = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "a"}))
    selected = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "b"}))
    adapter = SlackAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._app = SimpleNamespace(client=primary)
    adapter._channel_team = {"C_COLLIDING": "T_A"}
    adapter._team_clients = {"T_A": primary}
    if available:
        adapter._team_clients["T_B"] = selected
    fake_gateway = ModuleType("gateway.run")
    fake_gateway._gateway_runner_ref = lambda: SimpleNamespace(adapters={Platform.SLACK: adapter})
    monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway)

    result = await _send_to_platform(
        Platform.SLACK, SimpleNamespace(token="synthetic", extra={}),
        "C_COLLIDING", "reply", team_id="T_B",
    )

    primary.chat_postMessage.assert_not_awaited()
    if available:
        assert result.get("success") is True
        selected.chat_postMessage.assert_awaited_once()
    else:
        assert "error" in result
        assert "T_B" in result["error"]
        selected.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["text", "media"])
@pytest.mark.parametrize("credential_source", ["config", "environment"])
async def test_regression_pr4_standalone_configured_workspace_verified(
    monkeypatch, tmp_path, route, credential_source,
):
    """Configured tokens remain usable only after an auth.test workspace match."""
    from gateway.platform_registry import platform_registry
    import hermes_cli.plugins

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "synthetic-A,synthetic-B")
    calls = []
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    def post(url, *, headers, **kwargs):
        token = headers["Authorization"].removeprefix("Bearer ")
        calls.append((url.rsplit("/", 1)[-1], token))
        response = MagicMock(status=200)
        data = ({"ok": True, "team_id": "T_A" if token == "synthetic-A" else "T_B"}
                if url.endswith("auth.test") else {"ok": True, "ts": "sent"})
        response.json = AsyncMock(return_value=data)
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    session.post = MagicMock(side_effect=post)
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    fake_gateway = ModuleType("gateway.run")
    fake_gateway._gateway_runner_ref = lambda: None
    monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway)
    monkeypatch.setattr(hermes_cli.plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(platform_registry, "get", lambda name: SimpleNamespace(
        standalone_sender_fn=_standalone_send, max_message_length=4000,
    ))
    upload = AsyncMock(return_value={"ok": True, "file": {"timestamp": 1}})
    sdk_factory = MagicMock(return_value=SimpleNamespace(files_upload_v2=upload))
    monkeypatch.setattr("slack_sdk.web.async_client.AsyncWebClient", sdk_factory)
    file = tmp_path / "attachment.pdf"
    file.write_bytes(b"synthetic attachment")
    media = [(str(file), False)] if route == "media" else []
    token = "synthetic-A,synthetic-B" if credential_source == "config" else ""

    result = await _send_to_platform(
        Platform.SLACK, SimpleNamespace(token=token, extra={}), "C_B", "hello",
        team_id="T_B", media_files=media,
    )

    assert result.get("success") is True, result
    assert calls[:2] == [("auth.test", "synthetic-A"), ("auth.test", "synthetic-B")]
    if route == "media":
        sdk_factory.assert_called_once_with(token="synthetic-B")
        upload.assert_awaited_once()
    else:
        assert calls[2:] == [("chat.postMessage", "synthetic-B")]


@pytest.mark.asyncio
@pytest.mark.parametrize("verification", ["wrong_team", "api_error", "timeout"])
async def test_regression_pr4_unverified_standalone_workspace_refused(monkeypatch, tmp_path, verification):
    """Failed identity verification must not be followed by any Slack write."""
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    calls = []
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    def post(url, **kwargs):
        calls.append(url)
        assert url.endswith("auth.test"), "unverified token reached a Slack write"
        response = MagicMock(status=200)
        response.json = AsyncMock(return_value={"ok": verification != "api_error", "team_id": "T_A"})
        if verification == "timeout":
            response.json.side_effect = asyncio.TimeoutError()
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    session.post = MagicMock(side_effect=post)
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    result = await _standalone_send(
        SimpleNamespace(token="synthetic-A", extra={}), "C_COLLIDING", "reply", team_id="T_B",
    )
    assert "error" in result
    assert "T_B" in result["error"]
    assert all(url.endswith("auth.test") for url in calls)
