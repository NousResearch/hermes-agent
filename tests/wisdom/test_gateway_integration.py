from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from wisdom.integration import handle_gateway_command, maybe_capture_gateway_event


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-123",
        chat_id="chat-123",
        user_name="tester",
        chat_type="dm",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="msg-123")


def test_wisdom_is_gateway_known_command():
    from hermes_cli.commands import is_gateway_known_command, resolve_command

    assert resolve_command("wisdom").name == "wisdom"
    assert is_gateway_known_command("wisdom")


@pytest.mark.asyncio
async def test_gateway_command_returns_string_without_agent(isolated_env_db):
    gateway = SimpleNamespace(_session_key_for_source=lambda source: "session-key")
    response = await handle_gateway_command(_event("/wisdom status"), gateway)
    assert "Wisdom status" in response


@pytest.mark.asyncio
async def test_natural_explicit_capture_returns_confirmation(isolated_env_db):
    response = await maybe_capture_gateway_event(
        _event("Remember this: clients buy peace of mind."),
        _source(),
        session_key="session-key",
    )
    assert response is not None
    assert response.startswith("Captured #1")


@pytest.mark.asyncio
async def test_ordinary_chat_and_non_wisdom_slash_pass_through(isolated_env_db):
    assert await maybe_capture_gateway_event(_event("What do you think about this?"), _source()) is None
    assert await maybe_capture_gateway_event(_event("/todo now call Yash"), _source()) is None


@pytest.mark.asyncio
async def test_secret_like_natural_capture_silently_passes_through(isolated_env_db):
    response = await maybe_capture_gateway_event(
        _event("Remember this: Authorization: Bearer abcdefghijklmnopqrstuvwxyz"),
        _source(),
        session_key="session-key",
    )
    assert response is None


@pytest.mark.asyncio
async def test_natural_capture_exception_fails_open(monkeypatch, isolated_env_db):
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("wisdom.integration.capture_text", _raise)
    response = await maybe_capture_gateway_event(
        _event("Remember this: still normal chat if Wisdom fails."),
        _source(),
        session_key="session-key",
    )
    assert response is None


@pytest.mark.asyncio
async def test_command_exception_returns_concise_error(monkeypatch, isolated_env_db):
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("wisdom.integration.handle_wisdom_command", _raise)
    gateway = SimpleNamespace(_session_key_for_source=lambda source: "session-key")
    response = await handle_gateway_command(_event("/wisdom status"), gateway)
    assert response == "Wisdom command failed. Normal Hermes is still available."
